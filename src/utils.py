import torch
import torch.nn as nn
from torch.amp import autocast
from tqdm_loggable.auto import tqdm
import logging
import os
import datasets
from torch.utils.data import IterableDataset
from src.constants import (
    DATA_DIR,
    DEVICE,
    MODEL_PRECISION,
    MIN_INTERVAL_SEC,
    TEST_ENV,
    HAS_CUDA,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from itertools import islice


def load_model_and_tokenizer(path):
    """
    Load models locally because the supercloud doesn't support locking
    """
    logging.info(f"loading model from {path}")
    model_kwargs = {
        "torch_dtype": MODEL_PRECISION,
        "trust_remote_code": True,
        "cache_dir": DATA_DIR + "llm_cache/",
        "device_map": "auto",
    }

    # Get HF token from environment variables
    hf_token = os.getenv("HF_TOKEN", None)
    if hf_token:
        logging.info("Using Hugging Face authentication token")
        model_kwargs["token"] = hf_token

    # Only use flash attention if explicitly installed
    try:
        import flash_attn

        model_kwargs["attn_implementation"] = "flash_attention_2"
        logging.info("Using flash attention for faster inference")
    except ImportError:
        logging.info("Flash attention not available, using default attention")

    model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
    logging.info(f"loading tokenizer for {path}")

    tokenizer_kwargs = {
        "trust_remote_code": True,
    }

    # Add token to tokenizer if available
    if hf_token:
        tokenizer_kwargs["token"] = hf_token

    tokenizer = AutoTokenizer.from_pretrained(
        path,
        **tokenizer_kwargs,
    )
    return model, tokenizer


def save_encodings_chunk(encodings, save_path, chunk_counter):
    """
    Save a chunk of encodings to a file.

    Args:
        encodings (list): The list of encodings to save.
        save_path (str): The path to save the encodings.
        chunk_counter (int): The current chunk number for naming the file.
    """
    chunk_save_path = os.path.join(save_path, f"chunk_{chunk_counter}.pt")
    torch.save(encodings, chunk_save_path)


def tokenize_and_save_dataset(
    tokenizer, save_path, chunk_size, buffer_size, max_length
):
    """
    Tokenize a dataset and save the encodings to the specified path.

    Args:
        tokenizer: The tokenizer to use for encoding the dataset.
        save_path (str): The path to save the tokenized encodings.
        chunk_size (int): The size of chunks to save during tokenization.
        buffer_size (int): The size of the buffer to accumulate examples before tokenizing.
        max_length (int): The maximum length of the tokenized sequences.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logging.info("Tokenizing training dataset and saving encodings")
    ds = datasets.load_dataset(
        "codeparrot/github-code-clean",
        streaming=True,
        split="train",
        languages=["Python"],
        filter_languages=True,
        filter_licenses=True,
        licenses=["mit", "isc"],
        trust_remote_code=True,
        cache_dir=DATA_DIR + "datasets/",
    )
    os.makedirs(save_path, exist_ok=True)
    encodings = []
    chunk_counter = 0
    buffer = []

    logging.info(f"chunk size: {chunk_size}")

    for example in tqdm(
        islice(iter(ds), 30_000),
        desc="Tokenizing dataset",
    ):
        buffer.append(example["code"])
        if len(buffer) == buffer_size:
            encodings_batch = tokenizer(
                buffer,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
            )

            input_ids_list = encodings_batch["input_ids"].tolist()
            attention_mask_list = encodings_batch["attention_mask"].tolist()

            for input_ids, attention_mask in zip(input_ids_list, attention_mask_list):
                encodings.append(
                    {
                        "input_ids": torch.tensor(input_ids),
                        "attention_mask": torch.tensor(attention_mask),
                    }
                )
            buffer = []

        if len(encodings) >= chunk_size:
            logging.info(f"saving chunk {chunk_counter}.")
            save_encodings_chunk(encodings, save_path, chunk_counter)
            encodings = []
            chunk_counter += 1

    if buffer:
        encodings_batch = tokenizer(
            buffer,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )

        input_ids_list = encodings_batch["input_ids"].tolist()
        attention_mask_list = encodings_batch["attention_mask"].tolist()

        for input_ids, attention_mask in zip(input_ids_list, attention_mask_list):
            encodings.append(
                {
                    "input_ids": torch.tensor(input_ids),
                    "attention_mask": torch.tensor(attention_mask),
                }
            )

    if encodings:
        save_encodings_chunk(encodings, save_path, chunk_counter)


def load_encodings_from_files(encoding_path, batch_size, max_length, num_chunks):
    """
    Yields batches of encodings for training.
    """
    logging.info(f"Loading pretokenized encodings from {encoding_path}")
    encoding_files = sorted(
        [
            os.path.join(encoding_path, f)
            for f in os.listdir(encoding_path)
            if f.endswith(".pt")
        ]
    )

    # Ensure we have enough chunks
    if len(encoding_files) < num_chunks:
        logging.warning(
            f"Requested {num_chunks} chunks but only found {len(encoding_files)}."
        )
        num_chunks = len(encoding_files)

    # Use the first num_chunks files
    encoding_files = encoding_files[:num_chunks]

    batch_input_ids = []
    batch_attention_mask = []
    for encoding_file in tqdm(
        encoding_files, desc="Chunks", mininterval=MIN_INTERVAL_SEC
    ):
        encodings = torch.load(encoding_file, weights_only=False)
        for encoding in tqdm(encodings, desc="Training", mininterval=MIN_INTERVAL_SEC):
            batch_input_ids.append(encoding["input_ids"][:max_length].unsqueeze(0))
            batch_attention_mask.append(
                encoding["attention_mask"][:max_length].unsqueeze(0)
            )
            if len(batch_input_ids) == batch_size:
                yield {
                    "input_ids": torch.cat(batch_input_ids, dim=0),
                    "attention_mask": torch.cat(batch_attention_mask, dim=0),
                }
                batch_input_ids = []
                batch_attention_mask = []
                if TEST_ENV:
                    # In test environment, break early
                    break
    if batch_input_ids:
        yield {
            "input_ids": torch.cat(batch_input_ids, dim=0),
            "attention_mask": torch.cat(batch_attention_mask, dim=0),
        }


class StreamingTrainDataset(IterableDataset):
    def __init__(
        self,
        batch_size,
        max_length,
        num_chunks,
        encoding_dir=os.path.join(DATA_DIR, "encodings_train"),
    ):
        super().__init__()
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_chunks = num_chunks
        self.encoding_dir = encoding_dir

    def __iter__(self):
        # Stream batches from files
        return load_encodings_from_files(
            self.encoding_dir, self.batch_size, self.max_length, self.num_chunks
        )


def calculate_perplexity(
    model, tokenizer, stride=2048, max_length=4096, full_dataset=True
):
    """
    Calculate perplexity using the test dataset.

    Args:
        model: The model to evaluate
        tokenizer: The tokenizer to use
        stride: Stride for sliding window evaluation
        max_length: Maximum sequence length
        full_dataset: Whether to use the full test dataset or a subset

    Returns:
        Perplexity score
    """
    # Load the test encodings
    test_dir = os.path.join(DATA_DIR, "encodings_test")
    test_files = sorted([f for f in os.listdir(test_dir) if f.endswith(".pt")])

    if not test_files:
        logging.error("No test files found. Please run setup.py first.")
        return float("inf")

    # Load the first test file
    test_file = os.path.join(test_dir, test_files[0])
    logging.info(f"Loading test data from {test_file}")
    encodings = torch.load(test_file, weights_only=False)

    # Concatenate all input_ids
    all_input_ids = []
    for encoding in encodings:
        all_input_ids.extend(encoding["input_ids"].tolist())

    input_ids = torch.tensor([all_input_ids], device=model.device)

    if not full_dataset:
        # Use only a small portion for quick evaluation
        input_ids = input_ids[:, : input_ids.shape[1] // 20]

    # Set sequence length and compute the number of samples
    seqlen = max_length
    num_tokens = input_ids.size(1)

    # Prepare the model
    model.eval()
    nlls = []
    total_tokens = 0
    loss_fct = nn.CrossEntropyLoss(reduction="sum")

    # Evaluate the model on the dataset
    nsamples = (num_tokens - seqlen + stride) // stride  # Adjusted for overlap
    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    for i in tqdm(
        range(nsamples),
        desc="Perplexity",
        mininterval=MIN_INTERVAL_SEC,
    ):
        start_idx = i * stride
        end_idx = start_idx + seqlen
        batch = input_ids[:, start_idx:end_idx]

        with torch.no_grad():
            with autocast(device_type=device_str, dtype=MODEL_PRECISION):
                outputs = model(batch)
                lm_logits = outputs.logits

        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:]

        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        nlls.append(loss.float())
        total_tokens += shift_labels.numel()

        if TEST_ENV:  # only one iteration for testing purposes
            break

    # Compute perplexity
    total_nll = torch.stack(nlls).sum()
    ppl = torch.exp(total_nll / total_tokens)
    return ppl.item()

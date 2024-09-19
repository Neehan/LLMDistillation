import torch
import torch.nn as nn
from torch.amp import autocast
from tqdm import tqdm
import logging
import os
import datasets
from src.constants import (
    DATA_DIR,
    DEVICE,
    MODEL_PRECISION,
    MIN_INTERVAL_SEC,
    TQDM_OUTPUT,
    TEST_ENV,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from itertools import islice


def load_model_and_tokenizer(path):
    """
    load models locally because the supercloud doesn't support locking
    """
    logging.info(f"loading model from {path}")
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=MODEL_PRECISION,
        trust_remote_code=True,
        cache_dir=DATA_DIR + "llm_cache/",
    ).to(DEVICE)
    logging.info(f"loading tokenizer for {path}")
    tokenizer = AutoTokenizer.from_pretrained(
        path,
        trust_remote_code=True,  # cache_dir=DATA_DIR + "llm_cache/"
    )
    return model, tokenizer


def load_coding_dataset(
    tokenizer,
    save_path=DATA_DIR + "datasets/github_code/encodings",
    chunk_size=5000,
    batch_size=1,
    max_length=2048,
    buffer_size=5000,
    force_reload=False,
):
    """
    Load coding dataset either from pretokenized files or by tokenizing a new dataset.

    Args:
        tokenizer: The tokenizer to use for encoding the dataset.
        save_path (str): The path to save or load the dataset encodings.
        chunk_size (int): The size of chunks to save during tokenization.
        batch_size (int): The number of encodings to yield at once.
        max_length (int): The maximum length of the tokenized sequences.

    Yields:
        torch.Tensor: A batch of tokenized input IDs.
    """
    if not os.path.exists(save_path) or not os.listdir(save_path) or force_reload:
        tokenize_and_save_dataset(
            tokenizer, save_path, chunk_size, buffer_size, max_length
        )
    yield from load_encodings_from_files(save_path, batch_size, max_length)


def load_encodings_from_files(save_path, batch_size, max_length):
    """
    Load pretokenized encodings from files in the specified directory.

    Args:
        save_path (str): The path to the directory containing the encoding files.
        batch_size (int): The number of encodings to yield at once.

    Yields:
        dict: A batch of input IDs and attention masks loaded from the encoding files.
    """
    logging.info(f"Loading pretokenized encodings from {save_path}")
    encoding_files = sorted(
        [os.path.join(save_path, f) for f in os.listdir(save_path) if f.endswith(".pt")]
    )
    # skip the first one cause we ppl test on it
    assert len(encoding_files) > 1, "must tokenize at least 2 chunks"
    encoding_files = encoding_files[
        1:2
    ]  # use one chunk for training during early experiments
    batch_input_ids = []
    batch_attention_mask = []
    for encoding_file in tqdm(
        encoding_files, desc="Chunks", file=TQDM_OUTPUT, mininterval=MIN_INTERVAL_SEC
    ):
        encodings = torch.load(encoding_file, weights_only=False)
        for encoding in tqdm(
            encodings, desc="Training", file=TQDM_OUTPUT, mininterval=MIN_INTERVAL_SEC
        ):
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
                if TEST_ENV:  # only one iteration for testing purposes
                    break
    if batch_input_ids:
        yield {
            "input_ids": torch.cat(batch_input_ids, dim=0),
            "attention_mask": torch.cat(batch_attention_mask, dim=0),
        }


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

    logging.info("Tokenizing dataset and saving encodings")
    ds = datasets.load_dataset(
        "codeparrot/github-code",
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

    for example in tqdm(islice(iter(ds), 20_000), desc="Tokenizing dataset"):
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


def calculate_perplexity(model, tokenizer, stride=512, max_length=2048):
    # Load the dataset
    ds = datasets.load_dataset(
        "codeparrot/github-code",
        streaming=True,
        split="train",
        languages=["Python"],
        filter_languages=True,
        filter_licenses=True,
        licenses=["mit", "isc"],
        trust_remote_code=True,
        # Ensure DATA_DIR is defined or replace it with the desired cache directory
        cache_dir=DATA_DIR + "datasets/",
    )
    # Collect the text data
    texts = []
    for example in tqdm(islice(iter(ds), 100), desc="Loading dataset"):
        texts.append(example["code"])

    encodings = tokenizer("\n\n".join(texts), return_tensors="pt")
    input_ids = encodings["input_ids"].to(model.device)

    # Set sequence length and compute the number of samples
    seqlen = max_length
    num_tokens = input_ids.size(1)  # Assuming input_ids shape is [1, num_tokens]
    nsamples = (num_tokens + seqlen - 1) // seqlen  # Ceiling division

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
        file=TQDM_OUTPUT,
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

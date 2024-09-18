import torch
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
    yield from load_encodings_from_files(save_path, batch_size)


def load_encodings_from_files(save_path, batch_size):
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
    batch_input_ids = []
    batch_attention_mask = []
    for encoding_file in tqdm(
        encoding_files, desc="Chunks", file=TQDM_OUTPUT, mininterval=MIN_INTERVAL_SEC
    ):
        encodings = torch.load(encoding_file, weights_only=False)
        for encoding in tqdm(
            encodings, desc="Training", file=TQDM_OUTPUT, mininterval=MIN_INTERVAL_SEC
        ):
            batch_input_ids.append(encoding["input_ids"].unsqueeze(0))
            batch_attention_mask.append(encoding["attention_mask"].unsqueeze(0))
            if len(batch_input_ids) == batch_size:
                yield {
                    "input_ids": torch.cat(batch_input_ids, dim=0),
                    "attention_mask": torch.cat(batch_attention_mask, dim=0),
                }
                batch_input_ids = []
                batch_attention_mask = []
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

    for example in tqdm(islice(iter(ds), 100_000), desc="Tokenizing dataset"):
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


def calculate_perplexity(model, save_path, stride=512, max_length=2048, n_files=50):
    """
    Calculate the perplexity of the model over the dataset.

    Args:
        model: The language model.
        save_path (str): The path to the directory containing the encoding files.
        stride (int): The stride size for the sliding window.

    Returns:
        float: The calculated perplexity.
    """
    model.eval()
    encodings_dir = os.path.join(save_path, "encodings")
    encoding_file = sorted(
        [
            os.path.join(encodings_dir, f)
            for f in os.listdir(encodings_dir)
            if f.endswith(".pt")
        ]
    )[0]
    nlls = []
    total_length = 0

    encodings = torch.load(encoding_file, weights_only=False)
    input_ids_list = [e["input_ids"] for e in encodings[:n_files]]
    attention_masks_list = [e["attention_mask"] for e in encodings[:n_files]]

    # Concatenate all input_ids and attention masks
    input_ids = torch.cat(input_ids_list, dim=1).squeeze()
    attention_masks = torch.cat(attention_masks_list, dim=1).squeeze()
    l_stride = stride

    for i in tqdm(range(0, input_ids.size(0), l_stride), desc="Perplexity:"):
        begin_loc = max(i + l_stride - max_length, 0)
        end_loc = i + l_stride
        trg_len = end_loc - i  # Number of tokens to predict
        input_ids_slice = input_ids[begin_loc:end_loc]
        attention_mask_slice = attention_masks[begin_loc:end_loc]
        labels = input_ids_slice.clone()
        labels[:-trg_len] = -100  # Mask tokens not to predict

        # Ensure proper device placement
        input_ids_slice = input_ids_slice.unsqueeze(0).to(model.device)
        attention_mask_slice = attention_mask_slice.unsqueeze(0).to(model.device)
        labels = labels.unsqueeze(0).to(model.device)

        with torch.no_grad():
            outputs = model(
                input_ids_slice, attention_mask=attention_mask_slice, labels=labels
            )
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)
        total_length += trg_len

    ppl = torch.exp(torch.stack(nlls).sum() / total_length)
    return ppl.item()

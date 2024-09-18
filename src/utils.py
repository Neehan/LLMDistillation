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
    chunk_size=1 if TEST_ENV else 100_000,
    batch_size=1,
    max_length=2048,
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
    if os.path.exists(save_path) and os.listdir(save_path):
        yield from load_encodings_from_files(save_path, batch_size)
    else:
        yield from tokenize_and_save_dataset(
            tokenizer, save_path, chunk_size, batch_size, max_length
        )


def load_encodings_from_files(save_path, batch_size):
    """
    Load pretokenized encodings from files in the specified directory.

    Args:
        save_path (str): The path to the directory containing the encoding files.
        batch_size (int): The number of encodings to yield at once.

    Yields:
        torch.Tensor: A batch of input IDs loaded from the encoding files.
    """
    logging.info(f"Loading pretokenized encodings from {save_path}")
    encoding_files = sorted(
        [os.path.join(save_path, f) for f in os.listdir(save_path) if f.endswith(".pt")]
    )
    batch = []
    for encoding_file in tqdm(
        encoding_files, desc="Training", file=TQDM_OUTPUT, mininterval=MIN_INTERVAL_SEC
    ):
        encodings = torch.load(encoding_file, weights_only=False)
        for encoding in encodings:
            batch.append(encoding["input_ids"])
            if len(batch) == batch_size:
                yield torch.cat(batch, dim=0)
                batch = []
    if batch:
        yield torch.cat(batch, dim=0)


def tokenize_and_save_dataset(tokenizer, save_path, chunk_size, batch_size, max_length):
    """
    Tokenize a dataset and save the encodings to the specified path.

    Args:
        tokenizer: The tokenizer to use for encoding the dataset.
        save_path (str): The path to save the tokenized encodings.
        chunk_size (int): The size of chunks to save during tokenization.
        batch_size (int): The number of encodings to yield at once.
        max_length (int): The maximum length of the tokenized sequences.

    Yields:
        torch.Tensor: A batch of tokenized input IDs.
    """
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
    batch = []

    logging.info(f"chunk size: {chunk_size}")

    for i, example in enumerate(
        # tqdm(islice(iter(ds), 10_000), desc="Tokenizing dataset")
        tqdm(ds, desc="Tokenizing dataset")
    ):
        encoding = tokenizer(
            example["code"],
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        batch.append(encoding["input_ids"])

        if len(batch) == batch_size:
            yield torch.cat(batch, dim=0)
            batch = []

        encodings.append(encoding)
        if (i + 1) % chunk_size == 0:
            if chunk_size % 10 == 0:
                logging.info(f"tokenized chunk {chunk_size}")
            save_encodings_chunk(encodings, save_path, chunk_counter)
            encodings = []
            chunk_counter += 1

    if batch:
        yield torch.cat(batch, dim=0)

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


def calculate_perplexity(model, path, stride=1024):

    encodings_dir = os.path.join(path, "encodings")
    chunk_file = sorted(os.listdir(encodings_dir))[0]

    nlls = []

    chunk_path = os.path.join(encodings_dir, chunk_file)
    encodings = torch.load(chunk_path, weights_only=False)
    for encoding in encodings:
        input_ids = encoding["input_ids"]
        seq_len = input_ids.size(1)
        prev_end_loc = 0
        for begin_loc in tqdm(
            range(0, seq_len, stride), desc="Calculating Perplexity:", file=TQDM_OUTPUT
        ):
            end_loc = min(begin_loc + stride, seq_len)
            trg_len = (
                end_loc - prev_end_loc
            )  # may be different from stride on last loop
            input_ids_slice = input_ids[:, begin_loc:end_loc].to(model.device)
            target_ids = input_ids_slice.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids_slice, labels=target_ids)
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl

import torch
import torch.nn as nn
from tqdm import tqdm

import datasets
from constants import *
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd


def load_model_and_tokenizer(path):
    """
    load models locally because the supercloud doesn't support locking
    """
    logging.info(f"loading model from {path}")
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=MODEL_PRECISION, local_files_only=True
    ).to(DEVICE)
    logging.info(f"loading tokenizer for {path}")
    tokenizer = AutoTokenizer.from_pretrained(
        path,
        trust_remote_code=True,  # cache_dir=DATA_DIR + "llm_cache/"
        local_files_only=True,
    )
    return model, tokenizer


# def load_and_tokenize_dataset(path, split, tokenizer, dataset_name=None):
#     """
#     path: path locally, not in hf
#     split: test, train etc
#     dataset_name: some datasets have multiple versions, need to specify which
#     """

#     dataset = datasets.load_dataset(
#         path,
#         dataset_name,
#         split=split,
#         keep_in_memory=True,
#     )
#     logging.info(f"loading dataset from {path}.")
#     dataset = pd.read_parquet(path)
#     logging.info(f"tokenizing {path}.")
#     encodings = tokenizer("\n\n".join(dataset["text"].tolist()), return_tensors="pt")
#     return encodings


def prepare_and_save_chunks(path, split, tokenizer, dataset_name=None):
    """
    Tokenizes large text chunks and saves them separately to manage memory usage efficiently.

    path: path locally or the dataset identifier from Huggingface
    split: 'test', 'train' etc.
    dataset_name: name of the dataset if it has multiple versions
    tokenizer: tokenizer instance from Huggingface
    """
    encodings_dir = os.path.join(path, "encodings")
    # Load dataset in streaming mode
    dataset = datasets.load_dataset(
        path, dataset_name, split=split, streaming=True, trust_remote_code=True
    )

    large_text = []
    word_count = 0
    file_index = 0

    for example in tqdm(
        dataset,
        desc="Tokenizing Dataset",
        mininterval=3 * 60,
    ):
        large_text.append(example["text"])
        word_count += len(example["text"].split())
        # Check if the accumulated texts are roughly 500 MB in size (assuming ~1 byte per char)
        if word_count >= 64 * 64 * 1024:  # Approx 500MB
            concatenated_text = "\n\n".join(large_text)
            encoded_chunk = tokenizer(
                concatenated_text, return_tensors="pt", padding=False, truncation=False
            )
            torch.save(
                encoded_chunk,
                os.path.join(encodings_dir, f"{split}_chunk_{file_index}.pt"),
            )
            large_text = []
            word_count = 0
            file_index += 1

    if len(large_text) > 0:  # Tokenize and save any remaining text
        concatenated_text = "\n\n".join(large_text)
        encoded_chunk = tokenizer(
            concatenated_text, return_tensors="pt", padding=False, truncation=False
        )
        torch.save(
            encoded_chunk, os.path.join(encodings_dir, f"{split}_chunk_{file_index}.pt")
        )


def load_and_tokenize_dataset(
    path, split, tokenizer, max_length=1024, dataset_name=None, batch_size=128
):
    """
    Yields smaller batches of tokens from pre-tokenized and saved chunks.

    path: Local system path to the directory containing tokenized chunks
    split: 'test', 'train' etc. (used to locate the correct tokenized data)
    max_length: Maximum length of the concatenated token sequences
    """
    logging.info(f"loading dataset from {path}")
    encodings_dir = os.path.join(path, "encodings")
    # Ensure the directory exists before listing contents; if not, prepare the chunks
    if not os.path.exists(encodings_dir) or not os.listdir(encodings_dir):
        os.makedirs(encodings_dir, exist_ok=True)
        prepare_and_save_chunks(path, split, tokenizer, dataset_name)

    chunks = sorted(os.listdir(encodings_dir))  # Ensure chunks are processed in order

    # keep one out of 25 cause dataset is huge
    if "openwebtext" in path:
        chunks = list(chunks)[0::25]

    for chunk_file in tqdm(
        chunks,
        desc="Training",
        mininterval=15 * 60,
        file=TQDM_OUTPUT,
    ):
        if chunk_file.startswith(f"{split}_chunk") and chunk_file.endswith(".pt"):
            chunk_path = os.path.join(encodings_dir, chunk_file)
            chunk = torch.load(chunk_path)
            input_ids = chunk["input_ids"]
            start = 0
            while start < input_ids.size(1):
                end = start + max_length * batch_size
                if end >= input_ids.size(1):
                    break
                yield input_ids[:, start:end].reshape(batch_size, -1)
                start = end


def calculate_perplexity(model, encodings):
    model.seqlen = 1024
    model = model.eval()
    nlls = []

    nsamples = 0

    for batch in tqdm(
        encodings,
        desc="computing ppl...",
        file=TQDM_OUTPUT,
        dynamic_ncols=True,
        mininterval=5 * 60,  # seconds between two updates
    ):
        batch = batch.to(model.device)
        nsamples += batch.shape[1]

        with torch.no_grad():
            outputs = model(input_ids=batch)
            lm_logits = outputs.logits
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = batch[:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen)).item()
    return ppl

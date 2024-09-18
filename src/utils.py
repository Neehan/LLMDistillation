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
)
from transformers import AutoModelForCausalLM, AutoTokenizer


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
    tokenizer, save_path=DATA_DIR + "datasets/github_code/encodings", chunk_size=1000
):
    if os.path.exists(save_path):
        logging.info(f"Loading pretokenized encodings from {save_path}")
        encoding_files = sorted(
            [
                os.path.join(save_path, f)
                for f in os.listdir(save_path)
                if f.endswith(".pt")
            ]
        )
    else:
        logging.info("Tokenizing dataset and saving encodings")
        ds = datasets.load_dataset(
            "codeparrot/github-code",
            streaming=True,
            split="train",
            languages=["Python"],
            licenses=["mit", "isc"],
            cache_dir=DATA_DIR + "datasets/",
        )
        os.makedirs(save_path, exist_ok=True)
        encodings = []
        chunk_counter = 0
        for i, example in enumerate(
            tqdm(iter(ds).take(10_000), desc="Tokenizing dataset")
        ):
            encodings.append(tokenizer(example["code"], return_tensors="pt"))
            if (i + 1) % chunk_size == 0:
                chunk_save_path = os.path.join(save_path, f"chunk_{chunk_counter}.pt")
                torch.save(encodings, chunk_save_path)
                encodings = []
                chunk_counter += 1
        if encodings:  # Save any remaining encodings
            chunk_save_path = os.path.join(save_path, f"chunk_{chunk_counter}.pt")
            torch.save(encodings, chunk_save_path)
        encoding_files = sorted(
            [
                os.path.join(save_path, f)
                for f in os.listdir(save_path)
                if f.endswith(".pt")
            ]
        )
    return encoding_files


def calculate_perplexity(
    model, path, split, tokenizer, dataset_name=None, stride=1024, start_index=1
):
    if dataset_name is None:
        logging.info(f"computing perplexity on {path}/{split}")
    else:
        logging.info(f"computing perplexity on {path}/{dataset_name}/{split}")

    # dataset = datasets.load_dataset(path, dataset_name, split=split)
    # concatenated_text = "\n\n".join(
    #     dataset["text"]
    # )  # Concatenate all texts with separator
    # encodings = tokenizer(concatenated_text, return_tensors="pt")

    encodings_dir = os.path.join(path, "encodings")
    chunks = sorted(os.listdir(encodings_dir))[start_index::100]

    max_length = 1024  # model.config.n_positions
    nlls = []

    for chunk_file in tqdm(
        chunks,
        desc="Computing Perplexity",
        mininterval=MIN_INTERVAL_SEC,
        file=TQDM_OUTPUT,
    ):
        if chunk_file.startswith(f"{split}_chunk") and chunk_file.endswith(".pt"):
            chunk_path = os.path.join(encodings_dir, chunk_file)
            encodings = torch.load(chunk_path)
            seq_len = encodings.input_ids.size(1)
            prev_end_loc = 0
            for begin_loc in range(0, seq_len, stride):
                end_loc = min(begin_loc + max_length, seq_len)
                trg_len = (
                    end_loc - prev_end_loc
                )  # may be different from stride on last loop
                input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
                target_ids = input_ids.clone()
                target_ids[:, :-trg_len] = -100

                with torch.no_grad():
                    outputs = model(input_ids, labels=target_ids)
                    neg_log_likelihood = outputs.loss

                nlls.append(neg_log_likelihood)

                prev_end_loc = end_loc
                if end_loc == seq_len:
                    break

    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl

import torch
import torch.nn as nn
from tqdm import tqdm

# import datasets
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


def load_and_tokenize_dataset(path, split, tokenizer, dataset_name=None):
    """
    path: path locally, not in hf
    split: test, train etc
    dataset_name: some datasets have multiple versions, need to specify which
    """
    if dataset_name is None:
        path = path + f"/{split}-00000-of-00001.parquet"
    else:
        path = path + "/" + dataset_name + f"/{split}-00000-of-00001.parquet"

        # dataset = datasets.load_dataset(
        #     path, dataset_name, split=split, keep_in_memory=True
        # )
    logging.info(f"loading dataset from {path}.")
    dataset = pd.read_parquet(path)
    logging.info(f"tokenizing {path}.")
    encodings = tokenizer("\n\n".join(dataset["text"].tolist()), return_tensors="pt")
    return encodings


def calculate_perplexity(model, encodings):
    model.seqlen = 2048
    encodings = encodings.input_ids.to(model.device)
    nsamples = encodings.numel() // model.seqlen
    model = model.eval()
    nlls = []

    for i in tqdm(
        range(nsamples),
        desc="evaluating...",
        file=TQDM_OUTPUT,
        dynamic_ncols=True,
        mininterval=5 * 60,  # seconds between two updates
    ):
        batch = encodings[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(
            model.device
        )
        with torch.no_grad():
            lm_logits = model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = encodings[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][
            :, 1:
        ]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen)).item()
    return ppl

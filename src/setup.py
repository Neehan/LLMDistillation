#!/usr/bin/env python3
"""
Setup script for Skip Layer Distillation.

This script:
1. Downloads the dataset
2. Tokenizes the data
3. Splits it into train and test sets
4. Saves the tokenized data to disk
"""

import os
import torch
import logging
import argparse
from pathlib import Path
from itertools import islice
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset
from dotenv import load_dotenv

# Default configuration values
DEFAULT_LANGUAGE = "Python"
DEFAULT_LICENSES = ["mit", "isc"]
DEFAULT_MODEL = "microsoft/Phi-3-mini-128k-instruct"
DEFAULT_TRAIN_SAMPLES = 50_000
DEFAULT_TEST_SAMPLES = 10_000
DEFAULT_TRAIN_CHUNK_SIZE = 5000
DEFAULT_MAX_LENGTH = 4096
DEFAULT_BUFFER_SIZE = 5000

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Constants
DATA_DIR = "data/"
TRAIN_ENCODINGS_DIR = os.path.join(DATA_DIR, "encodings_train")
TEST_ENCODINGS_DIR = os.path.join(DATA_DIR, "encodings_test")
LOGS_DIR = os.path.join(DATA_DIR, "logs")
DATASETS_CACHE_DIR = os.path.join(DATA_DIR, "datasets")
# Get HF token from environment variables
HF_TOKEN = os.getenv("HF_TOKEN", None)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Setup for Skip Layer Distillation")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Model name or path for tokenizer",
    )
    parser.add_argument(
        "--train_samples",
        type=int,
        default=DEFAULT_TRAIN_SAMPLES,
        help="Number of files for training",
    )
    parser.add_argument(
        "--test_samples",
        type=int,
        default=DEFAULT_TEST_SAMPLES,
        help="Number of files for testing",
    )
    parser.add_argument(
        "--train_chunk_size",
        type=int,
        default=DEFAULT_TRAIN_CHUNK_SIZE,
        help="Size of each training data chunk",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help="Maximum sequence length for tokenization",
    )
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=DEFAULT_BUFFER_SIZE,
        help="Buffer size for tokenization",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even if data already exists",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=DEFAULT_LANGUAGE,
        help="Programming language to filter for",
    )
    parser.add_argument(
        "--licenses",
        type=str,
        nargs="+",
        default=DEFAULT_LICENSES,
        help="Licenses to filter for",
    )
    return parser.parse_args()


def load_tokenizer(model_name):
    """Load tokenizer for the specified model."""
    logging.info(f"Loading tokenizer for {model_name}")

    # Check if we have a token for gated models
    if HF_TOKEN:
        logging.info("Using Hugging Face authentication token")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=HF_TOKEN,
        )
    else:
        # Fallback to a public model if no token is provided
        if "Phi-3" in model_name and not HF_TOKEN:
            logging.warning(
                "No HF_TOKEN provided for gated model. Using a fallback public model."
            )
            fallback_model = "gpt2"
            logging.info(f"Falling back to {fallback_model}")
            tokenizer = AutoTokenizer.from_pretrained(
                fallback_model,
                trust_remote_code=True,
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
            )

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def save_encodings_chunk(encodings, save_dir, chunk_counter):
    """Save a chunk of encodings to a file."""
    os.makedirs(save_dir, exist_ok=True)
    chunk_save_path = os.path.join(save_dir, f"chunk_{chunk_counter}.pt")
    torch.save(encodings, chunk_save_path)
    logging.info(
        f"Saved chunk {chunk_counter} with {len(encodings)} examples to {chunk_save_path}"
    )


def tokenize_and_save_dataset(
    tokenizer, dataset_iter, num_samples, save_dir, chunk_size, buffer_size, max_length
):
    """Tokenize a dataset and save the encodings to the specified path."""
    os.makedirs(save_dir, exist_ok=True)
    encodings = []
    chunk_counter = 0
    buffer = []

    for example in tqdm(
        islice(dataset_iter, num_samples),
        desc=f"Tokenizing for {save_dir}",
        total=num_samples,
    ):
        buffer.append(example["code"])

        if len(buffer) == buffer_size:
            # Tokenize the buffer
            encodings_batch = tokenizer(
                buffer,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
            )

            # Process each example in the batch
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

        # Save chunk if it reaches the chunk size
        if len(encodings) >= chunk_size:
            save_encodings_chunk(encodings, save_dir, chunk_counter)
            encodings = []
            chunk_counter += 1

    # Process any remaining examples in the buffer
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

    # Save any remaining encodings
    if encodings:
        save_encodings_chunk(encodings, save_dir, chunk_counter)


def create_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        DATA_DIR,
        TRAIN_ENCODINGS_DIR,
        TEST_ENCODINGS_DIR,
        LOGS_DIR,
        DATASETS_CACHE_DIR,
    ]

    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            logging.info(f"Created directory: {directory}")
        else:
            logging.info(f"Directory already exists: {directory}")


def main():
    """Main function to prepare data."""
    args = parse_args()

    # Create necessary directories
    create_directories()

    # Check if data already exists
    train_files = list(Path(TRAIN_ENCODINGS_DIR).glob("chunk_*.pt"))
    test_files = list(Path(TEST_ENCODINGS_DIR).glob("chunk_*.pt"))

    if not args.force and train_files and test_files:
        logging.info(
            f"Found {len(train_files)} training chunks and {len(test_files)} test chunks."
        )
        logging.info("Data already prepared. Use --force to reprocess.")
        return

    # Load tokenizer
    tokenizer = load_tokenizer(args.model)

    # Load dataset
    logging.info("Loading dataset from HuggingFace")
    ds = load_dataset(
        "codeparrot/github-code-clean",
        streaming=True,
        split="train",
        languages=[args.language],
        filter_languages=True,
        filter_licenses=True,
        licenses=args.licenses,
        trust_remote_code=True,
        cache_dir=DATASETS_CACHE_DIR,
    )

    # Create an iterator for the dataset
    ds_iter = iter(ds)

    # Process test data first (to ensure we have a fixed test set)
    logging.info(f"Processing {args.test_samples} test samples...")
    tokenize_and_save_dataset(
        tokenizer=tokenizer,
        dataset_iter=ds_iter,
        num_samples=args.test_samples,
        save_dir=TEST_ENCODINGS_DIR,
        chunk_size=args.test_samples,  # Save test data as a single chunk
        buffer_size=args.buffer_size,
        max_length=args.max_length,
    )

    # Process training data
    logging.info(f"Processing {args.train_samples} training samples...")
    tokenize_and_save_dataset(
        tokenizer=tokenizer,
        dataset_iter=ds_iter,
        num_samples=args.train_samples,
        save_dir=TRAIN_ENCODINGS_DIR,
        chunk_size=args.train_chunk_size,
        buffer_size=args.buffer_size,
        max_length=args.max_length,
    )

    logging.info("Data preparation complete!")
    logging.info(f"Test data saved to: {TEST_ENCODINGS_DIR}")
    logging.info(f"Training data saved to: {TRAIN_ENCODINGS_DIR}")


if __name__ == "__main__":
    main()

"""
Main training loop for skip layer distillation.

This module handles the overall training process, including model loading,
distillation, and saving the resulting model.
"""

import logging
import torch
from src.constants import DATA_DIR, MODEL_PRECISION, TEST_ENV, MIN_INTERVAL_SEC
from src import utils
import gc
import copy
from src.base_distiller import BaseDistiller
from src.utils import load_model_and_tokenizer, StreamingTrainDataset
import os
from torch.utils.data import DataLoader
from accelerate import Accelerator, FullyShardedDataParallelPlugin

# Create directory for saving the student model if it doesn't exist
student_model_dir = os.path.join(DATA_DIR, "llm_cache/student_model")
if not os.path.exists(student_model_dir):
    os.makedirs(student_model_dir, exist_ok=True)
    logging.info(f"Created directory for student model: {student_model_dir}")

# Configure distributed training with FSDP
fsdp_plugin = FullyShardedDataParallelPlugin(
    sharding_strategy="FULL_SHARD",
    cpu_offload=True,
    limit_all_gathers=True,
)

# Initialize accelerator with appropriate precision
accelerator = Accelerator(
    mixed_precision="fp16" if MODEL_PRECISION == torch.float16 else "bf16",
    fsdp_plugin=fsdp_plugin,
)


def get_gpu_memory_usage():
    """
    Print current GPU memory usage statistics.
    """
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    print(f"Memory Allocated: {allocated:.2f} GB")
    print(f"Memory Reserved: {reserved:.2f} GB")


def training_loop(distiller_factory: BaseDistiller, args):
    """
    Main training loop for skip layer distillation.

    Args:
        distiller_factory: Factory function/class to create the distiller
        args: Command line arguments with training parameters
    """
    # Extract training parameters from args
    learning_rate = args.lr
    num_epochs = args.num_epochs
    max_seq_len = args.max_seq_len
    batch_size = args.batch_size
    num_data_chunks = 2  # Number of chunks to split the dataset into for streaming

    logging.info(f"Model precision: {MODEL_PRECISION}")

    # Load teacher model and tokenizer
    teacher_model, tokenizer = load_model_and_tokenizer(args.model)
    logging.info(f"Loaded teacher model: {teacher_model.__class__.__name__}")

    # Calculate teacher model's perplexity as baseline
    teacher_perplexity = utils.calculate_perplexity(teacher_model, tokenizer)
    logging.info(
        f"Teacher model's perplexity on evaluation dataset: {teacher_perplexity:.3f}"
    )

    # Create student model as a copy of teacher model
    student_model = copy.deepcopy(teacher_model)

    # Initialize distiller with student model
    distiller = distiller_factory(student_model)

    # Prepare models for distributed training
    teacher_model, student_model, distiller = accelerator.prepare(
        teacher_model, student_model, distiller
    )

    # Distill every other layer, starting from the second-to-last layer
    # We skip the last layer as it's typically the output layer
    for layer_id in range(distiller.num_layers - 2, 0, -2):
        # Create a streaming dataset and dataloader for memory efficiency
        train_dataset = StreamingTrainDataset(
            batch_size=batch_size,
            max_length=max_seq_len,
            num_chunks=num_data_chunks,
        )
        # Set batch_size=None here because dataset already yields batches
        train_dataloader = DataLoader(train_dataset, batch_size=None)
        train_dataloader = accelerator.prepare(train_dataloader)

        if TEST_ENV:
            logging.info("\nMemory usage after loading distiller:")
            get_gpu_memory_usage()

        logging.info(f"Loaded dataset for training layer {layer_id}")
        logging.info(f"Training student model layer {layer_id}")

        # Train the current layer
        student_model = distiller.train_layer_with_dataloader(
            train_dataloader,
            tokenizer,
            teacher_model,
            layer_id=layer_id,
            epochs=num_epochs,
            lr=learning_rate,
            accelerator=accelerator,
        )

        if TEST_ENV:
            logging.info("\nMemory usage after training loop:")
            get_gpu_memory_usage()

        # Cleanup to free memory
        del train_dataset
        del train_dataloader
        del distiller
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if TEST_ENV:
            logging.info("\nMemory usage after garbage collection:")
            get_gpu_memory_usage()

        # Save intermediate model checkpoint (main process only)
        if accelerator.is_main_process:
            student_model.save_pretrained(student_model_dir)
            logging.info(f"Saved intermediate checkpoint for layer {layer_id}")

    # Save final model (main process only)
    if accelerator.is_main_process:
        student_model.save_pretrained(student_model_dir)
        logging.info("Saved final student model")

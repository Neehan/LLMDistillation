# training_loop.py
import logging
import torch
from src.constants import DATA_DIR, MODEL_PRECISION, TEST_ENV
from src import utils
import gc
import copy
from src.base_distiller import BaseDistiller
from src.utils import load_model_and_tokenizer, StreamingEncodingsDataset
import os
from torch.utils.data import DataLoader
from accelerate import Accelerator, FullyShardedDataParallelPlugin

os.makedirs(DATA_DIR + "llm_cache/student_model", exist_ok=True)

fsdp_plugin = FullyShardedDataParallelPlugin(
    sharding_strategy="FULL_SHARD",
    cpu_offload=True,
    limit_all_gathers=True,
)

accelerator = Accelerator(
    mixed_precision="fp16" if MODEL_PRECISION == torch.float16 else "bf16",
    fsdp_plugin=fsdp_plugin,
)


def get_gpu_memory_usage():
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    print(f"Memory Allocated: {allocated:.2f} GB")
    print(f"Memory Reserved: {reserved:.2f} GB")


def training_loop(distiller_factory: BaseDistiller, args):
    lr = args.lr
    num_epochs = args.num_epochs
    max_seq_len = args.max_seq_len
    batch_size = args.batch_size
    num_chunks = 2  # adjust as needed

    logging.info(f"params not in training precision: {MODEL_PRECISION} bits")
    teacher_model, tokenizer = load_model_and_tokenizer(args.model)
    logging.info(teacher_model)

    ppl = utils.calculate_perplexity(teacher_model, tokenizer)
    logging.info(f"Teacher model's ppl on full dataset: {ppl:.3f}")

    student_model = copy.deepcopy(teacher_model)
    distiller = distiller_factory(student_model)

    teacher_model, student_model, distiller = accelerator.prepare(
        teacher_model, student_model, distiller
    )

    # Distill every other layer
    for layer_id in range(distiller.num_layers - 2, 0, -2):
        # Create a streaming dataset and dataloader
        train_dataset = StreamingEncodingsDataset(
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_length=max_seq_len,
            num_chunks=num_chunks,
        )
        # Set batch_size=1 here because dataset already yields batches
        train_dataloader = DataLoader(train_dataset, batch_size=None)
        train_dataloader = accelerator.prepare(train_dataloader)

        if TEST_ENV:
            logging.info("\nAFTER LOADING DISTILLER")
            get_gpu_memory_usage()

        logging.info("loaded the dataset")
        logging.info(f"Training student model {layer_id}.")

        student_model = distiller.train_layer(
            train_dataloader,
            tokenizer,
            teacher_model,
            layer_id=layer_id,
            epochs=num_epochs,
            lr=lr,
            accelerator=accelerator,
        )

        if TEST_ENV:
            logging.info("\nAFTER TRAINING LOOP")
            get_gpu_memory_usage()

        # Cleanup
        del distiller
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if TEST_ENV:
            logging.info("\nAFTER CALLING GC")
            get_gpu_memory_usage()

        if accelerator.is_main_process:
            student_model.save_pretrained(
                DATA_DIR + "llm_cache/student_model",
            )

    if accelerator.is_main_process:
        student_model.save_pretrained(
            DATA_DIR + "llm_cache/student_model",
        )

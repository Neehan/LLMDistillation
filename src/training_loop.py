import logging
import torch
import torch.nn as nn
from src.constants import DATA_DIR, MODEL_PRECISION, TEST_ENV
from src import utils
import gc
from src.base_distiller import BaseDistiller
from src.utils import load_model_and_tokenizer


def get_gpu_memory_usage():
    allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert bytes to GB
    reserved = torch.cuda.memory_reserved() / (1024**3)
    print(f"Memory Allocated: {allocated:.2f} GB")
    print(f"Memory Reserved: {reserved:.2f} GB")


def training_loop(distiller_factory: BaseDistiller, args, distiller_kwargs):
    """
    progressively distill a student model by distilling one MLP
    layer at a time and then using the resulting model as teacher
    """

    lr = args.lr
    num_epochs = args.num_epochs
    max_seq_len = args.max_seq_len
    batch_size = args.batch_size
    n_layers = args.num_layers

    logging.info(f"params not in training precision: {MODEL_PRECISION} bits")
    teacher_model, tokenizer = load_model_and_tokenizer(args.model)
    student_model = None
    logging.info(teacher_model)
    dataset_name = "datasets/github_code"

    ppl = utils.calculate_perplexity(
        teacher_model,
        tokenizer,
    )
    logging.info(f"Teacher model {n_layers - 2}'s ppl: {ppl:.3f}")

    for layer_id in range(n_layers - 2, 0, -1):
        distiller = distiller_factory(
            tokenizer,
            teacher_model,
            student_model,
            dataset_name=dataset_name,
            **distiller_kwargs,
        )

        if TEST_ENV:
            logging.info("\nAFTER LOADING DISTILLER")
            get_gpu_memory_usage()

        # Load the dataset each time cause it's a generator under the hood
        train_encodings = utils.load_coding_dataset(
            tokenizer=distiller.tokenizer, batch_size=batch_size, max_length=max_seq_len
        )

        logging.info("loaded the dataset")

        logging.info(f"Training student model {layer_id}.")

        student_model = distiller.train_layer(
            train_encodings,
            train_seq_len=max_seq_len,
            layer_id=layer_id,
            loss_fn=nn.MSELoss(),
            epochs=num_epochs,
            lr=lr,
        )

        if TEST_ENV:
            logging.info("\nAFTER TRAINING LOOP")
            # Before loading the model
            get_gpu_memory_usage()

        # del teacher_model
        del distiller

        gc.collect()  # Encourage garbage collector to release unreferenced memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear CUDA cache

        if TEST_ENV:
            logging.info("\nAFTER CALLING GC")
            get_gpu_memory_usage()

        # make current student the new teacher and create a new distiller
        # teacher_model = student_model

        torch.save(
            student_model,
            DATA_DIR + "llm_cache/model" + f"_matryoshka_student.pth",
        )

    torch.save(
        student_model,
        DATA_DIR + "llm_cache/model" + f"_matryoshka_student.pth",
    )

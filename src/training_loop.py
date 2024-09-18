import logging
import torch
import torch.nn as nn
from src.constants import DATA_DIR, TEST_ENV
from src import utils
import gc


def training_loop(teacher_model, tokenizer, distiller, dataset_name):
    """
    progressively distill a student model by distilling one MLP
    layer at a time and then using the resulting model as teacher
    """

    n_layers = distiller.num_layers

    for layer_id in range(n_layers - 1, -1, -1):
        # Load the dataset each time cause it's a generator under the hood
        train_encodings = utils.load_and_tokenize_dataset(
            DATA_DIR + dataset_name,
            "train",
            tokenizer,
            max_length=1024,
            batch_size=2 if TEST_ENV else 16,
        )

        logging.info("loaded the dataset")
        logging.info(f"Training student model {layer_id}.")

        distiller.train_layer(
            train_encodings, train_seq_len=2048, layer_id=layer_id, loss_fn=nn.MSELoss()
        )

        # calculate the ppl of the teacher model first
        ppl = utils.calculate_perplexity(
            teacher_model,
            DATA_DIR + dataset_name,
            "train",
            tokenizer,
            stride=1024,
            start_index=1,
        )
        logging.info(f"Teacher model {layer_id} ppl: {ppl:.3f}")

        # Use BaseDistiller to handle training
        student_model = distiller.train_layer(
            train_seq_len=2048,
            layer_id=layer_id,
            loss_fn=nn.MSELoss(),
            epochs=1,
            lr=0.0004,
        )

        # Save the model state dictionary
        torch.save(
            student_model,
            DATA_DIR + model_path + f"_matryoshka_student_{layer_id}.pth",
        )

        # delete current teacher which we don't need anymore
        del teacher_model
        gc.collect()  # Encourage garbage collector to release unreferenced memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear CUDA cache
        # make current student the new teacher
        teacher_model = student_model

    # compute the final student model ppl
    ppl = utils.calculate_perplexity(
        teacher_model,
        DATA_DIR + dataset_name,
        "train",
        tokenizer,
        stride=1024,
        start_index=1,
    )
    logging.info(f"Teacher model {layer_id} ppl: {ppl:.3f}")

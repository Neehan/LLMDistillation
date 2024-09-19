import logging
import torch
import torch.nn as nn
from src.constants import DATA_DIR, MODEL_PRECISION
from src import utils
import gc
from src.base_distiller import BaseDistiller
from src.utils import load_model_and_tokenizer


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
    logging.info(teacher_model)
    dataset_name = "datasets/github_code"

    for layer_id in range(n_layers - 2, 0, -1):
        distiller = distiller_factory(
            teacher_model, tokenizer, dataset_name=dataset_name, **distiller_kwargs
        )

        # Load the dataset each time cause it's a generator under the hood
        train_encodings = utils.load_coding_dataset(
            tokenizer=distiller.tokenizer, batch_size=batch_size, max_length=max_seq_len
        )

        logging.info("loaded the dataset")

        # calculate the ppl of the teacher model first
        # ppl = utils.calculate_perplexity(
        #     distiller.teacher_model,
        #     distiller.tokenizer,
        # )
        # logging.info(f"Teacher model {layer_id} ppl: {ppl:.3f}")

        logging.info(f"Training student model {layer_id}.")

        student_model = distiller.train_layer(
            train_encodings,
            train_seq_len=max_seq_len,
            layer_id=layer_id,
            loss_fn=nn.MSELoss(),
            epochs=num_epochs,
            lr=lr,
        )

        del teacher_model
        del distiller
        # make current student the new teacher and create a new distiller
        teacher_model = student_model

        # Save the model state dictionary
        # torch.save(
        #     student_model,
        #     DATA_DIR + "llm_cache/model" + f"_matryoshka_student_{layer_id}.pth",
        # )

        gc.collect()  # Encourage garbage collector to release unreferenced memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear CUDA cache

    # compute the final student model ppl
    ppl = utils.calculate_perplexity(
        distiller.teacher_model,
        distiller.tokenizer,
    )
    logging.info(f"Teacher model {layer_id} ppl: {ppl:.3f}")

    torch.save(
        student_model,
        DATA_DIR + "llm_cache/model" + f"_matryoshka_student.pth",
    )

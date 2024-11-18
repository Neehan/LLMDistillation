import logging
import torch
from src.constants import DATA_DIR, MODEL_PRECISION, TEST_ENV
from src import utils
import gc
import copy
from src.base_distiller import BaseDistiller
from src.utils import load_model_and_tokenizer


def get_gpu_memory_usage():
    allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert bytes to GB
    reserved = torch.cuda.memory_reserved() / (1024**3)
    print(f"Memory Allocated: {allocated:.2f} GB")
    print(f"Memory Reserved: {reserved:.2f} GB")


def get_teacher_logits(teacher_model, tokenizer, max_seq_len, batch_size):

    logging.info("computing teacher model's logits for the training dataset.")
    teacher_model.eval()

    train_encodings = utils.load_coding_dataset(
        tokenizer=tokenizer, batch_size=batch_size, max_length=max_seq_len
    )

    all_logits = []
    with torch.no_grad():
        for batch in train_encodings:
            input_ids = batch["input_ids"].to(teacher_model.device)
            attention_mask = batch["attention_mask"].to(teacher_model.device)

            outputs = teacher_model(input_ids, attention_mask)
            logits = outputs.logits
            all_logits.append(logits)

    return all_logits


def training_loop(distiller_factory: BaseDistiller, args):
    """
    progressively distill a student model by distilling one MLP
    layer at a time and then using the resulting model as teacher
    """

    lr = args.lr
    num_epochs = args.num_epochs
    max_seq_len = args.max_seq_len
    batch_size = args.batch_size

    logging.info(f"params not in training precision: {MODEL_PRECISION} bits")
    teacher_model, tokenizer = load_model_and_tokenizer(args.model)
    logging.info(teacher_model)

    ppl = utils.calculate_perplexity(
        teacher_model,
        tokenizer,
    )
    logging.info(f"Teacher model's ppl on full dataset: {ppl:.3f}")

    # compute teacher logits only once
    # teacher_logits = get_teacher_logits(
    #     teacher_model, tokenizer, max_seq_len, batch_size
    # )

    # now train a student model initialized as teacher
    student_model = copy.deepcopy(teacher_model)

    distiller = distiller_factory(student_model)

    # distill every other layer
    for layer_id in range(distiller.num_layers - 2, 0, -2):
        distiller = distiller_factory(student_model)

        if TEST_ENV:
            logging.info("\nAFTER LOADING DISTILLER")
            get_gpu_memory_usage()

        # Load the dataset each time cause it's a generator under the hood
        train_encodings = utils.load_coding_dataset(
            tokenizer=tokenizer, batch_size=batch_size, max_length=max_seq_len
        )

        logging.info("loaded the dataset")

        logging.info(f"Training student model {layer_id}.")

        student_model = distiller.train_layer(
            train_encodings,
            tokenizer,
            teacher_model,
            layer_id=layer_id,
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

        student_model.save_pretrained(
            DATA_DIR + "llm_cache/model" + f"_student.pth",
        )

    student_model.save_pretrained(
        DATA_DIR + "llm_cache/model" + f"_student.pth",
    )

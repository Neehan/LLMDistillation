import torch
import logging
import gc

from constants import *
import utils
from train import train


teacher_model, tokenizer = utils.load_model_and_tokenizer(
    DATA_DIR + "llm_cache/models--microsoft--phi-1_5"
)

# Load the dataset
train_encodings = utils.load_and_tokenize_dataset(
    DATA_DIR + "datasets/wikitext", "train", tokenizer, "wikitext-2-raw-v1"
)
test_encodings = utils.load_and_tokenize_dataset(
    DATA_DIR + "datasets/wikitext", "test", tokenizer, "wikitext-2-raw-v1"
)

n_layers = len(teacher_model.model.layers)

for i in range(n_layers):
    logging.info(f"Training student model {i}.")
    # student model's i-th layer's MLP has been shrunk and rest of the layers are identical to teacher model.
    # we can use this student model to train the next student model whose next layer will be shrunk
    student_model = train(
        teacher_model, train_encodings, layer_id=i, epochs=1, lr=0.0004
    ).to(MODEL_PRECISION)

    # Save the model state dictionary
    torch.save(
        student_model.state_dict(),
        DATA_DIR + "llm_cache/models--microsoft--phi-1_5_student.pth",
    )
    ppl = utils.calculate_perplexity(student_model, test_encodings)
    logging.info(f"Student model {i} ppl: {ppl:.3f}")

    # delete current teacher which we don't need anymore
    del teacher_model
    gc.collect()  # Encourage garbage collector to release unreferenced memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clear CUDA cache
    # make current student the new teacher
    teacher_model = student_model

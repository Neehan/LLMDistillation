import torch
import logging

from constants import *
import utils
from train import train


model, tokenizer = utils.load_model_and_tokenizer(
    DATA_DIR + "llm_cache/models--microsoft--phi-1_5"
)

# Load the dataset
train_encodings = utils.load_and_tokenize_dataset(
    DATA_DIR + "datasets/wikitext", "train", tokenizer, "wikitext-2-raw-v1"
)
test_encodings = utils.load_and_tokenize_dataset(
    DATA_DIR + "datasets/wikitext", "test", tokenizer, "wikitext-2-raw-v1"
)


student_model = train(model, train_encodings, layer_id=0, epochs=1, lr=0.0004).to(
    MODEL_PRECISION
)

# Save the model state dictionary
torch.save(
    student_model.state_dict(),
    DATA_DIR + "llm_cache/models--microsoft--phi-1_5_student_layer_0.pth",
)

ppl = utils.calculate_perplexity(student_model, test_encodings)
logging.info(f"student model ppl: {ppl:.3f}")

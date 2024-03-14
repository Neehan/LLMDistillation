import os
import torch
import logging

from constants import *
import utils
from train import train


model, tokenizer = utils.load_model_and_tokenizer("microsoft/phi-1_5")

logging.info("running a dummy prompt to test the model inference")

# inputs = tokenizer(
#     '''def print_prime(n):
#    """
#    Print all primes between 1 and n
#    """''',
#     return_tensors="pt",
#     return_attention_mask=False,
# ).to(DEVICE)

# outputs = model.generate(**inputs, max_length=200)
# text = tokenizer.batch_decode(outputs)[0]

# logging.info(text)

# for name, module in model.named_children():
#     logging.info(name, str(module))

# Load the dataset
train_encodings = utils.load_and_tokenize_dataset(
    "wikitext", "train", tokenizer, "wikitext-2-raw-v1"
)
test_encodings = utils.load_and_tokenize_dataset(
    "wikitext", "test", tokenizer, "wikitext-2-raw-v1"
)


student_model = train(model, train_encodings, layer_id=0, epochs=1, lr=0.0004).to(
    torch.float32
)

ppl = utils.calculate_perplexity(student_model, test_encodings)
logging.info(f"student model ppl: {ppl:.3f}")

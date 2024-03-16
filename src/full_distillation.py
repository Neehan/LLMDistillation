import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import gc
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import logging
import utils
from constants import *


def train(
    teacher_model,
    token_encodings,
    epochs=1,
    lr=0.0004,
    temperature=1.1,
    trainable_attention=False,
    load_student_from_file=DATA_DIR
    + "llm_cache/models--microsoft--phi-1_5_progressive_student.pth",
):
    device = teacher_model.device
    student_model = copy.deepcopy(teacher_model)
    for layer_id in range(len(student_model.model.layers)):
        student_model.model.layers[layer_id].mlp = (
            nn.Sequential(
                nn.Linear(2048, 4096, bias=True),
                nn.GELU(),
                nn.Linear(4096, 2048, bias=True),
            )
            .to(device)
            .to(torch.float16)
        )

    if load_student_from_file:
        state_dict = torch.load(
            load_student_from_file,
            map_location=torch.device("cpu"),
        )
        student_model = student_model.cpu()
        student_model.load_state_dict(state_dict)

    student_model = student_model.to(device).to(torch.float32)

    # Disable gradient updates for all parameters except for the MLP
    if not trainable_attention:
        for name, param in student_model.named_parameters():
            if f"mlp" not in name:
                param.requires_grad = False

    teacher_model.eval()
    student_model.train()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, student_model.parameters()), lr=lr
    )

    loss_fn = torch.nn.KLDivLoss(reduction="batchmean")
    input_ids = token_encodings.input_ids
    seqlen = 2048
    nsamples = input_ids.size(1) // seqlen  # Adjust based on actual shape
    for epoch in range(epochs):
        losses = []
        for i in tqdm(
            range(nsamples),
            desc=f"Epoch {epoch+1}/{epochs}",
            file=TQDM_OUTPUT,
            dynamic_ncols=True,
            mininterval=3 * 60,  # seconds between two updates
        ):
            batch = input_ids[:, i * seqlen : (i + 1) * seqlen].to(device)
            optimizer.zero_grad()

            # Process the teacher model's output
            with torch.no_grad():
                teacher_probs = F.softmax(
                    teacher_model(input_ids=batch).logits / temperature, dim=-1
                )

            if torch.cuda.is_available():
                student_log_probs = F.log_softmax(
                    student_model(input_ids=batch).logits / temperature, dim=-1
                )
                loss = loss_fn(student_log_probs, teacher_probs.to(torch.float32))
                loss.backward()
                optimizer.step()
            losses.append(loss.item())
        avg_loss = sum(losses) / len(losses)
        logging.info(f"Average Loss: {avg_loss}")

    # training is done
    # have the student model same set of training params as teacher
    # Assuming teacher_model and student_model are your model instances
    for teacher_param, student_param in zip(
        teacher_model.parameters(), student_model.parameters()
    ):
        student_param.requires_grad = teacher_param.requires_grad
    return student_model


def main(trainable_attention=False):
    """
    progressively distill a student model by distilling one MLP
    layer at a time and then using the resulting model as teacher
    """
    model_path = "llm_cache/models--microsoft--phi-1_5"
    teacher_model, tokenizer = utils.load_model_and_tokenizer(DATA_DIR + model_path)

    if trainable_attention:
        logging.info("attn and mlp layers will be trained.")
    else:
        logging.info("mlp layers will be trained.")

    # Load the dataset
    train_encodings = utils.load_and_tokenize_dataset(
        DATA_DIR + "datasets/wikitext", "train", tokenizer, "wikitext-2-raw-v1"
    )
    test_encodings = utils.load_and_tokenize_dataset(
        DATA_DIR + "datasets/wikitext", "test", tokenizer, "wikitext-2-raw-v1"
    )

    logging.info(f"Training student model")
    # student model's i-th layer's MLP has been shrunk and rest of the layers are identical to teacher model.
    # we can use this student model to train the next student model whose next layer will be shrunk
    student_model = train(
        teacher_model,
        train_encodings,
        epochs=1,
        lr=0.0004,
        trainable_attention=trainable_attention,
    ).to(MODEL_PRECISION)
    # Save the model state dictionary
    torch.save(
        student_model,
        DATA_DIR + model_path + "_full_student.pth",
    )
    ppl = utils.calculate_perplexity(student_model, test_encodings)
    logging.info(f"Student model ppl: {ppl:.3f}")

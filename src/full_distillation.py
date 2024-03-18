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

# Assume MLP_OUTPUT is a global variable for capturing outputs
MLP_OUTPUT = [0 for _ in range(24)]


def create_mlp_output_hook(layer_id):
    """
    create a hook function to capture MLP layer output
    """

    def extract_mlp_output_hook(module, input, output):
        MLP_OUTPUT[layer_id] = output

    return extract_mlp_output_hook


def train(
    teacher_model,
    token_encodings,
    epochs=1,
    lr=0.0004,
    temperature=1.1,
    trainable_attention=False,
    load_student_from_file=DATA_DIR
    + "llm_cache/models--microsoft--phi-1_5_matryoshka_student.pth",
):
    device = teacher_model.device
    teacher_model = teacher_model.to(MODEL_PRECISION)
    if load_student_from_file is None:
        student_model = copy.deepcopy(teacher_model)
        for layer_id in range(len(student_model.model.layers)):
            student_model.model.layers[layer_id].mlp = nn.Sequential(
                nn.Linear(2048, 4096, bias=True),
                nn.GELU(),
                nn.Linear(4096, 2048, bias=True),
            )

    else:
        student_model = torch.load(
            load_student_from_file,
            map_location=torch.device("cpu"),
        )

    student_model = student_model.to(device).to(torch.float32)

    # Disable gradient updates for all parameters except for the MLP
    if not trainable_attention:
        for name, param in student_model.named_parameters():
            if f"mlp" not in name:
                param.requires_grad = False

    # Register hooks to capture outputs from the teacher and student MLPs
    teacher_hooks = []
    student_hooks = []

    # create hooks to capture all outputs from the mlp layers
    for layer_id in range(24):
        teacher_hook = teacher_model.model.layers[
            layer_id
        ].resid_dropout.register_forward_hook(create_mlp_output_hook(layer_id))
        student_hook = student_model.model.layers[
            layer_id
        ].resid_dropout.register_forward_hook(create_mlp_output_hook(layer_id))

        teacher_hooks.append(teacher_hook)
        student_hooks.append(student_hook)

    teacher_model.eval()
    student_model.train()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, student_model.parameters()), lr=lr
    )

    kldiv_loss_fn = torch.nn.KLDivLoss(reduction="batchmean")
    mse_loss_fn = nn.MSELoss()

    input_ids = token_encodings.input_ids
    seqlen = 1024
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
            try:
                batch = input_ids[:, i * seqlen : (i + 1) * seqlen].to(device)
                optimizer.zero_grad()

                # Process the teacher model's output
                with torch.no_grad():
                    teacher_probs = F.softmax(
                        teacher_model(input_ids=batch).logits / temperature, dim=-1
                    )
                    teacher_mlp_outputs = torch.stack(MLP_OUTPUT, dim=0).to(
                        torch.float32
                    )  # shallow copy

                student_log_probs = F.log_softmax(
                    student_model(input_ids=batch).logits / temperature, dim=-1
                )
                student_mlp_outputs = torch.stack(MLP_OUTPUT, dim=0)

                mse_loss = mse_loss_fn(student_mlp_outputs, teacher_mlp_outputs)
                kldiv_loss = kldiv_loss_fn(
                    student_log_probs, teacher_probs.to(torch.float32)
                )
                loss = 3000 * mse_loss + kldiv_loss
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            except Exception as e:
                logging.error("GOT AN EXCEPTION")
                logging.error(e)
                # remove hooks to prevent memory leaks
                for hook in teacher_hooks + student_hooks:
                    hook.remove()
                raise e
        avg_loss = sum(losses) / len(losses)
        logging.info(f"Average Loss: {avg_loss}")

    for hook in teacher_hooks + student_hooks:
        hook.remove()

    # training is done
    # have the student model same set of training params as teacher
    # Assuming teacher_model and student_model are your model instances
    for teacher_param, student_param in zip(
        teacher_model.parameters(), student_model.parameters()
    ):
        student_param.requires_grad = teacher_param.requires_grad

    # reduce precision to save memory
    student_model = student_model.to(device).to(torch.float16)

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
        load_student_from_file=None,
    ).to(MODEL_PRECISION)
    # Save the model state dictionary
    torch.save(
        student_model,
        DATA_DIR + model_path + "_full_student.pth",
    )
    ppl = utils.calculate_perplexity(student_model, test_encodings)
    logging.info(f"Student model ppl: {ppl:.3f}")

import torch
import torch.nn as nn
import copy
import gc
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import logging
import utils
from constants import *


# Assume MLP_OUTPUT is a global variable for capturing outputs
MLP_OUTPUT = None


def extract_mlp_output_hook(module, input, output):
    global MLP_OUTPUT
    MLP_OUTPUT = output


def train(
    teacher_model,
    token_encodings,
    layer_id=0,
    epochs=1,
    lr=0.0004,
    trainable_attention=False,
):
    device = teacher_model.device
    teacher_model = teacher_model.to(MODEL_PRECISION)

    student_model = copy.deepcopy(teacher_model)
    student_model.model.layers[layer_id].mlp = (
        nn.Sequential(
            nn.Linear(2048, 4096, bias=True),
            nn.GELU(),
            nn.Linear(4096, 2048, bias=True),
        )
        .to(device)
        .to(torch.float32)
    )
    if trainable_attention:
        student_model.model.layers[layer_id].self_attn = student_model.model.layers[
            layer_id
        ].self_attn.to(torch.float32)

        # saves memory by forgetting activations during forward pass
        student_model.gradient_checkpointing_enable()

        # Disable gradient updates for all parameters except for layer_id
        for name, param in student_model.named_parameters():
            if (
                f"model.layers.{layer_id}.mlp" not in name
                and f"model.layers.{layer_id}.self_attn" not in name
            ):
                param.requires_grad = False
    else:
        # Disable gradient updates for all parameters except for the MLP
        for name, param in student_model.named_parameters():
            if f"model.layers.{layer_id}.mlp" not in name:
                param.requires_grad = False

    # Register hooks to capture outputs from the teacher and student MLPs
    teacher_hook = teacher_model.model.layers[
        layer_id
    ].resid_dropout.register_forward_hook(extract_mlp_output_hook)
    student_hook = student_model.model.layers[
        layer_id
    ].resid_dropout.register_forward_hook(extract_mlp_output_hook)

    teacher_model.eval()
    student_model.train()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, student_model.parameters()), lr=lr
    )
    loss_fn = nn.MSELoss()
    if torch.cuda.is_available():
        scaler = GradScaler()  # Initialize the GradScaler for handling mixed precision

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
            try:
                batch = input_ids[:, i * seqlen : (i + 1) * seqlen].to(device)
                optimizer.zero_grad()

                # Process the teacher model's output
                with torch.no_grad():
                    teacher_model(input_ids=batch)
                    teacher_mlp_output = MLP_OUTPUT

                if torch.cuda.is_available():
                    # Use autocast for the forward pass to manage mixed precision
                    with autocast():
                        student_model(input_ids=batch)
                        student_mlp_output = MLP_OUTPUT
                        loss = loss_fn(
                            student_mlp_output.float(), teacher_mlp_output.float()
                        )

                    # Scale loss and backward
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    student_model(input_ids=batch)
                    student_mlp_output = MLP_OUTPUT
                    loss = loss_fn(
                        student_mlp_output.float(), teacher_mlp_output.float()
                    )
                    loss.backward()
                    optimizer.step()

                losses.append(loss.item())

            except Exception as e:
                logging.error("GOT AN EXCEPTION")
                logging.error(e)
                # remove hooks to prevent memory leaks
                teacher_hook.remove()
                student_hook.remove()
                raise e

        avg_loss = sum(losses) / len(losses)
        logging.info(f"Average Loss: {avg_loss}")

    # remove hooks to prevent memory leaks
    teacher_hook.remove()
    student_hook.remove()

    # training is done
    # have the student model same set of training params as teacher
    # Assuming teacher_model and student_model are your model instances
    for teacher_param, student_param in zip(
        teacher_model.parameters(), student_model.parameters()
    ):
        student_param.requires_grad = teacher_param.requires_grad

    # reduce precision again to save memory
    student_model = student_model.to(device).to(MODEL_PRECISION)

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

    n_layers = len(teacher_model.model.layers)

    for i in range(n_layers):
        logging.info(f"Training student model {i}.")
        # student model's i-th layer's MLP has been shrunk and rest of the layers are identical to teacher model.
        # we can use this student model to train the next student model whose next layer will be shrunk
        student_model = train(
            teacher_model,
            train_encodings,
            layer_id=i,
            epochs=1,
            lr=0.0004,
            trainable_attention=trainable_attention,
        ).to(MODEL_PRECISION)

        # Save the model state dictionary
        torch.save(
            student_model,
            DATA_DIR + model_path + "_progressive_student.pth",
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

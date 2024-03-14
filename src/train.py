import torch
import torch.nn as nn
import copy
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import logging
from constants import *


# Assume MLP_OUTPUT is a global variable for capturing outputs
MLP_OUTPUT = None


def extract_mlp_output_hook(module, input, output):
    global MLP_OUTPUT
    MLP_OUTPUT = output


def train(teacher_model, token_encodings, layer_id=0, epochs=1, lr=0.0004):
    device = teacher_model.device
    teacher_model.to(device)
    student_model = copy.deepcopy(teacher_model)

    student_model = student_model.to(torch.float32)
    student_model.model.layers[layer_id].mlp = (
        nn.Sequential(
            nn.Linear(2048, 4096, bias=True),
            nn.GELU(),
            nn.Linear(4096, 2048, bias=True),
        )
        .to(device)
        .to(torch.float32)
    )

    # Disable gradient updates for all parameters except for the MLP
    for name, param in student_model.named_parameters():
        if f"model.layers.{layer_id}.mlp" not in name:
            param.requires_grad = False

    # Register hooks to capture outputs from the teacher and student MLPs
    teacher_hook = teacher_model.model.layers[layer_id].mlp.register_forward_hook(
        extract_mlp_output_hook
    )
    student_hook = student_model.model.layers[layer_id].mlp.register_forward_hook(
        extract_mlp_output_hook
    )

    teacher_model.eval()
    student_model.train()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, student_model.parameters()), lr=lr
    )
    loss_fn = nn.MSELoss()
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
        ):
            try:
                if i > 1:
                    # remove hooks to prevent memory leaks
                    teacher_hook.remove()
                    student_hook.remove()
                    break

                batch = input_ids[:, i * seqlen : (i + 1) * seqlen].to(device)
                optimizer.zero_grad()

                # Process the teacher model's output
                with torch.no_grad():
                    teacher_model(input_ids=batch)
                    teacher_mlp_output = MLP_OUTPUT

                # Use autocast for the forward pass to manage mixed precision
                with autocast():
                    student_model(input_ids=batch)
                    student_mlp_output = MLP_OUTPUT
                    loss = loss_fn(
                        student_mlp_output.float(), teacher_mlp_output.float()
                    )

                # Scale loss and backward
                scaler.scale(loss).backward()
                scaler.step(optimizer)  # Optimize
                scaler.update()  # Update the scaler

                losses.append(loss.item())

            except Exception as e:
                logging.error("GOT AN EXCEPTION")
                logging.error(e)
                # remove hooks to prevent memory leaks
                teacher_hook.remove()
                student_hook.remove()

        avg_loss = sum(losses) / len(losses)
        logging.info(f"Average Loss: {avg_loss}")

    # remove hooks to prevent memory leaks
    teacher_hook.remove()
    student_hook.remove()

    return student_model

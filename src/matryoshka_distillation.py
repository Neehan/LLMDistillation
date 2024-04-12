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
MLP_OUTPUT = None

MATRYOSHKA_SIZE = 1536
# Global variable to control the size of MLP
CURRENT_MATRYOSHKA_SIZE = None  # None implies use the full model


class MatryoshkaMLP(nn.Module):
    def __init__(self, original_mlp):
        super(MatryoshkaMLP, self).__init__()
        self.original_mlp = original_mlp

    def forward(self, x):
        global CURRENT_MATRYOSHKA_SIZE
        if CURRENT_MATRYOSHKA_SIZE is not None:
            # Ensure the current size is within the allowed dimensions
            hidden_dim = self.original_mlp.c_fc.weight.shape[1]
            assert (
                CURRENT_MATRYOSHKA_SIZE <= hidden_dim
            ), f"CURRENT_MATRYOSHKA_SIZE too large ({CURRENT_MATRYOSHKA_SIZE} <= {hidden_dim})"

            # Adjust the weights and biases for the c_fc layer
            weight_fc = self.original_mlp.c_fc.weight[:, :CURRENT_MATRYOSHKA_SIZE]
            bias_fc = (
                self.original_mlp.c_fc.bias[:CURRENT_MATRYOSHKA_SIZE]
                if self.original_mlp.c_fc.bias is not None
                else None
            )

            x = F.linear(
                x, weight_fc.T, bias_fc
            )  # Using F.linear because GPT-2 uses linear mappings reshaped, not actual Conv1D
            x = self.original_mlp.act(x)
            x = self.original_mlp.dropout(x)  # Apply dropout after activation

            # Adjust the weights for the c_proj layer
            weight_proj = self.original_mlp.c_proj.weight[:CURRENT_MATRYOSHKA_SIZE, :]
            bias_proj = self.original_mlp.c_proj.bias

            # Apply the second projection layer
            x = F.linear(x, weight_proj.T, bias_proj)
            x = self.original_mlp.dropout(x)  # Apply dropout after projection
        else:
            # Use the full MLP as is
            x = self.original_mlp(x)
        return x


def extract_mlp_output_hook(module, input, output):
    global MLP_OUTPUT
    MLP_OUTPUT = output


def get_student_model_mlp_loss(student_model, input_ids, teacher_mlp_output, loss_fn):
    global CURRENT_MATRYOSHKA_SIZE
    CURRENT_MATRYOSHKA_SIZE = None
    student_model(input_ids=input_ids)
    full_student_output = MLP_OUTPUT

    CURRENT_MATRYOSHKA_SIZE = MATRYOSHKA_SIZE
    student_model(input_ids=input_ids)
    small_student_output = MLP_OUTPUT

    loss1 = loss_fn(full_student_output, teacher_mlp_output)
    loss2 = loss_fn(small_student_output, teacher_mlp_output)
    return loss1 + loss2


def copy_smaller_mlp(student_model, layer_id):
    # we have trained the Matryoshka layer, now remove it with the smaller MLP
    new_mlp = nn.Sequential(
        nn.Linear(2048, MATRYOSHKA_SIZE, bias=True),
        nn.GELU(),
        nn.Linear(MATRYOSHKA_SIZE, 2048, bias=True),
    )
    learned_mlp = student_model.transformer.h[layer_id].mlp

    new_mlp[0].weight.data = learned_mlp.original_mlp.c_fc.weight[
        :MATRYOSHKA_SIZE, :
    ].data
    new_mlp[0].bias.data = learned_mlp.original_mlp.c_fc.bias[:MATRYOSHKA_SIZE].data

    new_mlp[2].weight.data = learned_mlp.original_mlp.c_proj.weight[
        :, :MATRYOSHKA_SIZE
    ].data
    new_mlp[2].bias.data = learned_mlp.original_mlp.c_proj.bias.data
    return new_mlp


def train(
    teacher_model,
    token_encodings,
    layer_id=0,
    epochs=1,
    lr=0.0004,
    trainable_attention=False,
    batch_size=1024,
):
    device = teacher_model.device
    teacher_model = teacher_model.to(MODEL_PRECISION)

    student_model = copy.deepcopy(teacher_model)
    student_model.transformer.h[layer_id].mlp = (
        MatryoshkaMLP(student_model.transformer.h[layer_id].mlp)
        .to(device)
        .to(torch.float32)
    )

    if trainable_attention:
        student_model.transformer.h[layer_id].self_attn = student_model.transformer.h[
            layer_id
        ].self_attn.to(torch.float32)

        # saves memory by forgetting activations during forward pass
        student_model.gradient_checkpointing_enable()

        # Disable gradient updates for all parameters except for layer_id
        for name, param in student_model.named_parameters():
            if (
                f"transformer.h.{layer_id}.mlp" not in name
                and f"transformer.h.{layer_id}.self_attn" not in name
            ):
                param.requires_grad = False
    else:
        # Disable gradient updates for all parameters except for the MLP
        for name, param in student_model.named_parameters():
            if f"transformer.h.{layer_id}.mlp" not in name:
                param.requires_grad = False

    # Register hooks to capture outputs from the teacher and student MLPs
    teacher_hook = teacher_model.transformer.h[
        layer_id
    ].mlp.c_proj.register_forward_hook(extract_mlp_output_hook)
    student_hook = student_model.transformer.h[
        layer_id
    ].mlp.original_mlp.c_proj.register_forward_hook(extract_mlp_output_hook)

    teacher_model.eval()
    student_model.train()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, student_model.parameters()), lr=lr
    )
    loss_fn = nn.MSELoss()
    if torch.cuda.is_available():
        scaler = GradScaler()  # Initialize the GradScaler for handling mixed precision

    for epoch in range(epochs):
        losses = []
        loss = 0
        batch_counter = 0
        batch_generator = tqdm(
            token_encodings,
            desc="Processing batches",
            file=TQDM_OUTPUT,
            dynamic_ncols=True,
            mininterval=3 * 60,
        )
        for encoding in batch_generator:
            try:
                batch = encoding["input_ids"].to(device)
                optimizer.zero_grad()

                # Process the teacher model's output
                with torch.no_grad():
                    teacher_model(input_ids=batch)
                    teacher_mlp_output = MLP_OUTPUT

                if torch.cuda.is_available():
                    # Use autocast for the forward pass to manage mixed precision
                    with autocast():
                        loss += get_student_model_mlp_loss(
                            student_model, batch, teacher_mlp_output, loss_fn
                        )

                    if batch_counter >= batch_size:
                        # Scale loss and backward
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                        loss = 0
                        batch_counter = 0
                    else:
                        batch_counter += 1

                else:
                    loss += get_student_model_mlp_loss(
                        student_model, batch, teacher_mlp_output, loss_fn
                    )
                    if batch_counter >= batch_size:
                        loss.backward()
                        optimizer.step()
                        loss = 0
                        batch_counter = 0
                    else:
                        batch_counter += 1

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

    student_model.transformer.h[layer_id].mlp = copy_smaller_mlp(
        student_model, layer_id
    )
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


def main(model_path, trainable_attention=False):
    """
    progressively distill a student model by distilling one MLP
    layer at a time and then using the resulting model as teacher
    """
    teacher_model, tokenizer = utils.load_model_and_tokenizer(DATA_DIR + model_path)

    if trainable_attention:
        logging.info("attn and mlp layers will be trained.")
    else:
        logging.info("mlp layers will be trained.")

    # Load the dataset
    train_encodings = utils.load_and_tokenize_dataset(
        DATA_DIR + "datasets/openwebtext",
        "train",
        tokenizer,
        max_length=1024,
        # dataset_name="wikitext-2-raw-v1",
    )

    test_encodings = utils.load_and_tokenize_dataset(
        DATA_DIR + "datasets/wikitext",
        "test",
        tokenizer,
        max_length=1024,
        dataset_name="wikitext-103-raw-v1",
    )

    n_layers = len(teacher_model.transformer.h)

    for i in range(n_layers - 1, -1, -1):
        logging.info(f"Training student model {n_layers - 1 - i}.")
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
            DATA_DIR + model_path + f"_matryoshka_student_{i}.pth",
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

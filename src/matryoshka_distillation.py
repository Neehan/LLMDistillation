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
CURRENT_MATRYOSHKA_SIZE = [None] * 12  # None implies use the full model


class MatryoshkaMLP(nn.Module):
    def __init__(self, original_mlp, layer_id):
        super(MatryoshkaMLP, self).__init__()
        self.original_mlp = original_mlp
        self.layer_id = layer_id

    def forward(self, x):
        distilled_hidden_size = CURRENT_MATRYOSHKA_SIZE[self.layer_id]
        if distilled_hidden_size is not None:
            # Ensure the current size is within the allowed dimensions
            hidden_dim = self.original_mlp.dense_h_to_4h.weight.shape[1]
            assert (
                distilled_hidden_size <= hidden_dim
            ), f"distilled_hidden_size too large ({distilled_hidden_size} <= {hidden_dim})"

            # Adjust the weights and biases for the dense_h_to_4h layer
            weight_fc = self.original_mlp.dense_h_to_4h.weight[
                :, :distilled_hidden_size
            ]
            bias_fc = (
                self.original_mlp.dense_h_to_4h.bias[:distilled_hidden_size]
                if self.original_mlp.dense_h_to_4h.bias is not None
                else None
            )

            x = F.linear(
                x, weight_fc.T, bias_fc
            )  # Using F.linear because GPT-2 uses linear mappings reshaped, not actual Conv1D
            x = self.original_mlp.act(x)

            # Adjust the weights for the dense_4h_to_h layer
            weight_proj = self.original_mlp.dense_4h_to_h.weight[
                :distilled_hidden_size, :
            ]
            bias_proj = self.original_mlp.dense_4h_to_h.bias

            # Apply the second projection layer
            x = F.linear(x, weight_proj.T, bias_proj)
            # x = self.original_mlp.dropout(x)  # Apply dropout after projection
        else:
            # Use the full MLP as is
            x = self.original_mlp(x)
        return x


def extract_mlp_output_hook(module, input, output):
    global MLP_OUTPUT
    MLP_OUTPUT = output


def get_student_model_mlp_loss(
    student_model,
    layer_id,
    input_ids,
    teacher_mlp_output,
    teacher_model_logits,
    loss_fn,
):

    # get output with full sized hidden
    CURRENT_MATRYOSHKA_SIZE[layer_id] = None
    full_student_model_logits = student_model(input_ids=input_ids).logits
    full_student_output = MLP_OUTPUT

    # get output with small sized hidden on only layer_id
    CURRENT_MATRYOSHKA_SIZE[layer_id] = MATRYOSHKA_SIZE
    small_student_model_logits = student_model(input_ids=input_ids).logits
    small_student_output = MLP_OUTPUT

    # revert to full sized hidden
    CURRENT_MATRYOSHKA_SIZE[layer_id] = None

    loss1 = loss_fn(full_student_output, teacher_mlp_output)
    loss2 = loss_fn(small_student_output, teacher_mlp_output)

    # also add final two layers outputs
    # loss3 = loss_fn(teacher_model_logits, full_student_model_logits)
    # loss4 = loss_fn(teacher_model_logits, small_student_model_logits)

    return loss1 + loss2  # + (loss3 + loss4) * 0.01


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
    student_model.gpt_neox.layers[layer_id].mlp = (
        MatryoshkaMLP(student_model.gpt_neox.layers[layer_id].mlp, layer_id)
        .to(device)
        .to(torch.float32)
    )

    # Disable gradient updates for all parameters except for the MLP
    for name, param in student_model.named_parameters():
        if f"gpt_neox.layers.{layer_id}.mlp" not in name:
            param.requires_grad = False

    # Register hooks to capture outputs from the teacher and student MLPs
    teacher_hook = teacher_model.gpt_neox.layers[layer_id].mlp.register_forward_hook(
        extract_mlp_output_hook
    )
    student_hook = student_model.gpt_neox.layers[layer_id].mlp.register_forward_hook(
        extract_mlp_output_hook
    )

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
        for encoding_id, encoding in enumerate(token_encodings):
            # compute perplexity from time to time
            if encoding_id % 2048 == 2047:
                logging.info("calculating intermediate perplexity.")
                ppl = utils.calculate_perplexity(
                    student_model,
                    DATA_DIR + "datasets/python-github-code",
                    "train",
                    tokenizer=None,
                    stride=1024,
                    start_index=1,
                )
                logging.info(f"Sudent model's ppl: {ppl:.3f}")

            try:
                batch = encoding.to(device)
                optimizer.zero_grad()

                # Process the teacher model's output
                with torch.no_grad():
                    teacher_model_logits = teacher_model(input_ids=batch).logits
                    teacher_mlp_output = MLP_OUTPUT

                if torch.cuda.is_available():
                    # Use autocast for the forward pass to manage mixed precision
                    with autocast():
                        loss = get_student_model_mlp_loss(
                            student_model,
                            layer_id,
                            batch,
                            teacher_mlp_output,
                            teacher_model_logits,
                            loss_fn,
                        )

                    # Scale loss and backward
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss = get_student_model_mlp_loss(
                        student_model,
                        layer_id,
                        batch,
                        teacher_mlp_output,
                        teacher_model_logits,
                        loss_fn,
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

    # have the student model same set of training params as teacher
    for teacher_param, student_param in zip(
        teacher_model.named_parameters(), student_model.named_parameters()
    ):
        student_param[1].requires_grad = teacher_param[1].requires_grad

    # reduce precision again to save memory
    student_model = student_model.to(device).to(MODEL_PRECISION)
    return student_model


def main(model_path, trainable_attention=False):
    """
    progressively distill a student model by distilling one MLP
    layer at a time and then using the resulting model as teacher
    """
    teacher_model, tokenizer = utils.load_model_and_tokenizer(DATA_DIR + model_path)

    print(teacher_model)

    if trainable_attention:
        logging.info("attn and mlp layers will be trained.")
    else:
        logging.info("mlp layers will be trained.")

    n_layers = len(teacher_model.gpt_neox.layers)

    for i in range(n_layers - 1, -1, -1):
        # Load the dataset each time cause it's a generator under the hood
        train_encodings = utils.load_and_tokenize_dataset(
            DATA_DIR + "datasets/python-github-code",
            "train",
            tokenizer,
            max_length=1024,
            batch_size=2 if TEST_ENV else 16,
            # dataset_name=,
        )

        logging.info("loaded the dataset")

        # make the teacher get output using smaller hidden size
        for j in range(i, n_layers - 1):
            CURRENT_MATRYOSHKA_SIZE[j] = MATRYOSHKA_SIZE

        # revert the changes
        for j in range(i, n_layers - 1):
            CURRENT_MATRYOSHKA_SIZE[j] = None

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
        )

        ppl = utils.calculate_perplexity(
            teacher_model,
            # DATA_DIR + "datasets/wikitext",
            DATA_DIR + "datasets/python-github-code",
            "train",
            tokenizer,
            stride=1024,
            start_index=1,
        )
        logging.info(f"Teacher model {i} ppl: {ppl:.3f}")

        # Save the model state dictionary
        torch.save(
            student_model,
            DATA_DIR + model_path + f"_matryoshka_student_{i}.pth",
        )

        # delete current teacher which we don't need anymore
        del teacher_model
        gc.collect()  # Encourage garbage collector to release unreferenced memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear CUDA cache
        # make current student the new teacher
        teacher_model = student_model

    for j in range(n_layers - 1):
        CURRENT_MATRYOSHKA_SIZE[j] = None

    # compute the final student model ppl
    ppl = utils.calculate_perplexity(
        teacher_model,
        # DATA_DIR + "datasets/wikitext",
        DATA_DIR + "datasets/python-github-code",
        "train",
        tokenizer,
        stride=1024,
        start_index=1,
    )
    logging.info(f"Teacher model {i} ppl: {ppl:.3f}")

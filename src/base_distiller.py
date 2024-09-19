import math
import torch
from torch.amp import autocast, GradScaler
import logging
from src.constants import MODEL_PRECISION
from src.utils import calculate_perplexity
import logging


class BaseDistiller:
    def __init__(self, teacher_model, tokenizer, dataset_name):
        self.teacher_model = teacher_model.to(MODEL_PRECISION)
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.device = teacher_model.device
        self.student_model = None
        self.teacher_hook = None
        self.student_hook = None
        self.num_layers = len(self.get_model_layers(teacher_model))
        self.teacher_mlp_outputs = [None] * self.num_layers
        self.student_mlp_outputs = [None] * self.num_layers

    def get_model_layers(self, model):
        raise NotImplementedError("must be implemented by the children class")

    def get_model_mlp(self, model, layer_id):
        """
        get the model mlp layer on layer with layer id
        """
        raise NotImplementedError("must be implemented by the children class")

    def prepare_student_model(self, layer_id):
        """
        replace the mlp on layer with layer_id
        disable gradients for all params except this new mlp
        """
        raise NotImplementedError("must be implemented by the children class")
        # self.student_model = copy.deepcopy(self.teacher_model)
        # self.student_model.model.layers[layer_id].mlp = (
        #     nn.Sequential(
        #         nn.Linear(2048, 4096, bias=True),
        #         nn.GELU(),
        #         nn.Linear(4096, 2048, bias=True),
        #     )
        #     .to(self.device)
        #     .to(torch.float32)
        # )

        # disable gradients for all layers except the mlp layer just replaced
        # for name, param in self.student_model.named_parameters():
        #     if f"model.layers.{layer_id}.mlp" not in name:
        #         param.requires_grad = False

    def compute_loss(self, layer_id, input_ids, attention_mask, loss_fn):
        raise NotImplementedError("must be implemented by the children class")

    def create_mlp_output_hook(self, layer_id, is_teacher):
        """
        create a hook function to capture MLP layer output
        """

        def extract_mlp_output_hook(module, input, output):
            if is_teacher:
                self.teacher_mlp_outputs[layer_id] = output
            else:
                self.student_mlp_outputs[layer_id] = output

        return extract_mlp_output_hook

    def register_hooks(self, layer_id):
        self.teacher_hook = self.get_model_mlp(
            self.teacher_model, layer_id
        ).register_forward_hook(self.create_mlp_output_hook(layer_id, is_teacher=True))
        self.student_hook = self.get_model_mlp(
            self.student_model, layer_id
        ).register_forward_hook(self.create_mlp_output_hook(layer_id, is_teacher=False))

    def remove_hooks(self):
        """
        Remove the hooks for the teacher and student models.
        """
        if self.teacher_hook:
            self.teacher_hook.remove()
        if self.student_hook:
            self.student_hook.remove()

    def train_layer(
        self,
        train_encodings,
        train_seq_len,
        layer_id,
        loss_fn,
        epochs,
        lr,
        teacher_ppl=None,
    ):
        """
        Train a specific layer of the student model.

        Parameters:
        - train_seq_len (int): The sequence length for training.
        - layer_id (int): The ID of the layer to train.
        - loss_fn (function): The loss function to calculate the training loss.
        - epochs (int): The number of epochs for training (default is 1).
        - lr (float): The learning rate for optimization (default is 0.0004).
        """
        self.prepare_student_model(layer_id)
        self.student_model = self.student_model.to(self.device)
        self.register_hooks(layer_id)

        # Get only the MLP layer parameters for the specified layer_id
        mlp_parameters = self.get_model_mlp(self.student_model, layer_id).parameters()
        # Set up the optimizer to only train the MLP layer's parameters
        optimizer = torch.optim.AdamW(mlp_parameters, lr=lr)

        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        scaler = GradScaler(device_str)

        last_student_ppl = None
        for epoch in range(epochs):
            losses = []
            for batch_idx, batch in enumerate(train_encodings):
                if batch_idx % 100 == 24:
                    logging.info(
                        f"layer {layer_id}: calculating intermediate perplexity."
                    )
                    student_ppl = calculate_perplexity(
                        self.student_model,
                        self.tokenizer,
                    )
                    logging.info(f"Sudent model's ppl: {student_ppl:.3f}")
                    if (
                        teacher_ppl is not None
                        and math.abs(teacher_ppl - student_ppl) <= 0.01
                    ):
                        logging.info(
                            f"stopping early because teacher ppl: ({teacher_ppl:.3f}) and student ppl: ({student_ppl:.3f}) are close."
                        )
                        # student is very close to the teacher so stop early
                        break

                    if last_student_ppl is None:
                        last_student_ppl = student_ppl
                    elif last_student_ppl < student_ppl:
                        # stop early because the student is getting worse
                        logging.info(
                            f"stopping early because last student ppl ({last_student_ppl:3f}) was better than current student ppl ({student_ppl:.3f})"
                        )
                        break

                try:
                    self.teacher_model.eval()
                    self.student_model.train()

                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    optimizer.zero_grad()

                    if MODEL_PRECISION == torch.float16:
                        # need to use autocast and it uses 1x memory
                        self.student_model = self.student_model.to(torch.float32)
                        with autocast(device_type=device_str, dtype=MODEL_PRECISION):
                            # hook saves the intermediate outputs to self.student_mlp_outputs
                            loss = self.compute_loss(
                                layer_id, input_ids, attention_mask, loss_fn=loss_fn
                            )
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:  # bfloat16 is safer and can be autocasted directly
                        loss = self.compute_loss(
                            layer_id, input_ids, attention_mask, loss_fn=loss_fn
                        )
                        loss.backward()

                except Exception as e:
                    logging.error("GOT AN EXCEPTION")
                    logging.error(e)
                    # remove the hooks else memory leak
                    self.remove_hooks()
                    raise e
            avg_loss = sum(losses) / len(losses)
            logging.info(f"Average Loss: {avg_loss}")

        self.remove_hooks()  # training complete remove the hooks
        # turn on gradients following the teacher model
        for teacher_param, student_param in zip(
            self.teacher_model.parameters(), self.student_model.parameters()
        ):
            student_param.requires_grad = teacher_param.requires_grad

        self.student_model = self.student_model.to(self.device).to(MODEL_PRECISION)
        return self.student_model

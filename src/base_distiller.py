import torch
from torch.amp import autocast, GradScaler
import logging
from src.constants import MODEL_PRECISION, DEVICE
from src.utils import calculate_perplexity


class BaseDistiller:
    def __init__(self, student_model):
        self.student_model = student_model
        self.device = student_model.device
        self.num_layers = len(self.get_model_layers(student_model))

    def get_model_layers(self, model):
        raise NotImplementedError("must be implemented by the children class")

    def augment_student_model(self, layer_id):
        """
        augment student model with matryoshka structure at layer_id
        and freeze all layers except this new layer
        """
        raise NotImplementedError("must be implemented by the children class")

    def compute_loss(self, layer_id, input_ids, attention_mask, teacher_model_logits):
        raise NotImplementedError("must be implemented by the children class")

    def train_layer(
        self,
        train_encodings,
        tokenizer,
        teacher_logits,
        layer_id,
        epochs,
        lr,
    ):
        """
        Train a specific layer of the student model.

        Parameters:
        - layer_id (int): The ID of the layer to train.
        - loss_fn (function): The loss function to calculate the training loss.
        - epochs (int): The number of epochs for training (default is 1).
        - lr (float): The learning rate for optimization (default is 0.0004).
        """
        self.augment_student_model(layer_id)
        self.student_model = self.student_model.to(torch.float32)

        # Get only the layer_id layer parameters for training
        layer_parameters = self.get_model_layers(self.student_model)[
            layer_id
        ].parameters()
        # Set up the optimizer to only train the layer's parameters
        optimizer = torch.optim.AdamW(layer_parameters, lr=lr)

        scaler = GradScaler(DEVICE)
        last_student_ppl = None
        for epoch in range(epochs):
            losses = []
            for batch_idx, batch in enumerate(train_encodings):
                batch_size = batch["input_ids"].shape[0]

                self.student_model.train()
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                teacher_logits_batch = teacher_logits[batch_idx]
                optimizer.zero_grad()

                with autocast(device_type=str(DEVICE), dtype=MODEL_PRECISION):
                    loss = self.compute_loss(
                        layer_id, input_ids, attention_mask, teacher_logits_batch
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                losses.append(loss)

                # calculate running ppl to see if the model is converging
                if batch_idx % (1200 // batch_size) == 24:
                    logging.info(
                        f"layer {layer_id}: calculating intermediate perplexity."
                    )
                    # use a small dataset, this is just to check if the
                    # training is converging
                    current_student_ppl = calculate_perplexity(
                        self.student_model, tokenizer, full_dataset=False
                    )
                    logging.info(
                        f"Student model {layer_id}'s ppl: {current_student_ppl:.3f}"
                    )
                    if (
                        last_student_ppl is not None
                        and last_student_ppl < current_student_ppl
                    ):
                        # stop early because the student is getting worse
                        logging.info(
                            f"last student ppl ({last_student_ppl}) < current student ppl ({current_student_ppl})"
                            + " stopping early!"
                        )
                        break
                    else:
                        last_student_ppl = current_student_ppl

            avg_loss = sum(losses) / len(losses)
            logging.info(f"Student {layer_id}'s average Loss: {avg_loss}")

        # turn on gradients after training
        for param in self.student_model.parameters():
            param.requires_grad = True

        self.student_model = self.student_model.to(self.device).to(MODEL_PRECISION)

        # calculate ppl on full dataset
        student_ppl = calculate_perplexity(
            self.student_model, tokenizer, full_dataset=True
        )
        logging.info(
            f"Student model {layer_id}'s ppl on full dataset: {student_ppl:.3f}"
        )
        return self.student_model

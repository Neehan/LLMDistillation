# base_distiller.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from src.constants import MODEL_PRECISION, TEST_ENV
from src.utils import calculate_perplexity


class BaseDistiller:
    """
    Base class for model distillation techniques.

    This class provides the foundation for implementing various distillation
    strategies by defining common methods and interfaces.
    """

    def __init__(self, student_model):
        """
        Initialize the distiller with a student model.

        Args:
            student_model: The model to be distilled (student)
        """
        self.student_model = student_model
        self.device = student_model.device
        self.model_layers = self.get_model_layers(student_model)
        self.num_layers = len(self.model_layers)

        # Augment all layers in the student model
        for layer_id in range(self.num_layers):
            self.augment_student_model(layer_id)

    def get_model_layers(self, model):
        """
        Extract the layers from the model.

        Args:
            model: The model to extract layers from

        Returns:
            List of model layers
        """
        # This method should be overridden by subclasses
        raise NotImplementedError("Subclasses must implement get_model_layers")

    def augment_student_model(self, layer_id):
        """
        Augment a specific layer in the student model.

        Args:
            layer_id: The ID of the layer to augment
        """
        # This method should be overridden by subclasses
        raise NotImplementedError("Subclasses must implement augment_student_model")

    def compute_loss(self, layer_id, input_ids, attention_mask, teacher_model_logits):
        """
        Compute the distillation loss for a specific layer.

        Args:
            layer_id: The ID of the layer being distilled
            input_ids: Input token IDs
            attention_mask: Attention mask for the inputs
            teacher_model_logits: Logits from the teacher model

        Returns:
            The computed loss
        """
        # This method should be overridden by subclasses
        raise NotImplementedError("Subclasses must implement compute_loss")

    def train_layer(
        self,
        train_encodings,
        tokenizer,
        teacher_model,
        layer_id,
        epochs,
        lr,
        accelerator,
    ):
        """
        Train a specific layer of the student model.

        Args:
            train_encodings: Dataset of encoded training examples
            tokenizer: Tokenizer for processing text
            teacher_model: The teacher model to distill from
            layer_id: The ID of the layer to train
            epochs: Number of training epochs
            lr: Learning rate
            accelerator: Accelerator for distributed training

        Returns:
            The trained student model
        """
        # Set up optimizer for the specific layer
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.student_model.parameters()), lr=lr
        )

        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            batch_count = 0

            for batch in train_encodings:
                # Get inputs from batch
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]

                # Get teacher model predictions
                with torch.no_grad():
                    teacher_outputs = teacher_model(
                        input_ids=input_ids, attention_mask=attention_mask
                    )
                    teacher_logits = teacher_outputs.logits

                # Compute loss and update model
                optimizer.zero_grad()
                loss = self.compute_loss(
                    layer_id, input_ids, attention_mask, teacher_logits
                )

                # Use accelerator for backward pass and optimization
                accelerator.backward(loss)
                optimizer.step()

                total_loss += loss.item()
                batch_count += 1

                if TEST_ENV and batch_count % 10 == 0:
                    logging.info(
                        f"Epoch {epoch+1}, Batch {batch_count}, Loss: {loss.item():.4f}"
                    )

            # Log epoch results
            avg_loss = total_loss / batch_count if batch_count > 0 else 0
            logging.info(
                f"Epoch {epoch+1}/{epochs}, Layer {layer_id}, Average Loss: {avg_loss:.4f}"
            )

        return self.student_model

    def train_layer_with_dataloader(
        self,
        train_dataloader,
        tokenizer,
        teacher_model,
        layer_id,
        epochs,
        lr,
        accelerator,
    ):
        """
        Train a specific layer using a dataloader.

        This method is optimized for memory efficiency with large datasets.

        Args:
            train_dataloader: DataLoader for training data
            tokenizer: Tokenizer for processing text
            teacher_model: The teacher model to distill from
            layer_id: The ID of the layer to train
            epochs: Number of training epochs
            lr: Learning rate
            accelerator: Accelerator for distributed training

        Returns:
            The trained student model
        """
        # Prepare the layer for training
        self.augment_student_model(layer_id)
        self.student_model = self.student_model.to(torch.float32)

        # Set up optimizer for the specific layer
        layer_parameters = self.get_model_layers(self.student_model)[
            layer_id
        ].parameters()
        optimizer = torch.optim.AdamW(layer_parameters, lr=lr)
        optimizer = accelerator.prepare(optimizer)

        # Track perplexity for early stopping
        last_student_ppl = None

        for epoch in range(epochs):
            losses = []
            self.student_model.train()

            for batch_idx, batch in enumerate(train_dataloader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                # Get teacher model predictions
                with torch.no_grad():
                    teacher_logits_batch = teacher_model(
                        input_ids=input_ids, attention_mask=attention_mask
                    ).logits

                # Compute loss and update model
                optimizer.zero_grad()
                loss = self.compute_loss(
                    layer_id, input_ids, attention_mask, teacher_logits_batch
                )
                accelerator.backward(loss)
                optimizer.step()

                losses.append(loss.detach().cpu())

                # Intermediate perplexity check (main process only)
                if batch_idx % 50 == 0 and accelerator.is_main_process:
                    logging.info(
                        f"Layer {layer_id}: calculating intermediate perplexity."
                    )
                    current_student_ppl = calculate_perplexity(
                        self.student_model, tokenizer, full_dataset=False
                    )
                    logging.info(
                        f"Student model {layer_id}'s ppl: {current_student_ppl:.3f}"
                    )

                    # Early stopping if perplexity gets worse
                    if (
                        last_student_ppl is not None
                        and last_student_ppl < current_student_ppl
                    ):
                        logging.info(
                            f"Last student ppl ({last_student_ppl}) < current student ppl ({current_student_ppl}) stopping early!"
                        )
                        break
                    else:
                        last_student_ppl = current_student_ppl

            # Log epoch results
            avg_loss = sum(losses) / len(losses)
            if accelerator.is_main_process:
                logging.info(f"Student {layer_id}'s average Loss: {avg_loss}")

        # Re-enable gradients for all parameters
        for param in self.student_model.parameters():
            param.requires_grad = True

        # Convert back to the original precision
        self.student_model = self.student_model.to(self.device).to(MODEL_PRECISION)

        # Full dataset perplexity calculation on main process
        if accelerator.is_main_process:
            student_ppl = calculate_perplexity(
                self.student_model, tokenizer, full_dataset=True
            )
            logging.info(
                f"Student model {layer_id}'s ppl on full dataset: {student_ppl:.3f}"
            )

        return self.student_model

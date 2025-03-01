"""
Skip Layer Distillation implementation.

This module implements a distillation technique that learns which layers
can be skipped during inference with minimal impact on model performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from src.constants import TEST_ENV
from src.base_distiller import BaseDistiller


import torch
import torch.nn as nn


class SkipLayerDistiller(BaseDistiller):
    """
    Distiller that learns which layers can be skipped during inference.

    This distiller trains the model to perform well both with and without
    specific layers, allowing for runtime decisions about which layers to skip.
    """

    def __init__(self, student_model):
        """
        Initialize the skip layer distiller.

        Args:
            student_model: The model to be distilled
        """
        super(SkipLayerDistiller, self).__init__(student_model)
        # In the beginning all layers are active
        self.active_layer_list = [True] * self.num_layers

    def compute_loss(self, layer_id, input_ids, attention_mask, teacher_model_logits):
        """
        Compute the distillation loss for a specific layer.

        The loss combines performance with the layer active and inactive.

        Args:
            layer_id: The ID of the layer being distilled
            input_ids: Input token IDs
            attention_mask: Attention mask for the inputs
            teacher_model_logits: Logits from the teacher model

        Returns:
            The combined distillation loss
        """
        # Activate student's layer with layer_id and get its outputs
        self.active_layer_list[layer_id] = True
        large_student_model_logits = self.student_model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits

        # Deactivate the layer and get outputs
        self.active_layer_list[layer_id] = False
        small_student_model_logits = self.student_model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits

        # Calculate KL divergence loss between teacher and both student versions
        kl_loss_fn = torch.nn.KLDivLoss(reduction="batchmean")

        # Loss with layer active
        loss_with_layer = kl_loss_fn(
            F.log_softmax(large_student_model_logits, dim=-1),
            F.softmax(teacher_model_logits, dim=-1),
        )

        # Loss with layer inactive
        loss_without_layer = kl_loss_fn(
            F.log_softmax(small_student_model_logits, dim=-1),
            F.softmax(teacher_model_logits, dim=-1),
        )

        if TEST_ENV:
            logging.info(f"Loss with layer {layer_id}: {loss_with_layer.item():.3f}")
            logging.info(
                f"Loss without layer {layer_id}: {loss_without_layer.item():.3f}"
            )

        # Note that the layer keeps being deactivated after loss computation
        # We weight the loss_without_layer less to prioritize performance with all layers
        return loss_with_layer + loss_without_layer * 0.01

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
        # Deactivate the layer
        # From now on, the layer is explicitly activated during loss computation
        # and deactivated during perplexity computation
        self.active_layer_list[layer_id] = False

        return super().train_layer(
            train_encodings,
            tokenizer,
            teacher_model,
            layer_id,
            epochs,
            lr,
            accelerator,
        )

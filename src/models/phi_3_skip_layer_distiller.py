"""
Phi-3 specific implementation of Skip Layer Distillation.

This module adapts the skip layer distillation technique for Phi-3 models,
implementing the specific layer structure and forward pass logic.
"""

import logging
from src.skip_layer_distiller import SkipLayerDistiller
from src.training_loop import training_loop
from src.argparser import parser
import torch
import torch.nn as nn
from typing import Optional, Tuple


class Phi3SkippableLayer(nn.Module):
    """
    A wrapper for Phi-3 model layers that can be conditionally skipped.

    This module wraps a Phi-3 transformer layer and either passes inputs through
    the layer or bypasses it based on the active_layer_list.
    """

    def __init__(self, original_layer, active_layer_list, layer_id):
        """
        Initialize a skippable layer.

        Args:
            original_layer: The original Phi-3 layer to wrap
            active_layer_list: List tracking which layers are active
            layer_id: ID of this layer in the model
        """
        super(Phi3SkippableLayer, self).__init__()
        self.original_layer = original_layer
        self.active_layer_list = active_layer_list
        self.layer_id = layer_id

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass that either uses or skips the layer.

        Args:
            hidden_states: Input tensor
            attention_mask: Attention mask
            position_ids: Position IDs
            past_key_value: Cached key/values for generation
            output_attentions: Whether to output attention weights
            use_cache: Whether to use cached key/values
            **kwargs: Additional arguments

        Returns:
            Tuple containing output tensor and optional cached values
        """
        is_active = self.active_layer_list[self.layer_id]
        if is_active:
            # Call the original layer's forward method
            return self.original_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )
        else:
            # Skip the layer and return outputs with the correct structure
            outputs = (hidden_states,)
            if output_attentions:
                outputs += (None,)
            if use_cache:
                # If use_cache is True, we need to return a placeholder for past_key_value
                outputs += (past_key_value,)
            return outputs


class Phi3SkipLayerDistiller(SkipLayerDistiller):
    """
    Skip Layer Distiller implementation specific to Phi-3 models.

    This class implements the model-specific methods needed for skip layer
    distillation with Phi-3 architecture.
    """

    def get_model_layers(self, model):
        """
        Extract the layers from a Phi-3 model.

        Args:
            model: The Phi-3 model

        Returns:
            List of transformer layers
        """
        return model.model.layers

    def augment_student_model(self, layer_id):
        """
        Replace a layer in the Phi-3 model with a skippable version.

        Args:
            layer_id: The ID of the layer to augment
        """
        # Replace the original layer with a skippable version
        self.student_model.model.layers[layer_id] = Phi3SkippableLayer(
            self.student_model.model.layers[layer_id],
            self.active_layer_list,
            layer_id,
        )

        # Disable gradients for all layers except the layer being trained
        for name, param in self.student_model.named_parameters():
            if f"model.layers.{layer_id}" not in name:
                param.requires_grad = False


if __name__ == "__main__":
    # Add command line arguments
    parser.add_argument(
        "--model",
        help="The model to distill",
        default="microsoft/Phi-3-mini-128k-instruct",
        type=str,
    )

    # parser.add_argument(
    #     "--num_layers",
    #     help="the number of layers in the model",
    #     default=32,
    #     type=int,
    # )

    # Parse arguments
    args = parser.parse_args()
    for arg, value in vars(args).items():
        logging.info(f"Argument: {arg}, Value: {value}")

    # Run the training loop with Phi3SkipLayerDistiller
    distiller_kwargs = {}
    training_loop(distiller_factory=Phi3SkipLayerDistiller, args=args)

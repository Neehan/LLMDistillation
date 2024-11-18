import logging
from src.skip_layer_distiller import SkipLayerDistiller
from src.training_loop import training_loop
from src.argparser import parser
import torch
import torch.nn as nn
from typing import Optional, Tuple


class Phi3SkippableLayer(nn.Module):
    def __init__(self, original_layer, active_layer_list, layer_id):
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
    def get_model_layers(self, model):
        return model.model.layers

    def augment_student_model(self, layer_id):
        self.student_model.model.layers[layer_id] = Phi3SkippableLayer(
            self.student_model.model.layers[layer_id],
            self.active_layer_list,
            layer_id,
        )

        # Disable gradients for all layers except the MLP layer just replaced
        for name, param in self.student_model.named_parameters():
            if f"model.layers.{layer_id}" not in name:
                param.requires_grad = False


if __name__ == "__main__":

    parser.add_argument(
        "--model",
        help="the model to distill",
        default="microsoft/Phi-3-mini-128k-instruct",
        type=str,
    )

    # parser.add_argument(
    #     "--num_layers",
    #     help="the number of layers in the model",
    #     default=32,
    #     type=int,
    # )

    args = parser.parse_args()
    for arg, value in vars(args).items():
        logging.info(f"Argument: {arg}, Value: {value}")

    distiller_kwargs = {}
    training_loop(distiller_factory=Phi3SkipLayerDistiller, args=args)

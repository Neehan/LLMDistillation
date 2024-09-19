import copy
import logging
import torch.nn.functional as F
from src.mat_distiller import MatDistiller, MatryoshkaMLP
from src.training_loop import training_loop
from src.argparser import parser
import torch
from typing import Optional


# copied from phi-3-small-8k model's script
@torch.jit.script
def _quick_gelu(x):
    return x * torch.sigmoid(1.702 * x)


@torch.jit.script
def _gegelu(input, limit: Optional[float] = None):
    a_gelu, a_linear = input[..., ::2], input[..., 1::2]
    if limit is not None:
        a_gelu = torch.where(
            torch.isinf(a_gelu), a_gelu, a_gelu.clamp(min=None, max=limit)
        )
        a_linear = torch.where(
            torch.isinf(a_linear), a_linear, a_linear.clamp(min=-limit, max=limit)
        )
    out_gelu = _quick_gelu(a_gelu)
    return out_gelu * (a_linear + 1)


class Phi3SmallMatMLP(MatryoshkaMLP):
    def _mat_forward(self, x):
        # Get the hidden dimension for the current layer, or default to the original MLP dimensions
        hidden_dim = (
            self.hidden_dim_list[self.layer_id]
            if self.hidden_dim_list[self.layer_id] is not None
            else self.original_mlp.up_proj.out_features
        )

        # Apply the first linear projection (up_proj) and slice the weights
        x = F.linear(
            x,
            self.original_mlp.up_proj.weight[: hidden_dim * 2, :],
            (
                self.original_mlp.up_proj.bias[: hidden_dim * 2]
                if self.original_mlp.up_proj.bias is not None
                else None
            ),
        )

        # Apply the gegelu activation function with the given limit
        x = _gegelu(x, limit=self.original_mlp.gegelu_limit)

        # Apply the second linear projection (down_proj), and slice the weights accordingly
        x = F.linear(
            x,
            self.original_mlp.down_proj.weight[:, :hidden_dim],
            self.original_mlp.down_proj.bias,  # Bias does not need slicing in this layer
        )

        # Apply the dropout after down_proj
        x = self.original_mlp.dropout(x)

        return x


class PhiMatDistiller(MatDistiller):
    def get_model_layers(self, model):
        return model.model.layers

    def get_model_mlp(self, model, layer_id):
        return model.model.layers[layer_id].mlp

    def prepare_student_model(self, layer_id):
        self.student_model.model.layers[layer_id].mlp = Phi3SmallMatMLP(
            self.student_model.model.layers[layer_id].mlp,
            self.hidden_dim_list,
            layer_id,
        )

        # Disable gradients for all layers except the MLP layer just replaced
        for name, param in self.student_model.named_parameters():
            if f"model.layers.{layer_id}.mlp" not in name:
                param.requires_grad = False


if __name__ == "__main__":

    parser.add_argument(
        "--model",
        help="the model to distill",
        default="microsoft/Phi-3-small-8k-instruct",
        type=str,
    )

    parser.add_argument(
        "--num_layers",
        help="the number of layers in the model",
        default=32,
        type=int,
    )

    args = parser.parse_args()
    for arg, value in vars(args).items():
        logging.info(f"Argument: {arg}, Value: {value}")

    distiller_kwargs = {"mat_dim": args.distill_dim}
    training_loop(
        distiller_factory=PhiMatDistiller, args=args, distiller_kwargs=distiller_kwargs
    )

import copy
import logging
import torch.nn.functional as F
from src.mat_distiller import MatDistiller, MatryoshkaMLP
from src.training_loop import training_loop
from src.utils import load_model_and_tokenizer
from src.argparser import parser


class PhiMatMLP(MatryoshkaMLP):
    def _mat_forward(self, x):
        hidden_dim = (
            self.hidden_dim_list[self.layer_id]
            if self.hidden_dim_list[self.layer_id] is not None
            else self.original_mlp.fc1.out_features
        )
        x = F.linear(
            x,
            self.original_mlp.fc1.weight[:hidden_dim, :],
            (
                self.original_mlp.fc1.bias[:hidden_dim]
                if self.original_mlp.fc1.bias is not None
                else None
            ),
        )
        x = self.original_mlp.activation_fn(x)
        x = F.linear(
            x,
            self.original_mlp.fc2.weight[:, :hidden_dim],
            self.original_mlp.fc2.bias,  # Bias doesn't need to be sliced for the second layer
        )
        return x


class PhiMatDistiller(MatDistiller):
    def get_model_layers(self, model):
        return model.model.layers

    def get_model_mlp(self, model, layer_id):
        return model.model.layers[layer_id].mlp

    def prepare_student_model(self, layer_id):
        self.student_model = copy.deepcopy(self.teacher_model)
        self.student_model.model.layers[layer_id].mlp = PhiMatMLP(
            self.student_model.model.layers[layer_id].mlp,
            self.hidden_dim_list,
            layer_id,
        )

        # Disable gradients for all layers except the MLP layer just replaced
        for name, param in self.student_model.named_parameters():
            if f"model.layers.{layer_id}.mlp" not in name:
                param.requires_grad = False


if __name__ == "__main__":

    args = parser.parse_args()
    for arg, value in vars(args).items():
        logging.info(f"Argument: {arg}, Value: {value}")
    teacher_model, tokenizer = load_model_and_tokenizer(args.model)
    print(teacher_model)
    dataset_name = "datasets/github_code"

    distiller = PhiMatDistiller(
        teacher_model, tokenizer, dataset_name=dataset_name, mat_dim=args.distill_dim
    )
    training_loop(
        teacher_model,
        tokenizer,
        distiller,
        dataset_name,
        lr=args.lr,
        num_epochs=args.num_epochs,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
    )

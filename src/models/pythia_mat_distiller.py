import copy
import torch.nn.functional as F
from src.mat_distiller import MatDistiller, MatryoshkaMLP
from src.training_loop import training_loop as base_training_loop
from src.utils import load_model_and_tokenizer


class PythiaMatMLP(MatryoshkaMLP):
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


class PythiaMatDistiller(MatDistiller):
    def get_model_layers(self, model):
        return model.transformer.h

    def get_model_mlp(self, model, layer_id):
        return model.transformer.h[layer_id].mlp

    def prepare_student_model(self, layer_id):
        self.student_model = copy.deepcopy(self.teacher_model)
        self.student_model.transformer.h[layer_id].mlp = PythiaMatMLP(
            self.student_model.transformer.h[layer_id].mlp
        )

        # Disable gradients for all layers except the MLP layer just replaced
        for name, param in self.student_model.named_parameters():
            if f"transformer.h.{layer_id}.mlp" not in name:
                param.requires_grad = False


def training_loop(model_path, dataset_name):
    teacher_model, tokenizer = load_model_and_tokenizer(model_path)
    print(teacher_model)

    distiller = PythiaMatDistiller(
        teacher_model, tokenizer, dataset_name=dataset_name, mat_dim=4096
    )
    base_training_loop(teacher_model, tokenizer, distiller, dataset_name)
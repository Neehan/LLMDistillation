import torch
import torch.nn as nn
from src.base_distiller import BaseDistiller


class MatryoshkaMLP(nn.Module):
    def __init__(self, original_mlp):
        super(MatryoshkaMLP, self).__init__()
        self.original_mlp = original_mlp
        self.current_hidden_dim = None

    def _mat_forward(self, x):
        """
        matriyoshka forward method for the specific model
        """
        raise NotImplementedError

    def forward(self, x):
        if self.current_hidden_dim is not None:
            return self._mat_forward(x)
        else:
            # Use the full MLP as is
            x = self.original_mlp(x)
        return x


class MatDistiller(BaseDistiller):
    def __init__(self, teacher_model, tokenizer, dataset_name, mat_dim: int):
        super(MatDistiller, self).__init__(teacher_model, tokenizer, dataset_name)
        self.mat_dim = mat_dim

    def compute_loss(self, layer_id, batch, loss_fn):
        # Implement logic to compute the loss for the given batch
        with torch.no_grad():
            teacher_model_logits = self.teacher_model(input_ids=batch).logits

        # set matryoshka dim to be none, using full matrix
        self.get_model_mlp(self.student_model, layer_id).current_hidden_dim = None
        large_student_model_logits = self.student_model(input_ids=batch).logits
        large_student_mlp_outputs = self.student_mlp_outputs

        # set matryoshka dim to be mat_dim, using the small matrix
        self.get_model_mlp(self.student_model, layer_id).current_hidden_dim = (
            self.mat_dim
        )
        small_student_model_logits = self.student_model(input_ids=batch).logits
        small_student_mlp_outputs = self.student_mlp_outputs

        loss1 = loss_fn(
            large_student_mlp_outputs[layer_id], self.teacher_mlp_outputs[layer_id]
        )
        loss2 = loss_fn(
            small_student_mlp_outputs[layer_id], self.teacher_mlp_outputs[layer_id]
        )

        # keep mat dim to be small so that when ppl is calculated, it is
        # calculated from the small model

        # also add final two layers outputs
        # loss3 = loss_fn(teacher_model_logits, full_student_model_logits)
        # loss4 = loss_fn(teacher_model_logits, small_student_model_logits)

        return loss1 + loss2  # + (loss3 + loss4) * 0.01

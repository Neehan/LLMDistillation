import torch
import torch.nn as nn
from src.base_distiller import BaseDistiller


class MatryoshkaMLP(nn.Module):
    def __init__(self, original_mlp, hidden_dim_list, layer_id):
        super(MatryoshkaMLP, self).__init__()
        self.original_mlp = original_mlp
        self.hidden_dim_list = hidden_dim_list
        self.layer_id = layer_id

    def _mat_forward(self, x):
        """
        matriyoshka forward method for the specific model
        """
        raise NotImplementedError

    def forward(self, x):
        current_hidden_dim = self.hidden_dim_list[self.layer_id]
        if current_hidden_dim is not None:
            return self._mat_forward(x)
        else:
            # Use the full MLP as is
            x = self.original_mlp(x)
        return x


class MatDistiller(BaseDistiller):
    def __init__(self, teacher_model, tokenizer, dataset_name, mat_dim: int):
        super(MatDistiller, self).__init__(teacher_model, tokenizer, dataset_name)
        self.mat_dim = mat_dim
        # matriyoshka always uses small matrix unless specified
        self.hidden_dim_list = [mat_dim] * self.num_layers

    def compute_loss(self, layer_id, input_ids, attention_mask, loss_fn):

        # set matryoshka dim to be none, using full matrix
        self.hidden_dim_list[layer_id] = None

        # Implement logic to compute the loss for the given batch
        with torch.no_grad():
            teacher_model_logits = self.teacher_model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits

        teacher_mlp_output = self.teacher_mlp_outputs[layer_id]

        large_student_model_logits = self.student_model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits
        large_student_mlp_output = self.student_mlp_outputs[layer_id]

        # set matryoshka dim to be mat_dim, using the small matrix
        self.hidden_dim_list[layer_id] = self.mat_dim
        small_student_model_logits = self.student_model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits
        small_student_mlp_output = self.student_mlp_outputs[layer_id]

        loss1 = loss_fn(large_student_mlp_output, teacher_mlp_output)
        loss2 = loss_fn(small_student_mlp_output, teacher_mlp_output)

        # keep mat dim to be small so that when ppl is calculated, it is
        # calculated from the small model

        # also add final two layers outputs
        # loss3 = loss_fn(teacher_model_logits, full_student_model_logits)
        # loss4 = loss_fn(teacher_model_logits, small_student_model_logits)

        return loss1 + loss2  # + (loss3 + loss4) * 0.01

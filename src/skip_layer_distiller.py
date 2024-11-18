import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from src.constants import TEST_ENV
from src.base_distiller import BaseDistiller


class SkippableLayer(nn.Module):
    def __init__(self, original_layer, active_layer_list, layer_id):
        super(SkippableLayer, self).__init__()
        self.original_layer = original_layer
        self.active_layer_list = active_layer_list
        self.layer_id = layer_id

    def forward(self, x):
        is_active: bool = self.active_layer_list[self.layer_id]
        if is_active:
            return self.original_layer(x)
        else:
            # skip this layer
            return x


class SkipLayerDistiller(BaseDistiller):
    def __init__(self, student_model):
        super(SkipLayerDistiller, self).__init__(student_model)
        # in the beginning all layers are active
        self.active_layer_list = [True] * self.num_layers

    def compute_loss(self, layer_id, input_ids, attention_mask, teacher_model_logits):

        # activate student's layer with layer_id and get its outputs
        self.active_layer_list[layer_id] = True
        large_student_model_logits = self.student_model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits

        # deactivate the layer and get outputs
        self.active_layer_list[layer_id] = False
        small_student_model_logits = self.student_model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits

        # also add final two layers outputs
        kl_loss_fn = torch.nn.KLDivLoss(reduction="batchmean")
        loss1 = kl_loss_fn(
            F.log_softmax(large_student_model_logits, dim=-1),
            F.softmax(teacher_model_logits, dim=-1),
        )

        loss2 = kl_loss_fn(
            F.log_softmax(small_student_model_logits, dim=-1),
            F.softmax(teacher_model_logits, dim=-1),
        )

        if TEST_ENV:
            logging.info(f"large student loss: {loss1.item():.3f}")
            logging.info(f"small student loss: {loss2.item():.3f}")

        # note that the layer keeps being deactivated after loss computation
        return loss1 + loss2 * 0.01

    def train_layer(
        self,
        train_encodings,
        tokenizer,
        teacher_logits,
        layer_id,
        epochs,
        lr,
    ):
        # deactivate the layer
        # from now on, the layer is explicitly activated during loss computation
        # and deactivated during ppl computation
        self.active_layer_list[layer_id] = False
        return super().train_layer(
            train_encodings,
            tokenizer,
            teacher_logits,
            layer_id,
            epochs,
            lr,
        )

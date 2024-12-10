# base_distiller.py
import torch
import logging
from src.constants import MODEL_PRECISION
from src.utils import calculate_perplexity


class BaseDistiller:
    def __init__(self, student_model):
        self.student_model = student_model
        self.device = student_model.device
        self.num_layers = len(self.get_model_layers(student_model))

    def get_model_layers(self, model):
        raise NotImplementedError("must be implemented by the children class")

    def augment_student_model(self, layer_id):
        raise NotImplementedError("must be implemented by the children class")

    def compute_loss(self, layer_id, input_ids, attention_mask, teacher_model_logits):
        raise NotImplementedError("must be implemented by the children class")

    def train_layer(
        self,
        train_dataloader,
        tokenizer,
        teacher_model,
        layer_id,
        epochs,
        lr,
        accelerator,
    ):
        self.augment_student_model(layer_id)
        self.student_model = self.student_model.to(torch.float32)

        layer_parameters = self.get_model_layers(self.student_model)[
            layer_id
        ].parameters()
        optimizer = torch.optim.AdamW(layer_parameters, lr=lr)
        optimizer = accelerator.prepare(optimizer)

        last_student_ppl = None
        for epoch in range(epochs):
            losses = []
            self.student_model.train()
            for batch_idx, batch in enumerate(train_dataloader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                with torch.no_grad():
                    teacher_logits_batch = teacher_model(
                        input_ids=input_ids, attention_mask=attention_mask
                    ).logits

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
                        f"layer {layer_id}: calculating intermediate perplexity."
                    )
                    current_student_ppl = calculate_perplexity(
                        self.student_model, tokenizer, full_dataset=False
                    )
                    logging.info(
                        f"Student model {layer_id}'s ppl: {current_student_ppl:.3f}"
                    )
                    if (
                        last_student_ppl is not None
                        and last_student_ppl < current_student_ppl
                    ):
                        logging.info(
                            f"last student ppl ({last_student_ppl}) < current student ppl ({current_student_ppl}) stopping early!"
                        )
                        break
                    else:
                        last_student_ppl = current_student_ppl

            avg_loss = sum(losses) / len(losses)
            if accelerator.is_main_process:
                logging.info(f"Student {layer_id}'s average Loss: {avg_loss}")

        for param in self.student_model.parameters():
            param.requires_grad = True

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

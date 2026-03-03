"""General distillation LightningModule — Stage 1

Responsibilities
----------------
* Orchestrate teacher (frozen) and student (trainable) forward passes
* Apply FitDense projections to align dimensions
* Compute combined intermediate distillation loss (embedding + hidden + attn)
* Log metrics to WandB / TensorBoard via Lightning's built-in logger

Modern improvements over original TinyBERT
-------------------------------------------
* Teacher is loaded once and frozen with ``torch.no_grad`` context — no custom
  ``requires_grad`` loops
* FitDense projections live inside the LightningModule and are checkpointed
  automatically
* Mixed-precision training via Lightning's ``precision`` flag (no manual
  ``apex.amp``)
* ``configure_optimizers`` centralises LR schedule + warmup
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
import torch
from transformers import AutoModel, BertForMaskedLM, get_linear_schedule_with_warmup

from src.distillation.losses import (
    attention_loss,
    embedding_loss,
    hidden_state_loss,
)
from src.models.fit_dense import FitDenseStack
from src.models.student import build_student_for_pretraining


class GeneralDistillModule(pl.LightningModule):
    """Lightning module for Stage 1 — general (pre-training) distillation

    Parameters
    ----------
    teacher_cfg : dict
        ``configs/model/teacher.yaml`` contents
    student_cfg : dict
        ``configs/model/student.yaml`` contents
    distill_cfg : dict
        ``distillation:`` section from ``configs/distill_general.yaml``
    training_cfg : dict
        ``training:`` section from ``configs/distill_general.yaml``
    """

    def __init__(
        self,
        teacher_cfg: Dict[str, Any],
        student_cfg: Dict[str, Any],
        distill_cfg: Dict[str, Any],
        training_cfg: Dict[str, Any],
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Config
        self.distill_cfg = distill_cfg
        self.training_cfg = training_cfg
        self.teacher_layer_indices: List[int] = distill_cfg["teacher_layer_indices"]

        # Teacher (frozen)
        self.teacher = AutoModel.from_pretrained(
            teacher_cfg["name_or_path"],
            attn_implementation="eager",
        )
        self.teacher.config.output_hidden_states = True
        self.teacher.config.output_attentions = True
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        # Student
        self.student: BertForMaskedLM = build_student_for_pretraining(student_cfg)

        # FitDense projections
        self.fit_dense = FitDenseStack(
            student_hidden=student_cfg["hidden_size"],
            teacher_hidden=teacher_cfg["hidden_size"],
            num_layers=len(self.teacher_layer_indices),
        )

    # Forward / Training
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        # Teacher forward (no grad)
        with torch.no_grad():
            t_out = self.teacher(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids"),
                output_hidden_states=True,
                output_attentions=True,
            )

        # Student forward
        s_out = self.student.bert(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids"),
            output_hidden_states=True,
            output_attentions=True,
        )

        # Gather matched layers
        t_emb = t_out.hidden_states[0]
        t_hiddens = [t_out.hidden_states[i + 1] for i in self.teacher_layer_indices]
        t_attns = [t_out.attentions[i] for i in self.teacher_layer_indices]

        # Student hidden states
        s_emb = s_out.hidden_states[0]
        s_hiddens = list(s_out.hidden_states[1:])
        s_attns = list(s_out.attentions)

        # Project student to teacher dim
        s_emb_proj = self.fit_dense.project_embedding(s_emb)
        s_hiddens_proj = self.fit_dense.project_hidden(s_hiddens)

        # Losses
        l_emb = embedding_loss(s_emb_proj, t_emb)
        l_hid = hidden_state_loss(s_hiddens_proj, t_hiddens)
        l_att = attention_loss(s_attns, t_attns)

        alpha_e = self.distill_cfg["alpha_embd"]
        alpha_h = self.distill_cfg["alpha_hidn"]
        alpha_a = self.distill_cfg["alpha_attn"]

        loss = alpha_e * l_emb + alpha_h * l_hid + alpha_a * l_att

        # Logging
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/loss_emb", l_emb)
        self.log("train/loss_hid", l_hid)
        self.log("train/loss_att", l_att)

        return loss

    # Optimiser & Scheduler
    def configure_optimizers(self):
        # Trainable params: student + fit_dense
        no_decay = {"bias", "LayerNorm.weight", "LayerNorm.bias"}
        grouped = [
            {
                "params": [
                    p
                    for n, p in list(self.student.named_parameters())
                    + list(self.fit_dense.named_parameters())
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.training_cfg["weight_decay"],
            },
            {
                "params": [
                    p
                    for n, p in list(self.student.named_parameters())
                    + list(self.fit_dense.named_parameters())
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            grouped, lr=self.training_cfg["learning_rate"]
        )

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * self.training_cfg["warmup_ratio"])
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

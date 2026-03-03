"""Task-specific distillation LightningModule — Stage 2

Responsibilities
----------------
* Load a **fine-tuned** teacher for the target GLUE task
* Load the general-distilled student and attach a classification head
* Implement the two-phase training loop described in the TinyBERT paper:
    - **Phase 2a** — intermediate distillation on task data (emb + hidden + attn)
    - **Phase 2b** — prediction-layer distillation (KL on soft logits) + hard-label CE
* Evaluate on the dev set with task-appropriate metrics

Modern improvements
-------------------
* ``torchmetrics`` for numerically correct metric accumulation across devices
* ``phase`` flag allows the same module to serve both 2a and 2b — no code duplication
* Automatic mixed precision via Lightning
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
import torch
import torchmetrics
from transformers import (
    AutoModelForSequenceClassification,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from src.distillation.losses import (
    attention_loss,
    embedding_loss,
    hard_label_loss,
    hidden_state_loss,
    prediction_loss,
)
from src.models.fit_dense import FitDenseStack
from src.models.student import build_student_for_classification


class TaskDistillModule(pl.LightningModule):
    """Lightning module for Stage 2 — task-specific distillation

    Parameters
    ----------
    teacher_cfg : dict
        Teacher model YAML section
    student_cfg : dict
        Student model YAML section
    distill_cfg : dict
        ``distillation:`` section from ``configs/distill_task.yaml``
    training_cfg : dict
        Either ``training.stage2a`` or ``training.stage2b`` sub-dict
    task_cfg : dict
        ``task:`` section (``name``, ``num_labels``, ``metric``)
    phase : str
        ``"intermediate"`` (2a) or ``"prediction"`` (2b)
    teacher_ckpt : str or None
        Path to a fine-tuned teacher checkpoint.  If ``None``, the teacher
        is loaded from HF Hub as-is (user must ensure it is already fine-tuned)
    student_general_ckpt : str or None
        Path to the general-distilled student checkpoint from Stage 1
    """

    def __init__(
        self,
        teacher_cfg: Dict[str, Any],
        student_cfg: Dict[str, Any],
        distill_cfg: Dict[str, Any],
        training_cfg: Dict[str, Any],
        task_cfg: Dict[str, Any],
        phase: str = "intermediate",
        teacher_ckpt: Optional[str] = None,
        student_general_ckpt: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.distill_cfg = distill_cfg
        self.training_cfg = training_cfg
        self.task_cfg = task_cfg
        self.phase = phase
        self.teacher_layer_indices: List[int] = distill_cfg["teacher_layer_indices"]

        num_labels: int = task_cfg["num_labels"]

        # Teacher (frozen)
        teacher_path = teacher_ckpt or teacher_cfg["name_or_path"]
        self.teacher = AutoModelForSequenceClassification.from_pretrained(
            teacher_path,
            num_labels=num_labels,
            attn_implementation="eager",
        )
        self.teacher.config.output_hidden_states = True
        self.teacher.config.output_attentions = True
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        # Student
        self.student: BertForSequenceClassification = build_student_for_classification(
            student_cfg,
            num_labels=num_labels,
            pretrained_path=student_general_ckpt,
        )

        # FitDense in phase 2a
        if self.phase == "intermediate":
            self.fit_dense = FitDenseStack(
                student_hidden=student_cfg["hidden_size"],
                teacher_hidden=teacher_cfg["hidden_size"],
                num_layers=len(self.teacher_layer_indices),
            )
        else:
            self.fit_dense = None

        # Metrics
        metric_name = task_cfg.get("metric", "accuracy")
        if metric_name == "accuracy":
            self.val_metric = torchmetrics.Accuracy(
                task="multiclass", num_classes=num_labels
            )
        elif metric_name == "f1":
            self.val_metric = torchmetrics.F1Score(
                task="multiclass", num_classes=num_labels
            )
        elif metric_name == "matthews_corrcoef":
            self.val_metric = torchmetrics.MatthewsCorrCoef(
                task="multiclass", num_classes=num_labels
            )
        else:
            self.val_metric = torchmetrics.Accuracy(
                task="multiclass", num_classes=num_labels
            )

    # Training
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        labels = batch.pop("labels", batch.pop("label", None))

        # Teacher forward (no grad)
        with torch.no_grad():
            t_out = self.teacher(
                **batch,
                output_hidden_states=True,
                output_attentions=True,
            )

        # Student forward
        s_out = self.student(
            **batch,
            output_hidden_states=True,
            output_attentions=True,
        )

        if self.phase == "intermediate":
            loss = self._intermediate_loss(s_out, t_out)
        else:
            loss = self._prediction_loss(s_out, t_out, labels)

        self.log("train/loss", loss, prog_bar=True)
        return loss

    def _intermediate_loss(self, s_out, t_out) -> torch.Tensor:
        """Phase 2a — embedding + hidden + attention MSE losses."""
        t_emb = t_out.hidden_states[0]
        t_hiddens = [t_out.hidden_states[i + 1] for i in self.teacher_layer_indices]
        t_attns = [t_out.attentions[i] for i in self.teacher_layer_indices]

        s_emb = s_out.hidden_states[0]
        s_hiddens = list(s_out.hidden_states[1:])
        s_attns = list(s_out.attentions)

        s_emb_proj = self.fit_dense.project_embedding(s_emb)
        s_hiddens_proj = self.fit_dense.project_hidden(s_hiddens)

        l_emb = embedding_loss(s_emb_proj, t_emb)
        l_hid = hidden_state_loss(s_hiddens_proj, t_hiddens)
        l_att = attention_loss(s_attns, t_attns)

        loss = (
            self.distill_cfg["alpha_embd"] * l_emb
            + self.distill_cfg["alpha_hidn"] * l_hid
            + self.distill_cfg["alpha_attn"] * l_att
        )
        self.log("train/loss_emb", l_emb)
        self.log("train/loss_hid", l_hid)
        self.log("train/loss_att", l_att)
        return loss

    def _prediction_loss(self, s_out, t_out, labels) -> torch.Tensor:
        """Phase 2b — KL on soft labels + optional hard-label CE."""
        temperature = self.distill_cfg["temperature"]
        alpha_pred = self.distill_cfg["alpha_pred"]
        alpha_hard = self.distill_cfg.get("alpha_hard", 0.0)

        l_pred = prediction_loss(s_out.logits, t_out.logits, temperature)
        loss = alpha_pred * l_pred
        self.log("train/loss_pred", l_pred)

        if alpha_hard > 0.0 and labels is not None:
            l_hard = hard_label_loss(s_out.logits, labels)
            loss = loss + alpha_hard * l_hard
            self.log("train/loss_hard", l_hard)

        return loss

    # Validation
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        labels = batch.pop("labels", batch.pop("label", None))
        s_out = self.student(**batch)
        preds = s_out.logits.argmax(dim=-1)
        self.val_metric.update(preds, labels)

    def on_validation_epoch_end(self) -> None:
        score = self.val_metric.compute()
        self.log("val/metric", score, prog_bar=True)
        self.val_metric.reset()

    # Optimiser & Scheduler
    def configure_optimizers(self):
        no_decay = {"bias", "LayerNorm.weight", "LayerNorm.bias"}

        # Collect trainable modules
        trainable_named: list = list(self.student.named_parameters())
        if self.fit_dense is not None:
            trainable_named += list(self.fit_dense.named_parameters())

        grouped = [
            {
                "params": [
                    p for n, p in trainable_named if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.training_cfg["weight_decay"],
            },
            {
                "params": [
                    p for n, p in trainable_named if any(nd in n for nd in no_decay)
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

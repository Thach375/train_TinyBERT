"""Custom PyTorch Lightning callbacks

Responsibilities
----------------
* ``DistillationProgressBar`` — tqdm bar showing loss components
* ``SaveStudentCallback`` — at end of training, export only the student
  weights (without teacher / fit_dense) as a stand-alone HF model
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class SaveStudentCallback(Callback):
    """Save the student model as a standalone Hugging Face checkpoint

    Exports ``student.save_pretrained(output_dir / "student_hf")`` at the end
    of training so the checkpoint is directly loadable via ``AutoModel.from_pretrained``

    Parameters
    ----------
    output_dir : str
        Root output directory
    """

    def __init__(self, output_dir: str) -> None:
        super().__init__()
        self.output_dir = Path(output_dir)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        save_path = self.output_dir / "student_hf"
        save_path.mkdir(parents=True, exist_ok=True)
        student = getattr(pl_module, "student", None)
        if student is not None and hasattr(student, "save_pretrained"):
            student.save_pretrained(str(save_path))
            print(f"[SaveStudentCallback] Student saved to {save_path}")


class LogLearningRateCallback(Callback):
    """Log each param-group learning rate every N steps"""

    def __init__(self, log_every_n_steps: int = 50) -> None:
        super().__init__()
        self.log_every = log_every_n_steps

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if trainer.global_step % self.log_every == 0:
            for i, pg in enumerate(trainer.optimizers[0].param_groups):
                pl_module.log(f"lr/group_{i}", pg["lr"], prog_bar=False)

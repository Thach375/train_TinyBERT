"""CLI entry-point for Stage 1 — General (pre-training) distillation.

Usage
-----
    python scripts/run_general_distill.py --config configs/distill_general.yaml

The script:
1. Loads configuration from YAML
2. Builds the ``PretrainDataModule``
3. Builds the ``GeneralDistillModule`` (teacher + student + fit-dense)
4. Trains via PyTorch Lightning ``Trainer``
5. Saves the student checkpoint for Stage 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

torch.set_float32_matmul_precision("medium")

from src.data.pretrain_data import PretrainDataModule
from src.distillation.general_distill import GeneralDistillModule
from src.utils.callbacks import SaveStudentCallback
from src.utils.helpers import load_yaml, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="TinyBERT General Distillation")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/distill_general.yaml",
        help="Path to the general-distillation YAML config.",
    )
    args = parser.parse_args()

    # Load config
    cfg = load_yaml(args.config)

    # Merge model sub-configs referenced via `defaults:`
    teacher_cfg = load_yaml("configs/model/teacher.yaml")
    student_cfg = load_yaml("configs/model/student.yaml")

    training_cfg = cfg["training"]
    distill_cfg = cfg["distillation"]
    data_cfg = cfg["data"]
    paths_cfg = cfg["paths"]
    log_cfg = cfg["logging"]

    seed_everything(training_cfg["seed"])

    # DataModule
    dm = PretrainDataModule(
        teacher_name_or_path=teacher_cfg["name_or_path"],
        dataset_name=data_cfg["dataset_name"],
        dataset_config=data_cfg.get("dataset_config"),
        max_seq_length=data_cfg["max_seq_length"],
        mlm_probability=data_cfg["mlm_probability"],
        batch_size=training_cfg["batch_size"],
        num_workers=training_cfg["num_workers"],
        cache_dir=paths_cfg.get("cache_dir"),
    )

    # LightningModule
    model = GeneralDistillModule(
        teacher_cfg=teacher_cfg,
        student_cfg=student_cfg,
        distill_cfg=distill_cfg,
        training_cfg=training_cfg,
    )

    # Logger
    output_dir = paths_cfg["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if log_cfg["logger"] == "wandb":
        logger = WandbLogger(project=log_cfg["project"], save_dir=output_dir)
    else:
        logger = TensorBoardLogger(save_dir=output_dir, name=log_cfg["project"])

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir,
            filename="general-{epoch:02d}-{step}",
            save_last=True,
            every_n_train_steps=5000,
        ),
        LearningRateMonitor(logging_interval="step"),
        SaveStudentCallback(output_dir=output_dir),
    ]

    # Trainer
    trainer = pl.Trainer(
        max_epochs=training_cfg["max_epochs"],
        precision=training_cfg["precision"],
        gradient_clip_val=training_cfg["gradient_clip_val"],
        accumulate_grad_batches=training_cfg["accumulate_grad_batches"],
        log_every_n_steps=log_cfg["log_every_n_steps"],
        logger=logger,
        callbacks=callbacks,
        deterministic="warn",  # True can silently crash with 16-mixed on some CUDA ops
    )

    trainer.fit(model, datamodule=dm)
    print(f"[Stage 1] Training complete. Outputs → {output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""CLI entry-point for Stage 2: Task-specific distillation

The script:
1. Loads configuration from YAML
2. Applies any --set KEY=VALUE overrides (dot-notation, e.g. distillation.alpha_attn=0)
3. **Phase 2a** — intermediate distillation on task data
4. **Phase 2b** — prediction-layer distillation (skippable via --skip-stage2b)
5. Saves the final distilled student
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

torch.set_float32_matmul_precision("medium")

from src.data.glue_data import GLUEDataModule
from src.distillation.task_distill import TaskDistillModule
from src.utils.callbacks import SaveStudentCallback
from src.utils.helpers import load_yaml, seed_everything


def _set_nested(d: Dict[str, Any], dotted_key: str, value: str) -> None:
    """Set a value in a nested dict using dot-notation key.

    Automatically casts value to int / float / bool / None where possible.
    Example: _set_nested(cfg, "distillation.alpha_attn", "0") → cfg["distillation"]["alpha_attn"] = 0.0
    """
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    leaf = keys[-1]

    # Type casting
    if value.lower() == "null" or value.lower() == "none":
        d[leaf] = None
    elif value.lower() == "true":
        d[leaf] = True
    elif value.lower() == "false":
        d[leaf] = False
    else:
        try:
            d[leaf] = int(value)
        except ValueError:
            try:
                d[leaf] = float(value)
            except ValueError:
                d[leaf] = value  # keep as string


def _build_logger(log_cfg: dict, output_dir: str, suffix: str):
    if log_cfg["logger"] == "wandb":
        return WandbLogger(
            project=log_cfg["project"], name=suffix, save_dir=output_dir
        )
    return TensorBoardLogger(save_dir=output_dir, name=f"{log_cfg['project']}_{suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TinyBERT Task Distillation (Stage 2)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/distill_task.yaml",
        help="Path to the task-distillation YAML config.",
    )
    parser.add_argument(
        "--set",
        nargs="+",
        metavar="KEY=VALUE",
        default=[],
        help=(
            "Override config values using dot-notation.\n"
            "Examples:\n"
            "  --set distillation.alpha_attn=0\n"
            "  --set distillation.alpha_embd=0 distillation.alpha_hidn=0\n"
            "  --set paths.student_general_ckpt=null  (ablate Stage 1)\n"
            "  --set training.stage2a.max_epochs=10\n"
        ),
    )
    parser.add_argument(
        "--skip-stage2b",
        action="store_true",
        default=False,
        help="Skip Phase 2b (prediction-layer distillation). Useful for ablating Stage 2b.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help=(
            "Tag this run (used as wandb run name and output sub-directory).\n"
            "Defaults to 'baseline' if no --set overrides are applied."
        ),
    )
    args = parser.parse_args()

    # Auto-generate run name from overrides if not provided
    run_name = args.run_name
    if run_name is None:
        if args.set or args.skip_stage2b:
            parts = [kv.split("=")[0].replace(".", "_") for kv in args.set]
            if args.skip_stage2b:
                parts.append("no_stage2b")
            run_name = "__".join(parts)
        else:
            run_name = "baseline"
    print(f"Run name: {run_name}")

    # Load config then apply --set overrides
    cfg = load_yaml(args.config)

    for kv in args.set:
        if "=" not in kv:
            parser.error(f"--set values must be KEY=VALUE, got: '{kv}'")
        key, value = kv.split("=", 1)
        _set_nested(cfg, key, value)
        print(f"  Override: {key} = {value}")

    teacher_cfg = load_yaml("configs/model/teacher.yaml")
    student_cfg = load_yaml("configs/model/student.yaml")

    distill_cfg = cfg["distillation"]
    task_cfg = cfg["task"]
    paths_cfg = cfg["paths"]
    log_cfg = cfg["logging"]

    seed = cfg["training"]["seed"]
    seed_everything(seed)

    # Each run gets its own sub-directory so ablations don't overwrite each other
    output_dir = str(Path(paths_cfg["output_dir"]) / run_name)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # DataModule
    dm = GLUEDataModule(
        teacher_name_or_path=teacher_cfg["name_or_path"],
        task_name=task_cfg["name"],
        max_seq_length=cfg["data"]["max_seq_length"],
        batch_size=cfg["training"]["stage2a"]["batch_size"],
        num_workers=cfg["training"]["num_workers"],
        cache_dir=paths_cfg.get("cache_dir"),
    )

    #  Phase 2a: Intermediate distillation
    print("Phase 2a: Intermediate Distillation")

    training_2a = cfg["training"]["stage2a"]
    training_2a.update({
        "gradient_clip_val": cfg["training"]["gradient_clip_val"],
        "accumulate_grad_batches": cfg["training"]["accumulate_grad_batches"],
        "precision": cfg["training"]["precision"],
        "seed": seed,
    })

    model_2a = TaskDistillModule(
        teacher_cfg=teacher_cfg,
        student_cfg=student_cfg,
        distill_cfg=distill_cfg,
        training_cfg=training_2a,
        task_cfg=task_cfg,
        phase="intermediate",
        teacher_ckpt=cfg.get("teacher_ckpt"),
        student_general_ckpt=paths_cfg.get("student_general_ckpt"),
    )

    logger_2a = _build_logger(log_cfg, output_dir, f"{run_name}__phase2a")
    ckpt_2a_dir = str(Path(output_dir) / "phase2a")
    trainer_2a = pl.Trainer(
        max_epochs=training_2a["max_epochs"],
        precision=training_2a["precision"],
        gradient_clip_val=training_2a["gradient_clip_val"],
        accumulate_grad_batches=training_2a["accumulate_grad_batches"],
        log_every_n_steps=log_cfg["log_every_n_steps"],
        logger=logger_2a,
        callbacks=[
            ModelCheckpoint(dirpath=ckpt_2a_dir, save_last=True),
            LearningRateMonitor(logging_interval="step"),
        ],
        deterministic="warn",
    )

    trainer_2a.fit(model_2a, datamodule=dm)

    if args.skip_stage2b:
        print("Phase 2b skipped (--skip-stage2b).")
        print(f"[Stage 2] Training complete. Final student → {output_dir}/student_hf")
        return

    #  Phase 2b: Prediction-layer distillation
    print("Phase 2b: Prediction-layer Distillation")

    training_2b = cfg["training"]["stage2b"]
    training_2b.update({
        "gradient_clip_val": cfg["training"]["gradient_clip_val"],
        "accumulate_grad_batches": cfg["training"]["accumulate_grad_batches"],
        "precision": cfg["training"]["precision"],
        "seed": seed,
    })

    # Warm-start student from the Phase-2a checkpoint
    model_2b = TaskDistillModule(
        teacher_cfg=teacher_cfg,
        student_cfg=student_cfg,
        distill_cfg=distill_cfg,
        training_cfg=training_2b,
        task_cfg=task_cfg,
        phase="prediction",
        teacher_ckpt=cfg.get("teacher_ckpt"),
        student_general_ckpt=None,  # will load 2a student weights below
    )

    # Transfer student weights from phase 2a
    model_2b.student.load_state_dict(model_2a.student.state_dict())

    logger_2b = _build_logger(log_cfg, output_dir, f"{run_name}__phase2b")
    ckpt_2b_dir = str(Path(output_dir) / "phase2b")
    trainer_2b = pl.Trainer(
        max_epochs=training_2b["max_epochs"],
        precision=training_2b["precision"],
        gradient_clip_val=training_2b["gradient_clip_val"],
        accumulate_grad_batches=training_2b["accumulate_grad_batches"],
        log_every_n_steps=log_cfg["log_every_n_steps"],
        logger=logger_2b,
        callbacks=[
            ModelCheckpoint(dirpath=ckpt_2b_dir, save_last=True),
            LearningRateMonitor(logging_interval="step"),
            SaveStudentCallback(output_dir=output_dir),
        ],
        deterministic="warn",
    )

    trainer_2b.fit(model_2b, datamodule=dm)
    print(f"[Stage 2] Training complete. Final student → {output_dir}/student_hf")


if __name__ == "__main__":
    main()

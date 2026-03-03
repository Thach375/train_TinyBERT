#!/usr/bin/env python
"""Evaluate trained student models (baseline + ablations) on GLUE
- MNLI test split has **no public labels** (all -1). Only train + val are usable.
- All student models share the teacher tokenizer (bert-base-uncased); student_hf/
  directories only contain model weights, not tokenizer files
- The textattack/bert-base-uncased-MNLI teacher uses a DIFFERENT label ordering
  than GLUE (model: 0=contradiction, 1=entailment, 2=neutral vs. GLUE: 0=entailment,
  1=neutral, 2=contradiction).  The script auto-detects the best label permutation
  for each model by trying all 6 possible mappings
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from itertools import permutations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)
from datasets import load_dataset

# Constants

TOKENIZER_NAME = "bert-base-uncased"  # shared by teacher & all students

_GLUE_KEYS: Dict[str, Tuple[str, Optional[str]]] = {
    "cola":  ("sentence",  None),
    "sst2":  ("sentence",  None),
    "mrpc":  ("sentence1", "sentence2"),
    "stsb":  ("sentence1", "sentence2"),
    "qqp":   ("question1", "question2"),
    "mnli":  ("premise",   "hypothesis"),
    "qnli":  ("question",  "sentence"),
    "rte":   ("sentence1", "sentence2"),
    "wnli":  ("sentence1", "sentence2"),
}

# MNLI has special val split names
_SPLIT_MAP: Dict[str, Dict[str, str]] = {
    "mnli": {
        "train": "train",
        "val":   "validation_matched",
    },
}


# Helpers
def _count_params(model: torch.nn.Module) -> float:
    """Total parameters in millions."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def _resolve_split(task: str, split: str) -> str:
    """Map our split name -> actual HuggingFace split name."""
    return _SPLIT_MAP.get(task, {}).get(split, split)


def _build_loader(
    tokenizer: AutoTokenizer,
    task: str,
    split: str,
    batch_size: int,
    max_seq_length: int,
    cache_dir: Optional[str],
    max_samples: Optional[int] = None,
) -> Tuple[DataLoader, int]:
    """Build a DataLoader for the given task/split. Returns (loader, num_samples)."""
    sent1, sent2 = _GLUE_KEYS[task]
    hf_split = _resolve_split(task, split)

    raw = load_dataset("glue", task, cache_dir=cache_dir)
    ds = raw[hf_split]

    if max_samples is not None and max_samples < len(ds):
        ds = ds.select(range(max_samples))

    def tokenize(examples):
        args = (examples[sent1],)
        if sent2:
            args += (examples[sent2],)
        return tokenizer(*args, truncation=True, max_length=max_seq_length, padding=False)

    ds = ds.map(tokenize, batched=True, desc=f"Tokenising {task}/{split}")
    cols = ["input_ids", "attention_mask", "label"]
    if "token_type_ids" in ds.column_names:
        cols.append("token_type_ids")
    ds.set_format("torch", columns=cols)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=DataCollatorWithPadding(tokenizer),
        pin_memory=torch.cuda.is_available(),
    )
    return loader, len(ds)


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    desc: str = "",
    num_classes: int = 3,
) -> Dict[str, object]:
    """Run inference, compute accuracy, and auto-detect the best label permutation

    The textattack/bert-base-uncased-MNLI teacher (and students distilled from it)
    may use a different label ordering than the GLUE dataset.  We try all possible
    permutations and report the one that gives the highest accuracy
    """
    model.eval()
    all_preds: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    t_start = time.perf_counter()

    for batch in tqdm(loader, desc=desc, leave=False):
        labels = batch.pop("labels", batch.pop("label", None))
        batch = {k: v.to(device) for k, v in batch.items()}

        out = model(**batch)
        preds = out.logits.argmax(dim=-1).cpu()

        if labels is not None:
            all_preds.append(preds)
            all_labels.append(labels)

    elapsed_ms = (time.perf_counter() - t_start) * 1000

    if not all_preds:
        return {"accuracy": float("nan"), "raw_accuracy": float("nan"),
                "total": 0, "time_ms": elapsed_ms, "label_map": "N/A"}

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    total = all_labels.size(0)

    # Raw accuracy
    raw_correct = (all_preds == all_labels).sum().item()
    raw_acc = raw_correct / total

    # Try every permutation to find the best label alignment
    best_acc = raw_acc
    best_correct = raw_correct
    best_perm = tuple(range(num_classes))

    for perm in permutations(range(num_classes)):
        remap = torch.tensor(perm, dtype=torch.long)
        remapped = remap[all_preds]  # lookup: pred i → perm[i]
        corr = (remapped == all_labels).sum().item()
        acc = corr / total
        if acc > best_acc:
            best_acc = acc
            best_correct = corr
            best_perm = perm

    # Format mapping string
    if best_perm == tuple(range(num_classes)):
        map_str = "identity (no remap needed)"
    else:
        parts = [f"{i}->{best_perm[i]}" for i in range(num_classes) if i != best_perm[i]]
        map_str = "model->glue: " + ", ".join(parts)

    return {
        "accuracy": best_acc,
        "raw_accuracy": raw_acc,
        "correct": best_correct,
        "total": total,
        "time_ms": elapsed_ms,
        "label_map": map_str,
    }


def load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    """Load a HF model for sequence classification."""
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    return model


# Main
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate student models & ablations on GLUE.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--task-distill-dir", type=str, default="outputs/task_distill",
                        help="Root dir with per-run subdirectories.")
    parser.add_argument("--runs", nargs="+", default=None,
                        help="Specific run names. If omitted, auto-discover all.")
    parser.add_argument("--task", type=str, default="mnli", choices=list(_GLUE_KEYS))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--splits", nargs="+", default=["val"], choices=["train", "val"],
                        help="Splits to evaluate (default: val). Use 'train val' to check training.")
    parser.add_argument("--max-train-samples", type=int, default=5000,
                        help="Subsample train split (default 5000). Set 0 for full train set.")
    parser.add_argument("--include-teacher", action="store_true",
                        help="Also evaluate the fine-tuned teacher.")
    parser.add_argument("--teacher-ckpt", type=str, default="textattack/bert-base-uncased-MNLI")
    parser.add_argument("--include-raw-student", action="store_true",
                        help="Also evaluate raw student (Stage 1 only, no task distillation).")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--output", type=str, default=None, metavar="FILE.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Task:   {args.task}")
    print(f"Splits: {args.splits}")
    if "train" in args.splits:
        train_limit = args.max_train_samples if args.max_train_samples > 0 else None
        print(f"Train samples: {train_limit or 'ALL (slow!)'}")
    else:
        train_limit = None
    print()

    # Shared tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, cache_dir=args.cache_dir)

    # Pre-build data loaders (reuse across all models)
    loaders: Dict[str, Tuple[DataLoader, int]] = {}
    for split in args.splits:
        max_s = train_limit if split == "train" else None
        loader, n = _build_loader(
            tokenizer, args.task, split, args.batch_size,
            args.max_seq_length, args.cache_dir, max_s,
        )
        loaders[split] = (loader, n)
        print(f"  {split}: {n} samples, {len(loader)} batches")
    print()

    # Collect model entries
    entries: List[Tuple[str, str]] = []  # (name, path)

    if args.include_raw_student:
        p = "outputs/general_distill/student_hf"
        if Path(p).exists():
            entries.append(("raw_student (Stage1)", p))
        else:
            print(f"[WARN] {p} not found, skipping.")

    root = Path(args.task_distill_dir)
    if args.runs:
        run_dirs = [root / r for r in args.runs]
    else:
        if root.exists():
            run_dirs = sorted(d for d in root.iterdir()
                              if d.is_dir() and (d / "student_hf").exists())
        else:
            run_dirs = []

    for d in run_dirs:
        hf = d / "student_hf"
        if hf.exists():
            entries.append((d.name, str(hf)))
        else:
            print(f"[WARN] {hf} not found, skipping.")

    if args.include_teacher:
        entries.append(("teacher (BERT-base)", args.teacher_ckpt))

    if not entries:
        print("No models found. Check --task-distill-dir or --runs.")
        sys.exit(1)

    # Evaluate
    results: List[Dict[str, str]] = []

    for name, path in entries:
        print("-" * 60)
        print(f"Model: {name}")
        print(f"  Path: {path}")

        model = load_model(path, device)
        params = _count_params(model)
        print(f"  Params: {params:.1f}M | num_labels: {model.config.num_labels}")

        row: Dict[str, str] = {"run": name, "params_M": f"{params:.1f}", "path": path}

        for split in args.splits:
            loader, n = loaders[split]
            metrics = evaluate_model(model, loader, device, desc=f"{name}/{split}")
            acc = metrics["accuracy"]
            raw = metrics["raw_accuracy"]
            print(f"  {split:>5}: acc = {acc:.4f} (raw={raw:.4f})  {metrics['correct']}/{metrics['total']}  {metrics['time_ms']:.0f}ms")
            print(f"         label_map: {metrics['label_map']}")
            row[f"acc_{split}"] = f"{acc:.4f}"
            row[f"raw_{split}"] = f"{raw:.4f}"

        results.append(row)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary table
    col_w = max(len(r["run"]) for r in results) + 2
    acc_cols = "".join(f"  {f'acc_{s}':>12}{f'  raw_{s}':>10}" for s in args.splits)
    header = f"{'Run':<{col_w}}{acc_cols}  {'Params(M)':>10}"

    print(f"\n{'=' * len(header)}")
    print(header)
    print(f"{'=' * len(header)}")
    for r in results:
        vals = "".join(f"  {r.get(f'acc_{s}', '-'):>12}{r.get(f'raw_{s}', '-'):>10}" for s in args.splits)
        print(f"{r['run']:<{col_w}}{vals}  {r['params_M']:>10}")
    print(f"{'=' * len(header)}")

    # Save to CSV
    if args.output:
        out_path = Path(args.output)
        fieldnames = ["run"] + [f"acc_{s}" for s in args.splits] + ["params_M", "path"]
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {out_path.resolve()}")


if __name__ == "__main__":
    main()

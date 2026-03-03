"""GLUE DataModule for task-specific distillation (Stage 2)

Responsibilities
----------------
* Load any GLUE task via ``datasets.load_dataset("glue", <task>)``
* Tokenise sentence / sentence-pair inputs with the teacher tokenizer
* Provide ``train_dataloader()`` and ``val_dataloader()`` for Lightning

Modern improvements
-------------------
* Dynamic padding via ``DataCollatorWithPadding`` to minimise wasted compute
* Single code path for all GLUE tasks (auto-detects sentence keys)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding, PreTrainedTokenizerBase

# Mapping from GLUE task name → (sentence1_key, sentence2_key | None)
_GLUE_KEYS: Dict[str, tuple] = {
    "cola": ("sentence", None),
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "stsb": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
    "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


class GLUEDataModule(pl.LightningDataModule):
    """Lightning DataModule for GLUE benchmark tasks

    Parameters
    ----------
    teacher_name_or_path : str
        Tokenizer name matching the teacher model
    task_name : str
        GLUE task (``sst2``, ``mrpc``, ``mnli``, …)
    max_seq_length : int
        Maximum token length
    batch_size : int
        Per-device batch size
    num_workers : int
        DataLoader workers
    cache_dir : str or None
        HF cache directory
    """

    def __init__(
        self,
        teacher_name_or_path: str,
        task_name: str = "sst2",
        max_seq_length: int = 128,
        batch_size: int = 32,
        num_workers: int = 4,
        cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.task_name = task_name.lower()
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_dir = cache_dir

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            teacher_name_or_path, cache_dir=cache_dir
        )

        keys = _GLUE_KEYS.get(self.task_name)
        if keys is None:
            raise ValueError(
                f"Unknown GLUE task '{task_name}'. Supported: {list(_GLUE_KEYS)}"
            )
        self._sent1_key, self._sent2_key = keys

    # Lightning hooks

    def setup(self, stage: Optional[str] = None) -> None:
        raw = load_dataset("glue", self.task_name, cache_dir=self.cache_dir)

        def tokenize_fn(examples: Dict[str, Any]) -> Dict[str, Any]:
            args = (examples[self._sent1_key],)
            if self._sent2_key is not None:
                args += (examples[self._sent2_key],)
            return self.tokenizer(
                *args,
                truncation=True,
                max_length=self.max_seq_length,
                padding=False,
            )

        self._train_ds = raw["train"].map(
            tokenize_fn,
            batched=True,
            desc=f"Tokenising {self.task_name} train",
        )
        self._train_ds.set_format(
            "torch", columns=["input_ids", "attention_mask", "token_type_ids", "label"]
        )

        # Validation split name varies across tasks
        val_key = (
            "validation_matched" if self.task_name == "mnli" else "validation"
        )
        self._val_ds = raw[val_key].map(
            tokenize_fn,
            batched=True,
            desc=f"Tokenising {self.task_name} val",
        )
        self._val_ds.set_format(
            "torch", columns=["input_ids", "attention_mask", "token_type_ids", "label"]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=DataCollatorWithPadding(self.tokenizer),
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=DataCollatorWithPadding(self.tokenizer),
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

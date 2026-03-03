"""Pre-training DataModule for general distillation (Stage 1).

Responsibilities
----------------
* Load a large unlabelled corpus (e.g. BookCorpus, Wikipedia) via
  ``datasets`` (Hugging Face Datasets library)
* Tokenise with the teacher's tokenizer
* Apply dynamic Masked Language Modelling (MLM) collation using
  ``DataCollatorForLanguageModeling``
* Expose ``train_dataloader()`` compatible with PyTorch Lightning

Modern improvements over the original TinyBERT
-----------------------------------------------
* Uses ``datasets.load_dataset`` for streaming large corpora
* ``DataCollatorForLanguageModeling`` handles dynamic masking per-epoch
* Configurable via YAML (``configs/distill_general.yaml``)
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import pytorch_lightning as pl
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerBase,
)


class PretrainDataModule(pl.LightningDataModule):
    """Lightning DataModule for unsupervised pre-training distillation

    Parameters
    ----------
    teacher_name_or_path : str
        Tokenizer name matching the teacher model
    dataset_name : str
        Hugging Face Datasets identifier (e.g. ``"bookcorpus"``)
    dataset_config : str or None
        Optional dataset configuration name
    max_seq_length : int
        Maximum token length per sample
    mlm_probability : float
        Fraction of tokens to mask
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
        dataset_name: str = "bookcorpus",
        dataset_config: Optional[str] = None,
        max_seq_length: int = 128,
        mlm_probability: float = 0.15,
        batch_size: int = 64,
        num_workers: int = 4,
        cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.teacher_name_or_path = teacher_name_or_path
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.max_seq_length = max_seq_length
        self.mlm_probability = mlm_probability
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_dir = cache_dir

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            teacher_name_or_path, cache_dir=cache_dir
        )

    # Lightning hooks

    def setup(self, stage: Optional[str] = None) -> None:
        """Download & tokenise the dataset"""
        raw = load_dataset(
            self.dataset_name,
            self.dataset_config,
            cache_dir=self.cache_dir,
        )
        # Use train split or first available split
        split = "train" if "train" in raw else list(raw.keys())[0]
        raw_train = raw[split]

        # Determine the text column
        text_col = "text"
        if text_col not in raw_train.column_names:
            text_col = raw_train.column_names[0]

        def tokenize_fn(examples: Dict[str, Any]) -> Dict[str, Any]:
            return self.tokenizer(examples[text_col],
                                  truncation=True,
                                  max_length=self.max_seq_length,
                                  padding="max_length",
                                  return_special_tokens_mask=True,
                                  )

        self.dataset = raw_train.map(
            tokenize_fn,
            batched=True,
            remove_columns=raw_train.column_names,
            desc="Tokenising",
        )
        self.dataset.set_format("torch")

    def train_dataloader(self) -> DataLoader:
        collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=self.mlm_probability,
        )
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collator,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

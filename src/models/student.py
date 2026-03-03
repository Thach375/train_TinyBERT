"""Student (TinyBERT) model builder

Responsibilities
----------------
* Construct a smaller BERT from a ``BertConfig`` derived from the YAML config
* Optionally load a warm-start checkpoint (general-distilled weights)
* Expose the same ``output_hidden_states=True, output_attentions=True``
  interface expected by the distillation losses
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from transformers import BertConfig, BertForMaskedLM, BertForSequenceClassification, BertModel


def build_student_config(cfg: Dict[str, Any]) -> BertConfig:
    """Create a ``BertConfig`` from the student YAML section

    Parameters
    ----------
    cfg : dict
        Flat dictionary with keys matching ``configs/model/student.yaml``

    Returns
    -------
    BertConfig
    """
    return BertConfig(
        vocab_size=cfg.get("vocab_size", 30522),
        hidden_size=cfg["hidden_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        intermediate_size=cfg["intermediate_size"],
        max_position_embeddings=cfg.get("max_position_embeddings", 512),
        hidden_act=cfg.get("hidden_act", "gelu"),
        hidden_dropout_prob=cfg.get("hidden_dropout_prob", 0.1),
        attention_probs_dropout_prob=cfg.get("attention_probs_dropout_prob", 0.1),
        attn_implementation="eager",
    )


def build_student_for_pretraining(cfg: Dict[str, Any]) -> BertForMaskedLM:
    """Build a student ``BertForMaskedLM`` (for general distillation)

    Parameters
    ----------
    cfg : dict
        Student model YAML section

    Returns
    -------
    BertForMaskedLM
        Randomly initialised student model
    """
    config = build_student_config(cfg)
    if cfg.get("name_or_path"):
        model = BertForMaskedLM.from_pretrained(cfg["name_or_path"], config=config)
    else:
        model = BertForMaskedLM(config)
    return model


def build_student_for_classification(
    cfg: Dict[str, Any],
    num_labels: int,
    pretrained_path: Optional[str] = None,
) -> BertForSequenceClassification:
    """Build a student ``BertForSequenceClassification`` (for task distillation)

    Parameters
    ----------
    cfg : dict
        Student model YAML section
    num_labels : int
        Number of target classes
    pretrained_path : str, optional
        Path to general-distilled checkpoint.  If given, the backbone weights
        are loaded (classification head is randomly initialised)

    Returns
    -------
    BertForSequenceClassification
    """
    config = build_student_config(cfg)
    config.num_labels = num_labels

    model = BertForSequenceClassification(config)

    if pretrained_path is not None:
        state = torch.load(pretrained_path, map_location="cpu", weights_only=False)
        # Lightning checkpoint wraps weights under "state_dict"
        if "state_dict" in state:
            state = state["state_dict"]

        # strip prefix "student." to "bert.*"
        cleaned: Dict[str, torch.Tensor] = {}
        for k, v in state.items():
            if not k.startswith("student."):
                continue  # bỏ teacher.*, fit_dense.*, optimizer states...
            new_k = k[len("student."):]  # "student.bert.encoder...." to "bert.encoder...."
            if new_k.startswith("cls."):
                continue  # bỏ MLM head, không dùng trong classification
            cleaned[new_k] = v

        missing, unexpected = model.load_state_dict(cleaned, strict=False)

        # pooler và classifier bị missing là bình thường
        expected_missing = {"bert.pooler.dense.weight", "bert.pooler.dense.bias",
                            "classifier.weight", "classifier.bias"}
        real_missing = [m for m in missing if m not in expected_missing]
        if real_missing:
            print(f"[Student] Missing keys (unexpected): {real_missing[:5]}…")
        if unexpected:
            print(f"[Student] Unexpected keys when loading ckpt: {unexpected[:5]}…")
        else:
            print(f"[Student] Backbone loaded successfully from: {pretrained_path}")

    return model


def build_student_encoder(cfg: Dict[str, Any]) -> BertModel:
    """Build a bare ``BertModel`` encoder for embedding/hidden-state extraction

    Useful when only the encoder backbone is needed (no head)
    """
    config = build_student_config(cfg)
    return BertModel(config)

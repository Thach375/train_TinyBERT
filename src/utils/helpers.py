"""Miscellaneous helpers

Responsibilities
----------------
* ``seed_everything`` — reproducibility across NumPy, PyTorch, and Python
* ``load_yaml`` — simple YAML loader with OmegaConf-style dot access
* ``build_layer_mapping`` — derive teacher→student layer index mapping from config, following the uniform strategy from the TinyBERT paper
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml


def seed_everything(seed: int = 42) -> None:
    """Set random seeds for reproducibility

    Parameters
    ----------
    seed : int
        Global seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a YAML file and return its contents as a dict

    Parameters
    ----------
    path : str or Path
        Path to the YAML file
        
    Returns
    -------
    dict
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_layer_mapping(
    teacher_num_layers: int,
    student_num_layers: int,
    strategy: str = "uniform",
) -> List[int]:
    """Create a teacher layer-index list for the student to mimic

    Parameters
    ----------
    teacher_num_layers : int
        Total number of transformer layers in the teacher
    student_num_layers : int
        Total number of transformer layers in the student
    strategy : str
        ``"uniform"`` — evenly spaced layers (default, matches TinyBERT paper)
        ``"last"``    — last *student_num_layers* of the teacher
        ``"first"``   — first *student_num_layers* of the teacher

    Returns
    -------
    list[int]
        0-indexed teacher layer indices of length ``student_num_layers``

    Example
    -------
    >>> build_layer_mapping(12, 4, "uniform")
    [2, 5, 8, 11]
    """
    if strategy == "uniform":
        step = teacher_num_layers / student_num_layers
        return [int(round((i + 1) * step)) - 1 for i in range(student_num_layers)]
    elif strategy == "last":
        return list(range(teacher_num_layers - student_num_layers, teacher_num_layers))
    elif strategy == "first":
        return list(range(student_num_layers))
    else:
        raise ValueError(f"Unknown mapping strategy: {strategy}")


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters

    Parameters
    ----------
    model : nn.Module
    trainable_only : bool
        If ``True``, count only parameters with ``requires_grad``

    Returns
    -------
    int
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

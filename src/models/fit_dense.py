"""Learnable linear projections for dimension alignment

Responsibilities
----------------
When the student hidden size (e.g. 312) differs from the teacher (e.g. 768),
we need *fit dense* (linear) layers to project student representations into
the teacher's space **before** computing the MSE loss

This module provides:
* ``FitDense``: a single ``nn.Linear`` projection
* ``FitDenseStack``: creates one ``FitDense`` per matched layer pair
  (embedding + N transformer layers)

Modern improvement over original TinyBERT
------------------------------------------
* Uses ``nn.ModuleDict`` so projections are named and checkpoint-friendly
* Optional layer-norm + activation after projection for smoother optimisation
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


class FitDense(nn.Module):
    """Single linear projection  student_dim → teacher_dim

    Parameters
    ----------
    in_features : int
        Student hidden dimension
    out_features : int
        Teacher hidden dimension
    use_layernorm : bool
        If ``True``, apply LayerNorm after the linear layer
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_layernorm: bool = False,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.layernorm: Optional[nn.LayerNorm] = (
            nn.LayerNorm(out_features) if use_layernorm else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project ``x`` from student space to teacher space"""
        out = self.linear(x)
        if self.layernorm is not None:
            out = self.layernorm(out)
        return out


class FitDenseStack(nn.Module):
    """A collection of ``FitDense`` modules — one per layer pair + embedding

    The stack has ``len(layer_indices) + 1`` projections:
    * Index ``0`` → embedding-layer projection
    * Index ``i`` (1-based) → i-th matched hidden-layer projection

    Parameters
    ----------
    student_hidden : int
        Student hidden size
    teacher_hidden : int
        Teacher hidden size
    num_layers : int
        Number of matched transformer-layer pairs (excluding embedding)
    use_layernorm : bool
        Passed to each ``FitDense``
    """

    def __init__(
        self,
        student_hidden: int,
        teacher_hidden: int,
        num_layers: int,
        use_layernorm: bool = False,
    ) -> None:
        super().__init__()
        # +1 for the embedding layer
        self.projections = nn.ModuleDict(
            {
                str(i): FitDense(student_hidden, teacher_hidden, use_layernorm)
                for i in range(num_layers + 1)
            }
        )

    def __getitem__(self, idx: int) -> FitDense:
        return self.projections[str(idx)]  # type: ignore[return-value]

    def __len__(self) -> int:
        return len(self.projections)

    def project_embedding(self, student_emb: torch.Tensor) -> torch.Tensor:
        """Project embedding-layer output"""
        return self.projections["0"](student_emb)

    def project_hidden(
        self,
        student_hiddens: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Project each matched hidden-layer output (1-indexed in the dict)"""
        return [
            self.projections[str(i + 1)](h)
            for i, h in enumerate(student_hiddens)
        ]

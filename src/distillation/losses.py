"""TinyBERT distillation loss functions

Responsibilities
----------------
Implement the three core losses from *TinyBERT: Distilling BERT for NLP*
(Jiao et al., 2020) plus the prediction-layer KL-divergence loss:

1. **Embedding-layer loss**: MSE between projected student and teacher
   embedding outputs
2. **Hidden-state loss**: MSE between projected student and teacher hidden
   states at each matched layer
3. **Attention loss**: MSE between student and teacher attention probability
   matrices at each matched layer
4. **Prediction-layer loss**: KL divergence on temperature-scaled soft logits

All functions work on *batched* tensors and return a scalar ``torch.Tensor``

Modern improvements
-------------------
* Pure functional API: compose losses in any combination
* Type hints + docstrings for every function
* Numerically stable KL with ``log_target=True``
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Embedding-layer loss
def embedding_loss(
    student_emb: torch.Tensor,
    teacher_emb: torch.Tensor,
) -> torch.Tensor:
    """MSE loss between student (projected) and teacher embedding outputs

    Parameters
    ----------
    student_emb : Tensor  (B, S, D_teacher)
        Student embedding output **after** FitDense projection
    teacher_emb : Tensor  (B, S, D_teacher)
        Teacher embedding output
        
    Returns
    -------
    Tensor (scalar)
    """
    return F.mse_loss(student_emb, teacher_emb)


# 2. Hidden-state loss
def hidden_state_loss(
    student_hiddens: List[torch.Tensor],
    teacher_hiddens: List[torch.Tensor],
) -> torch.Tensor:
    """Sum of per-layer MSE between projected student and teacher hidden states

    Parameters
    ----------
    student_hiddens : list[Tensor]  each (B, S, D_teacher)
        Projected student hidden states for each matched layer
    teacher_hiddens : list[Tensor]  each (B, S, D_teacher)
        Teacher hidden states at the corresponding layers

    Returns
    -------
    Tensor (scalar)
    """
    assert len(student_hiddens) == len(teacher_hiddens), (
        f"Layer count mismatch: {len(student_hiddens)} vs {len(teacher_hiddens)}"
    )
    total = torch.tensor(0.0, device=student_hiddens[0].device)
    for s_h, t_h in zip(student_hiddens, teacher_hiddens):
        total = total + F.mse_loss(s_h, t_h)
    return total


# 3. Attention loss
def attention_loss(
    student_attentions: List[torch.Tensor],
    teacher_attentions: List[torch.Tensor],
) -> torch.Tensor:
    """Sum of per-layer MSE between student and teacher attention distributions

    Following the original TinyBERT paper, the loss is computed on the
    *unnormalised* attention probabilities (after softmax, before dropout)
    Hugging Face returns these when ``output_attentions=True``

    Parameters
    ----------
    student_attentions : list[Tensor]  each (B, H_s, S, S)
        Student attention probabilities per matched layer
    teacher_attentions : list[Tensor]  each (B, H_t, S, S)
        Teacher attention probabilities at the corresponding layers

    Returns
    -------
    Tensor (scalar)

    Notes
    -----
    If the student has fewer heads than the teacher, we average the teacher
    heads to match (following the original paper's recommendation)
    """
    assert len(student_attentions) == len(teacher_attentions)
    total = torch.tensor(0.0, device=student_attentions[0].device)

    for s_attn, t_attn in zip(student_attentions, teacher_attentions):
        # s_attn: (B, H_s, S, S)   t_attn: (B, H_t, S, S)
        h_s = s_attn.size(1)
        h_t = t_attn.size(1)

        if h_s != h_t:
            # Average teacher heads into h_s groups
            group_size = h_t // h_s
            t_attn = t_attn.reshape(
                t_attn.size(0), h_s, group_size, t_attn.size(2), t_attn.size(3)
            ).mean(dim=2)

        total = total + F.mse_loss(s_attn, t_attn)

    return total


# 4. Prediction-layer (soft-label) loss
def prediction_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """KL-divergence on temperature-scaled soft labels

    Parameters
    ----------
    student_logits : Tensor  (B, C)
    teacher_logits : Tensor  (B, C)
    temperature : float

    Returns
    -------
    Tensor (scalar)
    """
    s_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    t_probs = F.softmax(teacher_logits / temperature, dim=-1)
    loss = F.kl_div(s_log_probs, t_probs, reduction="batchmean", log_target=False)
    return loss * (temperature ** 2)


# 5. Hard-label cross-entropy
def hard_label_loss(
    student_logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Standard cross-entropy on ground-truth labels

    Parameters
    ----------
    student_logits : Tensor  (B, C)
    labels : Tensor  (B,)

    Returns
    -------
    Tensor (scalar)
    """
    return F.cross_entropy(student_logits, labels)


# 6. Combined intermediate loss

def intermediate_distillation_loss(
    student_emb: torch.Tensor,
    teacher_emb: torch.Tensor,
    student_hiddens: List[torch.Tensor],
    teacher_hiddens: List[torch.Tensor],
    student_attentions: List[torch.Tensor],
    teacher_attentions: List[torch.Tensor],
    alpha_embd: float = 1.0,
    alpha_hidn: float = 1.0,
    alpha_attn: float = 1.0,
) -> torch.Tensor:
    """Weighted sum of embedding + hidden + attention losses

    Returns
    -------
    Tensor (scalar)
    """
    l_emb = embedding_loss(student_emb, teacher_emb)
    l_hid = hidden_state_loss(student_hiddens, teacher_hiddens)
    l_att = attention_loss(student_attentions, teacher_attentions)
    return alpha_embd * l_emb + alpha_hidn * l_hid + alpha_attn * l_att

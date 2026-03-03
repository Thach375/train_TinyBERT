# TinyBERT — Knowledge Distillation from BERT

A clean re-implementation of **TinyBERT** (Jiao et al., 2020) using PyTorch Lightning + HuggingFace Transformers.  
Compresses **BERT-base-uncased** (109 M params) into a **4-layer / 312-dim student** (~14.5 M params) in two stages.

---

## Problem

Knowledge distillation trains a small *student* to mimic a large *teacher* by matching intermediate representations in addition to output predictions.  TinyBERT extends this to every layer of BERT:

**Stage 1 — General Distillation** on a large unlabelled corpus (WikiText-103):

$$\mathcal{L}_{\text{gen}} = \alpha_{\text{emb}}\.\mathcal{L}_{\text{emb}} + \alpha_{\text{hid}}\.\mathcal{L}_{\text{hid}} + \alpha_{\text{att}}\.\mathcal{L}_{\text{att}}$$

**Stage 2 — Task Distillation** on a downstream GLUE task (MNLI), two phases:

- **Phase 2a** (intermediate): same $\mathcal{L}_{\text{emb}} + \mathcal{L}_{\text{hid}} + \mathcal{L}_{\text{att}}$ on task data  
- **Phase 2b** (prediction): soft-label KL + optional hard-label CE

$$\mathcal{L}_{\text{task}} = \alpha_{\text{pred}}\.\mathcal{L}_{\text{KL}}(T) + \alpha_{\text{hard}}\.\mathcal{L}_{\text{CE}}$$

---

## Directory Structure

```
train_TinyBERT/
├── configs/
│   ├── distill_general.yaml      # Stage 1 hyperparameters
│   ├── distill_task.yaml         # Stage 2 hyperparameters
│   └── model/
│       ├── teacher.yaml          # Teacher architecture (BERT-base, 768-dim, 12L)
│       └── student.yaml          # Student architecture (TinyBERT, 312-dim, 4L)
│
├── src/
│   ├── models/
│   │   ├── student.py            # Build BertForMaskedLM / BertForSequenceClassification
│   │   └── fit_dense.py          # FitDenseStack: linear projections 312→768 per layer
│   ├── distillation/
│   │   ├── losses.py             # All loss functions
│   │   ├── general_distill.py    # LightningModule: Stage 1
│   │   └── task_distill.py       # LightningModule: Stage 2 (phases 2a & 2b)
│   ├── data/
│   │   ├── pretrain_data.py      # DataModule: WikiText-103 + MLM collation
│   │   └── glue_data.py          # DataModule: any GLUE task
│   └── utils/
│       ├── callbacks.py          # SaveStudentCallback (export HF format each epoch)
│       └── helpers.py            # seed_everything, load_yaml, layer-mapping
│
├── scripts/
│   ├── run_general_distill.py    # Stage 1 CLI
│   ├── run_task_distill.py       # Stage 2 CLI (2a → 2b pipeline)
│   └── evaluate.py               # Evaluate all runs, print accuracy table
│
└── outputs/
    ├── general_distill/          # Stage 1 checkpoints + student_hf/
    └── task_distill/
        └── <run_name>/           # Per-run checkpoints + student_hf/
```

---

## Loss Functions

All implemented in [`src/distillation/losses.py`](src/distillation/losses.py).

| Loss | Formula | Used in |
|---|---|---|
| **Embedding MSE** | $\mathrm{MSE}(W_e\,h^S_0,\; h^T_0)$ | Stage 1, 2a |
| **Hidden-state MSE** | $\frac{1}{N}\sum_i \mathrm{MSE}(W_i\,h^S_i,\; h^T_{m(i)})$ | Stage 1, 2a |
| **Attention MSE** | $\frac{1}{N}\sum_i \mathrm{MSE}(A^S_i,\; A^T_{m(i)})$ | Stage 1, 2a |
| **Soft-label KL** | $\mathrm{KL}(\sigma(z^S/T) \,\|\, \sigma(z^T/T))$ | Stage 2b |
| **Hard-label CE** | $\mathrm{CE}(z^S, y)$ | Stage 2b (optional, `alpha_hard > 0`) |

- $W_e, W_i$: learnable **FitDense** projections (student 312-dim → teacher 768-dim)  
- $m(i)$: teacher layer mapped to student layer $i$ (`teacher_layer_indices: [2, 5, 8, 11]`)  
- $T$: distillation temperature (default 4.0)

---

## Models & Datasets

| Role | Model | Params |
|---|---|---|
| Teacher (general) | `bert-base-uncased` | 109 M |
| Teacher (MNLI fine-tuned) | `textattack/bert-base-uncased-MNLI` | 109 M |
| Student | TinyBERT-4L-312H (trained from scratch) | ~14.5 M |

> **Note:** `textattack/bert-base-uncased-MNLI` uses MultiNLI label ordering  
> (0=contradiction, 1=entailment, 2=neutral), which differs from GLUE  
> (0=entailment, 1=neutral, 2=contradiction).  
> The training code applies `teacher_label_remap: [1, 2, 0]` automatically.

| Dataset | Task | Size |
|---|---|---|
| WikiText-103 | Stage 1 pre-training (MLM) | ~103 M tokens |
| GLUE / MNLI | Stage 2 fine-tuning (NLI, 3-class) | 393 k train / 9.8 k val |

Datasets download automatically via HuggingFace `datasets` and cache at `~/.cache/huggingface/datasets/`.

---

## Install

```bash
pip install -r requirements.txt
```

---

## Training

### Stage 1 — General Distillation

```bash
python scripts/run_general_distill.py
```

Output: `outputs/general_distill/last.ckpt` and `outputs/general_distill/student_hf/`

### Stage 2 — Task Distillation (baseline)

```bash
python scripts/run_task_distill.py
```

Runs Phase 2a (20 epochs) then Phase 2b (3 epochs) automatically.  
Output: `outputs/task_distill/baseline/student_hf/`

### Stage 2 — Ablation Studies

```bash
# Skip Phase 2b
python scripts/run_task_distill.py --run-name no_stage2b --skip-stage2b

# No attention loss
python scripts/run_task_distill.py --run-name no_attn_loss \
    --set distillation.alpha_attn=0.0

# No attention + no embedding loss
python scripts/run_task_distill.py --run-name no_attn_embd \
    --set distillation.alpha_attn=0.0 distillation.alpha_embd=0.0

# No Stage 1 (random init student)
python scripts/run_task_distill.py --run-name no_stage1 \
    --set paths.student_general_ckpt=null
```

Any config key can be overridden with `--set key.subkey=value`.

---

## Evaluation

```bash
# Validation set only (default)
python scripts/evaluate.py

# Train (5 k samples) + val — check for training vs. random-chance
python scripts/evaluate.py --splits train val

# Include teacher as upper-bound
python scripts/evaluate.py --splits train val --include-teacher

# Include raw Stage-1-only student as lower bound
python scripts/evaluate.py --splits train val --include-raw-student

# Specific runs only
python scripts/evaluate.py --runs baseline no_attn_loss

# Save to CSV
python scripts/evaluate.py --splits train val --output results.csv
```

The script auto-discovers all runs under `outputs/task_distill/`, auto-detects the best label permutation per model, and prints a summary table with overfitting gap and sanity diagnostics.

---

## Key Hyperparameters

| File | Parameter | Default |
|---|---|---|
| `distill_task.yaml` | `distillation.temperature` | 4.0 |
| `distill_task.yaml` | `distillation.alpha_pred` | 1.0 |
| `distill_task.yaml` | `distillation.alpha_hard` | 0.0 |
| `distill_task.yaml` | `distillation.alpha_attn/embd/hidn` | 1.0 |
| `distill_task.yaml` | `training.stage2a.max_epochs` | 20 |
| `distill_task.yaml` | `training.stage2b.max_epochs` | 3 |
| `distill_general.yaml` | `training.max_epochs` | 3 |

---

## References

- Jiao, X., Yin, Y., Shang, L., Jiang, X., Chen, X., Li, L., Wang, F., & Liu, Q. (2020). **TinyBERT: Distilling BERT for Natural Language Understanding**. *EMNLP 2020*. [[paper]](https://arxiv.org/abs/1909.10351)
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**. *NAACL 2019*. [[paper]](https://arxiv.org/abs/1810.04805)
- Williams, A., Nangia, N., & Bowman, S. (2018). **MultiNLI: A Broad-Coverage Challenge Corpus for Sentence Understanding**. *NAACL 2018*. [[paper]](https://arxiv.org/abs/1704.05426)
- Original TinyBERT repo: [huawei-noah/Pretrained-Language-Model/TinyBERT](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT)

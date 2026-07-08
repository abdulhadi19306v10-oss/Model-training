# IMDB Sentiment Classification — Fine-Tuning Experiments

Fine-tuned transformer models on the [IMDB sentiment dataset](https://huggingface.co/datasets/stanfordnlp/imdb) (binary classification: positive/negative movie reviews) using Hugging Face `transformers` on Google Colab (free tier, T4 GPU).

## Objective

Compare accuracy vs. compute cost across model size, dataset size, and sequence length to understand diminishing returns in fine-tuning.

## Experiments

### Baseline — DistilBERT, 2k samples, 256 tokens
- Model: `distilbert-base-uncased`
- Train samples: 2,000 / 25,000
- Max sequence length: 256
- Epochs: 2

| | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Negative | 0.86 | 0.87 | 0.87 | 254 |
| Positive | 0.86 | 0.86 | 0.86 | 246 |

**Accuracy: 86%**

---

### Option 1 — DistilBERT, full dataset, 256 tokens
- Model: `distilbert-base-uncased`
- Train samples: 25,000 (full)
- Max sequence length: 256
- Epochs: 3
- Training time: ~20–30 min

| | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Negative | 0.91 | 0.92 | 0.92 | 1000 |
| Positive | 0.92 | 0.91 | 0.92 | 1000 |

**Accuracy: 92%**

---

### Option 2 — BERT-base, full dataset, 256 tokens
- Model: `bert-base-uncased`
- Train samples: 25,000 (full)
- Max sequence length: 256
- Epochs: 3
- Training time: ~45–70 min

| | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Negative | 0.92 | 0.93 | 0.93 | 1000 |
| Positive | 0.93 | 0.92 | 0.92 | 1000 |

**Accuracy: 93%**

---

### Option 3 — BERT-base, full dataset, 512 tokens
- Model: `bert-base-uncased`
- Train samples: 25,000 (full)
- Max sequence length: 512
- Epochs: 3
- Training time: ~90–150 min

| | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Negative | 0.94 | 0.95 | 0.94 | 1000 |
| Positive | 0.95 | 0.94 | 0.94 | 1000 |

**Accuracy: 94%**

## Summary

| Experiment | Model | Train Size | Seq Length | Time | Accuracy |
|---|---|---|---|---|---|
| Baseline | DistilBERT | 2k | 256 | ~5–10 min | 86% |
| Option 1 | DistilBERT | 25k | 256 | ~20–30 min | 92% |
| Option 2 | BERT-base | 25k | 256 | ~45–70 min | 93% |
| Option 3 | BERT-base | 25k | 512 | ~90–150 min | 94% |

## Key Findings

- **Data quantity** gave the largest single gain: +6% accuracy (86% → 92%) from scaling train set 2k → 25k, at low added compute cost.
- **Model size** (DistilBERT → BERT-base) added +1% accuracy for ~2x training time — diminishing returns.
- **Sequence length** (256 → 512 tokens) added +1% accuracy for another ~2x training time — most reviews already fit within 256 tokens, so gains were marginal.
- Overall curve shows fast returns from more data, then a flattening curve as model size and sequence length increase — larger models/longer sequences cost significantly more compute for progressively smaller gains.

## Environment

- Platform: Google Colab (Free Tier)
- GPU: NVIDIA T4
- Libraries: `transformers`, `datasets`, `accelerate`, `evaluate`, `scikit-learn`
- Mixed precision: fp16 enabled throughout

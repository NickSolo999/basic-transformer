# Building a Transformer from Scratch

A ground-up, educational implementation of a GPT-style decoder-only transformer
in PyTorch — no pre-trained weights, no black boxes, every line explained.

This project walks through the full stack: from raw text → tokens → embeddings →
attention → transformer blocks → a trained language model → attention visualisations.
Each notebook is self-contained and builds on the previous one.

---

## Project Layout

```
basic-transformer/
│
├── 01_tokenization_and_embeddings.ipynb   ← tiktoken, nn.Embedding, sinusoidal PE
├── 02_self_attention.ipynb                ← scaled dot-product attention from scratch
├── 03_multi_head_attention.ipynb          ← multi-head attention, per-head heatmaps
├── 04_transformer_block.ipynb             ← full decoder block (MHA + FFN + norms)
├── 05_training.ipynb                      ← train a GPT-style model, generate text
├── 06_visualising_attention.ipynb         ← load checkpoint, attention rollout, heatmaps
│
├── utils/
│   ├── __init__.py                        ← re-exports model + visualisation symbols
│   ├── model.py                           ← all model classes (importable)
│   └── visualisation.py                  ← shared plotting helpers
│
├── checkpoints/                           ← saved model weights (created by notebook 05)
├── requirements.txt
└── README.md
```

---

## Notebooks at a Glance

| # | Notebook | What you'll build | Key concepts |
|---|----------|-------------------|--------------|
| 01 | Tokenization & Embeddings | tiktoken tokenizer, learnable embeddings, sinusoidal PE | BPE, vocabulary, embedding space geometry, PCA |
| 02 | Self-Attention | Scaled dot-product attention from scratch | Q/K/V projections, softmax, causal mask, attention heatmap |
| 03 | Multi-Head Attention | Multi-head module from scratch | Head splitting/merging, per-head specialisation |
| 04 | Transformer Block | Full GPT-2-style decoder block | Pre-norm, residual stream, FFN with GELU, shape tracking |
| 05 | Training | Train a small LM on a text corpus | Cross-entropy loss, Adam + warmup/cosine LR, text generation |
| 06 | Visualising Attention | Interpret a trained model | Per-layer/head heatmaps, attention rollout, interactive input |

---

## Architecture — GPT-Style Decoder

```
  Input tokens  [batch, seq_len]
       │
  ┌────▼────────────────────────────────┐
  │  Token Embedding  (vocab → d_model) │  × sqrt(d_model) scaling
  │  + Sinusoidal PE  (pos  → d_model)  │
  └────┬────────────────────────────────┘
       │
  ┌────▼──────────────────────────┐  ┐
  │  LayerNorm                    │  │
  │  Multi-Head Attention         │  │  × n_layers
  │  + Residual                   │  │
  │  LayerNorm                    │  │
  │  Feed-Forward (GELU)          │  │
  │  + Residual                   │  │
  └────┬──────────────────────────┘  ┘
       │
  ┌────▼────────────────────────────────┐
  │  LayerNorm (final)                  │
  │  LM Head: Linear (d_model → vocab)  │  ← weights tied to embedding
  └────┬────────────────────────────────┘
       │
  Logits  [batch, seq_len, vocab_size]
```

**Default hyperparameters** (notebook 05 trains in < 5 min on CPU):

| Parameter | Value |
|-----------|-------|
| `d_model` | 64 |
| `n_heads` | 4 |
| `n_layers` | 2 |
| `d_ff` | 256 |
| `max_seq_len` | 128 |
| `dropout` | 0.1 |
| Tokenizer | tiktoken `gpt2` (vocab = 50 257) |

---

## Quick Start

### 1 — Install dependencies

```bash
pip install -r requirements.txt
```

Or with conda:

```bash
conda create -n transformer python=3.11
conda activate transformer
pip install -r requirements.txt
```

### 2 — Launch Jupyter

```bash
jupyter lab
# or
jupyter notebook
```

### 3 — Run notebooks in order

Open notebooks `01` → `06` in sequence.  
Notebook `05` will write a checkpoint to `checkpoints/gpt_checkpoint.pt`.  
Notebook `06` loads that checkpoint — run `05` first.

---

## Importing the Model Directly

The `utils` package exposes everything so you can use the model outside notebooks:

```python
import sys
sys.path.append('..')   # if calling from a subdirectory

from utils.model import GPTModel, DEFAULT_CONFIG
from utils.visualisation import plot_attention_heatmap

# Build a model with default config
model = GPTModel(DEFAULT_CONFIG)
print(model.count_parameters())   # ~200 k params

# Or load a trained checkpoint
import torch
checkpoint = torch.load('checkpoints/gpt_checkpoint.pt', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

---

## Key Ideas Covered

- **Byte-Pair Encoding (BPE)** via tiktoken — how raw text becomes integer tokens
- **Embeddings** as a lookup table that learns to place similar tokens nearby
- **Sinusoidal positional encoding** — inject position information without learning it
- **Scaled dot-product attention** — `softmax(QK^T / sqrt(d_k)) · V`
- **Causal masking** — upper-triangular -inf mask so position *i* can only attend to ≤ *i*
- **Multi-head attention** — split d_model into independent heads, attend in parallel
- **Residual connections** — gradients flow directly through the network depth
- **Pre-norm (LayerNorm before sub-layer)** — stabilises training vs post-norm
- **GELU activation** — smooth, probabilistic alternative to ReLU used in GPT-2
- **Weight tying** — embedding and LM-head share the same weight matrix
- **Warmup + cosine LR schedule** — standard recipe for transformer training
- **Temperature / top-k sampling** — control diversity in text generation
- **Attention rollout** (Abnar & Zuidema 2020) — propagate attention through layers

---

## Requirements

| Library | Purpose |
|---------|---------|
| `torch >= 2.0` | Tensor ops, autograd, `nn.Module` |
| `tiktoken >= 0.5` | GPT-2 BPE tokenizer |
| `matplotlib >= 3.7` | Base plotting |
| `seaborn >= 0.12` | Heatmaps |
| `numpy >= 1.24` | Array ops |
| `scikit-learn >= 1.3` | PCA for embedding visualisation |
| `tqdm >= 4.65` | Progress bars |
| `ipywidgets >= 8.0` | Interactive widgets in notebook 06 |
| `jupyter >= 1.0` | Notebook environment |

No GPU required — the small default config trains comfortably on CPU.

---

## References

- Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (2017)
- Radford et al., [*Language Models are Unsupervised Multitask Learners*](https://openai.com/research/language-unsupervised) — GPT-2 (2019)
- Abnar & Zuidema, [*Quantifying Attention Flow in Transformers*](https://arxiv.org/abs/2005.00928) (2020) — attention rollout
- Andrej Karpathy, [*nanoGPT*](https://github.com/karpathy/nanoGPT) — minimalist GPT implementation

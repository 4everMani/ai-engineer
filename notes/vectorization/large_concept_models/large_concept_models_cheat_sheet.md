# Large Concept Models — Cheat Sheet

> **One-Liner:** LCMs replace token-level autoregressive generation with concept-level (sentence-level) generation in a continuous SONAR embedding space, enabling language-agnostic reasoning.

---

## Core Concepts at a Glance

| Concept | What it is | Why it matters |
|---------|-----------|---------------|
| **Concept** | A single sentence, represented as a 1024-dim vector | The atomic unit of reasoning — higher-level than tokens |
| **SONAR** | Frozen sentence encoder/decoder (200 languages) | Provides the shared "language of thought" space |
| **Base LCM** | Transformer + MSE regression → predict next vector | ❌ Fails due to mean collapse |
| **Diffusion LCM** | Iterative denoising from noise → next vector | ✅ Star contribution, avoids mean collapse |
| **QLCM** | RVQ discretization + standard next-token prediction | ✅ Faster alternative to Diffusion LCM |
| **Mean Collapse** | MSE predicts the average of all valid outputs | Produces bland, meaningless midpoint vectors |
| **RVQ** | Residual Vector Quantization (8 codebooks × 8192) | Converts 1024 floats → 8 discrete codes |

---

## The Pipeline (5 Steps)

```
Input Sentences
    │
    ▼
┌──────────────────────┐
│  SONAR Encoder       │  Sentence → 1024-dim vector (frozen)
│  (200 languages)     │
└──────────────────────┘
    │
    ▼
┌──────────────────────┐
│  LCM Core            │  Sequence of vectors → predict next vector
│  (Diffusion / QLCM)  │
└──────────────────────┘
    │
    ▼
┌──────────────────────┐
│  SONAR Decoder       │  1024-dim vector → sentence in target language (frozen)
│  (any language)      │
└──────────────────────┘
    │
    ▼
Output Sentence (any of 200 languages!)
```

---

## Key Numbers

| Metric | Value |
|--------|-------|
| SONAR embedding dimension | 1024 |
| Languages supported (text) | 200 |
| Languages supported (speech) | 76 |
| Flagship model parameters | 1.6B |
| Training data | 1.3T tokens (multilingual) |
| VRAM for weights (FP16) | ~3.2 GB |
| RVQ codebooks | 8 |
| RVQ codebook size | 8,192 entries each |
| Diffusion denoising steps | ~100 |

---

## Why Not Just Use MSE?

```
After "Tim wasn't very athletic":

Valid next sentence A: [0.6, -1.0, 0.9]  "He joined a sport"
Valid next sentence B: [0.2,  0.5, -0.3]  "Friends teased him"
Valid next sentence C: [-0.4, 0.8, 0.1]  "He didn't care"

MSE prediction = Average = [0.13, 0.1, 0.23] → ❌ Meaningless!
Diffusion sample = One of A, B, or C    → ✅ Specific & valid!
```

---

## Complexity Comparison

| Model | Attention Cost | Steps per Concept |
|-------|---------------|-------------------|
| Token LLM | O(400N²) | ~20 |
| Diffusion LCM | O(N²) × 100 | ~100 |
| QLCM | O(N²) + O(64) | 8 |

---

## Quick Decision Tree

- **Need language-agnostic reasoning?** → LCM
- **Need highest quality generation?** → Diffusion LCM
- **Need fastest inference?** → QLCM (Two-Tower)
- **Need token-level precision?** → Standard LLM (LCM isn't there yet)

---

> **Status:** Phases 1–2. Will be updated as dissection continues.

# Large Concept Models — Glossary

| Term | Definition | Context / Example |
|------|-----------|-------------------|
| **LCM** | Large Concept Model — a generative model that predicts the next *concept vector* instead of the next token. | The central contribution of this paper. |
| **Concept** | The atomic unit of meaning in an LCM; operationally defined as a single sentence. | "Tim wasn't very athletic." is one concept. |
| **SONAR** | A pre-trained, frozen sentence encoder/decoder by Meta that maps sentences into a shared 1024-dim embedding space across 200 languages. | The "language of thought" that the LCM reasons in. |
| **Embedding Space** | A high-dimensional vector space where data points (sentences) are represented as coordinates. Similar meanings → nearby coordinates. | SONAR maps "cat on mat" (English) and "chat sur tapis" (French) to nearby points. |
| **Frozen** | A neural network whose weights are locked and not updated during training. | SONAR encoder/decoder are frozen; only the LCM core is trained. |
| **Autoregressive** | A generation strategy where each output depends on all previous outputs, produced one step at a time. | LLMs generate token-by-token; LCMs generate concept-by-concept. |
| **Self-Attention** | The Transformer mechanism where every position in a sequence computes relevance scores against every other position. Scales O(N²). | The reason why long sequences are computationally expensive. |
| **MSE (Mean Squared Error)** | A loss function that measures the average squared difference between predicted and target values. | Used by Base LCM; causes mean collapse. |
| **Mean Collapse** | A failure mode where MSE regression predicts the arithmetic average of all valid outputs, resulting in a meaningless blurry midpoint. | Base LCM produces bland, generic text because of this. |
| **Diffusion Model** | A generative model that creates outputs by iteratively denoising random noise into structured data. | Used by Stable Diffusion for images; used here for concept vectors. |
| **Forward Process** | The noise-adding phase of diffusion: progressively corrupts clean data with Gaussian noise across T timesteps. | $c^{(t)} = \alpha_t c + \sigma_t \epsilon$ |
| **Reverse Process** | The denoising phase of diffusion: starts from pure noise and iteratively removes noise to recover clean data. | The "sculpting from marble" phase during inference. |
| **Gaussian Noise** | Random values sampled from a normal distribution $\mathcal{N}(0, I)$. | The raw material that diffusion starts with and sculpts into concepts. |
| **Variance-Preserving Schedule** | A noise schedule where $\alpha_t^2 + \sigma_t^2 = 1$, ensuring the noisy signal maintains unit variance throughout the process. | Prevents the magnitude of vectors from exploding or collapsing during diffusion. |
| **RVQ (Residual Vector Quantization)** | A technique that converts a continuous vector into a sequence of discrete codes by iteratively quantizing residual errors across multiple codebooks. | Converts 1024 floats → 8 integer codes. |
| **Codebook** | A lookup table of pre-learned prototype vectors used in quantization. | Each of the 8 RVQ codebooks has 8,192 prototype entries. |
| **Residual** | The error remaining after approximating a vector with a codebook entry; passed to the next codebook for further refinement. | $r_1 = c - \text{Codebook}_1[q_1]$ |
| **QLCM** | Quantized Large Concept Model — uses RVQ to discretize SONAR vectors, then applies standard next-token prediction. | Faster alternative to Diffusion LCM. |
| **One-Tower QLCM** | A QLCM variant using a single Transformer that interleaves context and quantized codes in one sequence. | Simpler but longer sequence lengths. |
| **Two-Tower QLCM** | A QLCM variant using separate Transformers for context encoding and code generation. | More efficient; Tower 2 only predicts 8 codes. |
| **Pre-Layer-Norm** | A Transformer variant where LayerNorm is applied *before* the attention/FFN sublayer rather than after. Improves training stability. | Used in the Base LCM architecture (`norm_first=True`). |
| **Cosine Similarity** | A metric measuring the angular alignment between two vectors (1.0 = identical direction, 0.0 = orthogonal). | Used to verify that SONAR encodes same-meaning sentences nearby. |
| **Quadratic Scaling** | Computational cost grows as the square of the input size: doubling N quadruples cost. | Self-attention is O(N²); 10× more tokens = 100× more compute. |
| **Token** | A subword unit (e.g., "un", "##happy") produced by a tokenizer; the atomic unit of standard LLMs. | LLMs predict tokens; LCMs predict concepts (sentences). |
| **FP16** | Half-precision floating point (16-bit). Reduces VRAM usage by ~50% compared to FP32 with minimal quality loss. | 1.6B parameter model ≈ 3.2 GB at FP16. |

---

> **Status:** Phases 1–2 terms. Will be extended as dissection continues.

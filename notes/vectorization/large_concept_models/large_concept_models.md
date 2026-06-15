# Large Concept Models

> **Paper:** *Large Concept Models: Language Modeling in a Sentence Representation Space*
> **Authors:** Meta FAIR (Facebook AI Research)
> **Category:** Vectorization / Sentence-Level Generative Models
> **Status:** 🔬 Dissection In Progress (Phases 1–2 complete)

---

## Summary

This paper proposes **Large Concept Models (LCMs)** — a paradigm shift from token-level to sentence-level (concept-level) autoregressive generation. LCMs operate in the continuous SONAR embedding space (1024 dimensions, 200 languages) and predict the next concept vector using diffusion-based denoising, enabling language-agnostic, compute-efficient reasoning at a higher level of abstraction.

---

## Notes & Resources

### 📝 Written Notes
- [Detailed Notes](large_concept_models_detailed_notes.md) — Full exhaustive breakdown covering Phase 1 (Mental Model) and Phase 2 (Methodology & Math) with code examples
- [Cheat Sheet](large_concept_models_cheat_sheet.md) — Quick-reference summary of key concepts, pipeline, numbers, and decision tree
- [Glossary](large_concept_models_glossary.md) — Definitions of all technical terms with context and examples
- [Q&A Session](large_concept_models_qna.md) — All interactive questions asked during dissection with detailed answers

### 🖼️ Architecture Diagrams
- [LCM Architecture Overview](diagrams/lcm_architecture.png) — End-to-end pipeline: Encoder → LCM Core (3 variants) → Decoder
- [SONAR Embedding Space](diagrams/sonar_embedding_space.png) — Visualization of multilingual semantic clustering
- [Mean Collapse Problem](diagrams/mean_collapse_problem.png) — Why MSE regression fails and how Diffusion fixes it
- [Diffusion Denoising Process](diagrams/diffusion_denoising.png) — Step-by-step noise-to-concept sculpting
- [RVQ Quantization](diagrams/rvq_quantization.png) — How continuous vectors are discretized into 8 codes

### 📄 Source Paper
- [Facebook Large Concept Models.pdf](../../papers/vectorization/Facebook%20Large%20Concept%20Models.pdf)

---

## Dissection Progress

| Phase | Topic | Status |
|-------|-------|--------|
| Phase 1 | High-Level Mental Model | ✅ Complete |
| Phase 2 | Core Methodology & Mathematics | ✅ Complete |
| Phase 3 | The Data Pipeline | ⏳ Pending |
| Phase 4 | Real-World Impact & Deep Technical Dive | ⏳ Pending |
| Phase 5 | Fine-Tuning Patterns & Ecosystem Evolution | ⏳ Pending |
| Phase 6 | Results & Ablation Studies | ⏳ Pending |
| Phase 6.5 | Failure Modes & Boundary Analysis | ⏳ Pending |
| Phase 7 | Documentation & Engineering Critique | ⏳ Pending |

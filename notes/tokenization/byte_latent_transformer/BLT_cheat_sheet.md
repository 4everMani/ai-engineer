# Cheat Sheet: Byte Latent Transformer (BLT)

## The Core Concept
Standard tokenization groups characters based on rigid statistical dictionaries, causing issues with rare words, multilingual text, and reasoning. The **Byte Latent Transformer (BLT)** abandons tokenization entirely, reading raw bytes and dynamically grouping them into **Patches** based on their predictability (Entropy).

## Architecture Breakdown
1. **Local Encoder (The Scout):** A lightweight module that reads raw bytes and predicts the next byte to calculate entropy.
2. **Entropy Patcher:** Groups bytes into "Patches". 
   - *Low Surprise (Predictable)* $\rightarrow$ Groups many bytes into one massive Patch (saves compute).
   - *High Surprise (Complex/Novel)* $\rightarrow$ Creates boundaries, making tiny Patches (allocates heavy compute).
3. **Latent Transformer (The Brain):** A massive autoregressive model that processes the Patch representations. Bypasses the $O(N^2)$ bottleneck of byte-level processing.
4. **Local Decoder (The Translator):** Unpacks Patch representations back into raw bytes.

## Key Mathematical Concept
**Entropy Calculation for Boundaries:**
$$H_t = - \log P(b_{t+1} | b_{\le t})$$
If $H_t$ (surprise) crosses a dynamic threshold, a patch boundary is created.

## Engineering Impact
- **VRAM Massive Reduction:** Vocabulary drops from 100,000+ to exactly 256. Embedding matrices shrink by 99.7%.
- **Robustness:** Unfazed by obfuscation, typos, or completely novel data formats.
- **Multimodal Future:** By processing raw bytes natively, it lays the groundwork for a truly unified model handling text, audio, and visual binaries without specialized tokenizers.

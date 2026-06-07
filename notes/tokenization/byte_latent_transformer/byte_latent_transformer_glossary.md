# Glossary: Byte Latent Transformer

| Term | Definition | Context / Example |
|------|------------|-------------------|
| **Tokenizer** | A preprocessing algorithm that splits text into a fixed dictionary of integer IDs. | E.g., BPE or SentencePiece. Often struggles with typos or novel code syntax. |
| **Entropy** | A mathematical measure of unpredictability or "surprise" in a sequence. | In BLT, if the next byte is hard to guess (e.g., a random digit in math), entropy is high. |
| **Patch** | A dynamically sized chunk of bytes compressed into a single vector representation for the Latent Transformer to process. | "Hello world!" might be one single patch, while "3.14" might be split into 4 separate patches. |
| **Local Encoder** | A lightweight neural network component designed to read raw bytes sequentially and compress them. | Referred to as "The Scout" in our dissection; it handles the byte-level processing quickly. |
| **Latent Space** | An abstract, compressed mathematical representation of data. | The "Brain" of the BLT operates in the latent space (Patch Vectors) rather than raw byte space. |
| **O(N^2) Complexity** | The quadratic scaling bottleneck of standard Transformer attention mechanisms. | If sequence length doubles, compute time quadruples. BLT solves this by patching bytes together, shortening sequence length $N$. |
| **MinHash** | A technique used to quickly estimate how similar two sets are, widely used for deduplicating massive web corpuses. | Used by FineWeb-Edu to ensure the model doesn't overfit on repetitive internet boilerplate. |

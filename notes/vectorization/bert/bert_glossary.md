# BERT Glossary

| Term | Definition | Context/Example |
|------|------------|-----------------|
| **Autoregressive** | A model that predicts the next item in a sequence based only on the previous items. | GPT is autoregressive; it reads left-to-right to generate text. |
| **Bidirectional** | An architecture where the context of a word is derived from both its left and right surroundings simultaneously. | BERT understands "bank" by looking at words before and after it at the same time. |
| **Tokenization** | The process of converting raw text into smaller chunks (words or sub-words) that the model can process numerically. | BERT uses WordPiece, turning "playing" into "play" and "##ing". |
| **Masked Language Model (MLM)** | A training objective where random words are hidden from the input, and the model must predict them using context. | "The cat [MASK] on the mat." |
| **Embedding** | A mathematical vector (list of numbers) representing the meaning or position of a word in high-dimensional space. | BERT represents words as 768-dimensional embeddings. |
| **[CLS] Token** | A special classification token placed at the beginning of every sequence. | Used in NSP to aggregate the entire sequence's meaning into one vector. |
| **[SEP] Token** | A separator token used to mark the boundary between two different sentences. | `[CLS] Sentence A [SEP] Sentence B [SEP]` |
| **RAG (Retrieval-Augmented Gen.)** | An architecture where an embedding model retrieves relevant documents to provide context to a generation model. | Using a BERT descendant to find legal clauses, then using GPT-4 to summarize them. |

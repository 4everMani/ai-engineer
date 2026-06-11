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
| **Fine-tuning** | The process of taking a pre-trained model and training it further on a small, specific dataset for a specific task. | Adding a classification layer on top of BERT to classify IMDB movie reviews. |
| **LoRA / Adapters** | Techniques for Parameter-Efficient Fine-Tuning (PEFT) where the base model is frozen, and only tiny inject layers are trained. | Tuning BERT on a laptop by only updating 0.5% of its total parameters. |
| **RoBERTa** | A massively optimized version of BERT trained on 10x more data with the NSP task removed. | Used widely in industry because it strictly outperforms standard BERT. |
| **[UNK] Token** | The "Unknown" token. Used when a character is completely unrecognizable and isn't in the vocabulary. | If BERT encounters an incredibly rare foreign character, it replaces it with `[UNK]`. |

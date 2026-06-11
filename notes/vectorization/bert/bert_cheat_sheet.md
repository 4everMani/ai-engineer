# BERT Cheat-sheet

### The Core Problem Solved
Traditional language models read text left-to-right (like OpenAI GPT) or shallowly glued left-to-right and right-to-left together (like ELMo). This prevents them from truly understanding the context of a word based on its future surroundings.

### The BERT Solution
BERT uses a Deep Bidirectional Transformer Encoder. Every word attends to every other word simultaneously.

### The Two Pre-training Tasks
1.  **Masked Language Model (MLM):** 15% of tokens are chosen at random. Of those, 80% are replaced with `[MASK]`, 10% are replaced with a random word, and 10% are left unchanged. The model must guess the original word. This prevents the model from "cheating" while looking in both directions.
2.  **Next Sentence Prediction (NSP):** The model is fed two sentences. 50% of the time, the second sentence logically follows the first. 50% of the time, it's a random sentence. The `[CLS]` token acts as the classifier to predict if they belong together.

### The Three Embeddings
To prevent the model from treating sentences like a "bag of words", BERT adds three vectors together for every word:
1.  **Token Embedding:** The core meaning of the sub-word (from a ~30,522 WordPiece vocabulary).
2.  **Segment Embedding:** A binary indicator (A or B) telling the model which sentence the word belongs to.
3.  **Position Embedding:** The physical index of the word (0 to 512).

### Real-World Use Case
BERT is an **Encoder**. It is used for tasks that require deep comprehension, classification, or semantic search (Vector Databases/RAG pipelines). It is NOT used for text generation (which requires Decoders like GPT).

### Fine-Tuning & Variants
*   **Fine-Tuning:** Done by taking the `[CLS]` token or all token vectors and adding a tiny task-specific Neural Network layer on top. 
*   **Variants:** The community optimized BERT rapidly. **RoBERTa** trained on 10x data without NSP. **DistilBERT** shrunk it by 40% while retaining 97% performance. **mBERT** proved it could learn 104 languages simultaneously.

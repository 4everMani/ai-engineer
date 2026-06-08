# BERT Glossary

| Term | Definition | Context / Example |
| :--- | :--- | :--- |
| **Unidirectional Model** | A language model that processes text strictly in one direction (usually left-to-right). | OpenAI's original GPT. It can only use the words *before* a target word to understand its meaning. |
| **Shallow Bidirectional** | Training two separate networks (one L->R, one R->L) and concatenating their final outputs. | ELMo. The networks do not interact during the deep layers. |
| **Deep Bidirectional** | A model where every word attends to every other word (left and right) simultaneously at every layer. | BERT. The Transformer Encoder allows "bank" to mathematically pull context from both "river" (before) and "muddy" (after) at the same time. |
| **Masked Language Model (MLM)** | A training objective where a percentage of input tokens are hidden, and the model must predict them based on surrounding context. | `The cat [MASK] on the mat.` Forces the model to build a deep understanding of the sentence rather than just predicting the next word. |
| **Next Sentence Prediction (NSP)** | A training objective where the model receives two sentences and must predict if the second logically follows the first. | Helps the model learn the relationship between sentences, crucial for tasks like Question Answering. |
| **`[CLS]` Token** | Classification Token. Inserted at the beginning of every sequence. | BERT uses the final hidden state of this token as the aggregate representation for the entire sequence (used for NSP). |
| **`[SEP]` Token** | Separator Token. Used to explicitly separate two different sentences in a single input. | `[CLS] I love cats [SEP] They are cute [SEP]` |
| **Token Embedding** | A learned vector representation of a specific word or subword's semantic meaning. | The raw dictionary definition vector for "cats". |
| **Segment Embedding** | A learned vector that indicates which sentence a token belongs to (Sentence A or Sentence B). | Acts like a highlighter (e.g., yellow for Sentence A, blue for Sentence B) so the model can differentiate them in the "bag of words" soup. |
| **Position Embedding** | A learned vector that encodes the absolute sequential position of a token. | Acts as a GPS coordinate (e.g., "I am at Position 3"). Essential because Transformers process all tokens simultaneously and have no inherent concept of order. |

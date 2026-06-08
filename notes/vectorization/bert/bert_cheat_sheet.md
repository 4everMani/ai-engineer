# BERT: Cheat Sheet (Up to Phase 2)

## 1. The Core Problem
Older models like OpenAI's GPT were **unidirectional** (left-to-right), making them bad at understanding deep context. Models like ELMo were **shallow bidirectional** (concatenated independent left-to-right and right-to-left networks), which prevented true bidirectional feature learning.

## 2. The Innovation (Deep Bidirectionality)
BERT uses a Transformer Encoder to look at all words simultaneously in both directions. 

To prevent the model from "cheating" and simply copying the next word, BERT introduced two new pre-training tasks:
*   **Masked Language Model (MLM):** Randomly hides 15% of the words and forces the model to use the surrounding context to guess them (Fill-in-the-blank).
*   **Next Sentence Prediction (NSP):** Given Sentence A and Sentence B, predict if Sentence B logically follows A in the original text.

## 3. Input Representation (The 3 Embeddings)
BERT converts tokens into mathematical vectors by adding three specific embeddings together:
1.  **Token Embedding:** The raw dictionary definition of the word.
2.  **Segment Embedding:** Identifies which sentence the word belongs to (e.g., Sentence A vs Sentence B).
3.  **Position Embedding:** Acts as a GPS coordinate telling the model the absolute position of the word in the sequence.

$$ \text{Final Input Vector} = \text{Token Embedding} + \text{Segment Embedding} + \text{Position Embedding} $$

## 4. Special Tokens
*   `[CLS]`: Placed at the very start of the input. Its final output vector is used for the NSP classification task.
*   `[SEP]`: Used to separate Sentence A from Sentence B.
*   `[MASK]`: Used to hide a word for the MLM prediction task.

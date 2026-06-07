# Byte Pair Encoding (BPE) Cheat Sheet

## The Core Problem
Neural Machine Translation (NMT) traditionally used a fixed-size vocabulary, struggling with rare words, out-of-vocabulary words, compound words, and names. 

## The Solution
Instead of word-level tokenization, use **subword units**. 
BPE is an iterative compression algorithm adapted for text:
1. Initialize the vocabulary with characters (and `</w>` for word boundaries).
2. Count frequencies of all adjacent pairs.
3. Merge the most frequent pair into a single new symbol.
4. Repeat for $N$ iterations.

## Key Benefits
- **Open-Vocabulary**: Can represent any word, even unseen ones, via smaller subword units or characters.
- **Transparent Translation**: Can translate pieces of compound words independently (e.g., smart+est).
- **Efficiency**: Reduces vocabulary size while keeping text sequences reasonably short.

## Joint BPE
Learning BPE merges on the concatenated text of *both* the source and target languages to ensure consistent subword splitting across languages, improving alignment.

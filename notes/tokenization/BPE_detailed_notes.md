# Detailed Notes: Neural Machine Translation of Rare Words with Subword Units (The BPE Paper)

## Phase 1: High-Level Mental Model

### The Core Problem
Before this paper (circa 2015), Neural Machine Translation (NMT) models had a severe limitation: they operated with a **fixed vocabulary size** (typically around 30,000 to 50,000 words). However, language is fundamentally an *open-vocabulary* problem. 

This meant that whenever the network encountered a word it hadn't seen enough times during training, it would just emit a dreaded `<UNK>` (Unknown) token. This commonly happened with:
1. **Rare words** (words that just don't appear often in the training data).
2. **Names** (e.g., "Barack Obama").
3. **Compound words** (especially in languages like German, e.g., *Abwasserbehandlungsanlange* or "sewage water treatment plant").

The prevailing workaround at the time was to use a "back-off dictionary." If the model output `<UNK>`, the system would try to copy the source word over directly or look it up in a traditional translation dictionary. This was clunky, often failed with languages that had different alphabets (requiring transliteration), and completely failed to generate entirely *new* words.

### The Proposed Solution
The authors made a brilliant observation: **even if a word is "rare" or "unseen", a competent translator can often figure out how to translate it by looking at its subword units (morphemes or phonemes).**

To allow the neural network to do this, they adapted a simple data compression technique called **Byte Pair Encoding (BPE)**. Instead of treating text as a sequence of full words, BPE creates a vocabulary of *variable-length subword units*. High-frequency words (like "the") get merged back into single tokens, while rare or novel words (like "smartest") are broken down into logical subword components (e.g., "smart" and "est").

---

## Phase 2: Core Methodology, Mathematics & Code

### A Concrete BPE Walkthrough

Let's imagine we have a tiny training dataset containing only four words. We count how many times each word appears to get their frequencies:
- **hello** (appears 3 times)
- **head** (appears 2 times)
- **bed** (appears 1 time)
- **hard** (appears 1 time)

#### **Step 0: Initial Character Splitting**
First, we split every word into its individual characters. We also append a special end-of-word symbol, `</w>`, to the end of each word. (This symbol is crucial because it tells the model the difference between "est" at the end of "smart**est**" versus "est" at the beginning of "**est**ablish").

Our starting vocabulary looks like this:
```text
h e l l o </w>  (frequency: 3)
h e a d </w>    (frequency: 2)
b e d </w>      (frequency: 1)
h a r d </w>    (frequency: 1)
```

#### **Step 1: Merge Iteration 1**
We scan our dataset and count every pair of adjacent symbols:
- `(h, e)` appears 5 times (3 from hello, 2 from head).
- `(d, </w>)` appears 4 times (2 from head, 1 from bed, 1 from hard).
- `(l, l)` appears 3 times (from hello).

The most frequent pair is `(h, e)`. We create a new rule: **merge `h` and `e` into a single subword unit `he`**. We apply this rule to our dataset:
```text
he l l o </w>   (frequency: 3)
he a d </w>     (frequency: 2)
b e d </w>      (frequency: 1)
h a r d </w>    (frequency: 1)
```

#### **Step 2: Merge Iteration 2**
We count the pairs again based on the *new* state of the data:
- `(d, </w>)` is now the most frequent, appearing 4 times. 
- `(he, l)` appears 3 times.

We create our second rule: **merge `d` and `</w>` into `d</w>`**.
```text
he l l o </w>   (frequency: 3)
he a d</w>      (frequency: 2)
b e d</w>       (frequency: 1)
h a r d</w>     (frequency: 1)
```

#### **Step 3: Merge Iteration 3**
Count pairs again:
- `(he, l)` appears 3 times.
- `(l, l)` appears 3 times.
- `(l, o)` appears 3 times.

Since it's a tie, the algorithm picks one (e.g., `(he, l)`). Rule: **merge `he` and `l` into `hel`**.
```text
hel l o </w>    (frequency: 3)
he a d</w>      (frequency: 2)
b e d</w>       (frequency: 1)
h a r d</w>     (frequency: 1)
```

#### **The Resulting Vocabulary & Test Time Application**
Our base vocabulary is now `[h, e, l, o, a, d, b, r, </w>]` plus our newly minted subword tokens: `[he, d</w>, hel]`. 
At test time, if the network sees the completely novel word "**held**", it splits it: `h e l d </w>`, applies the merge rules, and translates the sequence `[hel, d</w>]`.

### Math-to-Code Translation (Algorithm 1 from the paper)
```python
import re, collections

def get_stats(vocab):
    """Counts the frequency of all adjacent symbol pairs in the vocabulary."""
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    """Replaces all occurrences of the most frequent pair with the merged symbol."""
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

# The initial dictionary state with frequencies calculated from the training corpus
vocab = {
    'h e l l o </w>': 3, 
    'h e a d </w>': 2,
    'b e d </w>': 1, 
    'h a r d </w>': 1
}

num_merges = 3 # Hyperparameter

for i in range(num_merges):
    pairs = get_stats(vocab)
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)
    print(f"Merge #{i+1}: {best[0]} + {best[1]} -> {''.join(best)}")
```

---

## Phase 3: The Data Pipeline (The Secret Sauce)

### 1. The Dataset
The authors used standard datasets from the WMT 2015 shared task:
- **English-to-German:** 4.2 million sentence pairs (about 100 million tokens).
- **English-to-Russian:** 2.6 million sentence pairs (about 50 million tokens).

### 2. "Joint BPE" and Cross-Alphabet Hacks
If you run BPE independently on English and German, you might split the same name differently in both languages (e.g., splitting "Barack" into `Ba` + `rack` in English, but `B` + `arack` in German). This makes it harder for the neural network to map them together. 

To fix this, they introduced **Joint BPE**: they concatenated the English and German vocabularies together and learned the BPE merge rules on the *union* of both languages.

For **English to Russian**, which uses a completely different Cyrillic alphabet, they used a very specific heuristic:
1. They transliterated the Russian vocabulary into Latin characters using ISO-9.
2. They combined this Latin-Russian text with the English text.
3. They learned the Joint BPE merge rules on this combined Latin text.
4. Finally, they transliterated the learned BPE merge operations *back* into Cyrillic to apply them to the actual Russian training text. 

### 3. The Critical Flaw: Where is the data cleaning?
Because this paper was published in 2015, they simply relied on the raw WMT 2015 competition data and the standard `Moses` tokenization scripts.
- **No Deduplication heuristics** are mentioned.
- **No Toxicity/PII filters** are documented.
By modern AI engineering standards, publishing a data pipeline without detailing exact deduplication ratios and safety filtering is considered a critical flaw. 

---

## Phase 4: Results & Pragmatic Engineering Critique

### The Results
The authors evaluated their success using the **Unigram F1 Score** to measure how many rare or unknown words were translated correctly. 
- Traditional back-off dictionaries struggled heavily, often just copy-pasting English names into Russian text.
- The **BPE Subword Model** drastically improved both precision and recall for rare and unseen words, effectively handling transliterations and generating new compound words it had never seen during training.
- Overall, it improved the BLEU score by up to 1.3 points over the baseline.

### Pragmatic Engineer Thoughts
- **VRAM vs. Accuracy Trade-off**: The old way of using massive 500,000-word vocabularies required gigantic embedding matrices, which eat up immense amounts of GPU VRAM. BPE elegantly compresses the vocabulary (e.g., to 30,000 subwords), shrinking the embedding matrix, reducing memory footprint, and massively speeding up the final softmax layer computation during inference.
- **Sequence Length ($O(N^2)$ Complexity)**: Character-level models blow up the sequence length (e.g., a 10-word sentence becomes 50 characters). Because Transformers and RNN attention mechanisms scale poorly with sequence length, processing 50 tokens is way slower than 10. BPE hits the perfect sweet spot: high-frequency words remain 1 token long, while only rare words are split into multi-token sequences.
- **Deployment Feasibility**: BPE tokenizers are incredibly lightweight. They are purely deterministic, fast, and don't require loading massive dictionaries or running complex morphological parsers in a production environment.

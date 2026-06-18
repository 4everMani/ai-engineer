# Module 1: Python Foundations — Complete Notes

> **Goal:** Master the Python syntax essentials needed to read and write AI/ML code fluently.
> **Platform:** All code runs on Google Colab — copy-paste into a cell and run.

---

## 🗂️ Quick-Reference Cheat Sheet

```python
# --- Variables & Types ---
x = 42                          # int (arbitrary precision)
lr = 3e-4                       # float (scientific notation)
name = "bert"                   # str
is_training = True              # bool
bias = None                     # NoneType

type(x)                         # <class 'int'>
isinstance(x, int)              # True

# --- F-Strings ---
print(f"Loss: {loss:.4f}")      # 4 decimal places
print(f"Acc: {acc:.1%}")        # percentage format
print(f"{'Name':>20}")          # right-align in 20 chars
print(f"{value!r}")             # repr() inside f-string

# --- Type Hints ---
def train(model: nn.Module, lr: float = 3e-4) -> float: ...
x: list[int] = [1, 2, 3]
y: Optional[str] = None         # from typing import Optional
z: dict[str, int] = {"a": 1}

# --- Conditionals ---
if x > 0:       ...
elif x == 0:    ...
else:           ...

# --- Loops ---
for i in range(10):             ...
for i, val in enumerate(lst):   ...
for a, b in zip(list1, list2):  ...
while condition:                ...

# --- Comprehensions ---
squares = [x**2 for x in range(10)]
evens = [x for x in range(10) if x % 2 == 0]
vocab = {tok: i for i, tok in enumerate(tokens)}
unique = {x.lower() for x in words}

# --- Unpacking ---
a, b, c = 1, 2, 3
first, *rest = [1, 2, 3, 4]
inputs, labels = batch          # DataLoader unpacking

# --- Walrus Operator ---
if (n := len(data)) > 100:  print(f"Large dataset: {n}")

# --- Ternary ---
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Debugging ---
print(f"{x=}")                  # Python 3.8+ debug shorthand
repr(obj)                       # unambiguous, developer-facing
str(obj)                        # human-readable
```

---

## Section 1: Python Syntax Essentials & Dynamic Typing

### 1.1 Indentation IS the Syntax

In Java/C# you use `{}` to define blocks. In Python, **indentation** (4 spaces by convention) defines blocks. This is enforced by the parser — not a style choice.

```python
# ✅ Correct — 4 spaces per indentation level
def train_one_epoch(model, dataloader, optimizer, criterion):
    """Train the model for one epoch and return average loss."""
    model.train()
    total_loss = 0.0
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)
```

```python
# ❌ IndentationError — inconsistent indentation
def broken():
    x = 1
      y = 2  # IndentationError: unexpected indent
```

### 1.2 Key Differences from Java/C#

| Feature | Java/C# | Python |
|---------|---------|--------|
| Block delimiters | `{ }` | Indentation (4 spaces) |
| Line terminator | `;` | Newline |
| Comments | `//` and `/* */` | `#` only |
| Variable declaration | `int x = 5;` | `x = 5` |
| Constants | `final int X = 5;` | `X = 5` (convention: UPPER_CASE) |
| Null | `null` | `None` |
| Boolean | `true` / `false` | `True` / `False` (capitalized!) |
| String concat | `+` | `+` or f-strings (preferred) |
| Multi-line | N/A | `\` or wrap in `()` |

```python
# Multi-line expressions using parentheses (very common in ML configs)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=3e-4,
    weight_decay=0.01,
    betas=(0.9, 0.999),
    eps=1e-8
)

# Or using backslash (less preferred)
total = first_term + \
        second_term + \
        third_term
```

### 1.3 Dynamic Typing

Python is **dynamically typed** — variables don't have declared types. A variable is just a name (label) pointing to an object in memory.

```python
# Variables are labels that can point to any object
x = 42          # x → int object
x = "hello"     # x → str object (int is garbage-collected)
x = [1, 2, 3]   # x → list object

# Check types at runtime
print(type(x))              # <class 'list'>
print(isinstance(x, list))  # True

# isinstance() works with inheritance — crucial for PyTorch!
# isinstance(model, nn.Module) → True for ANY PyTorch model
```

### 1.4 Primitive Types

#### `int` — Arbitrary Precision

```python
# Python ints have NO max size — no overflow!
num_params = 175_000_000_000   # 175B — underscores for readability
print(num_params)              # 175000000000

# Compare to Java: int max is 2^31 - 1 = ~2.1 billion
huge = 2 ** 1000               # Works in Python, impossible in Java int/long!

# Integer division vs float division
print(7 / 2)    # 3.5  (float division — always returns float)
print(7 // 2)   # 3    (floor division — returns int)
print(7 % 2)    # 1    (modulo)
print(2 ** 10)  # 1024 (exponentiation — not ^ like in some languages!)

# ⚠️ In Python, ^ is XOR, not exponentiation!
print(2 ^ 10)   # 8 (bitwise XOR, NOT 1024!)
print(2 ** 10)  # 1024 (THIS is exponentiation)
```

```python
import sys

# In Python, ints are objects with arbitrary precision. They grow as needed!
print(sys.getsizeof(0))             # 28 bytes (base object overhead)
print(sys.getsizeof(1))             # 28 bytes
print(sys.getsizeof(2**64))         # 36 bytes (grew larger to fit the number!)
print(sys.getsizeof(2**1000))       # 160 bytes (a massive number, perfectly fine)
```

**🔥 AI Context:** In pure Python, ints have infinite size and never overflow. However, **PyTorch and NumPy use fixed-size C-style integers** under the hood for GPU efficiency. A PyTorch integer tensor defaults to 64-bit (`torch.int64`), but you'll often cast to `torch.int32` or `torch.int8` to save memory! Parameter counts like `175_000_000_000` (GPT-3) use underscores for readability in configs.

#### `float` — 64-bit by Default

```python
# The 'e' stands for exponent (base 10). It is scientific notation:
learning_rate = 3e-4         # 3 * 10^(-4) = 0.0003
epsilon = 1e-8               # 1 * 10^(-8) = 0.00000001
loss = 2.4567
temperature = 0.7            # Softmax temperature for text generation

# ⚠️ Notice how they print differently!
print(f"LR: {learning_rate}")    # Output: LR: 0.0003
print(f"Epsilon: {epsilon}")     # Output: Epsilon: 1e-08

# Why? Python automatically formats numbers smaller than 0.0001
# using scientific notation to prevent printing too many zeros.
# You can force the format you want using f-string specifiers:
print(f"LR forced: {learning_rate:.1e}")   # 3.0e-04
print(f"Eps forced: {epsilon:.8f}")        # 0.00000001

# Special float values
import math
print(float('inf'))          # Infinity — used for "best loss so far"
print(float('-inf'))         # Negative infinity
print(math.isinf(float('inf')))  # True

# Initializing best loss tracking
best_loss = float('inf')     # Any real loss will be smaller
```

**Why do floats have precision issues?**
This is a hardware reality (IEEE 754 standard), not a Python bug! In Base-10 (decimal), fractions like `1/3` repeat infinitely (`0.333...`). Computers use Base-2 (binary), where fractions like `1/10` (0.1) and `1/5` (0.2) repeat infinitely (`0.000110011...`). Because a 64-bit float has finite memory, the computer chops off the repeating sequence, losing a tiny bit of precision. When you add these rounded-off numbers together, the errors compound!

```python
# ⚠️ GOTCHA: Float precision issues
print(0.1 + 0.2)              # 0.30000000000000004 (NOT 0.3!)
print(0.1 + 0.2 == 0.3)       # False!

# ✅ Use math.isclose() for float comparison
import math
print(math.isclose(0.1 + 0.2, 0.3))  # True

# 🔥 AI Context: This matters for loss comparison!
# ❌ if current_loss == best_loss:     # Unreliable!
# ✅ if math.isclose(current_loss, best_loss, rel_tol=1e-9):
```

#### `str` — Immutable Strings

**Why are strings immutable?**
1. **Dictionary Keys & Hashing:** Dictionaries map strings to numbers (e.g. `vocab = {"the": 0}`). This requires a stable "hash" value. If strings were mutable, their hash could change, breaking the dictionary.
2. **Memory Efficiency (Interning):** Python can safely reuse the same string object for multiple variables (e.g., `a = "NLP"` and `b = "NLP"` point to the same memory), saving massive amounts of RAM.
3. **Thread Safety:** Multiple threads can read the same string simultaneously without locking mechanisms.

**The Impact: Concatenation in Loops**
Because strings cannot be changed, operations like `+=` create a brand *new* string in memory.
```python
# ❌ BAD: This creates 10,000 new strings in memory, copying the old text every time.
# result = ""
# for word in words: result += word + " "

# ✅ GOOD: Put pieces in a list, then join. Python calculates the needed memory exactly once.
# result = " ".join(words)
```

```python
model_name = "bert-base-uncased"
paper_title = 'Attention Is All You Need'

# Multi-line strings (triple quotes — used for docstrings)
description = """
This model implements the Transformer architecture
described in Vaswani et al. (2017).
"""

# Strings are IMMUTABLE — you can't change characters in place
s = "hello"
# s[0] = "H"  # TypeError: 'str' object does not support item assignment
s = "H" + s[1:]  # Create a NEW string instead

# Common string methods you'll use constantly:

# 1. .lower() / .upper() -> Changes case (useful for case-insensitive NLP)
print("BERT".lower())                    # "bert"
print("bert".upper())                    # "BERT"

# 2. .strip() -> Removes leading/trailing whitespace and newlines
print("  bert  \n".strip())              # "bert"

# 3. .split(delimiter) -> Splits a string into a list based on the delimiter
print("a,b,c".split(","))                # ['a', 'b', 'c']
print("the cat sat".split())             # ['the', 'cat', 'sat'] (default splits on spaces)

# 4. "delimiter".join(list) -> The opposite of split. Joins a list into a single string
print(" ".join(["attention", "is", "all"]))  # "attention is all"

# 5. 'in' operator -> Checks if a substring exists within a string
print("transformer" in "the transformer model")  # True

# 6. .replace(old, new) -> Replaces all occurrences of 'old' with 'new'
print("bert-base".replace("-", "_"))     # "bert_base"

# 7. .startswith(prefix) / .endswith(suffix) -> Checks the beginning or end of a string
print("hello world".startswith("hello")) # True
print("model.pt".endswith(".pt"))        # True
```

**🔥 AI Context:** String operations appear everywhere in NLP:
```python
# Tokenizer-style operations
text = "Hello, World!"
tokens = text.lower().split()       # ['hello,', 'world!']

# Building vocabulary
sentences = ["the cat sat", "the dog ran"]
all_words = " ".join(sentences).split()
vocab = sorted(set(all_words))
print(vocab)  # ['cat', 'dog', 'ran', 'sat', 'the']
```

#### `bool` — Truthy and Falsy

```python
is_training = True
use_cuda = False

# Python has "truthy" and "falsy" values — very different from Java!
# Falsy values: 0, 0.0, "", [], {}, set(), None, False
# EVERYTHING else is truthy

# This is used CONSTANTLY in Python/ML code:
data = []
if data:
    print("has data")        # won't print — empty list is falsy
else:
    print("no data")         # prints this

# Common pattern: checking if optional argument was provided
def create_model(config=None):
    if config:               # None is falsy
        print("Using provided config")
    else:
        print("Using default config")

# Boolean operators: and, or, not (NOT &&, ||, !)
if is_training and use_cuda:
    print("Training on GPU")

if not is_training or not use_cuda:
    print("Not full GPU training")
```

#### `None` — Python's null

```python
bias = None  # "no value assigned"

# ✅ ALWAYS check None with `is`, NEVER `==`
if bias is None:
    print("No bias term")

if bias is not None:
    print(f"Bias: {bias}")

# Why `is` and not `==`?
# `is` checks identity (same object in memory)
# `==` checks equality (could be overridden by a class)
# None is a singleton — there's only ONE None object
```

**🔥 AI Context:** `None` is used everywhere for optional parameters:
```python
# From a typical model definition
class TransformerBlock:
    def __init__(self, d_model=512, dropout=0.1, bias=None, 
                 attention_mask=None, activation_fn=None):
        # None means "use default" or "not provided"
        if activation_fn is None:
            activation_fn = "gelu"
```

### 1.5 Constants in Python

**Python has no built-in `const` or `final` keyword.** Unlike Java's `final` or C#'s `const`, there's no way to make a variable truly immutable at the language level. Here are the three approaches, from most common to most strict:

#### Approach 1: UPPER_CASE Convention (99% of Python code)

```python
# Just name it in UPPER_CASE — everyone agrees not to reassign
MAX_SEQ_LENGTH = 512
LEARNING_RATE = 3e-4
PAD_TOKEN_ID = 0
D_MODEL = 768
NUM_HEADS = 12
VOCAB_SIZE = 30522
EPS = 1e-12

# ⚠️ Nothing stops you from reassigning — this is purely a convention
MAX_SEQ_LENGTH = 1024  # Works! Python won't complain.
```

This is what **all major ML codebases use** — PyTorch, HuggingFace, TensorFlow.

#### Approach 2: `Final` Type Hint (Python 3.8+) — IDE Warning

```python
from typing import Final

MAX_SEQ_LENGTH: Final = 512
LEARNING_RATE: Final[float] = 3e-4
MODEL_NAME: Final[str] = "bert-base-uncased"

# Your IDE (VS Code) will show a warning ⚠️ and mypy will flag it,
# but Python itself still allows it at runtime!
MAX_SEQ_LENGTH = 1024  # IDE warning, but runs fine
```

#### Approach 3: True Enforcement via `__setattr__`

```python
# If you truly need to prevent reassignment:
class _Constants:
    """Raises an error if you try to reassign an attribute."""
    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise AttributeError(f"Cannot reassign constant '{name}'")
        super().__setattr__(name, value)

CONST = _Constants()
CONST.MAX_SEQ_LENGTH = 512
CONST.PAD_TOKEN_ID = 0

print(CONST.MAX_SEQ_LENGTH)  # 512
# CONST.MAX_SEQ_LENGTH = 1024  # ❌ AttributeError: Cannot reassign constant 'MAX_SEQ_LENGTH'
```

**🔥 AI Context:** In practice, ML code uses UPPER_CASE convention and moves on:
```python
# Typical model config constants
D_MODEL = 768
N_HEADS = 12
N_LAYERS = 12
DROPOUT = 0.1
MAX_LEN = 512
BATCH_SIZE = 32
LR = 3e-4
WARMUP_STEPS = 1000
```

### ⚠️ Section 1 Pitfalls

```python
# PITFALL 1: Mutable default arguments (Lists, Dicts, Sets)
# ❌ NEVER use a mutable object as a default parameter value!
# This is because the default value is evaluated ONLY ONCE at function definition time,
# NOT each time the function is called.
#
# Why it happens:
# 1. At definition time: Python creates a single list object `[]` in memory and binds it to the parameter `layers`.
# 2. Call 1: Since no argument is passed, Python uses that specific list object in memory. `append` mutates it in-place.
# 3. Call 2: No argument is passed again, so Python re-uses the exact same list object, which now has data in it.
#
# Note: This pitfall only applies to MUTABLE objects (lists, dictionaries, sets, custom objects).
# Immutable objects (ints, strings, tuples, None) are perfectly safe to use as defaults
# because any operation on them creates a new object rather than modifying the existing one.
def add_layer(layers=[]):
    layers.append("linear")
    return layers

print(add_layer())  # ['linear']
print(add_layer())  # ['linear', 'linear']  ← BUG! Same list object!

# ✅ Use None + create inside the function
def add_layer(layers=None):
    if layers is None:
        layers = []
    layers.append("linear")
    return layers

print(add_layer())  # ['linear']
print(add_layer())  # ['linear']  ← Correct!


# PITFALL 2: `is` vs `==`
a = [1, 2, 3]
b = [1, 2, 3]
print(a == b)   # True  — same VALUE
print(a is b)   # False — different OBJECTS in memory

# Only use `is` for: None, True, False
# Use `==` for everything else


# PITFALL 3: ^ is XOR, not power
print(2 ^ 3)    # 1 (bitwise XOR)
print(2 ** 3)   # 8 (exponentiation — this is what you want!)


# PITFALL 4: Integer division
print(7 / 2)    # 3.5 (float division — different from Java's int/int!)
print(7 // 2)   # 3   (floor division — if you want integer result)
```

---

## Section 2: F-Strings & String Formatting

F-strings (formatted string literals) are Python's most powerful string formatting tool. They're used **everywhere** in ML code for logging, debugging, and output.

### 2.1 Basic F-Strings

```python
# Prefix string with f or F — expressions go inside { }
model_name = "GPT-3"
num_params = 175e9
print(f"Model: {model_name}, Parameters: {num_params}")
# Output: Model: GPT-3, Parameters: 175000000000.0

# Any valid Python expression works inside { }
batch_size = 32
seq_len = 512
print(f"Tokens per batch: {batch_size * seq_len}")
# Output: Tokens per batch: 16384

# Method calls work too
text = "attention is all you need"
print(f"Title: {text.title()}")
# Output: Title: Attention Is All You Need
```

### 2.2 Number Formatting (Critical for Training Logs!)

```python
loss = 2.456789123
accuracy = 0.9234
num_params = 175_000_000_000
epoch = 3
total_epochs = 100

# Fixed decimal places — THE most common format in ML
print(f"Loss: {loss:.4f}")          # Loss: 2.4568 (4 decimal places)
print(f"Loss: {loss:.2f}")          # Loss: 2.46
print(f"Loss: {loss:.6f}")          # Loss: 2.456789

# Percentage
print(f"Accuracy: {accuracy:.1%}")   # Accuracy: 92.3%
print(f"Accuracy: {accuracy:.2%}")   # Accuracy: 92.34%

# Comma separator for large numbers (the `:,` adds commas as thousands separators)
# This is crucial for readability when printing parameter counts or dataset sizes!
print(f"Parameters: {num_params:,}") # Parameters: 175,000,000,000

# Padding and alignment (useful for formatted tables)
print(f"Epoch: {epoch:03d}/{total_epochs}")  # Epoch: 003/100

# Scientific notation
print(f"LR: {3e-4:.1e}")            # LR: 3.0e-04

# Combined — a REAL training log line:
print(f"Epoch [{epoch:3d}/{total_epochs}] | "
      f"Loss: {loss:.4f} | "
      f"Acc: {accuracy:.2%} | "
      f"Params: {num_params:,.0f}")
# Output: Epoch [  3/100] | Loss: 2.4568 | Acc: 92.34% | Params: 175,000,000,000
```

### 2.3 Alignment & Padding

In f-strings, you can use alignment specifiers after the `:` to create perfectly lined-up columns without jagged text.
- `>` Right-align (adds spaces to the left). e.g., `{str(val):>10}` means "pad with spaces on the left until it is exactly 10 characters wide".
- `<` Left-align (adds spaces to the right).
- `^` Center-align (adds spaces to both sides).

```python
# Right-align with > (useful for formatted tables)
for name, val in [("loss", 2.45), ("accuracy", 0.93), ("lr", 0.0003)]:
    print(f"{name:>15}: {val:.4f}")
# Output:
#            loss: 2.4500
#        accuracy: 0.9300
#              lr: 0.0003

# Left-align with <
print(f"{'Model':<20}{'Params':>15}")
print(f"{'BERT-base':<20}{110_000_000:>15,}")
print(f"{'GPT-3':<20}{175_000_000_000:>15,}")
# Output:
# Model                         Params
# BERT-base                110,000,000
# GPT-3              175,000,000,000

# Center with ^
print(f"{'=== Training Started ===':^50}")
```

### 2.4 The Debug Shorthand (Python 3.8+)

```python
# The = sign inside f-strings prints variable_name=value
# This is INCREDIBLY useful for quick debugging
batch_size = 32
learning_rate = 3e-4
d_model = 512

print(f"{batch_size=}")        # batch_size=32
print(f"{learning_rate=}")     # learning_rate=0.0003
print(f"{d_model=}")           # d_model=512

# Works with expressions too!
print(f"{batch_size * 2 = }")  # batch_size * 2 = 64

# Combine with format specs
print(f"{learning_rate=:.1e}") # learning_rate=3.0e-04
```

### 2.5 Using `!r` and `!s` in F-Strings

```python
text = "hello\nworld"

# !s calls str() — human-readable
print(f"Text: {text!s}")    # Text: hello
                             #        world

# !r calls repr() — shows escape characters (developer view)
print(f"Text: {text!r}")    # Text: 'hello\nworld'

# This is crucial when debugging tokenizer outputs:
token = "\t"
print(f"Token: {token!r}")  # Token: '\t' — you can SEE the tab character
```

**🔥 AI Context — Real Training Loop:**
```python
# This is what a real training log looks like in practice
def train(model, dataloader, optimizer, criterion, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # F-string formatted log — you'll write this 1000 times
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1:3d}/{num_epochs}] "
                      f"Batch [{batch_idx:5d}/{len(dataloader)}] "
                      f"Loss: {loss.item():.4f} "
                      f"Acc: {100. * correct / total:.2f}%")

        avg_loss = running_loss / len(dataloader)
        avg_acc = correct / total
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1} Complete | "
              f"Avg Loss: {avg_loss:.4f} | "
              f"Avg Acc: {avg_acc:.2%}")
        print(f"{'='*60}\n")
```

### ⚠️ Section 2 Pitfalls

```python
# PITFALL 1: Forgetting the f prefix
name = "BERT"
print("Model: {name}")   # Prints literal: Model: {name}
print(f"Model: {name}")  # Prints: Model: BERT

# PITFALL 2: Backslashes inside f-string expressions
# ❌ This doesn't work:
# print(f"{'hello\nworld'}")  # SyntaxError

# ✅ Assign to a variable first:
text = "hello\nworld"
print(f"{text}")

# PITFALL 3: Curly braces in f-strings (e.g., printing dicts/JSON)
# ❌ print(f"Config: {"key": "value"}")  # SyntaxError
# ✅ Use double braces to escape:
print(f"Dict syntax: {{'key': 'value'}}")  # Dict syntax: {'key': 'value'}
```

---

## Section 3: Type Hints & Annotations

Type hints make Python code **readable and self-documenting**. They don't enforce types at runtime — they're for humans and IDEs. All modern ML codebases (HuggingFace, PyTorch, etc.) use them extensively.

### 3.1 Basic Type Hints

```python
# Variable annotations
model_name: str = "bert-base"
num_layers: int = 12
learning_rate: float = 3e-4
is_training: bool = True
bias: None = None  # rarely annotated this way

# Function annotations
def compute_loss(predictions: list, targets: list) -> float:
    """Compute mean squared error loss."""
    total = sum((p - t) ** 2 for p, t in zip(predictions, targets))
    return total / len(predictions)

# The -> float is the RETURN type hint
result: float = compute_loss([1.0, 2.0], [1.1, 1.9])
```

### 3.2 Collection Type Hints (Python 3.9+)

```python
# Python 3.9+ — use built-in types directly (lowercase)
scores: list[float] = [0.95, 0.87, 0.92]
config: dict[str, int] = {"d_model": 512, "nhead": 8}
unique_tokens: set[str] = {"the", "a", "an"}
shape: tuple[int, int, int] = (32, 128, 512)  # fixed-length tuple

# For Python 3.7-3.8, import from typing
from typing import List, Dict, Set, Tuple
scores: List[float] = [0.95, 0.87, 0.92]
config: Dict[str, int] = {"d_model": 512}
```

### 3.3 Optional, Union, and More

```python
from typing import Optional, Union, Any, Callable

# Optional[X] = X or None
# This is THE most common type hint in ML code
def create_model(
    d_model: int = 512,
    dropout: float = 0.1,
    pretrained_path: Optional[str] = None,    # might not be provided
    custom_head: Optional[Callable] = None     # optional function
) -> "TransformerModel":
    pass

# Union[X, Y] = X or Y
# Used when a parameter can be multiple types
def set_device(device: Union[str, int]) -> None:
    """Accept 'cuda', 'cpu', or GPU index like 0, 1."""
    pass

# Python 3.10+ — use | instead of Union
def set_device(device: str | int) -> None:
    pass

# Any — accepts anything (use sparingly!)
def log_metric(key: str, value: Any) -> None:
    pass

# Callable — for functions passed as arguments
from typing import Callable
def apply_transform(
    data: list[float],
    transform: Callable[[float], float]   # takes float, returns float
) -> list[float]:
    return [transform(x) for x in data]
```

### 3.4 Type Hints in Real AI Code

```python
from typing import Optional, Dict, List, Tuple, Union
# Imagine these imports exist:
# import torch
# import torch.nn as nn

# ----- A HuggingFace-style model config -----
class TransformerConfig:
    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        num_hidden_layers: int = 12,
        intermediate_size: int = 3072,
        hidden_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        use_cache: bool = True,
        classifier_dropout: Optional[float] = None,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        # ... etc

# ----- A training function signature -----
def train_epoch(
    model,                       # nn.Module
    dataloader,                  # DataLoader
    optimizer,                   # torch.optim.Optimizer
    scheduler: Optional[object] = None,
    device: str = "cuda",
    grad_clip: Optional[float] = 1.0,
    log_interval: int = 100,
) -> Dict[str, float]:
    """Returns dict like {'loss': 0.45, 'accuracy': 0.92}"""
    pass
```

### ⚠️ Section 3 Pitfalls

```python
# PITFALL 1: Type hints are NOT enforced at runtime!
def add(a: int, b: int) -> int:
    return a + b

print(add("hello", " world"))  # "hello world" — NO error!
# Type hints are for documentation and IDE support only

# PITFALL 2: Forward references (using a class before it's defined)
class TreeNode:
    def __init__(self, children: list["TreeNode"]):  # Quotes needed!
        self.children = children

# Python 3.10+ alternative:
from __future__ import annotations  # Put at top of file
class TreeNode:
    def __init__(self, children: list[TreeNode]):    # No quotes needed
        self.children = children
```

---

## Section 4: Conditional Statements

### 4.1 if / elif / else

```python
# Basic if/elif/else — note the colons and indentation!
loss = 2.5

if loss < 0.1:
    print("Excellent — model has converged!")
elif loss < 1.0:
    print("Good progress — keep training")
elif loss < 5.0:
    print("Still learning...")
else:
    print("Something might be wrong — check learning rate")

# No parentheses needed (unlike Java/C#)!
# ❌ if (loss < 0.1):    # Works but NOT Pythonic
# ✅ if loss < 0.1:      # Pythonic
```

### 4.2 Truthiness in Conditions

```python
# Python lets you use ANY value in a condition
# Falsy: 0, 0.0, "", [], {}, set(), None, False
# Everything else is truthy

# Common ML patterns using truthiness:
config = {}
if config:
    print("Config provided")
else:
    print("Using defaults")  # This prints — empty dict is falsy

# Checking for optional arguments
def build_model(pretrained_weights=None, extra_layers=None):
    if pretrained_weights:
        print("Loading pretrained weights")
    if extra_layers:
        print(f"Adding {len(extra_layers)} extra layers")
    else:
        print("Using base architecture")

build_model()  # Using base architecture
build_model(pretrained_weights="model.pt", extra_layers=["fc1", "fc2"])
```

### 4.3 Comparison Operators & Chaining

```python
# Standard comparisons
print(5 == 5)    # True
print(5 != 3)    # True
print(5 > 3)     # True
print(5 >= 5)    # True

# Python supports CHAINED comparisons (Java/C# can't do this!)
x = 5
print(1 < x < 10)        # True  — equivalent to (1 < x) and (x < 10)
print(0 <= x <= 100)      # True

# Very useful for value range checking:
learning_rate = 3e-4
assert 0 < learning_rate < 1, "Learning rate must be between 0 and 1"

temperature = 0.7
if 0.0 < temperature <= 2.0:
    print("Valid temperature")
```

### 4.4 Logical Operators: and, or, not

```python
# Python uses words, not symbols: and, or, not (NOT &&, ||, !)
use_gpu = True
has_gpu = True
is_training = True

if use_gpu and has_gpu:
    device = "cuda"
elif use_gpu and not has_gpu:
    print("Warning: GPU requested but not available, using CPU")
    device = "cpu"
else:
    device = "cpu"

# Short-circuit evaluation (same as Java/C#)
# `and` returns first falsy value, or last value if all truthy
# `or` returns first truthy value, or last value if all falsy

# Common pattern: default values with `or`
user_batch_size = 0  # User didn't set it
batch_size = user_batch_size or 32  # Falls back to 32
print(batch_size)  # 32

model_name = ""
name = model_name or "default-model"
print(name)  # "default-model"

# ⚠️ Be careful: `or` doesn't work well when 0 or "" are valid values!
# If 0 is a valid batch_size, use: batch_size = user_batch_size if user_batch_size is not None else 32
```

### 4.5 Membership & Identity

```python
# `in` operator — membership testing
valid_activations = ["relu", "gelu", "silu", "swish"]
if "gelu" in valid_activations:
    print("GELU is available")  # prints

# Works with strings too
if "transformer" in "the transformer model":
    print("Found it")  # prints

# `not in`
if "mish" not in valid_activations:
    print("Mish not supported")  # prints

# `is` vs `==` (identity vs equality)
a = None
if a is None:         # ✅ Check identity for None
    print("No value")

b = [1, 2, 3]
c = [1, 2, 3]
print(b == c)  # True  — same value
print(b is c)  # False — different objects
```

### ⚠️ Section 4 Pitfalls

```python
# PITFALL 1: Using = instead of == in conditions
# ❌ if x = 5:    # SyntaxError in Python (unlike C/C++ where this is a bug)
# ✅ if x == 5:   # Correct

# PITFALL 2: Truthiness with 0
count = 0
if count:
    print("has items")
else:
    print("empty")  # This prints! 0 is falsy.

# If 0 is a valid value, be explicit:
if count is not None:
    print(f"Count: {count}")  # Now 0 will work correctly

# PITFALL 3: `or` for defaults with valid falsy values
batch_size = 0  # User explicitly wants 0 (maybe for some config)
# ❌ size = batch_size or 32   # size = 32, not 0!
# ✅ size = batch_size if batch_size is not None else 32  # size = 0
```

---

## Section 5: Loops

### 5.1 `for` Loops & `range()`

```python
# Basic for loop with range
for i in range(5):
    print(i)  # 0, 1, 2, 3, 4  (NOT 5 — exclusive upper bound)

# range(start, stop, step)
for i in range(0, 10, 2):
    print(i)  # 0, 2, 4, 6, 8

# Counting down
for i in range(10, 0, -1):
    print(i)  # 10, 9, 8, ..., 1

# Iterating over collections (you almost NEVER use range for this!)
models = ["bert", "gpt2", "t5", "llama"]

# ❌ Java/C# style — DON'T do this in Python
for i in range(len(models)):
    print(models[i])

# ✅ Pythonic — iterate directly
for model in models:
    print(model)
```

### 5.2 `enumerate()` — Index + Value

```python
# When you NEED the index, use enumerate (not range(len(...)))
layers = ["embedding", "attention", "feedforward", "layernorm"]

for idx, layer in enumerate(layers):
    print(f"Layer {idx}: {layer}")
# Layer 0: embedding
# Layer 1: attention
# Layer 2: feedforward
# Layer 3: layernorm

# Start from a different index
for idx, layer in enumerate(layers, start=1):
    print(f"Layer {idx}: {layer}")
# Layer 1: embedding
# Layer 2: attention
# ...
```

**🔥 AI Context:** `enumerate` is used in every training loop:
```python
# Real training loop pattern
for batch_idx, (inputs, labels) in enumerate(dataloader):
    loss = train_step(model, inputs, labels)
    if batch_idx % 100 == 0:
        print(f"Batch {batch_idx}: loss = {loss:.4f}")
```

### 5.3 `zip()` — Parallel Iteration

```python
# zip() combines multiple iterables element-wise
models = ["BERT", "GPT-2", "T5"]
params = ["110M", "1.5B", "11B"]
years = [2018, 2019, 2019]

for model, param, year in zip(models, params, years):
    print(f"{model} ({year}): {param} parameters")
# BERT (2018): 110M parameters
# GPT-2 (2019): 1.5B parameters
# T5 (2019): 11B parameters

# ⚠️ zip stops at the SHORTEST iterable
list1 = [1, 2, 3]
list2 = [4, 5]
for a, b in zip(list1, list2):
    print(a, b)  # (1,4), (2,5) — 3 is SILENTLY dropped!

# ✅ Use zip with strict=True (Python 3.10+) to catch mismatches
# for a, b in zip(list1, list2, strict=True):  # ValueError!

# Creating dicts from two lists
keys = ["d_model", "nhead", "num_layers"]
values = [512, 8, 6]
config = dict(zip(keys, values))
print(config)  # {'d_model': 512, 'nhead': 8, 'num_layers': 6}
```

### 5.4 `while` Loops

```python
# while loop — less common in Python than for loops
patience = 5
best_loss = float('inf')
epochs_without_improvement = 0
epoch = 0

# Early stopping pattern
while epochs_without_improvement < patience:
    epoch += 1
    current_loss = simulate_training(epoch)  # imaginary function

    if current_loss < best_loss:
        best_loss = current_loss
        epochs_without_improvement = 0
        print(f"Epoch {epoch}: New best loss = {best_loss:.4f}")
    else:
        epochs_without_improvement += 1
        print(f"Epoch {epoch}: No improvement ({epochs_without_improvement}/{patience})")

print(f"Early stopping at epoch {epoch}")
```

### 5.5 `break`, `continue`, `else` on Loops

```python
# break — exit the loop early
for epoch in range(1000):
    loss = train_one_epoch()
    if loss < 0.001:
        print(f"Converged at epoch {epoch}!")
        break

# continue — skip to next iteration
for batch_idx, data in enumerate(dataloader):
    if data is None:
        continue  # skip corrupted batches
    train_step(data)

# for...else — the else block runs if the loop completed WITHOUT break
# This is unique to Python and useful for "search" patterns
target_lr = 1e-5
for param_group in optimizer.param_groups:
    if param_group['lr'] == target_lr:
        print("Found matching learning rate group")
        break
else:
    # This runs if we never hit break — i.e., target not found
    print("No parameter group with target learning rate")
```

### 5.6 Iterating Over Dictionaries

```python
config = {
    "model": "transformer",
    "d_model": 512,
    "nhead": 8,
    "num_layers": 6,
    "dropout": 0.1
}

# Iterate over keys (default)
for key in config:
    print(key)

# Iterate over values
for value in config.values():
    print(value)

# Iterate over key-value pairs (MOST common)
for key, value in config.items():
    print(f"{key}: {value}")

# 🔥 Real pattern: logging hyperparameters
print("=" * 40)
print("Training Configuration:")
print("=" * 40)
for param, value in config.items():
    print(f"  {param:>15}: {value}")
```

### ⚠️ Section 5 Pitfalls

```python
# PITFALL 1: Modifying a list while iterating over it
layers = ["conv1", "conv2", "fc1", "fc2"]
# ❌ This causes subtle bugs!
for layer in layers:
    if "fc" in layer:
        layers.remove(layer)   # Modifying while iterating!
print(layers)  # ['conv1', 'conv2', 'fc2']  ← BUG: fc2 not removed!

# ✅ Create a new list instead
layers = ["conv1", "conv2", "fc1", "fc2"]
layers = [l for l in layers if "fc" not in l]
print(layers)  # ['conv1', 'conv2']

# PITFALL 2: Using range(len()) instead of enumerate
# ❌ for i in range(len(items)):  item = items[i]
# ✅ for i, item in enumerate(items): ...

# PITFALL 3: Forgetting that range() is exclusive at the top
for i in range(3):
    print(i)  # 0, 1, 2 — NOT 3!
```

---

## Section 6: Comprehensions

Comprehensions are Python's most powerful one-liner tool. They replace `for` loops with concise, readable expressions. **You'll see them on every page of AI code.**

### 6.1 List Comprehensions

```python
# Basic syntax: [expression FOR variable IN iterable]
squares = [x ** 2 for x in range(10)]
print(squares)  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# With condition (filter): [expression FOR var IN iterable IF condition]
even_squares = [x ** 2 for x in range(10) if x % 2 == 0]
print(even_squares)  # [0, 4, 16, 36, 64]

# Transform + filter
lengths = [len(word) for word in ["the", "transformer", "is", "great"]
           if len(word) > 2]
print(lengths)  # [3, 11, 5]
```

**🔥 AI Context — Comprehensions in real ML code:**
```python
# Building vocabulary from tokens
tokens = ["the", "cat", "sat", "on", "the", "mat"]
vocab = sorted(set(tokens))
token_to_idx = {token: idx for idx, token in enumerate(vocab)}
print(token_to_idx)
# {'cat': 0, 'mat': 1, 'on': 2, 'sat': 3, 'the': 4}

# Encoding a sentence
sentence = "the cat sat"
encoded = [token_to_idx[word] for word in sentence.split()]
print(encoded)  # [4, 0, 3]

# Filtering layers by name (common in fine-tuning)
all_params = ["encoder.layer.0.weight", "encoder.layer.0.bias",
              "classifier.weight", "classifier.bias",
              "encoder.layer.1.weight", "encoder.layer.1.bias"]

# Only get classifier parameters (for fine-tuning just the head)
classifier_params = [p for p in all_params if "classifier" in p]
print(classifier_params)  # ['classifier.weight', 'classifier.bias']

# Flatten list of lists (common with batched data)
batches = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flat = [item for batch in batches for item in batch]
print(flat)  # [1, 2, 3, 4, 5, 6, 7, 8, 9]
# Read as: for each batch, for each item in batch, take item
```

### 6.2 Dict Comprehensions

```python
# {key_expr: value_expr FOR var IN iterable}
words = ["attention", "is", "all", "you", "need"]
word_lengths = {word: len(word) for word in words}
print(word_lengths)
# {'attention': 9, 'is': 2, 'all': 3, 'you': 3, 'need': 4}

# Inverting a dictionary (swap keys and values)
token_to_id = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
id_to_token = {idx: token for token, idx in token_to_id.items()}
print(id_to_token)
# {0: '[PAD]', 1: '[UNK]', 2: '[CLS]', 3: '[SEP]'}

# Filtering a dict
config = {"d_model": 512, "nhead": 8, "dropout": 0.1, "num_layers": 6}
# Only keep parameters > 10
large_params = {k: v for k, v in config.items() if isinstance(v, (int, float)) and v > 10}
print(large_params)  # {'d_model': 512, 'num_layers': 6}
```

### 6.3 Set Comprehensions

```python
# {expression FOR var IN iterable}
text = "the cat sat on the mat on the floor"
unique_words = {word for word in text.split()}
print(unique_words)  # {'floor', 'mat', 'cat', 'on', 'the', 'sat'}

# Unique word lengths
unique_lengths = {len(word) for word in text.split()}
print(unique_lengths)  # {2, 3, 5}
```

### 6.4 Nested Comprehensions

```python
# Matrix creation (list of lists)
# 3x4 matrix of zeros
matrix = [[0 for col in range(4)] for row in range(3)]
print(matrix)  # [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

# Identity matrix
n = 3
identity = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
print(identity)  # [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

# All pairs (like itertools.product)
batch_sizes = [16, 32, 64]
learning_rates = [1e-3, 3e-4, 1e-4]
experiments = [(bs, lr) for bs in batch_sizes for lr in learning_rates]
print(experiments)
# [(16, 0.001), (16, 0.0003), (16, 0.0001), (32, 0.001), ...]
# → Hyperparameter grid search!
```

### 6.5 Generator Expressions (Memory-Efficient Comprehensions)

```python
# Use () instead of [] — creates a generator (lazy evaluation)
# Useful for HUGE datasets that don't fit in memory

# List comprehension: creates entire list in memory
sum_of_squares = sum([x**2 for x in range(1_000_000)])

# Generator expression: computes values one at a time
sum_of_squares = sum(x**2 for x in range(1_000_000))
# Same result, but uses almost no memory!

# When passing to a function, you can omit the extra parentheses
total = sum(x**2 for x in range(1_000_000))
any_positive = any(x > 0 for x in losses)
all_valid = all(len(seq) > 0 for seq in sequences)
max_length = max(len(seq) for seq in sequences)
```

### ⚠️ Section 6 Pitfalls

```python
# PITFALL 1: Overly complex comprehensions — just use a loop!
# ❌ Hard to read:
result = [transform(x) for batch in data for x in batch if validate(x) and x.shape[0] > 0]

# ✅ Use a regular loop when it gets complex:
result = []
for batch in data:
    for x in batch:
        if validate(x) and x.shape[0] > 0:
            result.append(transform(x))

# Rule of thumb: if it's more than ~80 chars or has 2+ conditions, use a loop

# PITFALL 2: Side effects in comprehensions
# ❌ Don't use comprehensions for side effects (like printing)
[print(x) for x in range(5)]  # Works but creates a useless list of Nones

# ✅ Use a regular for loop for side effects
for x in range(5):
    print(x)

# PITFALL 3: Variable leaking (Python 2 issue — fixed in Python 3)
# In Python 3, comprehension variables don't leak
x = 10
squares = [x for x in range(5)]
print(x)  # 10 — x is unchanged (Python 3)
```

---

## Section 7: Tuple Unpacking & Multiple Assignment

### 7.1 Basic Unpacking

```python
# Multiple assignment
x, y, z = 1, 2, 3
print(x, y, z)  # 1 2 3

# Swap variables (no temp needed!)
a, b = 1, 2
a, b = b, a
print(a, b)  # 2 1

# Tuple unpacking
point = (3.0, 4.0)
x, y = point

# Works with any iterable
first, second, third = [10, 20, 30]
a, b, c = "xyz"   # a='x', b='y', c='z'
```

### 7.2 Star Unpacking (`*`)

```python
# * captures "the rest" into a list
first, *rest = [1, 2, 3, 4, 5]
print(first)  # 1
print(rest)   # [2, 3, 4, 5]

# Works at any position
first, *middle, last = [1, 2, 3, 4, 5]
print(first)   # 1
print(middle)  # [2, 3, 4]
print(last)    # 5

*beginning, last = [1, 2, 3, 4, 5]
print(beginning)  # [1, 2, 3, 4]
print(last)       # 5

# Ignoring values with _
_, important, _ = (1, 42, 3)
print(important)  # 42

# Ignoring multiple values
first, *_ = [1, 2, 3, 4, 5]
print(first)  # 1 (rest is discarded)
```

### 7.3 Unpacking in Real AI Code

```python
# ===== DataLoader unpacking — you'll do this 1000 times =====
# for inputs, labels in dataloader:
#     outputs = model(inputs)
#     loss = criterion(outputs, labels)

# ===== Function returns =====
def evaluate(model, dataloader):
    """Return loss and accuracy."""
    return 0.45, 0.92

loss, accuracy = evaluate(None, None)  # Unpack the return tuple

# ===== Ignoring unneeded values =====
# torch.max returns (values, indices) — often you only need one
# values, indices = torch.max(output, dim=1)
# _, predicted = torch.max(output, dim=1)  # ignore values, keep indices

# ===== Unpacking in enumerate =====
data = [("image1.jpg", "cat"), ("image2.jpg", "dog")]
for idx, (filename, label) in enumerate(data):
    print(f"Sample {idx}: {filename} -> {label}")
# Sample 0: image1.jpg -> cat
# Sample 1: image2.jpg -> dog

# ===== Nested unpacking =====
# HuggingFace-style batch
batch = {
    "input_ids": [101, 2023, 2003, 102],
    "attention_mask": [1, 1, 1, 1],
    "labels": [1]
}

# Unpack dict items
for key, value in batch.items():
    print(f"{key}: {value}")
```

### 7.4 Returning Multiple Values

```python
# Functions can return tuples (multiple values)
def get_model_stats(model_params):
    total = sum(model_params)
    mean = total / len(model_params)
    max_val = max(model_params)
    min_val = min(model_params)
    return total, mean, max_val, min_val

# Unpack all
total, mean, max_val, min_val = get_model_stats([100, 200, 300])

# Or keep as tuple
stats = get_model_stats([100, 200, 300])
print(stats[0])  # total

# Unpack only what you need
total, *_ = get_model_stats([100, 200, 300])
```

### ⚠️ Section 7 Pitfalls

```python
# PITFALL 1: Mismatched unpacking count
# ❌ x, y = [1, 2, 3]   # ValueError: too many values to unpack
# ❌ x, y, z = [1, 2]   # ValueError: not enough values to unpack
# ✅ x, y, *rest = [1, 2, 3]  # Use * to handle variable length

# PITFALL 2: Single-element tuple needs a comma
t = (42)     # This is just the integer 42 in parentheses!
t = (42,)    # THIS is a tuple with one element
print(type((42)))   # <class 'int'>
print(type((42,)))  # <class 'tuple'>
```

---

## Section 8: Walrus Operator (`:=`)

The walrus operator `:=` (Assignment Expression, Python 3.8+) lets you assign a value to a variable **as part of an expression**. It reduces duplication when you need to both compute and use a value.

### 8.1 Basic Usage

```python
# WITHOUT walrus operator (typical pattern)
data = get_data()
if data:
    process(data)

# WITH walrus operator — assign and check in one line
if (data := get_data()):
    process(data)

# The parentheses are required around the := expression in most cases
```

### 8.2 Practical Examples

```python
# ===== In while loops — the classic use case =====
# WITHOUT walrus
line = input("Enter command: ")
while line != "quit":
    process(line)
    line = input("Enter command: ")

# WITH walrus — no duplication!
while (line := input("Enter command: ")) != "quit":
    process(line)

# ===== Filtering with a computation =====
# WITHOUT walrus — compute len() twice
data = ["transformer", "is", "great", "at", "nlp"]
long_words = []
for word in data:
    if len(word) > 3:
        long_words.append(len(word))  # Recomputing len()!

# WITH walrus — compute once, use twice
long_words = [length for word in data if (length := len(word)) > 3]
print(long_words)  # [11, 5]

# ===== In conditions with computed values =====
import re
text = "Learning rate: 3e-4"
if (match := re.search(r"[\d.]+e?-?\d*", text)):
    print(f"Found number: {match.group()}")
# Without walrus, you'd need to call re.search() twice or use a temp variable
```

### 8.3 AI Engineering Example

```python
# Processing batches until empty
# while (batch := next(dataloader_iter, None)) is not None:
#     inputs, labels = batch
#     loss = train_step(model, inputs, labels)

# Logging only when a threshold is hit
metrics = [0.5, 0.3, 0.1, 0.05, 0.02, 0.001]
for metric in metrics:
    if (improved := metric < 0.1):
        print(f"Metric {metric:.4f} — below threshold! Improved={improved}")
```

### ⚠️ Walrus Operator Pitfalls

```python
# PITFALL 1: Don't overuse it — readability matters!
# ❌ Too clever:
# result = [(y := f(x)), y**2, y**3]

# ✅ Use a regular variable assignment when it's clearer:
y = f(x)
result = [y, y**2, y**3]

# PITFALL 2: Walrus operator has unusual scoping
# The variable "leaks" out of the comprehension (unlike regular comp variables)
[y := x for x in range(5)]
print(y)  # 4 — y is accessible here! (This is intentional but can be confusing)
```

---

## Section 9: Ternary Expressions

Python's equivalent of the `condition ? value_if_true : value_if_false` syntax from Java/C#.

### 9.1 Basic Syntax

```python
# value_if_true IF condition ELSE value_if_false
# (Note: the order is DIFFERENT from Java's ternary!)

x = 10
result = "positive" if x > 0 else "non-positive"
print(result)  # "positive"

# Java equivalent:  String result = x > 0 ? "positive" : "non-positive";
# Python:           result = "positive" if x > 0 else "non-positive"
```

### 9.2 AI Engineering Examples

```python
# THE most common ternary in all of ML:
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

# Activation function selection
activation = "gelu" if model_type == "bert" else "relu"

# Conditional learning rate
lr = 1e-5 if fine_tuning else 3e-4

# Mode string
mode = "train" if is_training else "eval"
print(f"Model in {mode} mode")

# Conditional formatting
loss = 0.001
status = f"✅ Converged ({loss:.4f})" if loss < 0.01 else f"⏳ Training ({loss:.4f})"
print(status)  # ✅ Converged (0.0010)

# In function arguments
# dropout_rate = 0.1 if is_training else 0.0
# model = create_model(dropout=0.1 if training else 0.0)

# Nested ternary (use sparingly!)
size = "small" if params < 1e8 else "medium" if params < 1e9 else "large"
# Better as if/elif/else for readability when there are many branches
```

### ⚠️ Ternary Pitfalls

```python
# PITFALL: Don't nest ternaries deeply — use if/elif/else instead
# ❌ Hard to read:
result = "a" if x > 10 else "b" if x > 5 else "c" if x > 0 else "d"

# ✅ Use if/elif/else:
if x > 10:
    result = "a"
elif x > 5:
    result = "b"
elif x > 0:
    result = "c"
else:
    result = "d"
```

---

## Section 10: `print()` Debugging, `repr()` vs `str()`

### 10.1 `print()` — Your First Debugging Tool

```python
# print() accepts multiple arguments separated by comma
x = 42
y = "hello"
print(x, y)           # 42 hello (space-separated by default)
print(x, y, sep=", ") # 42, hello
print(x, end="")      # 42 (no newline at end)

# Printing multiple values for debugging
batch_size = 32
seq_len = 128
d_model = 512
print(f"Shape check: ({batch_size}, {seq_len}, {d_model})")

# Python 3.8+ debug shorthand — THE BEST debugging trick
loss = 2.5678
epoch = 5
lr = 3e-4
print(f"{loss=}")      # loss=2.5678
print(f"{epoch=}")     # epoch=5
print(f"{lr=}")        # lr=0.0003
print(f"{lr=:.1e}")    # lr=3.0e-04

# Multiple on one line
print(f"{epoch=}, {loss=:.4f}, {lr=:.1e}")
# epoch=5, loss=2.5678, lr=3.0e-04
```

### 10.2 `repr()` vs `str()`

```python
# str() — human-readable, meant for end users
# repr() — unambiguous, meant for developers/debugging

# With strings, the difference is clear:
text = "hello\tworld\n"

print(str(text))    # hello    world     (interprets escape characters)
print(repr(text))   # 'hello\tworld\n'  (shows the raw escape characters)

# With numbers — usually the same
print(str(42))    # 42
print(repr(42))   # 42

# With objects — repr() shows more info
import datetime
now = datetime.datetime.now()
print(str(now))    # 2024-01-15 10:30:45.123456   (human-friendly)
print(repr(now))   # datetime.datetime(2024, 1, 15, 10, 30, 45, 123456)  (reconstructible)
```

### 10.3 Defining `__str__` and `__repr__` on Your Classes

```python
class ModelConfig:
    def __init__(self, name, d_model, nhead, num_layers):
        self.name = name
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers

    def __str__(self):
        """Human-readable — used by print() and str()"""
        return f"{self.name} (d={self.d_model}, h={self.nhead}, L={self.num_layers})"

    def __repr__(self):
        """Developer-readable — used in debuggers, logs, and interactive console"""
        return (f"ModelConfig(name={self.name!r}, d_model={self.d_model}, "
                f"nhead={self.nhead}, num_layers={self.num_layers})")

config = ModelConfig("BERT-base", 768, 12, 12)

print(config)       # BERT-base (d=768, h=12, L=12)         ← calls __str__
print(repr(config)) # ModelConfig(name='BERT-base', d_model=768, nhead=12, num_layers=12) ← calls __repr__

# In a list, Python uses repr() for each element:
configs = [config, ModelConfig("GPT-2", 768, 12, 12)]
print(configs)  # [ModelConfig(name='BERT-base', ...), ModelConfig(name='GPT-2', ...)]
```

### 10.4 Debugging Patterns You'll Use Daily

```python
# ===== Pattern 1: Shape debugging (THE most common debug in ML) =====
# In real code with PyTorch:
# print(f"Input shape: {x.shape}")     # Input shape: torch.Size([32, 128])
# print(f"Output shape: {out.shape}")  # Output shape: torch.Size([32, 128, 512])

# Simulating with plain Python:
x_shape = (32, 128)
out_shape = (32, 128, 512)
print(f"Input shape: {x_shape}")
print(f"Output shape: {out_shape}")

# ===== Pattern 2: Checkpoint logging =====
def debug_training_step(epoch, batch_idx, loss, grads=None):
    print(f"\n{'='*50}")
    print(f"  Epoch: {epoch} | Batch: {batch_idx}")
    print(f"  Loss:  {loss:.6f}")
    if grads is not None:
        print(f"  Grad norm: {sum(g**2 for g in grads)**0.5:.4f}")
    print(f"{'='*50}")

debug_training_step(1, 42, 2.3456, [0.1, 0.2, 0.3])

# ===== Pattern 3: Conditional debugging with a flag =====
DEBUG = True  # Set to False to disable all debug prints

def train_step(inputs, labels):
    # ... training code ...
    loss = 0.5  # placeholder
    if DEBUG:
        print(f"  [DEBUG] loss={loss:.4f}")
    return loss

# ===== Pattern 4: Progress logging =====
num_samples = 1000
for i in range(num_samples):
    # ... processing ...
    if i % 200 == 0:
        print(f"Processing: {i}/{num_samples} ({i/num_samples:.0%})")
# Processing: 0/1000 (0%)
# Processing: 200/1000 (20%)
# Processing: 400/1000 (40%)
# ...

# ===== Pattern 5: Quick type/value inspection =====
data = {"input_ids": [101, 2023, 102], "attention_mask": [1, 1, 1]}
for key, value in data.items():
    print(f"{key}: type={type(value).__name__}, len={len(value)}, value={value!r}")
# input_ids: type=list, len=3, value=[101, 2023, 102]
# attention_mask: type=list, len=3, value=[1, 1, 1]
```

### ⚠️ Section 10 Pitfalls

```python
# PITFALL 1: print() returns None!
result = print("hello")
print(result)  # None — don't assign print()'s return value

# PITFALL 2: print() in Colab/Jupyter has output limits
# Very long outputs get truncated. For large data, print a summary:
big_list = list(range(10000))
# ❌ print(big_list)  # Walls of text
# ✅ print(f"Length: {len(big_list)}, First 5: {big_list[:5]}, Last 5: {big_list[-5:]}")

# PITFALL 3: Forgetting that str() and repr() can differ significantly
# Always use repr() when debugging strings with invisible characters:
text = "hello\t\tworld\n"
print(f"Value: {text}")    # Looks like: hello		world  (hard to see issues)
print(f"Value: {text!r}")  # Shows: 'hello\t\tworld\n' (now you can see everything!)
```

---

## 🧪 Practice Challenge: Training Dashboard Simulator

Combine ALL concepts from this module into one exercise. Copy this into a Colab cell and run it!

```python
"""
Practice Challenge: Build a Training Dashboard Simulator
=========================================================
Uses: variables, types, f-strings, type hints, conditionals,
      loops, comprehensions, unpacking, ternary, walrus, debugging
"""

from typing import Optional
import random
import math

# ===== 1. Configuration (type hints, variables, ternary) =====
class TrainingConfig:
    def __init__(
        self,
        model_name: str = "MiniTransformer",
        num_epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 3e-4,
        use_gpu: bool = True,
        weight_decay: float = 0.01,
        patience: int = 3,
        warmup_steps: Optional[int] = None,
    ):
        self.model_name = model_name
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = "cuda" if use_gpu else "cpu"  # ternary
        self.weight_decay = weight_decay
        self.patience = patience
        self.warmup_steps = warmup_steps if warmup_steps is not None else num_epochs // 5

    def __repr__(self) -> str:
        return (f"TrainingConfig(model={self.model_name!r}, "
                f"epochs={self.num_epochs}, bs={self.batch_size}, "
                f"lr={self.learning_rate:.1e}, device={self.device!r})")

    def __str__(self) -> str:
        return f"{self.model_name} on {self.device} for {self.num_epochs} epochs"


# ===== 2. Simulated Training (loops, f-strings, conditionals) =====
def simulate_training(config: TrainingConfig) -> dict[str, list[float]]:
    """Simulate a training run and return metrics history."""

    print(f"\n{'='*60}")
    print(f"  🚀 Starting Training: {config}")
    print(f"{'='*60}")
    print(f"  Config: {config!r}")
    print()

    # Log all hyperparameters (dict iteration)
    hyperparams = {
        "Model": config.model_name,
        "Device": config.device,
        "Epochs": config.num_epochs,
        "Batch Size": config.batch_size,
        "Learning Rate": config.learning_rate,
        "Weight Decay": config.weight_decay,
        "Warmup Steps": config.warmup_steps,
        "Patience": config.patience,
    }

    print("  Hyperparameters:")
    for key, value in hyperparams.items():
        # f-string formatting: right-align key, format value
        formatted = f"{value:.1e}" if isinstance(value, float) and value < 0.01 else str(value)
        print(f"    {key:>15}: {formatted}")
    print()

    # Training loop
    history: dict[str, list[float]] = {"loss": [], "accuracy": []}
    best_loss = float('inf')
    epochs_without_improvement = 0

    random.seed(42)  # Reproducibility!

    for epoch in range(1, config.num_epochs + 1):
        # Simulate metrics (loss decreases, accuracy increases, with noise)
        base_loss = 3.0 * math.exp(-0.3 * epoch) + 0.1
        noise = random.gauss(0, 0.05)
        epoch_loss = max(0.01, base_loss + noise)

        base_acc = 1.0 - math.exp(-0.4 * epoch)
        acc_noise = random.gauss(0, 0.02)
        epoch_acc = min(0.99, max(0.0, base_acc + acc_noise))

        history["loss"].append(epoch_loss)
        history["accuracy"].append(epoch_acc)

        # Learning rate warmup (walrus operator for computed value)
        if (effective_lr := config.learning_rate * min(1.0, epoch / config.warmup_steps)) < config.learning_rate:
            lr_status = f"(warmup: {effective_lr:.1e})"
        else:
            lr_status = ""

        # Status emoji (chained conditionals)
        if epoch_loss < 0.1:
            status = "🎯"
        elif epoch_loss < 0.5:
            status = "✅"
        elif epoch_loss < 1.0:
            status = "📈"
        else:
            status = "⏳"

        # Print epoch summary (f-strings with formatting)
        print(f"  Epoch [{epoch:3d}/{config.num_epochs}] {status} "
              f"Loss: {epoch_loss:.4f} | "
              f"Acc: {epoch_acc:.2%} | "
              f"LR: {effective_lr:.1e} {lr_status}")

        # Early stopping check
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                print(f"\n  ⚠️  Early stopping at epoch {epoch} "
                      f"(no improvement for {config.patience} epochs)")
                break

    return history


# ===== 3. Analysis (comprehensions, unpacking, zip) =====
def analyze_results(history: dict[str, list[float]], config: TrainingConfig) -> None:
    """Analyze and display training results using comprehensions."""

    losses = history["loss"]
    accuracies = history["accuracy"]
    num_epochs = len(losses)

    print(f"\n{'='*60}")
    print(f"  📊 Training Analysis")
    print(f"{'='*60}\n")

    # Basic stats (unpacking)
    best_loss = min(losses)
    best_acc = max(accuracies)
    best_loss_epoch = losses.index(best_loss) + 1
    best_acc_epoch = accuracies.index(best_acc) + 1

    print(f"  Best Loss:     {best_loss:.4f} (epoch {best_loss_epoch})")
    print(f"  Best Accuracy: {best_acc:.2%} (epoch {best_acc_epoch})")
    print(f"  Final Loss:    {losses[-1]:.4f}")
    print(f"  Final Acc:     {accuracies[-1]:.2%}")

    # Epoch-over-epoch improvement (zip for parallel iteration)
    improvements = [prev - curr for prev, curr in zip(losses[:-1], losses[1:])]
    positive_improvements = [imp for imp in improvements if imp > 0]
    print(f"\n  Epochs with loss decrease: {len(positive_improvements)}/{len(improvements)}")

    # Find top-3 improvement epochs (enumerate + sorted + unpacking)
    indexed_improvements = list(enumerate(improvements, start=2))
    top_3 = sorted(indexed_improvements, key=lambda pair: pair[1], reverse=True)[:3]
    print(f"  Top 3 improvement epochs:")
    for epoch, improvement in top_3:
        print(f"    Epoch {epoch}: loss decreased by {improvement:.4f}")

    # Convergence check (ternary + walrus)
    if (final_loss := losses[-1]) < 0.2:
        verdict = "✅ Model converged successfully!"
    elif final_loss < 1.0:
        verdict = "⚠️  Model partially converged — try more epochs"
    else:
        verdict = "❌ Model did not converge — check hyperparameters"

    print(f"\n  Verdict: {verdict}")

    # Summary table using f-string alignment
    print(f"\n  {'Epoch':>6} {'Loss':>10} {'Accuracy':>10} {'Status':>8}")
    print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*8}")
    for epoch, (loss, acc) in enumerate(zip(losses, accuracies), start=1):
        status = "✅" if loss < 0.5 else "⏳"
        print(f"  {epoch:>6} {loss:>10.4f} {acc:>9.2%} {status:>8}")


# ===== Run it! =====
config = TrainingConfig(
    model_name="MiniTransformer-v1",
    num_epochs=15,
    batch_size=64,
    learning_rate=3e-4,
    use_gpu=True,
    patience=5,
)

history = simulate_training(config)
analyze_results(history, config)

print(f"\n🎉 Module 1 Practice Complete!")
print(f"Concepts used: variables, types, f-strings, type hints,")
print(f"conditionals, loops, enumerate, zip, comprehensions,")
print(f"unpacking, walrus operator, ternary, repr/str, print debugging")
```

---

## 🔗 Connection to Research Papers

Here's where Module 1 concepts show up in real AI paper code:

### F-Strings in Training Logs (Every Paper)
```python
# From any Transformer training script
print(f"Epoch {epoch}/{num_epochs} | "
      f"Train Loss: {train_loss:.4f} | "
      f"Val Loss: {val_loss:.4f} | "
      f"Perplexity: {math.exp(val_loss):.2f}")
```

### Comprehensions in BPE Tokenizer (GPT, BERT)
```python
# Building vocabulary from token frequencies
vocab = {token: idx for idx, token in enumerate(sorted_tokens)}

# Encoding text
encoded = [vocab.get(word, vocab["[UNK]"]) for word in text.split()]
```

### Type Hints in HuggingFace Transformers
```python
# From HuggingFace's modeling code
def forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    token_type_ids: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    labels: Optional[torch.LongTensor] = None,
) -> Union[Tuple, SequenceClassifierOutput]:
    ...
```

### Ternary for Device Selection (Every GPU Paper)
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

### Unpacking in DataLoaders (Every Training Loop)
```python
for batch_idx, (inputs, labels) in enumerate(train_loader):
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)
    loss = criterion(outputs, labels)
```

---

## 📋 Summary of What You Learned

| Concept | Python Syntax | Java/C# Equivalent |
|---------|--------------|---------------------|
| Blocks | Indentation (4 spaces) | `{ }` |
| Dynamic typing | `x = 42` | `int x = 42;` |
| F-strings | `f"Loss: {loss:.4f}"` | `String.format("Loss: %.4f", loss)` |
| Type hints | `def f(x: int) -> str:` | `String f(int x)` |
| Ternary | `"a" if cond else "b"` | `cond ? "a" : "b"` |
| Walrus | `if (n := len(x)) > 5:` | N/A |
| List comp | `[x*2 for x in lst]` | `lst.stream().map(x -> x*2).collect(...)` |
| Dict comp | `{k: v for k,v in items}` | N/A (use loop) |
| Unpacking | `a, b = 1, 2` | N/A |
| Star unpack | `first, *rest = lst` | N/A |
| None check | `if x is None:` | `if (x == null)` |
| Truthiness | `if lst:` (empty = false) | `if (lst != null && !lst.isEmpty())` |
| For-each | `for x in lst:` | `for (var x : lst)` |
| Enumerate | `for i, x in enumerate(lst):` | Manual counter |
| Zip | `for a, b in zip(l1, l2):` | Manual indexing |
| Debug print | `print(f"{x=}")` | `System.out.println("x=" + x);` |

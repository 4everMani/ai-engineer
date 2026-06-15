# 📘 Python for AI Engineers — Learning Path

> Complete these 15 modules in order before starting the PyTorch path.
> All code runs on **Google Colab**.

---

## Module Progress Tracker

| # | Module | Status | Notes |
|---|--------|--------|-------|
| 1 | [Python Foundations](#module-1-python-foundations) | ⏳ Pending | - |
| 2 | [Data Structures Deep Dive](#module-2-data-structures-deep-dive) | ⏳ Pending | - |
| 3 | [Functions & Functional Patterns](#module-3-functions--functional-patterns) | ⏳ Pending | - |
| 4 | [OOP — The PyTorch Way](#module-4-oop--the-pytorch-way) | ⏳ Pending | - |
| 5 | [Iterators, Generators & Context Managers](#module-5-iterators-generators--context-managers) | ⏳ Pending | - |
| 6 | [File I/O & Data Handling](#module-6-file-io--data-handling) | ⏳ Pending | - |
| 7 | [Error Handling & Debugging](#module-7-error-handling--debugging) | ⏳ Pending | - |
| 8 | [Modules, Packages & Environment](#module-8-modules-packages--environment) | ⏳ Pending | - |
| 9 | [NumPy — The Foundation of Everything](#module-9-numpy--the-foundation-of-everything) | ⏳ Pending | - |
| 10 | [Math in Python](#module-10-math-in-python) | ⏳ Pending | - |
| 11 | [Matplotlib & Data Visualization](#module-11-matplotlib--data-visualization) | ⏳ Pending | - |
| 12 | [Python Performance & Best Practices](#module-12-python-performance--best-practices) | ⏳ Pending | - |
| 13 | [Async Programming & Concurrency](#module-13-async-programming--concurrency) | ⏳ Pending | - |
| 14 | [Web APIs with FastAPI](#module-14-web-apis-with-fastapi) | ⏳ Pending | - |
| 15 | [Testing ML Code with pytest](#module-15-testing-ml-code-with-pytest) | ⏳ Pending | - |

---

## Module Details

### Module 1: Python Foundations
**Topics:**
- Python syntax essentials (indentation-based blocks, no semicolons/braces)
- Variables and dynamic typing
- Primitive types: `int`, `float`, `str`, `bool`, `None`
- String formatting with f-strings
- Type hints and annotations
- Conditional statements (`if/elif/else`)
- Loops (`for`, `while`, `range`, `enumerate`, `zip`)
- List comprehensions, dict comprehensions, set comprehensions
- Tuple unpacking and multiple assignment
- Walrus operator (`:=`)
- Ternary expressions
- `print()` debugging and `repr()` vs `str()`

**AI Engineering Context:**
- F-strings are used everywhere in training logs: `print(f"Epoch {epoch}, Loss: {loss:.4f}")`
- Type hints appear in all modern ML codebases (HuggingFace, PyTorch)
- Comprehensions are the idiomatic way to build vocabularies, filter datasets, and construct batches
- Unpacking is used in data loaders: `for inputs, labels in dataloader:`

---

### Module 2: Data Structures Deep Dive
**Topics:**
- Lists: creation, indexing, slicing (`[start:stop:step]`), negative indexing
- Tuples: immutability, as dict keys, named tuples
- Dictionaries: creation, `.get()`, `.items()`, `.keys()`, `.values()`, dict merging
- Sets: creation, operations (union, intersection, difference), frozensets
- Advanced slicing: multi-dimensional (NumPy preview), slice objects
- Nested data structures (list of dicts, dict of lists)
- `collections` module: `Counter`, `defaultdict`, `deque`, `OrderedDict`, `namedtuple`
- Sorting: `sorted()`, `list.sort()`, custom key functions
- Copying: shallow copy vs deep copy

**AI Engineering Context:**
- Slicing is THE fundamental operation in tensor manipulation: `tensor[0, :, 2:5]`
- Dictionaries are used for model configs, hyperparameters, and vocabulary mappings
- `Counter` is used in BPE to count token pair frequencies
- `defaultdict` is used in building adjacency lists for knowledge graphs
- Nested structures represent batched data: `[{"input_ids": [...], "attention_mask": [...]}]`

---

### Module 3: Functions & Functional Patterns
**Topics:**
- Function definition, parameters, return values
- Default arguments, keyword arguments
- `*args` and `**kwargs` (packing/unpacking)
- Lambda functions (anonymous functions)
- `map()`, `filter()`, `reduce()`
- Closures: functions that capture variables
- Decorators: `@decorator` syntax, writing custom decorators, stacking decorators
- `functools`: `partial`, `lru_cache`, `wraps`
- First-class functions (passing functions as arguments)

**AI Engineering Context:**
- `**kwargs` is everywhere in ML: `model = Transformer(d_model=512, nhead=8, **config)`
- Decorators: `@torch.no_grad()` is a decorator, `@property` used in model configs
- `functools.partial` is used to create pre-configured loss functions
- Lambda functions appear in custom sort keys, data transforms, and callbacks
- Closures are used in learning rate schedule functions

---

### Module 4: OOP — The PyTorch Way
**Topics:**
- Classes and objects: `class`, `self`, `__init__`
- `super().__init__()` and why it's critical
- Instance attributes vs class attributes
- Inheritance (single and multiple)
- Magic/dunder methods: `__len__`, `__getitem__`, `__repr__`, `__str__`, `__call__`, `__add__`, `__eq__`
- `@property` decorator (getters/setters)
- `@staticmethod` and `@classmethod`
- Abstract base classes (`abc` module)
- Composition vs inheritance
- Method Resolution Order (MRO)

**AI Engineering Context:**
- **This is THE most important Python module for PyTorch.** Every model is a class inheriting from `nn.Module`
- `__init__` + `super().__init__()` → defining model layers
- `forward()` → the forward pass (called via `__call__`)
- `__len__` and `__getitem__` → required for `torch.utils.data.Dataset`
- `__repr__` → how `print(model)` shows model architecture
- Abstract classes → HuggingFace's `PreTrainedModel` pattern

---

### Module 5: Iterators, Generators & Context Managers
**Topics:**
- The iteration protocol: `__iter__()` and `__next__()`
- Building custom iterators
- Generators with `yield`
- Generator expressions vs list comprehensions (memory efficiency)
- `itertools`: `chain`, `islice`, `product`, `combinations`
- The `with` statement and context managers
- `__enter__` and `__exit__` methods
- `contextlib`: `contextmanager` decorator, `suppress`
- Lazy evaluation concept

**AI Engineering Context:**
- `torch.no_grad()` is a context manager — understanding `with` is essential
- `torch.cuda.amp.autocast()` is a context manager for mixed precision
- Generators are used for streaming large datasets that don't fit in memory
- `DataLoader` is an iterator — it yields batches one at a time
- `itertools.chain` is used to combine parameter groups for optimizers

---

### Module 6: File I/O & Data Handling
**Topics:**
- Reading/writing text files (`open()`, `read()`, `readline()`, `readlines()`)
- File modes (`r`, `w`, `a`, `rb`, `wb`)
- JSON: `json.load()`, `json.dump()`, `json.loads()`, `json.dumps()`
- CSV: `csv.reader`, `csv.writer`, `csv.DictReader`
- `pathlib.Path`: modern file path handling
- `os` module essentials: `os.path`, `os.listdir`, `os.makedirs`
- Pickle: serialization/deserialization
- Working with binary files
- Downloading files in Colab (`wget`, `gdown`, `requests`)

**AI Engineering Context:**
- Model configs are stored as JSON files
- Datasets come as CSV, JSON, or text files
- `pickle` is how Python serializes objects — `torch.save()` uses pickle internally
- `pathlib` is used in HuggingFace's model/dataset caching
- In Colab, you'll download datasets with `!wget` or `gdown`

---

### Module 7: Error Handling & Debugging
**Topics:**
- `try`/`except`/`else`/`finally` blocks
- Common exceptions: `ValueError`, `TypeError`, `KeyError`, `IndexError`, `RuntimeError`
- Raising exceptions: `raise`
- Custom exception classes
- `assert` statements
- Reading tracebacks (stack traces)
- Debugging in Colab: `%debug` magic, `pdb`
- `logging` module basics
- `warnings` module

**AI Engineering Context:**
- `RuntimeError: CUDA out of memory` — the most common PyTorch error
- `assert` is used in shape checking: `assert x.shape == (batch_size, seq_len, d_model)`
- Custom exceptions in data pipeline validation
- `try/except` around GPU operations for graceful fallback to CPU
- Reading PyTorch tracebacks (they can be extremely long and confusing)

---

### Module 8: Modules, Packages & Environment
**Topics:**
- The `import` statement: `import`, `from ... import`, `as`
- Module vs package vs library
- `__init__.py` and package structure
- Relative vs absolute imports
- Virtual environments: `venv`, `conda`
- `pip install`, `requirements.txt`, `pip freeze`
- Google Colab specifics: `!pip install`, pre-installed packages, drive mounting
- Project structure for ML projects
- `__name__ == "__main__"` pattern

**AI Engineering Context:**
- Understanding imports is critical for using PyTorch, HuggingFace, etc.
- ML projects have specific structure: `model.py`, `train.py`, `data.py`, `config.py`
- Colab has most ML packages pre-installed (torch, numpy, matplotlib)
- `requirements.txt` is used in reproducible research
- Drive mounting in Colab for persistent storage of checkpoints

---

### Module 9: NumPy — The Foundation of Everything
**Topics:**
- `ndarray` creation: `np.array`, `np.zeros`, `np.ones`, `np.arange`, `np.linspace`, `np.random`
- Data types (`dtype`): `float32`, `float64`, `int64`, `bool`
- Shape, dimensions, and `ndim`
- Reshaping: `reshape`, `flatten`, `ravel`, `squeeze`, `expand_dims`, `transpose`
- Indexing: basic, fancy (integer array), boolean masking
- Slicing: multi-dimensional slicing
- Broadcasting rules (critical for understanding tensor operations)
- Vectorized operations: element-wise, aggregations (`sum`, `mean`, `max`, `argmax`)
- Matrix operations: `dot`, `matmul`, `@` operator
- `np.einsum` (Einstein summation — used in attention computations)
- Stacking and concatenation: `np.stack`, `np.concatenate`, `np.vstack`, `np.hstack`
- Random number generation: `np.random.randn`, `np.random.seed`, distributions
- Copy vs view semantics

**AI Engineering Context:**
- **PyTorch tensors mirror the NumPy API almost exactly** — learn NumPy, and you know 80% of PyTorch tensor ops
- Broadcasting is how operations work on tensors of different shapes (critical for attention masks)
- `argmax` is how you get predictions from model outputs
- `einsum` appears in FlashAttention and efficient Transformer implementations
- Random seeds ensure reproducible experiments
- Shape manipulation is the #1 source of bugs in deep learning code

---

### Module 10: Math in Python
**Topics:**
- Vectors and matrices in code (1D and 2D arrays)
- Matrix multiplication: `@` operator, `np.matmul`, `np.dot`
- Transpose: `.T`, `np.transpose`
- Vector norms: L1, L2 (`np.linalg.norm`)
- Dot product and cosine similarity
- Softmax function (implementing from scratch)
- Cross-entropy loss (implementing from scratch)
- Log and exponential operations (`np.log`, `np.exp`)
- Numerical stability tricks (log-sum-exp)
- Probability distributions: Gaussian/Normal, Uniform
- Mean, variance, standard deviation
- Normalization (min-max, z-score, layer normalization concept)
- Basic calculus intuition: derivatives as "rate of change", chain rule as "multiply the rates"

**AI Engineering Context:**
- Matrix multiplication is the core operation in every neural network
- Cosine similarity is how SONAR measures sentence meaning (your LCM notes!)
- Softmax converts raw scores to probabilities (used in attention and output layers)
- Cross-entropy is the standard loss function for classification (BERT, GPT)
- Log-sum-exp trick prevents numerical overflow in softmax
- Layer normalization is in every Transformer

---

### Module 11: Matplotlib & Data Visualization
**Topics:**
- Basic plots: `plt.plot()`, `plt.scatter()`, `plt.bar()`
- Customization: titles, labels, legends, colors, line styles
- Subplots: `plt.subplots()`, grid layouts
- Heatmaps: `plt.imshow()`, `sns.heatmap()` (seaborn)
- Histograms: `plt.hist()` for distribution visualization
- Saving figures: `plt.savefig()`
- Plotting in Colab: `%matplotlib inline`
- Multiple y-axes (for plotting loss + accuracy together)
- Annotating plots

**AI Engineering Context:**
- Plotting training loss curves is essential for debugging model training
- Attention heatmaps visualize what the model is "looking at" (BERT, Transformers)
- Histograms of gradient norms help diagnose vanishing/exploding gradients
- Embedding space visualizations (t-SNE/PCA plots of SONAR vectors)
- Plotting learning rate schedules (warmup + decay)

---

### Module 12: Python Performance & Best Practices
**Topics:**
- Profiling: `%timeit` (Colab magic), `time.time()`, `cProfile`
- Memory: `sys.getsizeof()`, generators vs lists for memory efficiency
- `multiprocessing` and `concurrent.futures` basics
- `dataclasses` for clean configuration objects
- Type hints for large codebases (mypy-style)
- Code organization patterns for ML projects
- Logging best practices for training runs
- `tqdm` progress bars for training loops
- Configuration management (argparse, YAML configs, dataclasses)

**AI Engineering Context:**
- `tqdm` is used in every training loop for progress visualization
- `dataclasses` are used for training configs, model configs, and experiment tracking
- `multiprocessing` is used in `DataLoader(num_workers=N)` for parallel data loading
- Type hints make large ML codebases maintainable
- Proper logging replaces `print()` in production training scripts

---

### Module 13: Async Programming & Concurrency
**Topics:**
- Synchronous vs asynchronous execution (intuitive explanation)
- `asyncio` basics: event loop, `async def`, `await`
- `asyncio.gather()`: running multiple async tasks concurrently
- `asyncio.create_task()`: scheduling coroutines
- `aiohttp`: making async HTTP requests (calling ML APIs)
- `asyncio.Queue`: producer-consumer patterns
- Threading vs multiprocessing vs asyncio: when to use which
- `concurrent.futures`: `ThreadPoolExecutor`, `ProcessPoolExecutor`
- `asyncio.Semaphore`: limiting concurrent operations
- Running async code in Colab (`await` works directly in Colab/Jupyter)

**AI Engineering Context:**
- **FastAPI (Module 14) is built on asyncio** — every model-serving endpoint is an async function
- Calling multiple ML APIs concurrently (e.g., ensemble inference from different models)
- Async data preprocessing pipelines for real-time inference
- `ThreadPoolExecutor` is used for I/O-bound model loading tasks
- `ProcessPoolExecutor` is used for CPU-bound data preprocessing
- Batch inference servers handle concurrent requests using asyncio

---

### Module 14: Web APIs with FastAPI
**Topics:**
- What is a REST API? (HTTP methods: GET, POST)
- FastAPI basics: creating an app, defining endpoints
- Path parameters and query parameters
- Request/response models with Pydantic (`BaseModel`)
- Serving ML predictions: loading a model and creating an inference endpoint
- Input validation (ensuring correct data shapes, types)
- File upload endpoints (for image/audio inference)
- Background tasks (`BackgroundTasks`)
- CORS middleware (allowing frontend apps to call your API)
- Running FastAPI with `uvicorn`
- Testing your API with `requests` and Swagger UI
- Deploying to cloud: basic concepts (Docker container → cloud run)

**AI Engineering Context:**
- **This is THE way AI engineers deploy models for real-world use**
- Every production AI system has a model behind an API endpoint
- Pydantic models enforce input validation (correct tensor shapes, valid text)
- File upload endpoints power image classification, OCR, and audio transcription APIs
- HuggingFace Inference API, OpenAI API, and Anthropic API are all REST APIs
- Understanding APIs is critical for building AI agents that call external tools

---

### Module 15: Testing ML Code with pytest
**Topics:**
- Why test ML code? (reproducibility, catching data pipeline bugs, preventing regressions)
- `pytest` basics: test functions, `assert`, running tests
- Test organization: `test_` prefix, test directories
- Fixtures: `@pytest.fixture` for setup/teardown (e.g., creating test models)
- Parameterized tests: `@pytest.mark.parametrize` (testing multiple inputs)
- Mocking: `unittest.mock`, `monkeypatch` (mocking API calls, GPU availability)
- Testing tensor shapes: asserting model output dimensions
- Testing determinism: seeding and verifying reproducible outputs
- Testing data pipelines: validating data loading, preprocessing, augmentation
- Testing model forward pass: verifying output shapes and value ranges
- Snapshot testing for model outputs
- `pytest` markers: `@pytest.mark.slow`, `@pytest.mark.gpu` for conditional test runs

**AI Engineering Context:**
- Production ML systems MUST have tests — untested pipelines break silently
- Shape tests catch the #1 source of bugs: `assert model(x).shape == (batch_size, num_classes)`
- Fixture patterns for expensive model loading: load once, use across many tests
- Mocking GPU availability lets you test CUDA code on CPU-only CI servers
- Data pipeline tests prevent silent data corruption (wrong normalization, broken tokenization)
- HuggingFace and PyTorch themselves use pytest extensively

# 📕 PyTorch for AI Engineers — Learning Path

> Complete the [Python path](../python/python_path.md) first, then work through these 21 modules in order.
> All code runs on **Google Colab**. Set Runtime → GPU for modules 11+.

---

## Module Progress Tracker

| # | Module | Status | Papers Unlocked | Notes |
|---|--------|--------|----------------|-------|
| 1 | [Tensors — The Atoms of Deep Learning](#module-1-tensors--the-atoms-of-deep-learning) | ⏳ Pending | ALL | - |
| 2 | [Autograd — How Machines Learn](#module-2-autograd--how-machines-learn) | ⏳ Pending | ALL | - |
| 3 | [nn.Module — Building Neural Networks](#module-3-nnmodule--building-neural-networks) | ⏳ Pending | ALL | - |
| 4 | [Loss Functions & Optimizers](#module-4-loss-functions--optimizers) | ⏳ Pending | RLHF, Distillation | - |
| 5 | [The Training Loop](#module-5-the-training-loop) | ⏳ Pending | ALL | - |
| 6 | [Data Pipeline](#module-6-data-pipeline) | ⏳ Pending | BPE, BERT, BLT | - |
| 7 | [Embeddings & Positional Encoding](#module-7-embeddings--positional-encoding) | ⏳ Pending | BERT, Transformers, RoPE | - |
| 8 | [Attention Mechanisms from Scratch](#module-8-attention-mechanisms-from-scratch) | ⏳ Pending | Attention Is All You Need, FlashAttention, MQA, GQA | - |
| 9 | [Building a Transformer from Scratch](#module-9-building-a-transformer-from-scratch) | ⏳ Pending | Core Architecture, LCM, BERT | - |
| 10 | [Advanced Layers & Architectures](#module-10-advanced-layers--architectures) | ⏳ Pending | MoE, FlashAttention, Titans | - |
| 11 | [GPU & Memory Management](#module-11-gpu--memory-management) | ⏳ Pending | FlashAttention, Optimizations | - |
| 12 | [Model Saving, Loading & Fine-Tuning](#module-12-model-saving-loading--fine-tuning) | ⏳ Pending | Distillation, BERT, RLHF | - |
| 13 | [Diffusion Models Fundamentals](#module-13-diffusion-models-fundamentals) | ⏳ Pending | Large Concept Models | - |
| 14 | [Computer Vision with PyTorch](#module-14-computer-vision-with-pytorch) | ⏳ Pending | ViT, CLIP, DINO, Image Transformers | - |
| 15 | [NLP Pipeline End-to-End](#module-15-nlp-pipeline-end-to-end) | ⏳ Pending | BPE, BERT, GPT, text generation | - |
| 16 | [Reinforcement Learning Basics](#module-16-reinforcement-learning-basics) | ⏳ Pending | RLHF, DeepSeek R1, PPO | - |
| 17 | [Distributed Training & Scaling](#module-17-distributed-training--scaling) | ⏳ Pending | Large-scale training, multi-GPU | - |
| 18 | [Model Optimization & Deployment](#module-18-model-optimization--deployment) | ⏳ Pending | Quantization, ONNX, TorchServe | - |
| 19 | [HuggingFace Ecosystem Deep Dive](#module-19-huggingface-ecosystem-deep-dive) | ⏳ Pending | Transformers, Datasets, Trainer, PEFT | - |
| 20 | [Experiment Tracking & MLOps](#module-20-experiment-tracking--mlops) | ⏳ Pending | wandb, TensorBoard, MLflow | - |
| 21 | [Paper Code Reading Walkthroughs](#module-21-paper-code-reading-walkthroughs) | ⏳ Pending | ALL dissected papers | - |

---

## Module Details

### Module 1: Tensors — The Atoms of Deep Learning
**Topics:**
- What is a tensor? (scalar → vector → matrix → tensor)
- Creating tensors: `torch.tensor`, `torch.zeros`, `torch.ones`, `torch.randn`, `torch.arange`, `torch.linspace`
- Tensor attributes: `.shape`, `.dtype`, `.device`, `.ndim`, `.numel()`
- Data types: `torch.float32`, `torch.float16`, `torch.int64`, `torch.bool`
- Type casting: `.float()`, `.long()`, `.to(dtype)`
- Device management: `.to('cuda')`, `.cpu()`, `torch.device`
- Reshaping: `.view()`, `.reshape()`, `.squeeze()`, `.unsqueeze()`, `.permute()`, `.transpose()`
- View vs copy: when does reshaping share memory?
- Indexing and slicing (same as NumPy)
- Boolean masking: `tensor[tensor > 0]`
- Broadcasting rules (how operations work on tensors of different shapes)
- Basic operations: `+`, `-`, `*`, `/`, `@` (matmul), `torch.matmul`, `torch.bmm`
- Aggregations: `.sum()`, `.mean()`, `.max()`, `.argmax()`, `.min()`, `.argmin()` with `dim` argument
- Concatenation: `torch.cat`, `torch.stack`
- In-place operations: `add_()`, `mul_()` (and why they can break autograd)
- Tensor ↔ NumPy conversion: `.numpy()`, `torch.from_numpy()` (shared memory!)

**Papers This Unlocks:**
Every single paper in your tracker uses tensors. After this module, you'll understand lines like:
```python
concepts = torch.randn(8, 20, 1024)  # 8 docs, 20 sentences each, 1024-dim vectors
input_concepts = concepts[:, :-1, :]  # all but last sentence
```

---

### Module 2: Autograd — How Machines Learn
**Topics:**
- What is a gradient? (intuitive: the "slope" that tells you which direction to adjust)
- Computational graphs: how PyTorch tracks operations
- `requires_grad=True`: telling PyTorch to watch a tensor
- `.backward()`: computing gradients automatically
- `.grad`: accessing the computed gradients
- `torch.no_grad()`: disabling gradient tracking (inference mode)
- `.detach()`: breaking a tensor out of the computational graph
- Gradient accumulation: why gradients ADD, not replace
- `optimizer.zero_grad()`: why you must zero gradients before each step
- Gradient clipping: `torch.nn.utils.clip_grad_norm_`
- The chain rule in code (how backpropagation actually works)
- `retain_graph=True` and when you need it

**Papers This Unlocks:**
Every training procedure in every paper. After this module, you'll understand:
```python
loss.backward()     # compute gradients
optimizer.step()    # update weights using gradients
optimizer.zero_grad()  # reset gradients for next iteration
```

---

### Module 3: nn.Module — Building Neural Networks
**Topics:**
- `nn.Module`: the base class for ALL PyTorch models
- The `__init__` + `super().__init__()` pattern
- `forward()` method: defining the computation
- Why you call `model(x)` not `model.forward(x)` (the `__call__` hook)
- `nn.Linear`: the fundamental building block (matrix multiplication + bias)
- `nn.ReLU`, `nn.GELU`, `nn.Sigmoid`, `nn.Tanh` — activation functions
- `nn.Sequential`: chaining layers together
- Parameters: `.parameters()`, `.named_parameters()`
- Buffers: `.register_buffer()` (non-trainable state that moves with the model)
- `nn.ModuleList` and `nn.ModuleDict` (for dynamic architectures)
- Model inspection: `print(model)`, parameter counting
- Nesting modules: models that contain other models
- `model.train()` vs `model.eval()` — what changes?

**Papers This Unlocks:**
Every model architecture. After this module, `BaseLCM(nn.Module)` from your LCM notes will make complete sense:
```python
class BaseLCM(nn.Module):
    def __init__(self, d_model=1024, n_heads=16, n_layers=26):
        super().__init__()  # ← you now know why this is here
        self.output_head = nn.Linear(d_model, d_model)  # ← you know what this does
```

---

### Module 4: Loss Functions & Optimizers
**Topics:**
- What is a loss function? (measures "how wrong" the model is)
- `nn.MSELoss()`: Mean Squared Error (regression tasks)
- `nn.CrossEntropyLoss()`: classification tasks (includes softmax internally!)
- `nn.BCEWithLogitsLoss()`: binary classification
- `nn.NLLLoss()`: Negative Log Likelihood
- `nn.CosineEmbeddingLoss()`: similarity-based loss
- Writing custom loss functions
- Reduction modes: `reduction='mean'` vs `reduction='sum'` vs `reduction='none'`
- `torch.optim.SGD`: Stochastic Gradient Descent
- `torch.optim.Adam` and `torch.optim.AdamW`: why AdamW is the default in modern AI
- Learning rate: what it is and why it matters
- Learning rate schedulers: `StepLR`, `CosineAnnealingLR`, warmup schedules
- Weight decay (L2 regularization)
- Parameter groups: different learning rates for different layers

**Papers This Unlocks:**
- LCM uses MSE loss (and explains why it fails → mean collapse)
- BERT uses CrossEntropyLoss for masked language modeling
- RLHF papers use custom reward-based losses
- Distillation papers use KL-divergence loss

---

### Module 5: The Training Loop
**Topics:**
- The complete training loop: forward → loss → backward → step
- Batching: why we train on batches, not single examples
- Epochs: iterating over the entire dataset multiple times
- Validation loop: evaluating without training (no gradients)
- `model.train()` vs `model.eval()` in the loop
- `torch.no_grad()` context manager for validation
- Gradient accumulation (simulating larger batch sizes)
- Early stopping: halting training when validation loss stops improving
- Logging metrics: tracking loss, accuracy, etc.
- `tqdm` progress bars for training visualization
- Saving checkpoints during training
- Reproducibility: setting seeds (`torch.manual_seed`)
- Common training bugs and how to diagnose them

**Papers This Unlocks:**
Every paper that trains a model. This module teaches you the "engine" behind all research:
```python
for epoch in range(num_epochs):
    model.train()
    for batch in dataloader:
        optimizer.zero_grad()
        output = model(batch)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
```

---

### Module 6: Data Pipeline
**Topics:**
- `torch.utils.data.Dataset`: abstract class you inherit from
- Implementing `__len__` and `__getitem__`
- `torch.utils.data.DataLoader`: automatic batching, shuffling, parallel loading
- `batch_size`, `shuffle`, `num_workers`, `drop_last`
- Custom `collate_fn`: how to handle variable-length sequences
- Padding sequences to equal length
- Attention masks: telling the model which tokens are real vs padding
- `torch.utils.data.TensorDataset`: quick dataset from tensors
- `torch.utils.data.random_split`: train/val/test splits
- Transforms: preprocessing data before feeding to model
- Working with text data: tokenization → encoding → batching pipeline
- Working with HuggingFace `datasets` library

**Papers This Unlocks:**
- Tokenization papers (BPE, BLT): how tokenized data flows into models
- BERT: masked language model data pipeline with attention masks
- Any paper with a training procedure

---

### Module 7: Embeddings & Positional Encoding
**Topics:**
- What is an embedding? (mapping discrete IDs to dense vectors)
- `nn.Embedding`: lookup table mechanics
- Vocabulary → embedding: the token-to-vector pipeline
- `nn.Embedding` parameters: `num_embeddings`, `embedding_dim`, `padding_idx`
- One-hot encoding vs dense embeddings (why dense is better)
- Sinusoidal positional encoding (from "Attention Is All You Need")
- Implementing positional encoding from scratch
- Learned positional embeddings (an alternative)
- Rotary Position Embeddings (RoPE) — high-level concept
- `nn.EmbeddingBag`: efficient embeddings for bag-of-words
- Pre-trained embeddings: loading GloVe/Word2Vec into `nn.Embedding`

**Papers This Unlocks:**
- BERT: token embeddings + segment embeddings + position embeddings
- Attention Is All You Need: sinusoidal positional encoding
- VideoRoPE: Rotary Position Embedding
- Large Concept Models: SONAR embeddings (sentence-level)

---

### Module 8: Attention Mechanisms from Scratch
**Topics:**
- Intuition: attention as "which parts of the input should I focus on?"
- Query, Key, Value (Q, K, V) — what they mean
- Scaled Dot-Product Attention: the math and the code
  ```
  Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V
  ```
- Implementing scaled dot-product attention from scratch
- Multi-Head Attention: running multiple attention heads in parallel
- `nn.MultiheadAttention`: PyTorch's built-in implementation
- Self-attention vs cross-attention
- Causal masking: preventing the model from "cheating" by looking ahead
- `generate_square_subsequent_mask()`: creating causal masks
- Attention weight visualization (extracting and plotting attention patterns)
- Multi-Query Attention (MQA): shared keys/values
- Grouped Query Attention (GQA): compromise between MHA and MQA

**Papers This Unlocks:**
- Attention Is All You Need (the foundational paper)
- FlashAttention (optimized attention computation)
- Multi Query Attention paper
- Grouped Query Attention paper
- BERT (encoder self-attention)
- Large Concept Models (context encoder uses attention)

---

### Module 9: Building a Transformer from Scratch
**Topics:**
- The Transformer architecture: encoder + decoder
- `nn.TransformerEncoderLayer`: what's inside (attention → add & norm → FFN → add & norm)
- `nn.TransformerEncoder`: stacking encoder layers
- `nn.TransformerDecoderLayer` and `nn.TransformerDecoder`
- Positional encoding integration
- Source mask vs target mask vs memory mask
- Building an encoder-only model (BERT-style)
- Building a decoder-only model (GPT-style)
- Building an encoder-decoder model (T5-style, original Transformer)
- Feed-forward network (FFN): the "thinking" layer between attention layers
- Pre-LayerNorm vs Post-LayerNorm (and why modern models use Pre-LN)
- The complete forward pass traced step by step

**Papers This Unlocks:**
- Attention Is All You Need (the original Transformer)
- BERT (encoder-only Transformer)
- Large Concept Models (uses `nn.TransformerEncoder` in BaseLCM)
- Google Titans
- All core architecture papers

---

### Module 10: Advanced Layers & Architectures
**Topics:**
- `nn.LayerNorm`: what it normalizes and why
- `nn.BatchNorm1d`/`nn.BatchNorm2d`: batch normalization
- `nn.Dropout`: randomly zeroing activations during training
- `nn.GELU` vs `nn.ReLU` vs `nn.SiLU` (activation function choices in modern models)
- Residual connections: `output = layer(x) + x` (the skip connection)
- Pre-norm vs post-norm patterns
- `nn.ModuleList`: dynamic layer stacking
- `nn.ModuleDict`: named layer collections
- Mixture of Experts (MoE) routing: gating networks and expert selection
- Building a simple MoE layer from scratch
- Multi-scale architectures: processing at different resolutions

**Papers This Unlocks:**
- MoE papers: Sparsely-Gated MoE, GShard, Switch Transformers
- FlashAttention (understanding the attention + norm + FFN block)
- Titans (advanced attention architectures)
- Every Transformer-based paper uses LayerNorm, Dropout, and residual connections

---

### Module 11: GPU & Memory Management
**Topics:**
- GPU basics: what a GPU is and why it's faster for matrix math
- `.to(device)` and `.cuda()`: moving tensors and models to GPU
- `torch.device('cuda' if torch.cuda.is_available() else 'cpu')` pattern
- Google Colab GPU: setting up, checking GPU type, monitoring memory
- `torch.cuda.memory_allocated()` and `torch.cuda.memory_reserved()`
- Mixed precision training: `torch.cuda.amp.autocast()` and `GradScaler`
- Why FP16 saves memory and speeds up training
- Gradient checkpointing: trading compute for memory
- `torch.cuda.empty_cache()`: freeing unused GPU memory
- Common CUDA errors and how to fix them
- Batch size tuning: finding the maximum batch size that fits in GPU memory
- Model parallelism vs data parallelism (high-level concepts)

**Papers This Unlocks:**
- FlashAttention: memory-efficient attention
- Optimization papers: 1-bit LLMs, quantization
- Any paper that reports VRAM requirements
- Understanding the "Pragmatic Engineer Thoughts" in your dissected notes

---

### Module 12: Model Saving, Loading & Fine-Tuning
**Topics:**
- `model.state_dict()`: the dictionary of all parameters
- `torch.save()` and `torch.load()`: saving/loading checkpoints
- Saving the optimizer state (for resuming training)
- Loading a pre-trained model onto a different device
- Freezing layers: `param.requires_grad = False`
- Transfer learning: using a pre-trained model on a new task
- Fine-tuning: unfreezing specific layers
- LoRA concept: low-rank adaptation (adding small trainable matrices)
- HuggingFace `from_pretrained()`: loading models from the Hub
- HuggingFace `AutoModel`, `AutoTokenizer`
- Saving/loading with HuggingFace's `save_pretrained()`

**Papers This Unlocks:**
- BERT: fine-tuning a pre-trained model for downstream tasks
- Distillation papers: teacher-student training
- RLHF: fine-tuning with human feedback
- LoRA/PEFT: parameter-efficient fine-tuning

---

### Module 13: Diffusion Models Fundamentals
**Topics:**
- What is a diffusion model? (the "sculptor from noise" analogy)
- Forward process: progressively adding Gaussian noise
- Noise schedule: `beta`, `alpha`, `alpha_cumprod`
- Forward process equation: `x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * epsilon`
- Reverse process: iteratively denoising
- The denoising network: predicting noise vs predicting the clean signal
- DDPM (Denoising Diffusion Probabilistic Models) — the foundation
- Training objective: simple MSE on predicted noise
- Inference: the denoising loop from pure noise to a clean sample
- Classifier-free guidance: controlling generation quality
- `register_buffer` for noise schedule parameters
- Building a simple diffusion model from scratch in PyTorch

**Papers This Unlocks:**
- Large Concept Models — Diffusion LCM (the star approach in your notes!)
- Inference-Time Scaling for Diffusion Models
- Understanding the `DiffusionLCM` code in your `large_concept_models_detailed_notes.md`

---

### Module 14: Computer Vision with PyTorch
**Topics:**
- How computers "see": images as tensors (C × H × W)
- `torchvision`: the computer vision toolkit
- Image transforms: `transforms.Compose`, `Resize`, `Normalize`, `ToTensor`, `RandomCrop`, `RandomHorizontalFlip`
- Convolutional Neural Networks (CNNs): `nn.Conv2d`, `nn.MaxPool2d`, kernels, strides, padding
- Building a CNN from scratch (LeNet → ResNet concepts)
- Residual connections in vision (ResNet's skip connections)
- Vision Transformer (ViT): patching images into sequences, treating patches as tokens
- CLIP concept: connecting images and text in a shared embedding space
- DINO/DINOv2: self-supervised visual features
- Feature extraction: using pre-trained vision models as feature extractors
- `torchvision.models`: loading pre-trained ResNet, ViT, etc.
- Image classification, object detection, and segmentation (high-level)

**Papers This Unlocks:**
- Image is 16×16 Words (ViT — patching images into tokens)
- CLIP (connecting vision and language)
- DINO (self-supervised vision)
- IMAGEBIND (multi-modal embeddings)
- DeepSeek image generation
- Any paper that processes images or video

---

### Module 15: NLP Pipeline End-to-End
**Topics:**
- The complete NLP pipeline: raw text → tokens → IDs → embeddings → model → output
- Tokenization in practice: BPE, WordPiece, SentencePiece
- Building a vocabulary from scratch
- Token-to-ID mapping and padding
- Text classification: sentiment analysis, topic detection
- Sequence-to-sequence: encoder-decoder for translation/summarization
- Autoregressive text generation: greedy decoding, top-k, top-p (nucleus) sampling
- Beam search: exploring multiple hypotheses
- Temperature: controlling randomness in generation
- KV-Cache: why cached key/value pairs speed up generation
- Perplexity: measuring language model quality
- Prompt engineering: structured input formatting

**Papers This Unlocks:**
- BPE and BLT (tokenization algorithms)
- BERT (masked language model, text classification)
- Attention Is All You Need (sequence-to-sequence)
- Large Concept Models (concept-level generation vs token-level)
- Chain-of-Thought (structured prompting)
- DSPy (programmatic prompt engineering)

---

### Module 16: Reinforcement Learning Basics
**Topics:**
- RL fundamentals: agent, environment, state, action, reward
- Policy: the function that decides what action to take
- Value function: estimating how good a state is
- Reward signal: the feedback mechanism
- Policy gradient methods: REINFORCE algorithm
- PPO (Proximal Policy Optimization): the algorithm behind ChatGPT's training
- Reward modeling: training a model to predict human preferences
- RLHF pipeline: SFT → Reward Model → PPO fine-tuning
- DPO (Direct Preference Optimization): a simpler alternative to RLHF
- KL-divergence penalty: preventing the model from diverging too far
- Implementing a simple policy gradient in PyTorch
- The exploration-exploitation tradeoff

**Papers This Unlocks:**
- Deep RL with Human Feedback
- Fine-Tuning Language Models with RLHF
- Training Language Models with RLHF
- DeepSeek R1 (RL-based reasoning)
- All RLHF papers — this module is essential for understanding how models are aligned

---

### Module 17: Distributed Training & Scaling
**Topics:**
- Why distribute? (models and datasets too large for one GPU)
- Data Parallelism: same model on multiple GPUs, different data
- `torch.nn.DataParallel` (DP): the simple but limited approach
- `torch.nn.parallel.DistributedDataParallel` (DDP): the production standard
- Setting up DDP: `init_process_group`, `DistributedSampler`, rank/world_size
- FSDP (Fully Sharded Data Parallelism): sharding model parameters across GPUs
- Tensor Parallelism vs Pipeline Parallelism (high-level concepts)
- Gradient synchronization: all-reduce operations
- Mixed precision + DDP: combining techniques for maximum throughput
- DeepSpeed and `accelerate`: high-level distributed training libraries
- Multi-node training concepts (training across multiple machines)
- Scaling laws: how performance improves with compute/data/parameters

**Papers This Unlocks:**
- GShard (distributed MoE training)
- Any paper that trains large models (they all use DDP or FSDP)
- TensorFlow paper (distributed computation graphs)
- Ray (distributed computing framework)
- Understanding VRAM estimates in your paper notes

---

### Module 18: Model Optimization & Deployment
**Topics:**
- Model quantization: FP32 → FP16 → INT8 → INT4 (reducing model size)
- Post-Training Quantization (PTQ) vs Quantization-Aware Training (QAT)
- `torch.quantization` API
- ONNX export: `torch.onnx.export()` (framework-agnostic model format)
- TorchScript: `torch.jit.script` and `torch.jit.trace` (ahead-of-time compilation)
- `torch.compile()`: the modern compilation approach (PyTorch 2.0+)
- TorchServe: serving PyTorch models in production
- Model pruning: removing unnecessary weights
- Knowledge distillation for deployment: training a small model to mimic a large one
- Inference optimization: batched inference, dynamic batching
- Latency vs throughput tradeoffs
- Speculative decoding: using a small model to speed up a large model

**Papers This Unlocks:**
- 1-bit LLMs / 1.58-bit quantization
- ByteDance 1.58
- Speculative Decoding
- Distillation papers (Distilling Knowledge, BYOL, DINO)
- FlashAttention-3 (hardware-aware optimization)
- 1B outperforms 405B (efficiency techniques)

---

### Module 19: HuggingFace Ecosystem Deep Dive
**Topics:**
- HuggingFace Hub: browsing, downloading, and uploading models/datasets
- `transformers` library: `AutoModel`, `AutoTokenizer`, `AutoConfig`
- Pipeline API: quick inference (`pipeline('sentiment-analysis')`, `pipeline('text-generation')`)
- Loading specific model architectures: `BertModel`, `GPT2LMHeadModel`, etc.
- `datasets` library: loading, streaming, processing, and filtering datasets
- `Trainer` and `TrainingArguments`: the high-level training API
- Custom training with `Trainer`: custom loss functions, metrics, callbacks
- PEFT library: LoRA, QLoRA, prefix tuning, adapter layers
- `accelerate` library: multi-GPU and mixed precision with minimal code changes
- `tokenizers` library: fast tokenizer training and usage
- `evaluate` library: metrics (BLEU, ROUGE, accuracy, F1)
- Model cards and documentation best practices

**Papers This Unlocks:**
- BERT (HuggingFace's flagship model)
- All fine-tuning and transfer learning papers
- RLHF (TRL library built on top of HuggingFace)
- PEFT/LoRA papers
- The modern AI engineer's daily toolkit

---

### Module 20: Experiment Tracking & MLOps
**Topics:**
- Why track experiments? (reproducibility, comparison, collaboration)
- TensorBoard: logging scalars, images, histograms, model graphs
- `torch.utils.tensorboard.SummaryWriter`
- Weights & Biases (wandb): logging, visualization, sweeps, artifacts
- `wandb.init()`, `wandb.log()`, `wandb.watch()`
- MLflow: experiment tracking, model registry, deployment
- Hyperparameter sweeps: grid search, random search, Bayesian optimization
- Configuration management: YAML configs, Hydra, OmegaConf
- Reproducibility checklist: seeds, deterministic mode, environment capture
- Model versioning and artifact management
- CI/CD for ML: automated testing, training, and deployment pipelines
- Monitoring deployed models: drift detection, performance degradation

**Papers This Unlocks:**
- Any paper with extensive ablation studies (they all track experiments)
- Evaluation & Benchmarking papers (MMLU, HumanEval, MT-Bench)
- Case studies (Meta, Netflix, Uber, Swiggy — all use MLOps)
- Understanding how to reproduce paper results

---

### Module 21: Paper Code Reading Walkthroughs
**Topics:**
- Line-by-line walkthrough of BPE implementation code
- Line-by-line walkthrough of BERT architecture code
- Line-by-line walkthrough of `BaseLCM` from your LCM notes
- Line-by-line walkthrough of `DiffusionLCM` from your LCM notes
- Line-by-line walkthrough of `ResidualVectorQuantizer` from your LCM notes
- Pattern recognition: identifying common architecture patterns across papers
- Reading HuggingFace model source code
- Debugging research paper code: common issues and fixes
- Building your own implementations from research paper descriptions
- Contributing to open-source AI projects

**Papers This Unlocks:**
All dissected papers — this is the capstone module where everything comes together. You'll be able to:
- Read any PyTorch code block in your paper notes with full understanding
- Modify and experiment with the code
- Build your own implementations from scratch
- Contribute to open-source AI projects like HuggingFace Transformers

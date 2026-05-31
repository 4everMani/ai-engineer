---
name: build-paper-project
description: A specialized agentic skill that builds a functional, self-contained mini-project for an academic research paper to demonstrate its core concepts using Python and scikit-learn.
---

# Build Paper Project Skill

You are a Lead AI Prototyper. When the user invokes this skill for a specific research paper, your goal is to build a highly-commented, functional "toy" implementation of the paper's core architecture or algorithm.

## Technical Constraints
- **Language**: Python
- **Libraries (Context-Dependent)**: Choose the tech stack dynamically based on the paper's core concept:
  - For **Neural Architectures & Deep Learning** (e.g., Transformers, Attention, MoE): You MUST strictly use **PyTorch** (`torch`). You must explicitly write code that handles GPU device placement (`.to('cuda')`), VRAM, and tensor shapes.
  - For **Algorithms & Data Processing** (e.g., Tokenization, BPE, distance metrics): Use standard Python, `numpy`, or `scikit-learn` where appropriate, as these concepts do not fundamentally require GPU tensors.

## Workflow

### 1. Concept Isolation
Identify the single most critical mathematical or architectural contribution of the paper (e.g., iterative token merging in BPE, cosine similarity attention mechanisms).

### 2. Project Scaffolding
Create a dedicated folder for the project within the user's current workspace: `projects/<paper_name_snake_case>/`.

### 3. Implementation
Write a self-contained Python script (e.g., `main.py`) inside the folder. 
**CRITICAL RULE**: You must explicitly map your code variables to the exact mathematical notations used in the paper. Use heavy inline comments to explain how the code executes the theoretical math.

### 4. Execution / Test Block
The script must include an `if __name__ == "__main__":` block that runs a hardcoded, toy example. It should print out the initial state, the intermediate mathematical transformations, and the final result.

### 5. Documentation
Generate a brief `README.md` in the project folder containing:
- The core concept being demonstrated.
- A translation map from the paper's mathematical variables to your Python code variables.
- The terminal command to run the script.

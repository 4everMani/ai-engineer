# 🧠 Your AI Engineering Learning Workflow

Welcome to your custom-built AI Engineering curriculum! This repository contains over 70 state-of-the-art research papers. 

To help you transition from Software Engineer to AI Engineer, we have built a highly specific set of AI agent skills. Here is how you use them so you never forget!

---

## 🛠️ The Custom Skills

Whenever you want to learn a paper or test your knowledge, simply type a prompt telling the agent to "use" one of these skills:

### 1. `dissect-paper` (The Theory Deep-Dive)
**When to use it:** When you are starting a brand new paper.
**How to use it:** *"Use the `dissect-paper` skill on the Byte Latent Transformer paper."*
**What it does:** 
- It breaks the paper down interactively, section-by-section.
- It provides a "Math-to-Code" translation for every complex formula (Python pseudo-code).
- It provides pragmatic, real-world engineering critiques (VRAM, latency, etc.).
- If it's an architecture paper, it will automatically use the `drawio-skill` to generate visual diagrams for you.

### 2. `build-paper-project` (The Practical Implementation)
**When to use it:** After dissecting a paper, when you want to see the code work.
**How to use it:** *"Use the `build-paper-project` skill on the paper we just read."*
**What it does:** 
- It sets up a dedicated folder in `projects/`.
- It writes a 100% functional, highly-commented Python/scikit-learn script that mirrors the paper's math.
- It gives you a runnable toy example so you can experiment with the algorithm yourself.

### 3. `test-me` (The Feynman Technique)
**When to use it:** When you think you understand a concept, but want to prove it.
**How to use it:** *"Use the `test-me` skill for the Attention mechanism."*
**What it does:** 
- The agent stops teaching and acts as a strict examiner.
- You must explain the concept back to the agent in your own words.
- The agent will grade you and aggressively hunt for hidden flaws in your mental model.

---

## 🗺️ The Learning Tracker & Synthesis

- **Tracker**: Open [learning_tracker.md](learning_tracker.md) to track your progress.
- **Synthesis Milestones**: You will notice "Category Synthesis Sessions" at the end of every category in the tracker. When you hit one, **do not read a new paper**. Instead, ask the agent to *"Run a Synthesis Session"* to help you connect the dots between all the previous papers you just read.

---

## 📖 How to Actually Read an AI Paper (The 3-Pass Method)

Academic papers are **not** meant to be read linearly from start to finish like a novel. If you try to read them from page 1 to page 10, you will get bogged down in math and lose the big picture. 

Here is the exact order you should read them in:

### Pass 1: The "What and Why" (5-10 minutes)
**Goal:** Decide if the paper is worth your time and understand the high-level problem.
1. **Title & Abstract:** Read carefully. What is the core claim?
2. **Introduction:** Read the first paragraph (the background) and the last paragraph (their specific contribution).
3. **Conclusion:** Read this to see what they actually achieved.
4. **Figures & Diagrams:** Skim the images. A good architecture diagram (like the Transformer block in *Attention Is All You Need*) usually explains 80% of the paper.
*👉 Stop here and ask yourself: "Do I understand the problem they are solving?"*

### Pass 2: The "How" (The Core Algorithm)
**Goal:** Understand the mechanics without getting lost in the proofs.
1. **Methodology / Architecture Section:** This is the meat of the paper. Read this carefully. 
2. **Equations:** Do **not** try to memorize the math proofs. Look at the variables and figure out the inputs and outputs. (This is where the `dissect-paper` skill's "Math-to-Code" translation is a lifesaver).
*👉 You can safely SKIP:* "Related Work" (unless you are a researcher looking for citations) and complex mathematical proofs in the Appendix.

### Pass 3: The "Does it actually work?" (The Pragmatic Engineer view)
**Goal:** Evaluate the paper for real-world production.
1. **Experiments / Results:** Skim the tables. Did it beat the baseline? 
2. **Ablation Studies:** This is the most important part of the results! Ablation studies are where authors remove parts of their architecture one by one to see what breaks. It tells you which part of their code is actually doing the heavy lifting.
3. **Limitations:** Good papers tell you where their model fails. Pay attention to this for production engineering.

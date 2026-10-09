# MiniTransformer Lab 

A from-scratch implementation and exploration of Transformer architecture, starting from the mathematical foundations of attention and gradually progressing toward RAG, Agentic AI, and AI evaluation.

## 'Project Goal':

The goal of this project is to understand how modern LLM systems work by implementing their core concepts step by step rather than treating them as black boxes.

### Roadmap

* Phase 1 — Scaled Dot-Product Attention
* Phase 2 — Self-Attention
* Phase 3 — Positional Encoding
* Phase 4 — Multi-Head Attention
* Phase 5 — Feed-Forward Network
* Phase 6 — Residual Connections & Layer Normalization
* Phase 7 — Tokenization
* Phase 8 — Transformer Architecture
* Phase 9 — Causal Masking
* Phase 10 — Training a Mini Transformer
* Phase 11 — Text Generation
* Phase 12 — Attention Visualization
* Phase 13+ — RAG, Agentic AI & Evaluation

---

## Phase 1 — Scaled Dot-Product Attention

Implemented the core attention mechanism from scratch using NumPy.

### Formula

Attention is calculated as:

`Attention(Q, K, V) = softmax(QKᵀ / √dₖ)V`

### Concepts Covered

* Query (Q)
* Key (K)
* Value (V)
* Dot-product similarity
* Scaling by √dₖ
* Softmax
* Attention weights
* Weighted sum of values

### Implementation

The core implementation is available in:

`src/attention.py`

An experiment demonstrating the mechanism is available in:

`experiments/01_attention.py`

### Example

The experiment creates Query, Key, and Value matrices and calculates:

1. Query-Key similarity
2. Scaled attention scores
3. Softmax attention weights
4. Final attention output

### Technologies

* Python
* NumPy

## 🎯 Why I'm Building This

Instead of only using Transformer libraries, I'm building the architecture step by step to understand the mathematics, implementation, debugging, and evaluation of modern AI systems.

This project will eventually connect Transformer fundamentals with RAG, Agentic AI, and AI evaluation.

---

## 👨‍💻 Author

Yatharth Saini

GitHub: [Yatharth0143](https://github.com/Yatharth0143)

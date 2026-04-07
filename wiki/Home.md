# 🧠 AI-Essentials — Deep-Dive Course Wiki

> **Welcome!** This wiki is structured as a progressive course. Start at Module 1 and work your way through. Each module builds on the previous one, so reading in order is strongly recommended for beginners.

---

## 📚 Table of Contents

| Module | Title | What You Will Learn |
|--------|-------|---------------------|
| [1](01-Introduction.md) | **Introduction** | What the project is, why it exists, how to install it, and the big picture of deep learning |
| [2](02-Scalar-Autodiff.md) | **Scalar Automatic Differentiation** | The `Value` class, computation graphs, the chain rule, and reverse-mode backpropagation |
| [3](03-Activation-Functions.md) | **Activation Functions** | Why nonlinearity matters, tanh vs. relu, the `activations.py` registry |
| [4](04-Neural-Network-Building-Blocks.md) | **Neural Network Building Blocks** | The `Neuron` and `Layer` classes, He initialization, forward and backward passes |
| [5](05-MLP.md) | **Multi-Layer Perceptron** | Composing layers into an MLP, layer specs, parameter counting |
| [6](06-Loss-Functions.md) | **Loss Functions** | MSE for regression, binary cross-entropy for classification, gradient flow through losses |
| [7](07-Optimization.md) | **Optimization** | SGD, momentum, gradient clipping, learning-rate decay |
| [8](08-Math-Utilities.md) | **Math Utilities** | Dot product, matrix multiplication, shape helpers |
| [9](09-Training-Loop.md) | **The Training Loop** | Complete end-to-end walkthrough of `main.py`, mini-batch SGD, live visualization |
| [10](10-Advanced-Topics.md) | **Advanced Topics** | He initialization deep-dive, data normalization, extending the library |

---

## 🗺️ Architecture at a Glance

```
Value  ──►  Neuron  ──►  Layer  ──►  MLP
              ▲
         Activations
              │
         Math (dot, matmul)

Loss (MSE, BCE)  ──►  Optimizer (SGD)
```

---

## 🧭 How to Use This Wiki

- **Beginner?** Read every module in order, run the code snippets in a Python REPL or Jupyter notebook as you go.
- **Experienced reader?** Jump directly to any module using the table above.
- **Looking for a specific API?** See the module that covers the relevant file.

| File | Covered In |
|------|-----------|
| `value.py` | [Module 2](02-Scalar-Autodiff.md) |
| `activations.py` | [Module 3](03-Activation-Functions.md) |
| `neuron.py` | [Module 4](04-Neural-Network-Building-Blocks.md) |
| `layer.py` | [Module 4](04-Neural-Network-Building-Blocks.md) |
| `mlp.py` | [Module 5](05-MLP.md) |
| `loss.py` | [Module 6](06-Loss-Functions.md) |
| `optimizer.py` | [Module 7](07-Optimization.md) |
| `math.py` | [Module 8](08-Math-Utilities.md) |
| `main.py` | [Module 9](09-Training-Loop.md) |

---

*Next → [Module 1: Introduction](01-Introduction.md)*

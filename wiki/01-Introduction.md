# Module 1 — Introduction

> **Prerequisites:** None. This is the starting point.

---

## 1.1 What Is AI-Essentials?

**AI-Essentials** is a pure-Python neural network library built entirely from scratch — no PyTorch, TensorFlow, or any other machine-learning framework. Its only dependencies are Python's built-in `math` module (for the core engine), plus `numpy` and `matplotlib` used exclusively in the training demo.

The library demonstrates, step by step, how the core machinery of modern deep learning really works:

1. How numbers flow through a network (the **forward pass**).
2. How gradients flow back through the network (the **backward pass**, a.k.a. backpropagation).
3. How those gradients are used to adjust weights and reduce the loss (**optimization**).

---

## 1.2 Why Build From Scratch?

Most tutorials teach deep learning by calling `model.fit()` in a few lines. That works for building apps, but it hides everything interesting. When you implement a neural network from scratch you are forced to understand:

- What a **gradient** actually is and where it comes from.
- How the **chain rule** makes backpropagation possible.
- Why **weight initialization** matters.
- What "training" really means mathematically.

AI-Essentials was inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) and extends that idea with a more complete feature set: momentum SGD, gradient clipping, binary cross-entropy, He initialization, and a realistic training demo.

---

## 1.3 The Big Picture of Deep Learning

Before diving into code, let's place deep learning in context.

### What problem are we solving?

We have a dataset of input–output pairs `(x, y)` and we want to find a function `f` such that `f(x) ≈ y` for all pairs. For example:

- Predict house prices from features (`x = [size, rooms, ...]`, `y = price`).
- Classify images as cat or dog (`x = pixel values`, `y ∈ {0, 1}`).
- Fit a noisy curve (`x = scalar`, `y = scalar`) — the demo in `main.py`.

### What is a neural network?

A neural network is a parameterized function `f(x; θ)` where `θ` is a large collection of learnable numbers (called **weights** and **biases**). By stacking simple operations (linear combinations + nonlinear activations) in many layers, the network can approximate almost any function — this is the **Universal Approximation Theorem**.

### How does it learn?

1. **Forward pass:** Feed input `x` through the network, compute prediction `ŷ = f(x; θ)`.
2. **Loss computation:** Measure how wrong the prediction is: `L = loss(ŷ, y)`.
3. **Backward pass:** Compute `∂L/∂θ` — how much each parameter contributed to the error.
4. **Parameter update:** Nudge every parameter in the direction that reduces `L`.
5. Repeat thousands of times.

This cycle is called **gradient descent**, and the gradient computation in step 3 is handled automatically by the `Value` class via **automatic differentiation** (the topic of Module 2).

---

## 1.4 Project Structure

```
AI-Essentials/
├── src/
│   ├── main.py                    # Demo: train MLP on noisy sine data
│   └── ai_essentials/
│       ├── __init__.py            # Package version
│       ├── value.py               # Scalar autodiff Value node  ← Module 2
│       ├── activations.py         # tanh / relu registry        ← Module 3
│       ├── neuron.py              # Single perceptron            ← Module 4
│       ├── layer.py               # Layer of neurons             ← Module 4
│       ├── mlp.py                 # Multi-layer perceptron       ← Module 5
│       ├── loss.py                # MSE and binary cross-entropy ← Module 6
│       ├── optimizer.py           # SGD with momentum            ← Module 7
│       └── math.py                # dot, matmul, shape utils     ← Module 8
├── tests/
│   ├── test_value.py
│   ├── test_neuron.py
│   ├── test_layer.py
│   ├── test_mlp.py
│   ├── test_loss.py
│   ├── test_optimizer.py
│   ├── test_math.py
│   └── test_main.py
├── pyproject.toml
└── README.md
```

---

## 1.5 Installation

### Requirements

- Python **3.10+**
- `matplotlib` and `numpy` (for `main.py` only)
- `pytest` and `pytest-cov` (for the test suite)

### Steps

```bash
# Clone the repository
git clone https://github.com/RealYonoveleta/AI-Essentials.git
cd AI-Essentials

# Install the package in editable mode with all dev dependencies
pip install -e ".[dev]"
```

After installation every module is importable:

```python
from ai_essentials.value import Value
from ai_essentials.mlp import MLP
from ai_essentials.optimizer import SGD
```

### Running the Demo

```bash
# Launch the live training visualization
python src/main.py
# or via the CLI entry point installed by pyproject.toml:
ai-essentials
```

### Running the Tests

```bash
pytest                                          # run everything
pytest tests/test_value.py -v                  # one file, verbose
pytest --cov=ai_essentials --cov-report=term-missing  # with coverage
```

---

## 1.6 What You Will Build Understanding Of

By the end of this course you will be able to:

- Explain what automatic differentiation is and implement it yourself.
- Build a multi-layer perceptron from individual neurons.
- Train a network to fit a nonlinear function.
- Understand why momentum, He initialization, and data normalization matter.
- Extend the library with new activations, losses, and optimizers.

---

*← [Home](Home.md) | Next → [Module 2: Scalar Automatic Differentiation](02-Scalar-Autodiff.md)*

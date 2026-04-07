# 🧠 AI-Essentials

> A pure-Python, from-scratch neural network library featuring scalar automatic differentiation, a fully composable MLP, and an SGD optimizer with momentum — no external ML frameworks required.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Architecture & Components](#-architecture--components)
- [Theory & Design Principles](#-theory--design-principles)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [Usage](#-usage)
  - [Training a Model](#training-a-model)
  - [Building a Custom Network](#building-a-custom-network)
  - [Using Loss Functions](#using-loss-functions)
  - [Using the Optimizer](#using-the-optimizer)
- [Configuration & Parameters](#-configuration--parameters)
- [Running the Test Suite](#-running-the-test-suite)
- [Known Limitations & Tips](#-known-limitations--tips)

---

## 🌟 Project Overview

**AI-Essentials** is an educational, fully self-contained neural network engine written in pure Python. It is inspired by projects like [micrograd](https://github.com/karpathy/micrograd) and demonstrates, from first principles, how modern deep learning libraries work under the hood.

**What makes it unique:**

- 🔬 **Zero ML dependencies** — no PyTorch, TensorFlow, or JAX. Just Python's `math` module.
- 🌊 **Scalar reverse-mode autodiff** — every operation on a `Value` node records its own backward function, enabling exact gradient computation via backpropagation.
- 🧱 **Composable architecture** — `Neuron → Layer → MLP` are independently usable and fully interoperable.
- 🚀 **Ready-to-run demo** — `main.py` trains a 3-hidden-layer MLP to fit a noisy `sin(3x) + 0.3x² − 0.5x` curve with live matplotlib visualization.
- ✅ **Comprehensive test suite** — over 60 unit tests covering every module.

---

## 🏗 Architecture & Components

```
Value  ──►  Neuron  ──►  Layer  ──►  MLP
              ▲
         Activations
              │
         Math (dot, matmul)

Loss (MSE, BCE)  ──►  Optimizer (SGD)
```

### `Value` — Scalar Autodiff Node (`value.py`)

The foundation of the entire library. A `Value` wraps a single floating-point number and tracks the computational graph needed for backpropagation.

| Feature | Details |
|---|---|
| Supported ops | `+`, `-`, `*`, `/`, `**`, `tanh`, `relu`, `log` |
| Backward pass | Iterative topological sort (avoids Python recursion limit) |
| Gradient accumulation | Gradients accumulate correctly for shared nodes |

```python
from ai_essentials.value import Value

a = Value(2.0)
b = Value(3.0)
c = a * b + b        # builds the computation graph
c.backward()         # reverse-mode autodiff
print(a.grad)        # 3.0  (∂c/∂a = b)
print(b.grad)        # 3.0  (∂c/∂b = a + 1)
```

---

### `Neuron` — Single Perceptron (`neuron.py`)

A single neuron with learnable weights and bias. Weights are initialized using **He initialization** (`scale = sqrt(2 / n_inputs)`) for well-conditioned gradients.

```python
from ai_essentials.neuron import Neuron

n = Neuron(num_inputs=3, activation='tanh')
output = n([1.0, 2.0, -1.0])   # returns a Value
params = n.parameters()         # list of Value objects (weights + bias)
```

---

### `Layer` — Collection of Neurons (`layer.py`)

A layer groups `num_neurons` neurons, each processing the same input vector and producing one output per neuron.

```python
from ai_essentials.layer import Layer

layer = Layer(num_inputs=4, num_neurons=8, activation='relu')
outputs = layer([1.0, 2.0, 3.0, 4.0])   # returns list of 8 Values
```

---

### `MLP` — Multi-Layer Perceptron (`mlp.py`)

Chains multiple `Layer` objects sequentially. Each layer specification is a `(num_neurons, activation)` tuple.

```python
from ai_essentials.mlp import MLP

model = MLP(num_inputs=2, layers=[
    (16, 'tanh'),   # hidden layer 1: 16 neurons, tanh activation
    (16, 'relu'),   # hidden layer 2: 16 neurons, relu activation
    (1,  None),     # output layer:   1 neuron,  linear (no activation)
])

output = model([0.5, -1.2])   # returns a list of Value objects
```

---

### `Activations` — Activation Functions (`activations.py`)

| Name | Formula | Default Bias Init | Best For |
|---|---|---|---|
| `'tanh'` | `tanh(z)` | `0.0` | Smooth gradients, bounded output `(-1, 1)` |
| `'relu'` | `max(0, z)` | `0.01` | Sparse activations, deep networks |
| `None` | identity | `0.0` | Linear output layer |

---

### `Loss` — Loss Functions (`loss.py`)

| Function | Use Case | Formula |
|---|---|---|
| `mse_loss(predictions, targets)` | Regression | `mean((ŷ - y)²)` |
| `cross_entropy(predictions, targets)` | Binary classification | `-mean(y·log(ŷ) + (1−y)·log(1−ŷ))` |

```python
from ai_essentials.loss import mse_loss, cross_entropy
from ai_essentials.value import Value

preds   = [Value(0.9), Value(0.1)]
targets = [Value(1.0), Value(0.0)]

loss = cross_entropy(preds, targets)
loss.backward()
```

---

### `Optimizer` — SGD with Momentum (`optimizer.py`)

| Method | Description |
|---|---|
| `step()` | Apply gradient update: `v = momentum·v + grad`, `p -= lr·v` |
| `zero_grad()` | Reset all parameter gradients to `0.0` |
| `clip_grad(max_norm)` | L2 gradient clipping to prevent exploding gradients |

```python
from ai_essentials.optimizer import SGD

optimizer = SGD(model.parameters(), lr=0.05, momentum=0.9)

# Inside training loop:
loss.backward()
optimizer.clip_grad(5.0)   # optional: clip gradients
optimizer.step()
optimizer.zero_grad()
```

---

### `Math` Utilities (`math.py`)

| Function | Description |
|---|---|
| `dot(a, b)` | Dot product of two lists (supports `Value` elements) |
| `matmul(a, b)` | Matrix multiplication of 2D lists |
| `get_shape(a)` | Returns shape tuple of a nested list |
| `get_row(a, i)` | Returns the i-th row of a 2D list |
| `get_column(a, j)` | Returns the j-th column of a 2D list |

---

## 🔬 Theory & Design Principles

### Reverse-Mode Automatic Differentiation (Autodiff)

Every `Value` stores:
- `data` — the scalar forward value
- `grad` — the accumulated gradient (∂L/∂this)
- `_backward` — a closure that propagates gradients to children
- `_prev` — the set of input `Value` nodes

Calling `loss.backward()` triggers an **iterative post-order traversal** of the computation graph (from output to inputs), invoking each node's `_backward()` in reverse topological order. This is mathematically equivalent to the chain rule applied recursively.

### He Initialization

Neuron weights are sampled from `Uniform(-s, s)` where `s = sqrt(2 / num_inputs)`. This keeps the variance of activations constant across layers, preventing vanishing/exploding gradients — especially important with ReLU activations.

### Normalization in the Training Demo

`main.py` normalizes both inputs and targets to zero mean and unit variance before training. This keeps pre-activations in the linear region of `tanh`, enabling faster convergence.

### SGD with Momentum

Standard SGD can oscillate. Momentum accumulates a velocity vector in the gradient direction: `v = β·v + g`, `θ -= α·v`. This dampens oscillations and accelerates convergence along consistent gradient directions. The demo uses `momentum=0.9`.

### Learning Rate Step Decay

The demo halves the learning rate at epochs 400 and 700 to allow fine-tuning once the coarse shape of the curve is learned.

### Mini-Batch Training

Training uses mini-batches of 16 samples, shuffled every epoch. This introduces enough stochasticity to escape local minima while keeping gradient estimates stable.

---

## 📁 Project Structure

```
AI-Essentials/
├── src/
│   ├── main.py                    # Demo: train MLP on noisy sine data
│   └── ai_essentials/
│       ├── __init__.py            # Package version
│       ├── value.py               # Scalar autodiff Value node
│       ├── neuron.py              # Single perceptron
│       ├── layer.py               # Layer of neurons
│       ├── mlp.py                 # Multi-layer perceptron
│       ├── activations.py         # tanh / relu activation registry
│       ├── loss.py                # MSE and binary cross-entropy losses
│       ├── optimizer.py           # SGD with momentum + gradient clipping
│       └── math.py                # dot, matmul, shape utilities
├── tests/
│   ├── __init__.py
│   ├── test_value.py              # Autodiff correctness tests
│   ├── test_neuron.py             # Neuron forward/backward tests
│   ├── test_layer.py              # Layer structure/output tests
│   ├── test_mlp.py                # MLP structure, forward, backward tests
│   ├── test_loss.py               # MSE and cross-entropy tests
│   ├── test_optimizer.py          # SGD step/zero_grad tests
│   ├── test_math.py               # dot/matmul/shape utility tests
│   └── test_main.py               # Integration smoke tests
├── pyproject.toml
└── README.md
```

---

## ⚙️ Setup & Installation

### Requirements

- Python **3.10+**
- `matplotlib` (for the live training visualization in `main.py`)
- `numpy` (for data generation in `main.py`)
- `pytest` + `pytest-cov` (for running tests)

### Install

```bash
# Clone the repository
git clone https://github.com/RealYonoveleta/AI-Essentials.git
cd AI-Essentials

# Install the package and all dev dependencies
pip install -e ".[dev]"
```

This installs `ai-essentials` as an editable package, making all imports (`from ai_essentials.mlp import MLP`, etc.) available immediately.

---

## 🚀 Usage

### Training a Model

Run the built-in demo that trains a 3-hidden-layer MLP on noisy `sin(3x) + 0.3x² − 0.5x` data with live matplotlib visualization:

```bash
# Via the installed CLI entry point
ai-essentials

# Or directly
python src/main.py
```

You will see a live-updating plot of the model's predictions vs. the true curve, updating every 25 epochs. Training runs for 1,000 epochs with mini-batch SGD.

---

### Building a Custom Network

```python
from ai_essentials.mlp import MLP
from ai_essentials.optimizer import SGD
from ai_essentials.loss import mse_loss

# Define a 2-input → 32 tanh → 32 tanh → 1 linear network
model = MLP(num_inputs=2, layers=[
    (32, 'tanh'),
    (32, 'tanh'),
    (1,  None),
])

# Inspect parameters
print(f"Total parameters: {len(model.parameters())}")
# Layer 1: 32 * (2 weights + 1 bias) = 96
# Layer 2: 32 * (32 weights + 1 bias) = 1056
# Layer 3:  1 * (32 weights + 1 bias) = 33
# Total: 1185
```

---

### Running a Training Loop

```python
from ai_essentials.mlp import MLP
from ai_essentials.optimizer import SGD
from ai_essentials.loss import mse_loss

model = MLP(num_inputs=1, layers=[(16, 'tanh'), (16, 'tanh'), (1, None)])
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

# Example data: fit y = x^2 on [-1, 1]
X = [[x / 10.0] for x in range(-10, 11)]
y = [[(x[0] ** 2)] for x in X]
y_flat = [yi[0] for yi in y]

for epoch in range(200):
    preds = [model(xi)[0] for xi in X]
    loss = mse_loss(preds, y_flat)

    loss.backward()
    optimizer.clip_grad(5.0)
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.data:.6f}")
```

---

### Using Loss Functions

```python
from ai_essentials.loss import mse_loss, cross_entropy
from ai_essentials.value import Value

# Mean Squared Error (regression)
preds   = [Value(0.8), Value(0.2), Value(0.9)]
targets = [1.0,        0.0,        1.0]
loss = mse_loss(preds, targets)

# Binary Cross-Entropy (binary classification)
probs   = [Value(0.9), Value(0.1)]
labels  = [1.0,        0.0]
bce = cross_entropy(probs, labels)
bce.backward()
```

---

### Using the Optimizer

```python
from ai_essentials.optimizer import SGD
from ai_essentials.value import Value

params = model.parameters()
optimizer = SGD(params, lr=0.05, momentum=0.9)

# Standard training step
loss.backward()              # compute gradients
optimizer.clip_grad(5.0)     # optional: prevent exploding gradients
optimizer.step()             # apply parameter update
optimizer.zero_grad()        # clear gradients for the next step

# Adjust learning rate manually (step decay)
optimizer.lr *= 0.3
```

---

## 🔧 Configuration & Parameters

### MLP Layer Specification

Each layer in the `layers` list is a `(num_neurons, activation)` tuple:

| Parameter | Type | Description |
|---|---|---|
| `num_neurons` | `int` | Number of neurons in the layer |
| `activation` | `'tanh'`, `'relu'`, or `None` | Activation function; `None` = linear |

### SGD Optimizer

| Parameter | Default | Description |
|---|---|---|
| `lr` | `0.01` | Learning rate |
| `momentum` | `0.0` | Momentum coefficient (0 = vanilla SGD, 0.9 = heavy ball) |

### Gradient Clipping

```python
optimizer.clip_grad(max_norm=5.0)   # clips the global L2 gradient norm
```

Recommended when training on noisy or high-frequency targets.

### Demo `main.py` Key Settings

| Setting | Value | Rationale |
|---|---|---|
| Architecture | `[64, 64, 64, 1]` (tanh hidden, linear output) | Enough basis functions for multi-frequency signals |
| Learning rate | `0.05 → 0.015 → 0.0045` | Step decay at epochs 400, 700 |
| Momentum | `0.9` | Smooth progress over oscillatory loss landscape |
| Batch size | `16` | Covers ~1–2 oscillation periods locally |
| Gradient clip | `5.0` | Prevents exploding gradients on sharp sine peaks |
| Epochs | `1,000` | Sufficient for near-zero MSE on normalized targets |

---

## 🧪 Running the Test Suite

```bash
# Run all tests with verbose output
pytest

# Run a specific test file
pytest tests/test_value.py -v

# Run with coverage report
pytest --cov=ai_essentials --cov-report=term-missing
```

### Test Coverage Overview

| Module | Test File | What's Tested |
|---|---|---|
| `value.py` | `test_value.py` | Add, mul, pow, tanh, relu, log — forward & backward; grad accumulation; topological order |
| `neuron.py` | `test_neuron.py` | Initialization, He scaling, activation selection, parameter count |
| `layer.py` | `test_layer.py` | Output shape, neuron count, parameter collection |
| `mlp.py` | `test_mlp.py` | Layer count, forward shape, grad population, bounded tanh outputs |
| `loss.py` | `test_loss.py` | MSE correctness, BCE correctness, empty/mismatched input errors |
| `optimizer.py` | `test_optimizer.py` | SGD step, zero_grad, step+zero_grad together |
| `math.py` | `test_math.py` | dot product, matmul, shape, row/column extraction |
| `main.py` | `test_main.py` | Integration smoke tests |

---

## ⚠️ Known Limitations, Tips & Best Practices

### Limitations

- **Scalar autodiff only** — the `Value` class tracks individual scalars, not tensors. Training is CPU-bound and slower than vectorized libraries for large models.
- **No batched operations** — the forward pass iterates over samples in Python loops. For datasets with > 10 000 samples, training will be noticeably slow.
- **Limited activation registry** — only `tanh` and `relu` are built in. Adding new activations requires editing `activations.py`.

### Tips

- **Normalize your data.** Zero-mean, unit-variance inputs keep `tanh` pre-activations in the linear region and prevent saturation.
- **Use gradient clipping** (`optimizer.clip_grad(5.0)`) when fitting high-frequency or noisy targets to avoid exploding gradients.
- **Start with `tanh` hidden layers** for smooth, bounded activations. Use `None` (linear) for the output layer in regression tasks.
- **Widen, don't just deepen.** For tasks with multiple frequency components (e.g., sine waves), wider layers provide more basis functions. The demo uses 64 neurons per layer.
- **Step-decay the learning rate** after the model has learned the coarse shape of the target function. This allows fine-tuning without oscillating around the minimum.

### Extending the Library

- **New activation:** Add an entry to `ACTIVATIONS` in `activations.py` with a `(function, default_bias)` tuple.
- **New loss:** Add a function to `loss.py` following the `mse_loss` pattern — accept lists of `Value` objects/scalars and return a single `Value`.
- **New optimizer:** Subclass `Optimizer` in `optimizer.py` and implement `step()`.
- **New layer type:** Create a class with `__call__(self, x)` and `parameters(self)` methods; it will be compatible with `MLP` when passed as `neuron_class`.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
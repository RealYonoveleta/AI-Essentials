# Module 10 — Advanced Topics

> **Prerequisites:** All previous modules. This module deepens your understanding and shows how to extend the library.

---

## 10.1 He Initialization — Deep Dive

We briefly introduced He initialization in Module 4. Here we explain it more rigorously.

### The Problem: Variance Explosion / Vanishing

Consider a layer with `n` inputs `x₁, ..., xₙ` and weights `w₁, ..., wₙ` drawn i.i.d. from some distribution with variance `σ²`. The pre-activation is:

```
z = Σᵢ wᵢxᵢ
```

Assuming inputs and weights are independent and zero-mean:

```
Var(z) = n · Var(wᵢ) · Var(xᵢ)
```

If `Var(xᵢ) = 1` (which normalization ensures for the first layer), then `Var(z) = n · σ²`.

- If `σ²` is too small: `Var(z) → 0` as we go deeper → activations die → vanishing gradients.
- If `σ²` is too large: `Var(z) → ∞` as we go deeper → activations explode → exploding gradients.

### The He Solution (for ReLU)

He et al. (2015) showed that for ReLU (which zeroes out half its inputs on average), the correct initialization is:

```
σ² = 2 / n   →   σ = sqrt(2 / n)
```

This keeps `Var(z) ≈ 1` at every layer, regardless of network depth or width.

### Implementation in AI-Essentials

```python
# src/ai_essentials/neuron.py
scale = (2.0 / num_inputs) ** 0.5 if num_inputs > 0 else 1.0
self.w = [Value(random.uniform(-scale, scale)) for _ in range(num_inputs)]
```

The weights are drawn from `Uniform(-scale, scale)` where `scale = sqrt(2/n)`. For a uniform distribution on `[-s, s]`, the variance is `s²/3`, not `s²`. However, in practice this constant factor is absorbed during early training and He initialization still performs well.

The `if num_inputs > 0 else 1.0` guard prevents division by zero for edge-case neurons with no inputs.

---

## 10.2 Data Normalization — Why and How

### Z-Score Normalization (Standard Scaling)

```python
x_mean, x_std = x_raw.mean(), x_raw.std()
x_norm = (x_raw - x_mean) / x_std
```

After this transformation:
- Mean of `x_norm` ≈ 0
- Standard deviation of `x_norm` ≈ 1

**What this does for training:**

1. **Prevents tanh saturation:** For tanh, the useful gradient range is roughly `|z| < 2`. With normalized inputs, initial pre-activations are in this range.
2. **Equalizes feature scales:** If inputs had very different magnitudes, weights on large-magnitude features would dominate. Normalization puts all features on equal footing.
3. **Stabilizes gradient magnitudes:** Extreme input values cause extreme gradients which cause extreme parameter updates.

### Target Normalization

The same logic applies to the output. If targets are in `[−100, 100]` but the network output starts near 0, the initial loss is enormous. After normalization, targets are in approximately `[−3, 3]`, close to the network's initial output range.

After training, predictions are **denormalized**:
```python
y_original = y_normalized * y_std + y_mean
```

### A Note on Normalization vs. Batch Normalization

The normalization in `main.py` is **dataset-level normalization** — it is computed once from the training data before training begins. This is different from **batch normalization** (a technique where each layer normalizes its outputs per mini-batch during training). AI-Essentials does not implement batch normalization, but it could be added as a new layer type.

---

## 10.3 The Iterative Topological Sort — Why and How

The `backward()` method in `Value` uses an **iterative** post-order traversal rather than a recursive one. Let's understand why.

### The Naive Recursive Approach (Don't Do This)

```python
def backward_naive(self):
    self.grad = 1.0
    def build_topo(v, topo, visited):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child, topo, visited)
            topo.append(v)
    topo = []
    build_topo(self, topo, set())
    for v in reversed(topo):
        v._backward()
```

This works for small graphs but **hits Python's recursion limit** for deep networks. Python's default recursion depth is 1000. A 3-layer MLP with 64 neurons and a 200-sample batch can easily have thousands of chained operations.

### The Iterative Approach

```python
def backward(self):
    topo = []
    visited = set()
    stack = [self]
    while stack:
        v = stack[-1]
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                if child not in visited:
                    stack.append(child)
        else:
            stack.pop()
            topo.append(v)
    self.grad = 1.0
    for v in reversed(topo):
        v._backward()
```

This simulates the call stack explicitly using a Python `list` (as a stack), which can grow to any size without hitting Python's recursion limit. The algorithm:

1. Push the root node onto the stack.
2. If the top node is **not yet visited**: mark it visited and push its unvisited children.
3. If the top node **is already visited** (all children processed): pop it and append to `topo`.

After the loop, `topo` is in post-order — every child appears before its parent. Reversing it gives us the order we need: process the root first, leaves last.

---

## 10.4 Shared Nodes and Gradient Accumulation

When a `Value` is used in multiple operations, gradients from all downstream paths must be summed. This is the **multivariate chain rule**:

```
∂L/∂x = Σₖ (∂L/∂yₖ) · (∂yₖ/∂x)
```

where the sum is over all downstream nodes `yₖ` that depend on `x`.

This is why `_backward` closures always use `+=`:

```python
# correct: accumulate
self.grad += other.data * out.grad

# wrong: would overwrite contributions from other paths
self.grad = other.data * out.grad
```

And it's why `zero_grad()` resets gradients before each step — to clear the accumulated values from the previous batch.

---

## 10.5 Extending the Library

### Adding a New Activation Function

1. Add a method to `Value` in `value.py`:

```python
def sigmoid(self):
    import math
    s = 1.0 / (1.0 + math.exp(-self.data))
    out = Value(s, (self,), "sigmoid")
    def _backward():
        self.grad += s * (1.0 - s) * out.grad
    out._backward = _backward
    return out
```

2. Register it in `activations.py`:

```python
def sigmoid(x):
    return x.sigmoid()

ACTIVATIONS['sigmoid'] = (sigmoid, 0.0)
```

3. Now you can use it anywhere:

```python
n = Neuron(num_inputs=3, activation='sigmoid')
layer = Layer(num_inputs=4, num_neurons=8, activation='sigmoid')
model = MLP(2, [(16, 'sigmoid'), (1, None)])
```

### Adding a New Loss Function

Follow the pattern in `loss.py`:

```python
def hinge_loss(predictions, targets):
    """
    Hinge loss for SVM-style binary classification.
    targets should be +1 or -1.
    """
    check_loss_inputs(predictions, targets)
    predictions = to_values(predictions)
    targets     = to_values(targets)

    # hinge: max(0, 1 - y*p)
    total = Value(0.0)
    for p, t in zip(predictions, targets):
        margin = Value(1.0) - t * p
        # max(0, margin) using relu
        total = total + margin.relu()
    return total * (1 / len(predictions))
```

### Adding a New Optimizer

Subclass `Optimizer` and implement `step()`:

```python
from ai_essentials.optimizer import Optimizer

class Adam(Optimizer):
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(parameters, lr)
        self.beta1  = beta1
        self.beta2  = beta2
        self.eps    = eps
        self.m      = [0.0] * len(self.parameters)   # first moment
        self.v      = [0.0] * len(self.parameters)   # second moment
        self.t      = 0                               # time step

    def step(self):
        self.t += 1
        for i, p in enumerate(self.parameters):
            g = p.grad
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g * g
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p.data -= self.lr * m_hat / (v_hat ** 0.5 + self.eps)
```

### Adding a New Layer Type

Any class with `__call__(self, x)` and `parameters(self)` methods can be used as a layer building block:

```python
class DropoutLayer:
    """Randomly zeros activations during training."""
    def __init__(self, p=0.5):
        self.p = p
        self.training = True

    def __call__(self, x):
        import random
        if not self.training:
            return x
        return [v * (0.0 if random.random() < self.p else 1.0 / (1.0 - self.p))
                for v in x]

    def parameters(self):
        return []   # no learnable parameters
```

---

## 10.6 Performance and Limitations

### Scalar Autodiff Is Slow

Every number in AI-Essentials is a separate Python object. A mini-batch of 16 samples through an 8513-parameter network creates tens of thousands of `Value` objects per step, plus the associated closure objects for backward. This has significant overhead compared to vectorized libraries like PyTorch that process entire batches as tensor operations in C++/CUDA.

**Practical limits:**
- Models with up to ~10,000 parameters train comfortably on CPU.
- Datasets with up to ~10,000 samples work well with mini-batches.
- Beyond these sizes, consider a vectorized framework.

### No GPU Support

The library runs on CPU only. Adding GPU support would require replacing scalar `Value` operations with tensor operations on CUDA arrays — essentially reimplementing it as a tensor library, which is a fundamentally different design.

### Limited Activation Registry

Only `tanh` and `relu` are built in. Adding more requires editing `activations.py` (see Section 10.5 above).

---

## 10.7 Testing Philosophy

The library has over 60 unit tests covering every module. The test suite follows these principles:

1. **Structure tests**: verify that components are initialized correctly (e.g., parameter counts).
2. **Forward tests**: verify that the output has the right type, shape, and value range.
3. **Backward tests**: verify that gradients are non-zero and flow to all parameters.
4. **Edge case tests**: verify that errors are raised for invalid inputs.

### Running the Tests

```bash
pytest                                               # all tests
pytest tests/test_value.py -v                       # one file, verbose
pytest --cov=ai_essentials --cov-report=term-missing # with coverage
```

### Test Coverage Overview

| Module | Test File | Key Tests |
|--------|-----------|-----------|
| `value.py` | `test_value.py` | All ops forward & backward, grad accumulation, topological sort |
| `neuron.py` | `test_neuron.py` | He scaling, activation selection, parameter count |
| `layer.py` | `test_layer.py` | Output shape, neuron count, parameter collection |
| `mlp.py` | `test_mlp.py` | Layer count, forward shape, grad population, tanh bounds |
| `loss.py` | `test_loss.py` | MSE correctness, BCE correctness, empty/mismatched errors |
| `optimizer.py` | `test_optimizer.py` | SGD step, zero_grad, momentum |
| `math.py` | `test_math.py` | dot, matmul, shape, row/column |
| `main.py` | `test_main.py` | Integration smoke tests |

---

## 10.8 Glossary

| Term | Definition |
|------|-----------|
| **Autodiff** | Automatic differentiation: computing derivatives of functions expressed as programs |
| **Backpropagation** | The algorithm that applies the chain rule to a computation graph to compute gradients |
| **Batch normalization** | Normalizing layer inputs per mini-batch during training |
| **Bias** | A learnable offset added to the weighted sum in a neuron |
| **Chain rule** | `∂f/∂x = (∂f/∂g) · (∂g/∂x)` — the foundation of backpropagation |
| **Computation graph** | A DAG where nodes are values and edges are operations |
| **Epoch** | One complete pass through the training dataset |
| **Gradient** | A vector of partial derivatives indicating the direction of steepest increase in the loss |
| **Gradient clipping** | Rescaling gradients so their global L2 norm does not exceed a threshold |
| **He initialization** | Weight initialization with `σ = sqrt(2/n)`, designed for ReLU |
| **Learning rate** | The step size `α` in `θ ← θ − α · ∇L` |
| **Mini-batch** | A small random subset of the training data used for one gradient step |
| **Momentum** | An optimization technique that accumulates velocity across gradient steps |
| **Neuron** | A computational unit that computes `activation(w · x + b)` |
| **Overfitting** | When a model memorizes the training data and fails to generalize |
| **Pre-activation** | The value `z = w · x + b` before the activation function is applied |
| **ReLU** | Rectified Linear Unit: `max(0, z)` |
| **Reverse-mode autodiff** | Computing gradients by traversing the computation graph from output to input |
| **Saturation** | When an activation function's gradient approaches zero for large input magnitudes |
| **SGD** | Stochastic Gradient Descent: gradient descent with mini-batches |
| **Tanh** | Hyperbolic tangent: `(eᶻ − e⁻ᶻ) / (eᶻ + e⁻ᶻ)`, output in `(−1, 1)` |
| **Topological sort** | Ordering nodes of a DAG such that every node comes before its dependents |
| **Universal Approximation** | Theorem stating that a sufficiently wide neural network can approximate any continuous function |
| **Vanishing gradient** | When gradients become extremely small deep in a network, slowing or stopping learning |
| **Weight** | A learnable scalar multiplied by an input in a neuron |
| **Zero-centered** | An activation whose outputs are symmetric around 0 (e.g., tanh) |

---

## 10.9 Further Reading

- [Andrej Karpathy — micrograd](https://github.com/karpathy/micrograd) — the direct inspiration for this library
- [Andrej Karpathy — Neural Networks: Zero to Hero (YouTube)](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) — lectures that explain autodiff from scratch
- [Deep Learning (Goodfellow, Bengio, Courville)](https://www.deeplearningbook.org/) — comprehensive textbook, freely available online
- [He et al. (2015) — Delving Deep into Rectifiers](https://arxiv.org/abs/1502.01852) — the paper introducing He initialization
- [Rumelhart et al. (1986) — Learning representations by back-propagating errors](https://www.nature.com/articles/323533a0) — the original backpropagation paper

---

*← [Module 9: The Training Loop](09-Training-Loop.md) | [Home](Home.md)*

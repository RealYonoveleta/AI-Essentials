# Module 7 — Optimization

> **Prerequisites:** Modules 1–6. You should understand gradients, the `Value` class, and loss functions.

---

## 7.1 The Goal of Optimization

We have a loss function `L(θ)` where `θ` is the vector of all model parameters. We want to find `θ*` that minimises `L`. Since `L` is a complicated nonlinear function with millions of values, we can't solve for `θ*` analytically. Instead, we use **iterative gradient-based optimization**.

The most basic algorithm is **gradient descent**:

```
θ  ←  θ − α · ∇L(θ)
```

where `α` (the **learning rate**) controls how large a step we take. Too large and we overshoot; too small and training is painfully slow.

---

## 7.2 Problems with Vanilla Gradient Descent

Standard gradient descent has known weaknesses:

1. **Oscillations in ravines:** In a loss landscape shaped like a narrow valley, gradients zigzag across the valley walls instead of flowing smoothly down the valley floor.
2. **Local minima and saddle points:** Gradients are zero at local minima and saddle points, so vanilla GD can get stuck.
3. **Sensitivity to learning rate:** A fixed learning rate that works early in training may be too large (oscillating) or too small (stuck) later.

**Momentum** addresses problem 1; **learning rate scheduling** addresses problem 3.

---

## 7.3 SGD with Momentum — Theory

**Stochastic Gradient Descent (SGD)** processes one sample or mini-batch at a time, introducing noise that can help escape flat regions and saddle points.

**Momentum** augments SGD with a "velocity" term that accumulates gradients across steps, like a ball rolling down a hill that builds up speed:

```
v_t  =  β · v_{t-1} + g_t
θ_t  =  θ_{t-1} − α · v_t
```

where:
- `g_t = ∂L/∂θ` — the gradient at step `t`
- `v_t` — the velocity (exponentially weighted average of past gradients)
- `β` — the momentum coefficient (typically 0.9)
- `α` — the learning rate

**Effect of momentum:**
- Gradients in consistent directions accumulate, accelerating movement in those directions.
- Gradients that oscillate (pointing in opposite directions in alternate steps) cancel out, damping oscillations.
- With `β = 0.9`, the effective step size is `1/(1-0.9) = 10×` larger in the direction of sustained gradient, and oscillations are strongly suppressed.

---

## 7.4 Gradient Clipping — Theory

If the gradient norm becomes very large (e.g., when fitting sharp peaks in a noisy function), a single update step can push parameters far from their current values, potentially ruining previous progress. This is called the **exploding gradient** problem.

**Gradient clipping** rescales the gradient vector so its L2 norm never exceeds a threshold `max_norm`:

```
If ‖g‖₂ > max_norm:
    g  ←  g · (max_norm / ‖g‖₂)
```

This preserves the gradient **direction** while capping its **magnitude**. It is a simple but effective safeguard, especially for noisy or high-frequency targets.

---

## 7.5 The `Optimizer` Base Class

**File:** `src/ai_essentials/optimizer.py`

```python
import math


class Optimizer:
    def __init__(self, parameters, lr=0.01):
        self.parameters = list(parameters)
        self.lr = lr

    def step(self):
        raise NotImplementedError("step() must be implemented by subclasses.")

    def zero_grad(self):
        for p in self.parameters:
            p.grad = 0.0

    def clip_grad(self, max_norm=1.0):
        total_norm = math.sqrt(sum(p.grad**2 for p in self.parameters))
        if total_norm > max_norm:
            scale = max_norm / total_norm
            for p in self.parameters:
                p.grad *= scale
```

### Constructor

The optimizer stores a **Python list** (not a generator!) of parameter `Value` objects. Converting to a list is important — `parameters` might be a one-time generator from `model.parameters()`, so we consume it immediately to ensure we can iterate over it multiple times.

`self.lr` is public and mutable — you can change it mid-training for learning rate scheduling:

```python
optimizer.lr *= 0.3   # step decay
```

### `zero_grad()`

```python
def zero_grad(self):
    for p in self.parameters:
        p.grad = 0.0
```

Resets every parameter's gradient to 0. This must be called after each optimizer step because `Value._backward` uses `+=` — if you don't clear gradients, they accumulate across training steps, which is almost always wrong.

**Typical training loop order:**

```python
loss.backward()       # 1. compute gradients
optimizer.step()      # 2. update parameters
optimizer.zero_grad() # 3. clear gradients for next step
```

### `clip_grad(max_norm)`

```python
def clip_grad(self, max_norm=1.0):
    total_norm = math.sqrt(sum(p.grad**2 for p in self.parameters))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        for p in self.parameters:
            p.grad *= scale
```

Computes the **global L2 gradient norm** across all parameters (not per-parameter clipping). If it exceeds `max_norm`, scales every gradient down proportionally. This call goes between `loss.backward()` and `optimizer.step()`:

```python
loss.backward()
optimizer.clip_grad(5.0)   # <-- here
optimizer.step()
optimizer.zero_grad()
```

### `step()` — Abstract

The base class declares `step()` as abstract (raises `NotImplementedError`). Subclasses must override it.

---

## 7.6 The `SGD` Class

```python
class SGD(Optimizer):
    def __init__(self, parameters, lr=0.01, momentum=0.0):
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.velocity = [0.0] * len(self.parameters)

    def step(self):
        for i, p in enumerate(self.parameters):
            self.velocity[i] = self.momentum * self.velocity[i] + p.grad
            p.data -= self.lr * self.velocity[i]
```

### Constructor

`self.velocity` is a list of floats, one per parameter, initialised to 0. This is the "ball rolling" state that persists across training steps. Note: it stores raw Python floats — not `Value` objects — because velocities are not part of the computation graph (we don't need gradients of velocities).

### `step()`

For each parameter `p` at position `i`:

1. Update velocity: `v_i = β · v_i + p.grad`
2. Update parameter: `p.data -= lr · v_i`

Note the direct modification of `p.data` — bypassing the `Value` computation graph. This is intentional: the optimizer update is not a differentiable operation we want to track. We only need the gradient for the current step.

With `momentum = 0.0`, `velocity[i]` is always just `p.grad` and the update reduces to vanilla SGD:

```
p.data -= lr * p.grad
```

### Usage Example

```python
from ai_essentials.mlp import MLP
from ai_essentials.optimizer import SGD
from ai_essentials.loss import mse_loss

model = MLP(1, [(16, 'tanh'), (1, None)])
optimizer = SGD(model.parameters(), lr=0.05, momentum=0.9)

X = [[float(i) / 10] for i in range(-10, 11)]
y = [xi[0] ** 2 for xi in X]

for epoch in range(300):
    preds = [model(xi)[0] for xi in X]
    loss  = mse_loss(preds, y)

    loss.backward()
    optimizer.clip_grad(5.0)
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 100 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.data:.6f}")
```

---

## 7.7 Learning Rate Scheduling

The library does not have a built-in LR scheduler, but since `optimizer.lr` is a public attribute you can implement step decay inline:

```python
for epoch in range(1000):
    # Step decay
    if epoch == 400:
        optimizer.lr *= 0.3
    if epoch == 700:
        optimizer.lr *= 0.3

    # ... training step ...
```

This is exactly what `main.py` does. The rationale: a high LR early in training explores the loss landscape quickly; reducing it later allows fine-tuning once the coarse shape is learned.

---

## 7.8 Extending: Adding a New Optimizer

To add Adam, RMSProp, or any other optimizer, subclass `Optimizer` and implement `step()`:

```python
class RMSProp(Optimizer):
    def __init__(self, parameters, lr=0.001, alpha=0.99, eps=1e-8):
        super().__init__(parameters, lr)
        self.alpha = alpha
        self.eps   = eps
        self.sq    = [0.0] * len(self.parameters)

    def step(self):
        for i, p in enumerate(self.parameters):
            self.sq[i] = self.alpha * self.sq[i] + (1 - self.alpha) * p.grad**2
            p.data -= self.lr * p.grad / (self.sq[i]**0.5 + self.eps)
```

The rest of the library (loss, model, gradient clipping, zero_grad) works without any changes because the optimizer contract is just: call `step()` to update `p.data`, call `zero_grad()` to clear `p.grad`.

---

## 7.9 Key Takeaways

- The `Optimizer` base class manages parameter lists, `zero_grad()`, and `clip_grad()`.
- `SGD` adds a velocity vector that accumulates gradients across steps — this is momentum.
- Gradient clipping rescales the global gradient norm to prevent exploding gradients.
- Learning rate scheduling is done by manually updating `optimizer.lr`.
- New optimizers can be added by subclassing `Optimizer` and implementing `step()`.

---

*← [Module 6: Loss Functions](06-Loss-Functions.md) | Next → [Module 8: Math Utilities](08-Math-Utilities.md)*

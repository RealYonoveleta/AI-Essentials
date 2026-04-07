# Module 9 — The Training Loop

> **Prerequisites:** All previous modules. This module ties everything together.

---

## 9.1 The Complete Picture

At this point you understand every individual component:

| Component | Module |
|-----------|--------|
| `Value` — differentiable scalar | [Module 2](02-Scalar-Autodiff.md) |
| Activation functions | [Module 3](03-Activation-Functions.md) |
| `Neuron`, `Layer` | [Module 4](04-Neural-Network-Building-Blocks.md) |
| `MLP` | [Module 5](05-MLP.md) |
| Loss functions | [Module 6](06-Loss-Functions.md) |
| `SGD` optimizer | [Module 7](07-Optimization.md) |

Now let's see how they connect in the complete training loop — the code in `src/main.py`.

---

## 9.2 The Problem: Fitting a Noisy Nonlinear Function

`main.py` trains a neural network to fit the function:

```
y = sin(3x) + 0.3x² − 0.5x + ε
```

where `ε ~ N(0, 0.2)` is Gaussian noise, sampled at 200 points `x ∈ [−5, 5]`.

This is a challenging regression task because the target has:
- **High-frequency oscillation** (from `sin(3x)`)
- **A polynomial trend** (from `0.3x² − 0.5x`)
- **Noise** that the model must not overfit to

---

## 9.3 Step 1 — Data Generation and Normalization

```python
import numpy as np

np.random.seed(42)
x_raw = np.linspace(-5, 5, 200)
y_raw = (
    np.sin(3 * x_raw)
    + 0.3 * x_raw**2
    - 0.5 * x_raw
    + np.random.normal(0, 0.2, size=x_raw.shape)
)
```

### Input Normalization

```python
x_mean, x_std = x_raw.mean(), x_raw.std()
x_norm = (x_raw - x_mean) / x_std
```

**Why normalize inputs?**

The raw inputs span `[−5, 5]`. The tanh activation saturates when its input is far from 0 — at `z = ±5`, `tanh(5) ≈ 0.9999` and the gradient `1 − tanh²(5) ≈ 0.0001`. The network would receive near-zero gradients from the start.

After normalization, inputs have **zero mean and unit variance**, ensuring pre-activations start in tanh's linear region `[−1, 1]` where gradients are largest.

### Target Normalization

```python
y_mean, y_std = y_raw.mean(), y_raw.std()
y_norm = (y_raw - y_mean) / y_std
```

**Why normalize targets?**

- The raw outputs span roughly `[−6, 6]`. A linear output neuron starts with weights near 0, so its initial predictions are near 0 too. Without target normalization, the initial loss would be huge and gradients might explode.
- Normalized targets are approximately in `[−3, 3]`, and the network's output can reach this range early in training.
- After training, predictions are **denormalized**: `y_plot = yp.data * y_std + y_mean` to recover the original scale for plotting.

### Preparing the Data

```python
X = [[float(xi)] for xi in x_norm]
y_targets = [float(yi) for yi in y_norm]
```

The MLP expects inputs as lists of lists (each inner list is one sample). Since this is a 1D regression, each sample is `[scalar]`.

---

## 9.4 Step 2 — Model and Optimizer Setup

```python
from ai_essentials.mlp import MLP
from ai_essentials.optimizer import SGD

model = MLP(
    num_inputs=1,
    layers=[(64, "tanh"), (64, "tanh"), (64, "tanh"), (1, None)]
)
optimizer = SGD(model.parameters(), lr=0.05, momentum=0.9)
```

### Why This Architecture?

| Choice | Reason |
|--------|--------|
| 3 hidden layers of 64 neurons | Enough capacity to represent multi-frequency signals. Each neuron contributes one "basis function". |
| tanh hidden activations | Smooth, zero-centered. Works well with normalized data. |
| Linear output (no activation) | Regression: predict real-valued numbers, not probabilities. |
| lr = 0.05 | Fast initial convergence; decayed later. |
| momentum = 0.9 | Smooth progress through an oscillatory loss landscape. |

**Parameter count:**
```
Layer 1: 64 × (1 + 1) =  128
Layer 2: 64 × (64 + 1) = 4160
Layer 3: 64 × (64 + 1) = 4160
Layer 4:  1 × (64 + 1) =   65
Total:                   8513 parameters
```

---

## 9.5 Step 3 — The Training Loop

```python
import random

batch_size = 16
indices = list(range(len(X)))

plt.ion()
fig, ax = plt.subplots()

for epoch in range(1, 1001):
    # Learning rate step decay
    if epoch == 400:
        optimizer.lr *= 0.3
    if epoch == 700:
        optimizer.lr *= 0.3

    # Shuffle data each epoch (true stochastic mini-batch)
    random.shuffle(indices)

    for start in range(0, len(X), batch_size):
        batch_idx = indices[start : start + batch_size]
        X_batch   = [X[i] for i in batch_idx]
        y_batch   = [y_targets[i] for i in batch_idx]

        # ── Forward pass ──────────────────────────────────────
        y_pred_batch = [model(xi)[0] for xi in X_batch]
        loss = mse_loss(y_pred_batch, y_batch)

        # ── Backward pass ─────────────────────────────────────
        loss.backward()
        optimizer.clip_grad(5.0)

        # ── Optimizer step ────────────────────────────────────
        optimizer.step()
        optimizer.zero_grad()
```

### Anatomy of One Training Step

Let's walk through each line in detail:

#### Forward Pass

```python
y_pred_batch = [model(xi)[0] for xi in X_batch]
loss = mse_loss(y_pred_batch, y_batch)
```

1. For each sample `xi` in the mini-batch, run the forward pass through the MLP. This builds a computation graph in memory.
2. `model(xi)` returns a list of 1 `Value` (the output neuron). We take `[0]` to get the scalar.
3. `mse_loss` computes the average squared error across the mini-batch, producing one `Value` at the graph's root.

#### Backward Pass

```python
loss.backward()
```

Traverses the entire computation graph from `loss` back to every parameter, filling in `.grad` for all 8513 parameters.

```python
optimizer.clip_grad(5.0)
```

If the global gradient norm exceeds 5.0, scale all gradients down. This is important here because the sine function has sharp peaks where gradients can spike.

#### Optimizer Step

```python
optimizer.step()
optimizer.zero_grad()
```

Update every parameter using `v = 0.9·v + grad`, `p -= 0.05·v`. Then reset all gradients to 0 for the next mini-batch.

---

## 9.6 Mini-Batch Training Explained

**Why mini-batches?** There are three strategies:

| Strategy | Batch Size | Gradient estimate | Speed |
|----------|-----------|-------------------|-------|
| Full-batch GD | All samples | Exact | Slow per update |
| Stochastic GD (SGD) | 1 sample | Very noisy | Fast per update |
| Mini-batch SGD | 16–512 | Good balance | Fast |

With batch size 16 and 200 samples:
- Each epoch has `ceil(200/16) = 13` mini-batch updates.
- Each update sees ~1–2 full oscillation periods of the sine function.
- The stochasticity helps escape saddle points and local minima.

**Shuffling each epoch** ensures the network sees samples in a different order every epoch, preventing it from learning patterns tied to sample order.

---

## 9.7 Learning Rate Step Decay

```python
if epoch == 400:
    optimizer.lr *= 0.3    # 0.05 → 0.015
if epoch == 700:
    optimizer.lr *= 0.3    # 0.015 → 0.0045
```

**Rationale:**
- Early training (epochs 1–400): high LR explores the loss landscape quickly, learning the coarse shape.
- Mid training (epochs 400–700): reduced LR allows fine-tuning without overshooting.
- Late training (epochs 700–1000): very low LR polishes the fit without oscillating.

This is a simple but effective schedule. More sophisticated schedules (cosine annealing, warmup) are possible by modifying `optimizer.lr` in the same inline fashion.

---

## 9.8 Visualization

```python
if epoch % 25 == 0 or epoch == 1:
    y_pred_full = [model(xi)[0] for xi in X]
    full_loss = mse_loss(y_pred_full, y_targets)
    ax.clear()
    ax.scatter(x_raw, y_raw, color="blue", s=5, label="True")
    y_plot = [yp.data * y_std + y_mean for yp in y_pred_full]
    ax.plot(x_raw, y_plot, color="red", linewidth=2, label="Predicted")
    ax.set_title(f"Epoch {epoch} | Loss (normalized): {full_loss.data:.4f}")
    ax.legend()
    plt.pause(0.1)
```

Every 25 epochs:
1. Evaluate the model on all 200 samples (full dataset, no randomness).
2. Compute the full-dataset loss for display.
3. Denormalize predictions: `yp.data * y_std + y_mean` converts from normalized space back to the original scale.
4. Plot true data (blue scatter) vs. model predictions (red line).
5. `plt.pause(0.1)` flushes the plot update and pauses 0.1 seconds.

---

## 9.9 Putting It All Together — Pseudocode

```
SETUP:
  generate data (x_raw, y_raw)
  normalize: x_norm = (x - μ_x) / σ_x
             y_norm = (y - μ_y) / σ_y
  build MLP(1, [64 tanh, 64 tanh, 64 tanh, 1 linear])
  build SGD(params, lr=0.05, momentum=0.9)

TRAIN for epoch in 1..1000:
  decay lr at epoch 400 and 700
  shuffle sample indices

  FOR each mini-batch of size 16:
    forward:  preds = [MLP(x)[0] for x in batch]
    loss:     L = MSE(preds, targets)
    backward: L.backward()
    clip:     clip_grad(5.0)
    update:   SGD.step()
    reset:    SGD.zero_grad()

  EVERY 25 epochs:
    evaluate on full dataset
    plot: true data (blue) vs predictions (red, denormalized)

DONE.
```

---

## 9.10 Key Takeaways

- Data normalization is not optional — it prevents saturation and controls the scale of gradients.
- Mini-batch SGD with shuffling balances gradient quality, speed, and escape from local minima.
- The four-step training step — forward, backward, clip, step, zero_grad — is universal and applies to any loss/model combination in the library.
- Learning rate decay allows large initial steps for exploration and small final steps for fine-tuning.
- `main.py` is a complete, self-contained demonstration of every concept in this course.

---

*← [Module 8: Math Utilities](08-Math-Utilities.md) | Next → [Module 10: Advanced Topics](10-Advanced-Topics.md)*

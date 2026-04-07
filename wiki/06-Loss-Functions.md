# Module 6 — Loss Functions

> **Prerequisites:** Modules 1–5. You should understand `Value` and the MLP forward pass.

---

## 6.1 What Is a Loss Function?

A **loss function** (also called a **cost function** or **objective function**) measures how wrong the network's predictions are. It maps a set of predictions `ŷ` and true labels `y` to a single non-negative scalar. The training process minimises this scalar.

A good loss function must:
1. Return `0` when predictions are perfect.
2. Return a larger value the worse the predictions.
3. Be **differentiable** — so gradients can flow back through it to the parameters.

Different tasks call for different loss functions. The library implements two fundamental ones.

---

## 6.2 Mean Squared Error (MSE) — For Regression

### Theory

For a batch of `n` predictions, MSE is:

```
MSE = (1/n) · Σᵢ (ŷᵢ − yᵢ)²
```

**Why squared?** The square ensures:
- Errors are always non-negative.
- Larger errors are penalised more heavily (squaring amplifies them).
- The function is smooth and differentiable everywhere.

**Why mean (÷n)?** Dividing by the number of samples makes the loss independent of batch size, so hyperparameters (like learning rate) don't need to be re-tuned when you change the batch size.

**Gradient with respect to a prediction:**

```
∂MSE/∂ŷᵢ = (2/n) · (ŷᵢ − yᵢ)
```

The gradient is positive if `ŷᵢ > yᵢ` (overestimating → push parameter down) and negative if `ŷᵢ < yᵢ` (underestimating → push parameter up). This is the intuitive "pull toward the target" behaviour.

### Implementation

**File:** `src/ai_essentials/loss.py`

```python
def mse_loss(predictions, targets):
    """
    Mean Squared Error Loss.
    predictions and targets: lists of Value objects or scalars
    """
    check_loss_inputs(predictions, targets)

    predictions = to_values(predictions)
    targets     = to_values(targets)

    return sum(
        ((p - t) ** 2 for p, t in zip(predictions, targets)),
        Value(0.0)
    ) * (1 / len(predictions))
```

**Line by line:**

1. `check_loss_inputs` validates that both lists are non-empty and equal length.
2. `to_values` promotes any raw floats in `targets` to `Value` objects (targets usually don't need gradients, but they must be `Value` to participate in operations with prediction `Value` objects).
3. `(p - t) ** 2` — uses `Value.__sub__` then `Value.__pow__`, both of which register backward closures.
4. `sum(..., Value(0.0))` — Python's built-in `sum` with a `Value(0.0)` start accumulates the squared errors using `Value.__add__`.
5. `* (1 / len(predictions))` — divides by `n` using `Value.__mul__`.

The result is a single `Value` node that sits at the top of the computation graph. Calling `.backward()` on it propagates gradients all the way back to every model parameter.

### Usage Example

```python
from ai_essentials.loss import mse_loss
from ai_essentials.value import Value

preds   = [Value(0.9), Value(0.1), Value(0.8)]
targets = [1.0,        0.0,        1.0]

loss = mse_loss(preds, targets)
print(loss.data)   # (0.01 + 0.01 + 0.04) / 3 ≈ 0.0200

loss.backward()
print(preds[0].grad)   # 2/3 * (0.9 - 1.0) = -0.0667  (push up)
print(preds[1].grad)   # 2/3 * (0.1 - 0.0) =  0.0667  (push down)
```

---

## 6.3 Binary Cross-Entropy (BCE) — For Binary Classification

### Theory

When we frame a problem as binary classification (output ∈ {0, 1}), we want the network to output a **probability** `p = ŷᵢ ∈ (0, 1)` that the example belongs to class 1.

The **Binary Cross-Entropy** loss comes from the negative log-likelihood of a Bernoulli distribution:

```
BCE = -(1/n) · Σᵢ [ yᵢ · log(ŷᵢ) + (1 − yᵢ) · log(1 − ŷᵢ) ]
```

**Intuition:**
- When `yᵢ = 1`: loss = `−log(ŷᵢ)`. This is 0 when `ŷᵢ = 1` and → +∞ as `ŷᵢ → 0`. It strongly penalises confident wrong predictions.
- When `yᵢ = 0`: loss = `−log(1 − ŷᵢ)`. This is 0 when `ŷᵢ = 0` and → +∞ as `ŷᵢ → 1`.

BCE is more appropriate than MSE for classification because:
1. It is derived from the maximum likelihood principle for Bernoulli outcomes.
2. Its gradient does not saturate the way MSE does for predictions near 0 or 1.

**Gradient with respect to a prediction:**

```
∂BCE/∂ŷᵢ = (1/n) · (ŷᵢ − yᵢ) / (ŷᵢ · (1 − ŷᵢ))
```

Note: predictions must be in (0, 1) — typically the output layer uses a sigmoid activation when using BCE. The `log` in `Value` clamps to `eps=1e-12` to avoid `log(0)`.

### Implementation

```python
def cross_entropy(predictions, targets):
    """
    Binary Cross Entropy Loss.
    predictions: list of Value objects or scalars (predicted probabilities in (0, 1))
    targets: list of Value objects or scalars (true labels: 0 or 1)
    """
    check_loss_inputs(predictions, targets)

    predictions = to_values(predictions)
    targets     = to_values(targets)

    loss = sum(
        (t * p.log() + (1 - t) * (1 - p).log()
         for p, t in zip(predictions, targets)),
        Value(0.0),
    )
    return loss * (-1 / len(predictions))
```

**Line by line:**

1. Each term `t * p.log() + (1 - t) * (1 - p).log()` computes the per-sample log-likelihood using `Value.log()`, `Value.__mul__`, and `Value.__add__`.
2. All terms are summed into a single `Value`.
3. Multiplying by `(-1 / n)` converts the sum of log-likelihoods to the mean negative log-likelihood (BCE).

### Usage Example

```python
from ai_essentials.loss import cross_entropy
from ai_essentials.value import Value

preds   = [Value(0.9), Value(0.1)]   # model's predicted probabilities
targets = [1.0,        0.0]          # true labels

loss = cross_entropy(preds, targets)
print(loss.data)   # -(log(0.9) + log(0.9)) / 2 ≈ 0.105

loss.backward()
```

---

## 6.4 Input Validation: `check_loss_inputs`

```python
def check_loss_inputs(predictions, targets):
    if not predictions or not targets:
        raise ValueError("Predictions and targets must not be empty")
    if len(predictions) != len(targets):
        raise ValueError(
            f"Predictions and targets must have the same length "
            f"(got {len(predictions)} and {len(targets)})."
        )
```

Both loss functions call this helper first. It guards against the two most common usage mistakes:
1. Passing an empty list (nothing to compute).
2. Mismatched list lengths (a sign of a data pipeline bug).

---

## 6.5 Which Loss Function to Use?

| Task | Loss Function | Output Activation |
|------|---------------|-------------------|
| Regression (continuous output) | `mse_loss` | `None` (linear) |
| Binary classification | `cross_entropy` | `sigmoid` (not built-in yet) or `tanh` with labels ∈ {-1, +1} |

The demo `main.py` uses `mse_loss` because it fits a continuous function (sine wave). For a classification task, you would use `cross_entropy` with a sigmoid output.

---

## 6.6 Loss as the Root of the Computation Graph

The loss `Value` is the **root** of the entire computation graph for a training step. When you call `loss.backward()`:

1. The topological sort traverses the graph from the loss backward through the entire network.
2. Every parameter's `.grad` is filled in with `∂loss/∂parameter`.
3. The optimizer then uses those gradients to update parameters.

This is why it's critical to call `optimizer.zero_grad()` after each update step — to reset all gradients to 0 before the next forward pass.

---

## 6.7 Key Takeaways

- MSE is appropriate for regression. It penalises the squared difference between prediction and target.
- BCE is appropriate for binary classification. It penalises confident wrong predictions via logarithms.
- Both loss functions return a single `Value` that is the root of the computation graph.
- `check_loss_inputs` guards against empty or mismatched inputs.
- Both functions accept lists of raw Python floats or `Value` objects as targets.

---

*← [Module 5: Multi-Layer Perceptron](05-MLP.md) | Next → [Module 7: Optimization](07-Optimization.md)*

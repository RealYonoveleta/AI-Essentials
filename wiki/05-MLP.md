# Module 5 — The Multi-Layer Perceptron (MLP)

> **Prerequisites:** Modules 1–4. You should understand `Value`, activation functions, `Neuron`, and `Layer`.

---

## 5.1 From Neurons to Networks — Theory

A single neuron is a linear classifier with a nonlinear output. On its own, it can only separate data with a single hyperplane. The power of neural networks comes from **stacking layers**:

```
Input  →  [Hidden Layer 1]  →  [Hidden Layer 2]  →  ...  →  [Output Layer]
```

Each hidden layer transforms its input into a new representation. Early layers detect simple patterns; later layers combine those patterns into more abstract features. This hierarchical representation is what gives deep networks their expressiveness.

A **Multi-Layer Perceptron (MLP)** is the simplest such architecture: a sequence of fully-connected layers (also called **dense** layers), where every neuron in layer `k` is connected to every neuron in layer `k+1`.

---

## 5.2 Universal Approximation

The **Universal Approximation Theorem** (Hornik, 1989; Cybenko, 1989) states that a single hidden layer with enough neurons can approximate any continuous function on a compact domain to arbitrary precision — provided the activation function is not a polynomial.

In practice, **depth** (more layers) is usually more efficient than **width** (more neurons per layer) for learning structured functions. This is why real networks tend to be deep rather than wide. The AI-Essentials demo uses 3 hidden layers of 64 neurons each to fit a multi-frequency signal.

---

## 5.3 The `MLP` Class — Full Walkthrough

**File:** `src/ai_essentials/mlp.py`

```python
from ai_essentials.layer import Layer


def _parse_layer(l, n_args=3):
    """Normalize a layer spec to a tuple of length n_args, padding with None."""
    t = (l,) if isinstance(l, int) else tuple(l)
    if len(t) > n_args:
        raise ValueError(f"Layer spec {t} has too many arguments (max {n_args}).")
    return t + (None,) * (n_args - len(t))


class MLP:
    def __init__(self, num_inputs, layers):
        parsed = [_parse_layer(l) for l in layers]
        sizes = [num_inputs] + [size for size, _, _ in parsed]
        self.layers = [
            Layer(sizes[i], sizes[i + 1], activation=act, **({'bias_init': b} if b is not None else {}))
            for i, (_, act, b) in enumerate(parsed)
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
```

### `_parse_layer` — Flexible Layer Specification

The `_parse_layer` helper normalizes layer specifications into a uniform `(size, activation, bias_init)` triple:

| Input | Parsed as |
|-------|-----------|
| `4` | `(4, None, None)` |
| `(4,)` | `(4, None, None)` |
| `(4, 'tanh')` | `(4, 'tanh', None)` |
| `(4, 'relu', 0.0)` | `(4, 'relu', 0.0)` |

This flexibility allows concise or explicit layer specs depending on what the user needs.

### Constructor: `__init__`

```python
parsed = [_parse_layer(l) for l in layers]
sizes = [num_inputs] + [size for size, _, _ in parsed]
```

`sizes` is a list of layer widths: `[input_width, hidden1_width, hidden2_width, ..., output_width]`. For example, `MLP(2, [(4, 'tanh'), (3, None)])` produces `sizes = [2, 4, 3]`.

```python
self.layers = [
    Layer(sizes[i], sizes[i + 1], activation=act, ...)
    for i, (_, act, b) in enumerate(parsed)
]
```

Layer `i` takes `sizes[i]` inputs and produces `sizes[i+1]` outputs. This "sliding window over sizes" pattern is the standard way to connect layers in sequence.

### Forward Pass: `__call__`

```python
def __call__(self, x):
    for layer in self.layers:
        x = layer(x)
    return x
```

The input `x` is threaded through each layer in turn. Each layer transforms a list of `Value` objects (of length `in_width`) into a new list (of length `out_width`). After all layers, `x` is the network's output — a list of `Value` objects.

### `parameters()`

```python
def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]
```

A flat list of every learnable parameter in the network. This is what you pass to the optimizer.

---

## 5.4 Building an MLP — Examples

### Regression Network (single output)

```python
from ai_essentials.mlp import MLP

model = MLP(num_inputs=1, layers=[
    (16, 'tanh'),    # 16 neurons, tanh activation
    (16, 'tanh'),    # 16 neurons, tanh activation
    (1,  None),      # 1 output neuron, linear (no activation)
])

output = model([0.5])    # returns a list of 1 Value
print(output[0].data)    # the scalar prediction
```

### Binary Classification Network

```python
model = MLP(num_inputs=2, layers=[
    (8, 'relu'),
    (8, 'relu'),
    (1, 'tanh'),   # output in (-1, 1); compare to label ∈ {-1, +1}
                   # or use sigmoid in output + BCE loss
])
```

### Shorthand Notation (all defaults)

```python
# Plain integers use Layer defaults (relu activation, relu-default bias)
model = MLP(2, [4, 4, 1])
```

---

## 5.5 Parameter Counting Example

```python
model = MLP(2, [(4, 'tanh'), (3, None)])
print(len(model.parameters()))   # should be 27

# Layer 1: 4 neurons × (2 inputs + 1 bias) = 12
# Layer 2: 3 neurons × (4 inputs + 1 bias) = 15
# Total = 27
```

Let's verify with the actual library:

```python
model = MLP(2, [(3, 'tanh'), (1, None)])
# Layer 1: 3 × (2+1) = 9
# Layer 2: 1 × (3+1) = 4
# Total   = 13
assert len(model.parameters()) == 13   # passes ✓
```

The test suite verifies this in `tests/test_mlp.py`:

```python
def test_mlp_parameters_count():
    mlp = MLP(2, [(3, 'tanh'), (1, None)])
    assert len(mlp.parameters()) == 13
```

---

## 5.6 MLP Forward Pass — Step by Step

Let's trace `MLP(2, [(3, 'tanh'), (1, None)])` called with input `[0.5, -1.0]`:

```
Input:  x = [Value(0.5), Value(-1.0)]

──── Layer 1 (3 tanh neurons) ────
Neuron 0:  z₀ = w₀₀·0.5 + w₀₁·(-1.0) + b₀  →  tanh(z₀)  →  h₀
Neuron 1:  z₁ = w₁₀·0.5 + w₁₁·(-1.0) + b₁  →  tanh(z₁)  →  h₁
Neuron 2:  z₂ = w₂₀·0.5 + w₂₁·(-1.0) + b₂  →  tanh(z₂)  →  h₂

Intermediate: h = [h₀, h₁, h₂]    (list of 3 Values)

──── Layer 2 (1 linear neuron) ────
Neuron 0:  z = w₀·h₀ + w₁·h₁ + w₂·h₂ + b  →  (no activation)  →  output

Output: [output]                   (list of 1 Value)
```

The entire computation is one connected DAG containing 13 parameter nodes and all intermediate nodes.

---

## 5.7 Backward Pass through the MLP

When `loss.backward()` is called (after computing some loss from the MLP's output), gradients flow back through the entire computation graph:

```
∂L/∂output
  ↓ (through linear layer)
∂L/∂h₀, ∂L/∂h₁, ∂L/∂h₂
  ↓ (through tanh)
∂L/∂z₀, ∂L/∂z₁, ∂L/∂z₂
  ↓ (through dot product and addition)
∂L/∂w₀₀, ∂L/∂w₀₁, ∂L/∂b₀, ..., ∂L/∂w₂₂, ∂L/∂b₂
```

The optimizer then uses these gradients to update the parameters. See Module 7 for details on the optimizer.

---

## 5.8 Key Takeaways

- An `MLP` is a sequence of `Layer` objects connected in series.
- The `_parse_layer` helper normalizes diverse layer specification formats to a uniform `(size, activation, bias_init)` triple.
- Layer `i` takes `sizes[i]` inputs and produces `sizes[i+1]` outputs, automatically wired by the constructor.
- The forward pass threads the input list through each layer in order.
- `MLP.parameters()` returns a flat list of all `Value` objects — the interface used by the optimizer.

---

*← [Module 4: Neural Network Building Blocks](04-Neural-Network-Building-Blocks.md) | Next → [Module 6: Loss Functions](06-Loss-Functions.md)*

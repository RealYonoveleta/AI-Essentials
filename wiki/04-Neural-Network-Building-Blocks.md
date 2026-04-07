# Module 4 — Neural Network Building Blocks: Neuron and Layer

> **Prerequisites:** Modules 1–3. You should understand `Value`, `_backward`, and activation functions before continuing.

---

## 4.1 The Biological Inspiration (and Its Limits)

A biological neuron receives signals from other neurons through **dendrites**, combines them, and fires an electrical signal along its **axon** if the combined signal exceeds a threshold. Artificial neurons are a mathematical abstraction of this: they take a weighted sum of their inputs, add a bias, and pass the result through a nonlinear activation function.

This analogy is useful for building intuition but should not be taken too literally — modern deep learning has diverged significantly from neuroscience.

---

## 4.2 The Perceptron (Single Neuron) — Theory

A single artificial neuron with `n` inputs computes:

```
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
  = w · x + b          (dot product notation)

output = activation(z)
```

where:
- `x = [x₁, x₂, ..., xₙ]` — the input vector
- `w = [w₁, w₂, ..., wₙ]` — the learnable **weight** vector
- `b` — the learnable **bias** scalar
- `z` — the **pre-activation** (also called the **logit**)
- `activation` — a nonlinear function (tanh, relu, or identity)

The weights and bias together are the **parameters** of the neuron. Training adjusts them to minimise the loss.

---

## 4.3 He Initialization — Theory

How should we initialize the weights? If we initialize them all to zero, every neuron computes the same thing and the network can never differentiate. If we initialize them too large, the pre-activations saturate the tanh/relu, and gradients vanish.

**He initialization** (Kaiming He, 2015) is designed for ReLU-like activations. It initializes weights uniformly from:

```
w_i ~ Uniform(-s, s),   where s = sqrt(2 / n_inputs)
```

**Intuition:** With `n` inputs, the variance of the pre-activation `z = Σ wᵢxᵢ` depends on both the variance of `w` and the variance of `x`. He initialization is chosen so that the variance of `z` stays approximately 1 regardless of the layer width — preventing activations from shrinking (vanishing gradients) or growing (exploding gradients) as we go deeper.

For tanh, Xavier initialization (`s = sqrt(1 / n_inputs)`) is the traditional choice, but He initialization works well in practice for both activations.

---

## 4.4 The `Neuron` Class — Full Walkthrough

**File:** `src/ai_essentials/neuron.py`

```python
import random
from ai_essentials.value import Value, to_value
from ai_essentials.math import dot
from ai_essentials.activations import get_activation


class Neuron:
    def __init__(self, num_inputs, activation='relu', bias_init=None):
        scale = (2.0 / num_inputs) ** 0.5 if num_inputs > 0 else 1.0  # He initialization
        self.w = [Value(random.uniform(-scale, scale)) for _ in range(num_inputs)]
        self.activation, default_bias = get_activation(activation)
        self.b = Value(bias_init if bias_init is not None else default_bias)

    def __call__(self, x):
        x = [to_value(v) for v in x]
        z = dot(self.w, x) + self.b
        return self.activation(z) if self.activation else z

    def parameters(self):
        return self.w + [self.b]
```

### Constructor: `__init__`

| Line | What it does |
|------|-------------|
| `scale = (2.0 / num_inputs) ** 0.5` | He initialization scale factor |
| `self.w = [Value(random.uniform(-scale, scale)) ...]` | Initialize one weight per input as a `Value` |
| `self.activation, default_bias = get_activation(activation)` | Look up the activation function and its default bias |
| `self.b = Value(bias_init if bias_init is not None else default_bias)` | Initialize bias (override default if provided) |

The weights are plain `Value` objects — they participate in the computation graph just like any other `Value`. When `loss.backward()` is called, gradients flow back to `self.w[i].grad` and `self.b.grad`.

### Forward Pass: `__call__`

```python
def __call__(self, x):
    x = [to_value(v) for v in x]   # promote any raw floats to Value
    z = dot(self.w, x) + self.b     # weighted sum + bias → Value
    return self.activation(z) if self.activation else z
```

1. Promotes every input to a `Value` (if not already), so that the computation graph is connected.
2. Computes `z = w · x + b` using the `dot` utility (Module 8).
3. Applies the activation function (or returns `z` directly for linear output).

The result is a single `Value` that is connected to all of `self.w`, `self.b`, and the input `x` through the computation graph.

### `parameters()`

```python
def parameters(self):
    return self.w + [self.b]
```

Returns a flat list of all learnable parameters: `n_inputs` weights + 1 bias = `n_inputs + 1` parameters total. This is the interface that the optimizer uses to update parameters.

### Usage Example

```python
from ai_essentials.neuron import Neuron

n = Neuron(num_inputs=3, activation='tanh')
print(len(n.parameters()))   # 4  (3 weights + 1 bias)

output = n([1.0, 2.0, -1.0])
print(output)                # <Value: data=..., grad=0.0>

# After training:
output.backward()            # (usually called on the loss, not on one neuron)
print(n.w[0].grad)           # ∂output/∂w[0]
```

---

## 4.5 The `Layer` Class — Full Walkthrough

**File:** `src/ai_essentials/layer.py`

```python
from ai_essentials.neuron import Neuron


class Layer:
    def __init__(self, num_inputs, num_neurons, neuron_class=Neuron, activation='relu', bias_init=None):
        self.neurons = [neuron_class(num_inputs, activation, bias_init) for _ in range(num_neurons)]

    def __call__(self, x):
        return [neuron(x) for neuron in self.neurons]

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
```

A `Layer` is simply a collection of `num_neurons` independent neurons that all receive **the same input** `x`. Each neuron produces one scalar output, so the layer's output is a list of `num_neurons` values.

### Constructor: `__init__`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_inputs` | — | Width of the input vector |
| `num_neurons` | — | Number of neurons (= output width) |
| `neuron_class` | `Neuron` | Class to instantiate (allows custom neurons) |
| `activation` | `'relu'` | Activation function for all neurons |
| `bias_init` | `None` | Override bias initialization |

### Forward Pass: `__call__`

```python
def __call__(self, x):
    return [neuron(x) for neuron in self.neurons]
```

Each of the `num_neurons` neurons independently processes the entire input vector `x` and returns one `Value`. The result is a Python list of `Value` objects, one per neuron.

**Why a list and not a matrix?** Because the library uses scalar autodiff — each number is tracked individually. There is no tensor type; lists of `Value` objects play the role of vectors.

### `parameters()`

```python
def parameters(self):
    return [p for neuron in self.neurons for p in neuron.parameters()]
```

A flat list of all parameters across all neurons. A layer with `n` inputs and `m` neurons has `m * (n + 1)` parameters.

### Usage Example

```python
from ai_essentials.layer import Layer

layer = Layer(num_inputs=4, num_neurons=8, activation='relu')
print(len(layer.parameters()))   # 8 * (4 + 1) = 40

outputs = layer([1.0, 2.0, 3.0, 4.0])
print(len(outputs))               # 8 — one Value per neuron
```

---

## 4.6 The Data Flow: From Input to Layer Output

Let's trace what happens when we call `layer([1.0, 2.0])` with a 2-input, 3-neuron layer:

```
Input: x = [1.0, 2.0]

Neuron 0:  z₀ = w₀₀·1.0 + w₀₁·2.0 + b₀  →  relu(z₀)  →  out₀
Neuron 1:  z₁ = w₁₀·1.0 + w₁₁·2.0 + b₁  →  relu(z₁)  →  out₁
Neuron 2:  z₂ = w₂₀·1.0 + w₂₁·2.0 + b₂  →  relu(z₂)  →  out₂

Output: [out₀, out₁, out₂]   — all Value objects
```

Each of the 9 parameters (`w₀₀, w₀₁, b₀, ...`) is a `Value` connected to the output through the computation graph.

---

## 4.7 Parameter Counting

Understanding parameter counts is important for reasoning about model capacity and memory usage.

For a layer with `n_inputs` and `m` neurons:

```
parameters = m × (n_inputs + 1)
                          └── the +1 is the bias
```

Example: `Layer(num_inputs=10, num_neurons=5)` has `5 × 11 = 55` parameters.

For a full MLP (computed in Module 5):

```
MLP(2, [(4, 'tanh'), (3, 'relu'), (1, None)])

Layer 1: 4 × (2 + 1)  =  12 parameters
Layer 2: 3 × (4 + 1)  =  15 parameters
Layer 3: 1 × (3 + 1)  =   4 parameters
Total:                    31 parameters
```

---

## 4.8 Key Takeaways

- A `Neuron` computes `activation(w · x + b)`. It stores learnable `Value` objects for weights and bias.
- A `Layer` is a list of independent neurons that process the same input and each contribute one output value.
- He initialization keeps activation variance stable across layers, preventing vanishing and exploding gradients.
- Both `Neuron.parameters()` and `Layer.parameters()` return flat lists of `Value` objects — the interface used by optimizers.

---

*← [Module 3: Activation Functions](03-Activation-Functions.md) | Next → [Module 5: Multi-Layer Perceptron](05-MLP.md)*

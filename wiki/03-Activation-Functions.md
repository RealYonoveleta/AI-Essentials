# Module 3 — Activation Functions

> **Prerequisites:** Module 2 (especially `Value.tanh()` and `Value.relu()`).

---

## 3.1 Why Do We Need Activation Functions?

Consider a network where every layer simply computes a weighted sum of its inputs:

```
Layer 1:  h₁ = W₁·x + b₁
Layer 2:  h₂ = W₂·h₁ + b₂
Output:   y  = W₃·h₂ + b₃
```

Substituting:

```
y = W₃·(W₂·(W₁·x + b₁) + b₂) + b₃
  = (W₃W₂W₁)·x + (W₃W₂b₁ + W₃b₂ + b₃)
  = W_eff·x + b_eff
```

No matter how many layers we stack, a purely linear network collapses to a single linear transformation. It can only fit straight lines (or hyperplanes in multiple dimensions) — useless for any interesting problem.

**Activation functions introduce nonlinearity.** They are applied element-wise to the output of each neuron, breaking the linearity and giving the network the power to approximate arbitrary functions.

---

## 3.2 Desirable Properties of an Activation Function

| Property | Why it Matters |
|----------|----------------|
| **Nonlinear** | Enables the network to learn complex patterns |
| **Differentiable (almost everywhere)** | Required for gradient-based training |
| **Gradient doesn't vanish for large inputs** | Prevents gradients from dying during backprop |
| **Computationally cheap** | Applied billions of times during training |
| **Output range bounded or semi-bounded** | Prevents activations from exploding |

---

## 3.3 Tanh

### Theory

The hyperbolic tangent function:

```
tanh(z) = (eᶻ - e⁻ᶻ) / (eᶻ + e⁻ᶻ)
```

Its output is always in the interval `(-1, 1)`, making it **zero-centered** — outputs from a tanh neuron are symmetric around 0, which can help with gradient flow.

**Derivative:**

```
d/dz tanh(z) = 1 - tanh²(z)
```

At `z = 0` the derivative is 1 (maximum — gradients flow freely). As `|z|` grows, the derivative approaches 0 — this is the **saturation** or **vanishing gradient** problem. Neurons whose pre-activation is far from 0 contribute almost nothing to training.

**Shape:**

```
z:       -3    -2    -1     0     1     2     3
tanh(z): -0.995 -0.964 -0.762  0  0.762  0.964  0.995
```

### Implementation

```python
# src/ai_essentials/value.py
def tanh(self):
    t = math.tanh(self.data)      # forward value
    out = Value(t, (self,), "tanh")

    def _backward():
        self.grad += (1 - t * t) * out.grad   # d/dz tanh(z) = 1 - tanh²(z)

    out._backward = _backward
    return out
```

### When to Use Tanh

- Hidden layers where you want **bounded, zero-centered** activations.
- The demo `main.py` uses tanh throughout the hidden layers — the normalized inputs keep pre-activations near 0, avoiding saturation.

---

## 3.4 ReLU (Rectified Linear Unit)

### Theory

```
relu(z) = max(0, z)
```

ReLU outputs 0 for negative inputs and passes positive inputs unchanged. It is **not bounded above**, which means the activations can grow without limit — this is usually managed by weight initialization and normalization.

**Derivative (subgradient):**

```
d/dz relu(z) = 1  if z > 0
               0  if z ≤ 0
```

ReLU does not saturate for positive inputs (gradient is always 1 or 0), which makes it much better than tanh for very deep networks. However, neurons where `z ≤ 0` produce zero gradient — they can permanently stop learning if they fall into this regime. This is called the **dying ReLU problem**.

**Shape:**

```
z:        -3    -2    -1    0    1    2    3
relu(z):   0     0     0    0    1    2    3
```

### Implementation

```python
# src/ai_essentials/value.py
def relu(self):
    out = Value(max(0.0, self.data), (self,), "relu")

    def _backward():
        self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

    out._backward = _backward
    return out
```

### When to Use ReLU

- Deep hidden layers where vanishing gradients are a concern.
- The default activation when you specify a `Neuron` without an explicit activation.

---

## 3.5 Linear (No Activation)

Sometimes we want the output of a layer to be an **unbounded linear combination** of its inputs — no squashing. This is used for the **output layer** in regression problems, where we want to predict real-valued numbers.

In AI-Essentials, passing `activation=None` to a Neuron or Layer skips the activation step:

```python
# src/ai_essentials/neuron.py
def __call__(self, x):
    z = dot(self.w, x) + self.b
    return self.activation(z) if self.activation else z   # <-- if None, return z directly
```

---

## 3.6 The Activation Registry (`activations.py`)

**File:** `src/ai_essentials/activations.py`

```python
def tanh(x):
    return x.tanh()

def relu(x):
    return x.relu()

# (function, default_bias_init)
ACTIVATIONS = {
    'tanh': (tanh, 0.0),
    'relu': (relu, 0.01),
}

def get_activation(name):
    if name is None:
        return None, 0.0
    if name not in ACTIVATIONS:
        raise ValueError(
            f"Unknown activation '{name}'. Available: {list(ACTIVATIONS.keys())}"
        )
    return ACTIVATIONS[name]
```

### Design Notes

- **`ACTIVATIONS` dictionary:** Maps a string name to a `(function, default_bias_init)` tuple. The string name is what you pass to `Neuron`, `Layer`, or `MLP`.
- **Default bias:** Each activation has a different default bias. For ReLU, the default bias is `0.01` (a small positive value to avoid dead neurons at initialization). For tanh, `0.0` is correct because tanh is zero-centered.
- **`get_activation(name)`:** Returns both the function and the default bias. If `name` is `None`, it returns `(None, 0.0)` — which the `Neuron` interprets as "no activation" (linear output).

### Extending the Registry

To add a new activation (e.g., sigmoid):

```python
# First, add the method to Value in value.py:
def sigmoid(self):
    s = 1 / (1 + math.exp(-self.data))
    out = Value(s, (self,), "sigmoid")
    def _backward():
        self.grad += s * (1 - s) * out.grad
    out._backward = _backward
    return out

# Then register it in activations.py:
def sigmoid(x):
    return x.sigmoid()

ACTIVATIONS['sigmoid'] = (sigmoid, 0.0)
```

---

## 3.7 Tanh vs. ReLU — Comparison Table

| | tanh | ReLU |
|-|------|------|
| Output range | `(-1, 1)` | `[0, ∞)` |
| Zero-centered | ✅ | ❌ |
| Saturates for large inputs | ✅ (vanishing gradient) | ❌ (only for negatives) |
| Dying neuron problem | ❌ | ✅ |
| Default bias in library | `0.0` | `0.01` |
| Best for | Shallow networks, smooth targets | Deep networks, sparse activations |

---

## 3.8 Key Takeaways

- Without activation functions, deep networks collapse to single-layer linear models.
- `tanh` squashes to `(-1, 1)` with a smooth, zero-centered output — great when inputs are normalized.
- `relu` is computationally simpler and avoids saturation for positive inputs — preferred for deeper networks.
- The `ACTIVATIONS` dictionary in `activations.py` is the single place to register new activation functions for use throughout the library.

---

*← [Module 2: Scalar Automatic Differentiation](02-Scalar-Autodiff.md) | Next → [Module 4: Neural Network Building Blocks](04-Neural-Network-Building-Blocks.md)*

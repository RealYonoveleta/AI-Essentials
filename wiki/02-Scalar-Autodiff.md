# Module 2 — Scalar Automatic Differentiation

> **Prerequisites:** Module 1. Basic Python (functions, classes, closures).

---

## 2.1 Why Do We Need Gradients?

Training a neural network means finding the set of parameters `θ` that minimises a loss function `L(θ)`. We do this by repeatedly moving every parameter slightly in the direction that decreases `L`:

```
θ  ←  θ − α · ∂L/∂θ
```

where `α` is the learning rate and `∂L/∂θ` is the **gradient** of the loss with respect to that parameter. Without gradients, we can't train.

For a tiny network with 2 parameters you could derive the gradients by hand. A real network has thousands or millions of parameters — we need a computer to compute gradients automatically. That's what **automatic differentiation** (autodiff) does.

---

## 2.2 Three Ways to Compute Gradients

| Method | How | Pros | Cons |
|--------|-----|------|------|
| **Symbolic** | Manipulate algebraic expressions | Exact | Exponential expression growth |
| **Numerical** | `(f(x+ε) − f(x)) / ε` | Simple | Slow, floating-point errors |
| **Automatic** | Track computation graph + chain rule | Exact & efficient | Requires implementation |

AI-Essentials uses **reverse-mode automatic differentiation**, the same approach used by PyTorch and TensorFlow. It is the most efficient method when there are many parameters and one scalar loss output — which is exactly the neural network training scenario.

---

## 2.3 The Chain Rule — the Heart of Backpropagation

If `c = f(b)` and `b = g(a)`, then:

```
∂c/∂a = (∂c/∂b) · (∂b/∂a)
```

This is the chain rule from calculus. Backpropagation is nothing more than the systematic application of the chain rule across every operation in the computation graph, starting from the loss output and working backwards to the inputs.

### A Concrete Example

Suppose we compute:

```
a = 2.0
b = 3.0
c = a * b        # c = 6.0,  ∂c/∂a = b = 3,  ∂c/∂b = a = 2
d = c + b        # d = 9.0,  ∂d/∂c = 1,      ∂d/∂b = 1
```

We want `∂d/∂a` and `∂d/∂b`. Using the chain rule:

```
∂d/∂a = (∂d/∂c) · (∂c/∂a) = 1 · 3 = 3
∂d/∂b = (∂d/∂c) · (∂c/∂b) + (∂d/∂b directly)
       = 1 · 2 + 1 = 3
```

Note that `b` is used twice, so its gradient accumulates from both paths — this is **gradient accumulation** and is handled automatically by the `Value` class.

---

## 2.4 The Computation Graph

Every operation creates a node in a **directed acyclic graph (DAG)**:

```
   a(2.0) ──┐
             ├─► [*] c(6.0) ──┐
   b(3.0) ──┘                  ├─► [+] d(9.0)
   b(3.0) ────────────────────┘
```

Each node stores:
- Its **forward value** (`data`)
- Its **gradient** (`grad`) — how much the loss changes when this node's value increases by 1
- A reference to its **parent nodes** (`_prev`)
- A **closure** (`_backward`) that knows how to push gradients back to parents

During the **backward pass**, we traverse this graph in reverse topological order and call each node's `_backward()` closure.

---

## 2.5 The `Value` Class — Full Walkthrough

**File:** `src/ai_essentials/value.py`

### Construction

```python
class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = data          # the scalar number
        self.grad = 0.0           # ∂Loss/∂self, starts at 0
        self._backward = lambda: None   # no-op until an operation sets it
        self._prev = set(_children)     # parent nodes in the graph
        self._op = _op            # label for debugging ("*", "+", "tanh", ...)
```

`_children` and `_op` are internal (note the underscore prefix) — users never set them directly. They are populated automatically when operations are performed.

### Helper: `_wrap` / `to_value`

```python
def _wrap(self, other):
    return to_value(other)

def to_value(x):
    return x if isinstance(x, Value) else Value(x)
```

This lets you write `Value(2.0) + 3` — the raw Python float `3` is automatically promoted to `Value(3.0)`.

---

### Operation: Addition

```python
def __add__(self, other):
    other = self._wrap(other)
    out = Value(self.data + other.data, (self, other), "+")

    def _backward():
        self.grad  += 1.0 * out.grad
        other.grad += 1.0 * out.grad

    out._backward = _backward
    return out
```

**Math:** If `out = self + other`, then:
- `∂out/∂self  = 1`
- `∂out/∂other = 1`

The `_backward` closure multiplies `out.grad` (which is `∂Loss/∂out`, delivered from the node above) by the local gradient `1`, and **accumulates** it into `self.grad` and `other.grad`. The `+=` (not `=`) is critical: if `self` appears in multiple places in the graph, gradients from all paths must be summed.

---

### Operation: Multiplication

```python
def __mul__(self, other):
    other = self._wrap(other)
    out = Value(self.data * other.data, (self, other), "*")

    def _backward():
        self.grad  += other.data * out.grad
        other.grad += self.data  * out.grad

    out._backward = _backward
    return out
```

**Math:** If `out = self * other`, then:
- `∂out/∂self  = other.data`
- `∂out/∂other = self.data`

Notice the closures capture `self.data` and `other.data` **by reference** from the enclosing scope — Python closures close over variables, not values. This is correct because `self.data` won't change between the forward pass and when `_backward` is called.

---

### Operation: Power

```python
def __pow__(self, n):
    if not isinstance(n, (int, float)):
        raise ValueError(...)
    out = Value(self.data**n, (self,), f"**{n}")

    def _backward():
        if self.data == 0 and n < 1:
            return   # gradient undefined: 0^(negative)
        self.grad += n * (self.data ** (n - 1)) * out.grad

    out._backward = _backward
    return out
```

**Math:** If `out = self^n`, then `∂out/∂self = n · self^(n-1)`. The special case for `self.data == 0` and `n < 1` prevents a division-by-zero when computing `0^(n-1)` for fractional `n`.

---

### Operations: Subtraction, Division, Negation, Reflected Ops

These are built from the primitives above:

```python
def __neg__(self):    return self * -1
def __sub__(self, other): return self + (-other)
def __truediv__(self, other): return self * other**-1
def __radd__(self, other): return self + other     # e.g. 1 + Value(2)
def __rmul__(self, other): return self * other
def __rsub__(self, other): return Value(other) + (-self)
```

This is a key design principle: **reuse already-differentiated primitives rather than writing new backward functions**. Division is implemented as `a * b^(-1)`, so its gradient comes for free.

---

### Activation: tanh

```python
def tanh(self):
    t = math.tanh(self.data)
    out = Value(t, (self,), "tanh")

    def _backward():
        self.grad += (1 - t * t) * out.grad

    out._backward = _backward
    return out
```

**Math:** If `out = tanh(self)`, then `∂out/∂self = 1 - tanh²(self) = 1 - t²`. This is the derivative of tanh — it's 1 at `self=0` (maximum gradient) and approaches 0 as `|self|` grows (the **vanishing gradient** problem).

---

### Activation: relu

```python
def relu(self):
    out = Value(max(0.0, self.data), (self,), "relu")

    def _backward():
        self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

    out._backward = _backward
    return out
```

**Math:** If `out = max(0, self)`, then the gradient is `1` if `self > 0`, else `0` (the step function). This is the **subgradient** at the non-differentiable point `self = 0` — we simply choose `0`.

---

### Activation: log (natural logarithm)

```python
def log(self, eps=1e-12):
    val = max(self.data, eps)   # clamp to avoid log(0) = -inf
    l   = math.log(val)
    out = Value(l, (self,), "log")

    def _backward():
        self.grad += (1 / val) * out.grad

    out._backward = _backward
    return out
```

**Math:** `∂log(x)/∂x = 1/x`. The `eps` guard prevents numerical catastrophe when `self.data ≈ 0` (which can happen with poorly-calibrated prediction probabilities in BCE loss).

---

### The Backward Pass: `backward()`

```python
def backward(self):
    topo = []
    visited = set()
    stack = [self]
    # Iterative post-order traversal — avoids Python recursion limit on deep graphs
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

    self.grad = 1.0       # seed: ∂Loss/∂Loss = 1
    for v in reversed(topo):
        v._backward()
```

**Step 1 — Topological sort:** The algorithm builds a post-order traversal of the DAG using an explicit stack (instead of recursion, to avoid Python's default recursion limit of ~1000 frames on deep networks). After the traversal, `topo` contains nodes in an order where every parent appears before its children (or equivalently, when reversed, every node appears after all nodes that depend on it).

**Step 2 — Gradient seeding:** `self.grad = 1.0` because `∂L/∂L = 1` — the loss is its own derivative.

**Step 3 — Backward traversal:** Iterate `reversed(topo)` — from output to inputs — and call each node's `_backward()`. Because of the topological order, by the time we call `v._backward()`, all gradients from nodes that consume `v` have already been added to `v.grad`. The chain rule is satisfied.

---

## 2.6 A Complete Example

```python
from ai_essentials.value import Value

a = Value(2.0)
b = Value(3.0)
c = a * b        # c = 6.0
d = c + b        # d = 9.0
d.backward()

print(a.grad)    # 3.0  →  ∂d/∂a = ∂d/∂c · ∂c/∂a = 1 · b = 3
print(b.grad)    # 3.0  →  ∂d/∂b = ∂d/∂c · ∂c/∂b + ∂d/∂b = 1·2 + 1 = 3
print(c.grad)    # 1.0  →  ∂d/∂c = 1
```

---

## 2.7 Gradient Accumulation for Shared Nodes

```python
a = Value(3.0)
b = a + a        # b = 6.0,  but 'a' is used twice
b.backward()
print(a.grad)    # 2.0  →  ∂b/∂a = 1 + 1 = 2  (both paths sum)
```

Because `_backward` uses `+=`, each path through the graph contributes correctly to `a.grad`.

---

## 2.8 Key Takeaways

- Every `Value` is a node in a computation graph.
- Every operation creates a new `Value` and registers a backward closure.
- Calling `.backward()` does a topological traversal in reverse, applying the chain rule automatically.
- Gradients accumulate (`+=`) so shared nodes receive gradients from all paths.
- The implementation is ~127 lines of pure Python — that is all you need for a working autodiff engine.

---

*← [Module 1: Introduction](01-Introduction.md) | Next → [Module 3: Activation Functions](03-Activation-Functions.md)*

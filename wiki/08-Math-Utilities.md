# Module 8 — Math Utilities

> **Prerequisites:** Module 2 (`Value` and how it participates in arithmetic).

---

## 8.1 Overview

**File:** `src/ai_essentials/math.py`

The `math.py` module provides low-level numeric utilities used throughout the library, particularly by the `Neuron` class. It is intentionally minimal — just the primitives needed to support the core functionality.

```python
def get_shape(a):  ...   # shape of a nested list
def get_row(a, i):  ...  # i-th row
def get_column(a, j): ...# j-th column
def dot(a, b):  ...      # dot product of two lists
def matmul(a, b):  ...   # matrix multiplication
```

---

## 8.2 `dot` — The Core Operation

```python
def dot(a, b):
    """Calculate dot product between a and b"""
    from ai_essentials.value import Value

    if len(a) != len(b):
        raise ValueError("a and b must have the same size")

    return sum((x * y for x, y in zip(a, b)), Value(0.0))
```

**What it computes:**

```
dot(a, b) = a₀·b₀ + a₁·b₁ + ... + aₙ₋₁·bₙ₋₁
```

**Why import `Value` inside the function?** This is a **deferred import** to avoid circular imports. `math.py` is imported by `neuron.py`, which is imported by `layer.py`, which is imported by `mlp.py`. Importing `Value` at module level would create a circular dependency. Importing it inside the function body defers it to first use, by which time all modules are loaded.

**Why `sum(..., Value(0.0))`?** Python's built-in `sum` takes an optional `start` argument. Using `Value(0.0)` as the start ensures the result is always a `Value` even when `a` and `b` are empty (though `dot` raises if lengths differ, so this mainly provides type consistency).

**Usage in `Neuron`:**

```python
z = dot(self.w, x) + self.b
```

This one line computes `w₀x₀ + w₁x₁ + ... + wₙxₙ + b`, building the entire pre-activation subgraph of the neuron.

---

## 8.3 `matmul` — Matrix Multiplication

```python
def matmul(a, b):
    """Multiply two matrices a and b"""

    if len(a[0]) != len(b):
        raise ValueError("Columns in a must match rows in b")

    aRows = len(a)
    bCols = len(b[0])

    result = [[0] * bCols for _ in range(aRows)]

    for i in range(aRows):
        for j in range(bCols):
            result[i][j] = dot(a[i], get_column(b, j))

    return result
```

Computes `C = A × B` where `A` is `(aRows × k)` and `B` is `(k × bCols)`, producing `C` of shape `(aRows × bCols)`.

Each element `C[i][j]` is the dot product of row `i` of `A` and column `j` of `B`, using the `dot` function above. Because `dot` returns a `Value`, the result matrix is a 2D list of `Value` objects — the entire matmul is differentiable.

**Note:** `matmul` is provided as a utility but is not used internally by the current `Layer`/`Neuron` implementation, which processes neurons one at a time via individual `dot` calls. It is available for users who want to implement batch operations manually.

---

## 8.4 `get_shape` — Shape Introspection

```python
def get_shape(a):
    """Return the shape of a as a tuple for n-dimensional arrays."""

    shape = []
    while isinstance(a, list):
        shape.append(len(a))
        a = a[0] if a else []

    return tuple(shape)
```

Works like `numpy.shape` for nested Python lists:

```python
get_shape([[1, 2, 3], [4, 5, 6]])   # → (2, 3)
get_shape([1, 2, 3])                # → (3,)
get_shape([[[1, 2], [3, 4]]])       # → (1, 2, 2)
```

**Limitations:** It infers shape from the first element of each dimension, so it assumes a **regular (rectangular) array**. Jagged lists will produce incorrect results.

---

## 8.5 `get_row` and `get_column`

```python
def get_row(a, i=0):
    """Return the i-th row of a"""
    return a[i]

def get_column(a, j=0):
    """Return the j-th column of a"""
    return [row[j] for row in a]
```

Simple helpers for 2D list indexing. `get_column` is used by `matmul` to extract columns before computing dot products. Since 2D Python lists are row-major (a list of rows), column access requires a list comprehension.

---

## 8.6 Usage Examples

### Direct Use of `dot`

```python
from ai_essentials.math import dot
from ai_essentials.value import Value

a = [Value(1.0), Value(2.0), Value(3.0)]
b = [Value(4.0), Value(5.0), Value(6.0)]

result = dot(a, b)
print(result.data)   # 1·4 + 2·5 + 3·6 = 32.0

result.backward()
print(a[0].grad)     # 4.0   (∂(a·b)/∂a₀ = b₀)
print(b[2].grad)     # 3.0   (∂(a·b)/∂b₂ = a₂)
```

### Matrix Multiplication

```python
from ai_essentials.math import matmul, get_shape
from ai_essentials.value import Value

A = [[Value(1.0), Value(2.0)],
     [Value(3.0), Value(4.0)]]

B = [[Value(5.0), Value(6.0)],
     [Value(7.0), Value(8.0)]]

C = matmul(A, B)
print(get_shape(C))        # (2, 2)
print(C[0][0].data)        # 1·5 + 2·7 = 19.0
print(C[1][1].data)        # 3·6 + 4·8 = 50.0
```

---

## 8.7 Key Takeaways

- `dot(a, b)` is the mathematical backbone of every `Neuron` — it computes the weighted sum and returns a differentiable `Value`.
- `matmul` extends `dot` to full matrix multiplication, using `Value` elements for differentiability.
- `get_shape`, `get_row`, `get_column` are simple helpers for working with 2D list structures.
- The deferred import of `Value` inside `dot` is a deliberate design to avoid circular imports.

---

*← [Module 7: Optimization](07-Optimization.md) | Next → [Module 9: The Training Loop](09-Training-Loop.md)*

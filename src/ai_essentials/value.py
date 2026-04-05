import math


def to_value(x):
    return x if isinstance(x, Value) else Value(x)


def to_values(xs):
    return [to_value(x) for x in xs]


class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def _wrap(self, other):
        return to_value(other)

    def __add__(self, other):
        other = self._wrap(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = self._wrap(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, n):
        if not isinstance(n, (int, float)):
            raise ValueError(f"Power must be a scalar, got {type(n).__name__}")
        out = Value(self.data ** n, (self,), f"**{n}")

        def _backward():
            self.grad += n * (self.data ** (n - 1)) * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        other = self._wrap(other)
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return Value(other) + (-self)

    def __truediv__(self, other):
        other = self._wrap(other)
        return self * other ** -1

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t * t) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def log(self, eps=1e-12):
        val = max(self.data, eps)
        l = math.log(val)
        out = Value(l, (self,), "log")

        def _backward():
            self.grad += (1 / val) * out.grad

        out._backward = _backward
        return out
        
    def backward(self):
        topo = []
        visited = set()
        
        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
                
        build(self)
        
        self.grad = 1.0
        r_topo = reversed(topo)
        
        for v in r_topo:
            v._backward()
                    
        return r_topo

    
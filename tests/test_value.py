import math
import pytest
from ai_essentials.value import Value

def test_value_addition():
    a = Value(2.0)
    b = Value(3.0)
    c = a + b
    assert c.data == 5.0
    c.backward()
    assert a.grad == 1.0
    assert b.grad == 1.0

def test_value_multiplication():
    a = Value(2.0)
    b = Value(3.0)
    c = a * b
    assert c.data == 6.0
    c.backward()
    assert a.grad == 3.0
    assert b.grad == 2.0

def test_value_addition_with_scalar():
    a = Value(2.0)
    c = a + 5
    assert c.data == 7.0
    c.backward()
    assert a.grad == 1.0

def test_value_multiplication_with_scalar():
    a = Value(2.0)
    c = a * 4
    assert c.data == 8.0
    c.backward()
    assert a.grad == 4.0

def test_chain_operations():
    a = Value(2.0)
    b = Value(3.0)
    c = a * b + b + 1
    assert c.data == 2.0 * 3.0 + 3.0 + 1.0
    c.backward()
    # c = a*b + b + 1
    # dc/da = b
    # dc/db = a + 1
    assert a.grad == 3.0
    assert b.grad == 2.0 + 1.0

def test_tanh_forward():
    a = Value(0.5)
    t = math.tanh(0.5)
    out = a.tanh()
    assert pytest.approx(out.data, rel=1e-6) == t

def test_tanh_backward():
    a = Value(0.5)
    out = a.tanh()
    out.backward()
    expected_grad = 1 - math.tanh(0.5) ** 2
    assert pytest.approx(a.grad, rel=1e-6) == expected_grad

def test_backward_topological_order():
    # Test that backward works for a small graph
    a = Value(2.0)
    b = Value(3.0)
    c = a * b
    d = c + a
    d.backward()
    # d = c + a = a*b + a
    # dd/da = b + 1
    # dd/db = a
    assert a.grad == 3.0 + 1.0
    assert b.grad == 2.0

def test_grad_accumulation():
    # Test that gradients accumulate properly
    a = Value(2.0)
    b = Value(3.0)
    c1 = a * b
    c2 = a * b
    d = c1 + c2
    d.backward()
    # d = a*b + a*b = 2*a*b
    # dd/da = 2*b
    # dd/db = 2*a
    assert a.grad == 2 * 3.0
    assert b.grad == 2 * 2.0

def test_repr_and_op():
    a = Value(1.0)
    b = Value(2.0)
    c = a + b
    d = a * b
    assert c._op == "+"
    assert d._op == "*"
    assert isinstance(c._prev, set)
    assert isinstance(d._prev, set)
import pytest
from math import tanh
from ai_essentials.neuron import Neuron
from ai_essentials.value import Value


class DummyDot:
    """Helper to patch dot product for deterministic testing."""
    def __init__(self, value):
        self.value = value
    def __call__(self, w, x):
        return Value(self.value)


def test_neuron_initialization(monkeypatch):
    monkeypatch.setattr("random.uniform", lambda a, b: 0.5)
    n = Neuron(3)
    assert all(isinstance(w, Value) for w in n.w)
    assert all(w.data == 0.5 for w in n.w)
    assert isinstance(n.b, Value)
    assert n.b.data == 0.5


def test_neuron_call(monkeypatch):
    monkeypatch.setattr("ai_essentials.neuron.dot", DummyDot(2.0))
    n = Neuron(2)
    n.w = [Value(0.1), Value(0.2)]
    n.b = Value(0.3)
    x = [1.0, 2.0]
    result = n(x)
    expected = tanh(2.0 + 0.3)
    assert isinstance(result, Value)
    assert result.data == pytest.approx(expected)


def test_neuron_weights_and_bias_are_random():
    n1 = Neuron(2)
    n2 = Neuron(2)
    assert [w.data for w in n1.w] != [w.data for w in n2.w] or n1.b.data != n2.b.data


def test_neuron_call_with_zero_inputs(monkeypatch):
    monkeypatch.setattr("ai_essentials.neuron.dot", DummyDot(0.0))
    n = Neuron(0)
    n.w = []
    n.b = Value(0.0)
    x = []
    result = n(x)
    assert isinstance(result, Value)
    assert result.data == pytest.approx(0.0)


def test_neuron_call_input_length_mismatch():
    n = Neuron(3)
    x = [1.0, 2.0]  # Should be length 3
    with pytest.raises(Exception):
        n(x)


def test_neuron_parameters():
    n = Neuron(3)
    params = n.parameters()
    assert len(params) == 4  # 3 weights + 1 bias
    assert all(isinstance(p, Value) for p in params)
    assert params[-1] is n.b


def test_neuron_backward():
    """Test that gradients flow back through the neuron."""
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("random.uniform", lambda a, b: 0.5)
    n = Neuron(2)
    x = [Value(1.0), Value(1.0)]
    out = n(x)
    out.backward()
    for p in n.parameters():
        assert p.grad != 0.0

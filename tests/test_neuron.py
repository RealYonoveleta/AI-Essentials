import pytest
from math import tanh
from ai_essentials.neuron import Neuron

class DummyDot:
    """Helper to patch dot product for deterministic testing."""
    def __init__(self, value):
        self.value = value
    def __call__(self, w, x):
        return self.value

def test_neuron_initialization(monkeypatch):
    # Patch random.uniform to return fixed values
    monkeypatch.setattr("random.uniform", lambda a, b: 0.5)
    n = Neuron(3)
    assert n.w == [0.5, 0.5, 0.5]
    assert n.b == 0.5

def test_neuron_call(monkeypatch):
    # Patch dot to return a fixed value
    monkeypatch.setattr("ai_essentials.neuron.dot", DummyDot(2.0))
    n = Neuron(2)
    n.w = [0.1, 0.2]
    n.b = 0.3
    x = [1.0, 2.0]
    result = n(x)
    expected = tanh(2.0 + 0.3)
    assert result == pytest.approx(expected)

def test_neuron_weights_and_bias_are_random():
    n1 = Neuron(2)
    n2 = Neuron(2)
    # It's very unlikely that two neurons have exactly the same weights and bias
    assert n1.w != n2.w or n1.b != n2.b

def test_neuron_call_with_zero_inputs(monkeypatch):
    monkeypatch.setattr("ai_essentials.neuron.dot", DummyDot(0.0))
    n = Neuron(0)
    n.w = []
    n.b = 0.0
    x = []
    result = n(x)
    assert result == 0.0

def test_neuron_call_input_length_mismatch():
    n = Neuron(3)
    x = [1.0, 2.0]  # Should be length 3
    with pytest.raises(Exception):
        n(x)
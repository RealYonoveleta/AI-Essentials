import pytest
from ai_essentials.mlp import MLP
from ai_essentials.layer import Layer
from ai_essentials.loss import mse_loss
from ai_essentials.value import Value


# ── Structure tests ──────────────────────────────────────────────────────────

def test_mlp_initializes_correct_number_of_layers():
    mlp = MLP(4, [5, 3, 2])
    assert len(mlp.layers) == 3


def test_mlp_layers_are_layer_instances():
    mlp = MLP(2, [3, 4])
    for layer in mlp.layers:
        assert isinstance(layer, Layer)


def test_mlp_no_layers_returns_input_unchanged():
    mlp = MLP(3, [])
    x = [1.0, 2.0, 3.0]
    assert mlp(x) == x


def test_mlp_parameters_count():
    # MLP(2, [3, 1]):
    # Layer 1: 3 neurons * (2 weights + 1 bias) = 9
    # Layer 2: 1 neuron  * (3 weights + 1 bias) = 4
    mlp = MLP(2, [3, 1])
    assert len(mlp.parameters()) == 13


def test_mlp_all_parameters_are_values():
    mlp = MLP(2, [3])
    assert all(isinstance(p, Value) for p in mlp.parameters())


# ── Forward pass tests ───────────────────────────────────────────────────────

def test_mlp_output_is_list_of_values():
    mlp = MLP(2, [3, 1])
    out = mlp([1.0, 2.0])
    assert isinstance(out, list)
    assert all(isinstance(v, Value) for v in out)


def test_mlp_output_size_matches_last_layer():
    mlp = MLP(2, [4, 3])
    out = mlp([1.0, 2.0])
    assert len(out) == 3


def test_mlp_accepts_value_inputs():
    mlp = MLP(2, [3, 1])
    x = [Value(1.0), Value(-1.0)]
    out = mlp(x)
    assert all(isinstance(v, Value) for v in out)


def test_mlp_output_is_bounded_by_tanh():
    # tanh output is always in (-1, 1)
    mlp = MLP(3, [4, 2])
    out = mlp([10.0, -10.0, 5.0])
    for v in out:
        assert -1.0 < v.data < 1.0


# ── Backward pass tests ──────────────────────────────────────────────────────

def test_mlp_backward_populates_gradients():
    mlp = MLP(2, [3, 1])
    loss = mse_loss(mlp([1.0, -1.0]), [0.5])
    loss.backward()
    assert any(abs(p.grad) > 0 for p in mlp.parameters())


def test_mlp_all_parameters_receive_gradients():
    mlp = MLP(2, [3, 1])
    loss = mse_loss(mlp([1.0, -1.0]), [0.5])
    loss.backward()
    assert all(p.grad != 0.0 for p in mlp.parameters())


# ── Integration test: training scenario ─────────────────────────────────────

def test_mlp_loss_decreases_after_gradient_step():
    """
    A single gradient descent step should reduce the loss.
    This is the most fundamental check for a working training loop.
    """
    mlp = MLP(2, [4, 1])
    x = [1.0, -1.0]
    y_true = [1.0]
    lr = 0.1

    # First forward pass
    loss_before = mse_loss(mlp(x), y_true)
    loss_before.backward()

    # Gradient descent step
    for p in mlp.parameters():
        p.data -= lr * p.grad

    # Second forward pass (gradients zeroed out implicitly by new graph)
    loss_after = mse_loss(mlp(x), y_true)

    assert loss_after.data < loss_before.data


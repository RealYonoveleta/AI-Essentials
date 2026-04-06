import pytest
from ai_essentials.loss import mse_loss, cross_entropy
from ai_essentials.value import Value

def test_mse_loss_identical_predictions_and_targets():
    """Test MSE loss when predictions match targets (should be 0)."""
    predictions = [1.0, 2.0, 3.0]
    targets = [1.0, 2.0, 3.0]
    assert mse_loss(predictions, targets).data == 0.0


def test_mse_loss_simple_case():
    """Test MSE loss with simple known values."""
    predictions = [1.0, 2.0, 3.0]
    targets = [2.0, 3.0, 4.0]
    # ((1-2)^2 + (2-3)^2 + (3-4)^2) / 3 = (1 + 1 + 1) / 3 = 1.0
    assert mse_loss(predictions, targets).data == 1.0


def test_mse_loss_negative_differences():
    """Test MSE loss with negative differences."""
    predictions = [0.0, 0.0]
    targets = [1.0, 1.0]
    # ((0-1)^2 + (0-1)^2) / 2 = (1 + 1) / 2 = 1.0
    assert mse_loss(predictions, targets).data == 1.0


def test_mse_loss_single_element():
    """Test MSE loss with single element."""
    predictions = [5.0]
    targets = [3.0]
    # (5-3)^2 / 1 = 4.0
    assert mse_loss(predictions, targets).data == 4.0


def test_mse_loss_shape_mismatch():
    """Test MSE loss raises error when shapes don't match."""
    predictions = [1.0, 2.0, 3.0]
    targets = [1.0, 2.0]
    with pytest.raises(ValueError, match="Predictions and targets must have the same length"):
        mse_loss(predictions, targets)


def test_mse_loss_empty_lists():
    """Test MSE loss with empty lists."""
    predictions = []
    targets = []
    with pytest.raises(ValueError, match="Predictions and targets must not be empty"):
        mse_loss(predictions, targets)


def test_mse_loss_returns_value():
    """Test that mse_loss returns a Value object."""
    loss = mse_loss([1.0], [2.0])
    assert isinstance(loss, Value)


def test_mse_loss_backward():
    """Test that gradients flow back through mse_loss."""
    p = Value(3.0)
    t = Value(1.0)
    loss = mse_loss([p], [t])
    loss.backward()
    # d/dp (p - t)^2 = 2*(p - t) = 2*(3-1) = 4.0
    assert pytest.approx(p.grad, rel=1e-6) == 4.0


def test_cross_entropy_perfect_prediction():
    """Test cross entropy with perfect predictions."""
    import math
    predictions = [1 - 1e-12, 1e-12]
    targets = [1.0, 0.0]
    loss = cross_entropy(predictions, targets)
    assert pytest.approx(loss.data, abs=1e-6) == 0.0


def test_cross_entropy_returns_value():
    """Test that cross_entropy returns a Value object."""
    loss = cross_entropy([0.7], [1.0])
    assert isinstance(loss, Value)


def test_cross_entropy_backward():
    """Test that gradients flow back through cross_entropy."""
    p = Value(0.8)
    t = Value(1.0)
    loss = cross_entropy([p], [t])
    loss.backward()
    # d/dp -[t*log(p) + (1-t)*log(1-p)] = -t/p = -1/0.8 = -1.25
    assert pytest.approx(p.grad, rel=1e-4) == -1.25


def test_cross_entropy_empty_lists():
    """Test cross entropy raises error with empty lists."""
    with pytest.raises(ValueError, match="Predictions and targets must not be empty"):
        cross_entropy([], [])


def test_cross_entropy_shape_mismatch():
    """Test cross entropy raises error when shapes don't match."""
    with pytest.raises(ValueError, match="Predictions and targets must have the same length"):
        cross_entropy([0.5, 0.5], [1.0])

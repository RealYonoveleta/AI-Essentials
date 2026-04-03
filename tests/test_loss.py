import pytest
from ai_essentials.loss import mse_loss

def test_mse_loss_identical_predictions_and_targets():
    """Test MSE loss when predictions match targets (should be 0)."""
    predictions = [1.0, 2.0, 3.0]
    targets = [1.0, 2.0, 3.0]
    assert mse_loss(predictions, targets) == 0.0


def test_mse_loss_simple_case():
    """Test MSE loss with simple known values."""
    predictions = [1.0, 2.0, 3.0]
    targets = [2.0, 3.0, 4.0]
    # ((1-2)^2 + (2-3)^2 + (3-4)^2) / 3 = (1 + 1 + 1) / 3 = 1.0
    assert mse_loss(predictions, targets) == 1.0


def test_mse_loss_negative_differences():
    """Test MSE loss with negative differences."""
    predictions = [0.0, 0.0]
    targets = [1.0, 1.0]
    # ((0-1)^2 + (0-1)^2) / 2 = (1 + 1) / 2 = 1.0
    assert mse_loss(predictions, targets) == 1.0


def test_mse_loss_single_element():
    """Test MSE loss with single element."""
    predictions = [5.0]
    targets = [3.0]
    # (5-3)^2 / 1 = 4.0
    assert mse_loss(predictions, targets) == 4.0


def test_mse_loss_shape_mismatch():
    """Test MSE loss raises error when shapes don't match."""
    predictions = [1.0, 2.0, 3.0]
    targets = [1.0, 2.0]
    with pytest.raises(ValueError, match="Predictions and targets must have the same shape"):
        mse_loss(predictions, targets)


def test_mse_loss_empty_lists():
    """Test MSE loss with empty lists."""
    predictions = []
    targets = []
    with pytest.raises(ValueError, match="Predictions and targets must not be empty"):
        mse_loss(predictions, targets)
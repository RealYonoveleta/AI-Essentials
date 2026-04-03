from ai_essentials.math import get_shape
import math


def check_loss_inputs(predictions, targets):
    """Check if predictions and targets are valid for loss calculation."""

    if not predictions or not targets:
        raise ValueError("Predictions and targets must not be empty")

    if get_shape(predictions) != get_shape(targets):
        raise ValueError("Predictions and targets must have the same shape.")


def mse_loss(predictions, targets):
    """
    Mean Squared Error Loss.
    """
    check_loss_inputs(predictions, targets)

    return sum((p - t) ** 2 for p, t in zip(predictions, targets)) / len(predictions)


def cross_entropy(predictions, targets):
    """
    Binary Cross Entropy Loss.
    """
    check_loss_inputs(predictions, targets)
    eps = 1e-12
    return -sum(
        t * math.log(max(min(p, 1 - eps), eps)) + (1 - t) * math.log(max(min(1 - p, 1 - eps), eps))
        for p, t in zip(predictions, targets)
    ) / len(predictions)

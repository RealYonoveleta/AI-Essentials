from ai_essentials.value import Value, to_values


def check_loss_inputs(predictions, targets):
    """Check if predictions and targets are valid for loss calculation."""

    if not predictions or not targets:
        raise ValueError("Predictions and targets must not be empty")

    if len(predictions) != len(targets):
        raise ValueError(
            f"Predictions and targets must have the same length "
            f"(got {len(predictions)} and {len(targets)})."
        )


def mse_loss(predictions, targets):
    """
    Mean Squared Error Loss.
    predictions and targets: lists of Value objects or scalars
    """
    check_loss_inputs(predictions, targets)

    predictions = to_values(predictions)
    targets = to_values(targets)

    return sum(((p - t) ** 2 for p, t in zip(predictions, targets)), Value(0.0)) * (
        1 / len(predictions)
    )


def cross_entropy(predictions, targets):
    """
    Binary Cross Entropy Loss.
    predictions: list of Value objects or scalars (predicted probabilities in (0, 1))
    targets: list of Value objects or scalars (true labels: 0 or 1)
    """
    check_loss_inputs(predictions, targets)

    predictions = to_values(predictions)
    targets = to_values(targets)

    loss = sum(
        (t * p.log() + (1 - t) * (1 - p).log() for p, t in zip(predictions, targets)),
        Value(0.0),
    )
    return loss * (-1 / len(predictions))

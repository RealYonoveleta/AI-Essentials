def tanh(x):
    return x.tanh()


def relu(x):
    return x.relu()


ACTIVATIONS = {
    "tanh": tanh,
    "relu": relu,
}


def get_activation(name):
    if name is None:
        return None
    if name not in ACTIVATIONS:
        raise ValueError(
            f"Unknown activation '{name}'. Available: {list(ACTIVATIONS.keys())}"
        )
    return ACTIVATIONS[name]

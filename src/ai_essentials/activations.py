def tanh(x):
    return x.tanh()


def relu(x):
    return x.relu()


# (function, default_bias_init)
ACTIVATIONS = {
    'tanh': (tanh, 0.0),
    'relu': (relu, 0.01),
}


def get_activation(name):
    if name is None:
        return None, 0.0
    if name not in ACTIVATIONS:
        raise ValueError(
            f"Unknown activation '{name}'. Available: {list(ACTIVATIONS.keys())}"
        )
    return ACTIVATIONS[name]

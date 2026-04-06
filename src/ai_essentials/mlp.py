from ai_essentials.layer import Layer


def _parse_layer(l, n_args=3):
    """Normalize a layer spec to a tuple of length n_args, padding with None."""
    t = (l,) if isinstance(l, int) else tuple(l)
    if len(t) > n_args:
        raise ValueError(f"Layer spec {t} has too many arguments (max {n_args}).")
    return t + (None,) * (n_args - len(t))


class MLP:
    def __init__(self, num_inputs, layers):
        """
        layers: list of sizes or (size,) or (size, activation) or (size, activation, bias_init) tuples.
        Plain int uses Layer defaults for activation and bias.
        Examples:
            MLP(2, [4, 4, (1, None)])                  # hidden layers use defaults, linear output
            MLP(2, [(4, 'tanh'), (1, None)])            # explicit activation
            MLP(2, [(4, 'relu', 0.0), (1, None, 0.0)]) # explicit activation and bias
        """
        parsed = [_parse_layer(l) for l in layers]
        sizes = [num_inputs] + [size for size, _, _ in parsed]
        self.layers = [
            Layer(sizes[i], sizes[i + 1], activation=act, **({'bias_init': b} if b is not None else {}))
            for i, (_, act, b) in enumerate(parsed)
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

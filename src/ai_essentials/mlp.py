from ai_essentials.layer import Layer


class MLP:
    def __init__(self, num_inputs, layers):
        """
        layers: list of sizes or (size, activation) tuples.
        Plain int uses the Layer default activation.
        Examples:
            MLP(2, [4, 4, (1, None)])        # hidden layers use default, linear output
            MLP(2, [(4, 'tanh'), (1, None)])  # fully explicit
        """
        sizes = [num_inputs] + [l if isinstance(l, int) else l[0] for l in layers]
        self.layers = [
            Layer(sizes[i], sizes[i + 1]) if isinstance(layers[i], int)
            else Layer(sizes[i], sizes[i + 1], activation=layers[i][1])
            for i in range(len(layers))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

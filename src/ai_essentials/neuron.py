import random

from ai_essentials.value import Value, to_value
from ai_essentials.math import dot
from ai_essentials.activations import get_activation


class Neuron:
    def __init__(self, num_inputs, activation='relu', bias_init=None):
        scale = (2.0 / num_inputs) ** 0.5 if num_inputs > 0 else 1.0  # He initialization
        self.w = [Value(random.uniform(-scale, scale)) for _ in range(num_inputs)]
        self.activation, default_bias = get_activation(activation)
        self.b = Value(bias_init if bias_init is not None else default_bias)

    def __call__(self, x):
        x = [to_value(v) for v in x]
        z = dot(self.w, x) + self.b
        return self.activation(z) if self.activation else z

    def parameters(self):
        return self.w + [self.b]

import random

from ai_essentials.value import Value, to_value
from ai_essentials.math import dot
from ai_essentials.activations import get_activation


class Neuron:
    def __init__(self, num_inputs, activation='relu'):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(num_inputs)]
        self.b = Value(random.uniform(-1, 1))
        self.activation = get_activation(activation)

    def __call__(self, x):
        x = [to_value(v) for v in x]
        z = dot(self.w, x) + self.b
        return self.activation(z) if self.activation else z

    def parameters(self):
        return self.w + [self.b]

import random

from ai_essentials.value import Value, to_value
from ai_essentials.math import dot


class Neuron:
    def __init__(self, num_inputs):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(num_inputs)]
        self.b = Value(random.uniform(-1, 1))
        
    def __call__(self, x):
        x = [to_value(v) for v in x]
        z = dot(self.w, x) + self.b
        return z.tanh()

    def parameters(self):
        return self.w + [self.b]

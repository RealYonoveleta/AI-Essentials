from math import tanh
import random

from ai_essentials.math import dot


class Neuron:
    def __init__(self, num_inputs):
        self.w = [random.uniform(-1, 1) for _ in range(num_inputs)]
        self.b = random.uniform(-1, 1)
        
    def __call__(self, x):
        z = dot(self.w, x) + self.b
        return tanh(z)
        
        
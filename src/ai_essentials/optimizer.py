import math


class Optimizer:
    def __init__(self, parameters, lr=0.01):
        self.parameters = list(parameters)
        self.lr = lr

    def step(self):
        raise NotImplementedError("step() must be implemented by subclasses.")

    def zero_grad(self):
        for p in self.parameters:
            p.grad = 0.0

    def clip_grad(self, max_norm=1.0):
        total_norm = math.sqrt(sum(p.grad**2 for p in self.parameters))
        if total_norm > max_norm:
            scale = max_norm / total_norm
            for p in self.parameters:
                p.grad *= scale


class SGD(Optimizer):
    def step(self):
        for p in self.parameters:
            p.data -= self.lr * p.grad

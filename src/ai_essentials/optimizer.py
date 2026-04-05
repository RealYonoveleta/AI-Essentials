class Optimizer:
    def __init__(self, parameters, lr=0.01):
        self.parameters = list(parameters)
        self.lr = lr

    def step(self):
        raise NotImplementedError("step() must be implemented by subclasses.")

    def zero_grad(self):
        for p in self.parameters:
            p.grad = 0.0

class SGD(Optimizer):
    def step(self):
        for p in self.parameters:
            p.data -= self.lr * p.grad

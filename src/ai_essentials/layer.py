from ai_essentials.neuron import Neuron


class Layer:
    def __init__(self, num_inputs, num_neurons, neuron_class=Neuron, activation='relu'):
        self.neurons = [neuron_class(num_inputs, activation) for _ in range(num_neurons)]

    def __call__(self, x):
        return [neuron(x) for neuron in self.neurons]

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
            
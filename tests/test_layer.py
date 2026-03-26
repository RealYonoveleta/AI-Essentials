from ai_essentials.layer import Layer


class DummyNeuron:
    """A deterministic neuron for testing."""
    def __init__(self, num_inputs):
        self.num_inputs = num_inputs
        self.called_with = []

    def __call__(self, x):
        self.called_with.append(x)
        return sum(x)


def test_layer_creates_correct_number_of_neurons():
    """Layer should create num_neurons neurons."""
    l = Layer(3, 4, neuron_class=DummyNeuron)
    assert len(l.neurons) == 4


def test_layer_creates_neurons_with_correct_num_inputs():
    """Each neuron should be initialised with num_inputs."""
    l = Layer(5, 3, neuron_class=DummyNeuron)
    assert all(n.num_inputs == 5 for n in l.neurons)


def test_layer_neurons_are_correct_type():
    """Each neuron should be an instance of the provided neuron_class."""
    l = Layer(2, 3, neuron_class=DummyNeuron)
    assert all(isinstance(n, DummyNeuron) for n in l.neurons)


def test_layer_call_returns_one_output_per_neuron():
    """__call__ should return a list with one value per neuron."""
    l = Layer(2, 3, neuron_class=DummyNeuron)
    outputs = l([1, 2])
    assert len(outputs) == 3


def test_layer_call_passes_input_to_each_neuron():
    """__call__ should forward x to every neuron."""
    l = Layer(2, 3, neuron_class=DummyNeuron)
    x = [1, 2]
    l(x)
    assert all(n.called_with == [x] for n in l.neurons)


def test_layer_call_output_values():
    """__call__ output should equal what each neuron returns."""
    l = Layer(2, 3, neuron_class=DummyNeuron)
    outputs = l([1, 2])
    assert outputs == [3, 3, 3]


def test_layer_zero_neurons():
    """A layer with zero neurons should return an empty list."""
    l = Layer(2, 0, neuron_class=DummyNeuron)
    assert l.neurons == []
    assert l([1, 2]) == []


def test_layer_zero_inputs():
    """A layer with zero inputs per neuron should still work."""
    l = Layer(0, 2, neuron_class=DummyNeuron)
    outputs = l([])
    assert outputs == [0, 0]


def test_layer_default_neuron_class():
    """Layer should use Neuron by default without neuron_class argument."""
    from ai_essentials.neuron import Neuron
    l = Layer(3, 2)
    assert len(l.neurons) == 2
    assert all(isinstance(n, Neuron) for n in l.neurons)

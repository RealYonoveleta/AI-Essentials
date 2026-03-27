import pytest
from unittest.mock import MagicMock, patch
from ai_essentials.mlp import MLP


@pytest.fixture
def mock_layer():
    with patch("ai_essentials.mlp.Layer") as MockLayer:
        yield MockLayer


def test_mlp_initializes_layers_correctly(mock_layer):
    num_inputs = 4
    layer_sizes = [5, 3, 2]
    mlp = MLP(num_inputs, layer_sizes)
    # Should create len(layer_sizes) layers
    assert len(mlp.layers) == len(layer_sizes)
    # Check that Layer was called with correct sizes
    expected_calls = [
        (4, 5),
        (5, 3),
        (3, 2),
    ]
    actual_calls = [call.args for call in mock_layer.call_args_list]
    assert actual_calls == expected_calls


def test_mlp_call_passes_through_layers(mock_layer):
    # Setup: each mock layer returns its input + 1
    def layer_side_effect(x):
        return x + 1

    mock_layer.side_effect = lambda in_size, out_size: MagicMock(
        side_effect=layer_side_effect
    )
    mlp = MLP(2, [2, 2])
    result = mlp(0)
    # Should pass through 2 layers, so result = 0 + 1 + 1 = 2
    assert result == 2


def test_mlp_with_no_layers(mock_layer):
    mlp = MLP(3, [])
    x = [1, 2, 3]
    result = mlp(x)
    # No layers, should return input unchanged
    assert result == x
    assert mlp.layers == []


def test_mlp_layers_are_instances_of_layer(mock_layer):
    mlp = MLP(2, [3, 4])
    for layer in mlp.layers:
        assert isinstance(layer, mock_layer.return_value.__class__)


def test_mlp_repr_and_str(mock_layer):
    mlp = MLP(2, [3, 4])
    # Just check that repr and str do not raise
    repr_str = repr(mlp)
    str_str = str(mlp)
    assert isinstance(repr_str, str)
    assert isinstance(str_str, str)

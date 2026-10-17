"""Tests for the forward, gradient, and Adam behavior of masked weights."""
import numpy as np

from engine.gradcheck import numerical_gradient
from engine.tensor import Tensor
from nn.module import Parameter
from optim.adam import Adam


def test_fixed_mask_has_zero_forward_value_and_zero_gradient():
    weights = np.array([2.0, 3.0, 4.0])
    mask = np.array([1.0, 0.0, 1.0])
    parameter = Tensor(weights, requires_grad=True)

    effective_weights = parameter * mask
    loss = (effective_weights * effective_weights).sum()
    loss.backward()

    # The second connection is deleted in the forward pass and receives no gradient.
    numerical = numerical_gradient(
        lambda values: np.sum((values * mask) ** 2), weights
    )
    np.testing.assert_allclose(parameter.grad, numerical, atol=1e-7)
    assert effective_weights.data[1] == 0
    assert parameter.grad[1] == 0


def test_pruned_weight_stays_zero_and_revives_with_fresh_adam_state():
    parameter = Parameter([1.0, 2.0], prunable=True)
    optimizer = Adam([parameter])

    # Give both weights nonzero Adam history before pruning the first one.
    for _ in range(5):
        parameter.grad = np.array([2.0, 3.0])
        optimizer.step()

    optimizer.set_mask(parameter, [0, 1])

    # While masked, the weight gets zero gradient, stays zero, and has no stale state.
    for _ in range(3):
        optimizer.zero_grad()
        masked_loss = ((parameter * parameter.mask) * 2).sum()
        masked_loss.backward()

        assert parameter.grad[0] == 0
        optimizer.step()
        assert parameter.data[0] == 0
        for state_array in optimizer.state[parameter].values():
            assert state_array[0] == 0

    # Revival starts the first connection as a fresh weight at zero.
    optimizer.set_mask(parameter, [1, 1])
    fresh_parameter = Parameter([0.0])
    fresh_optimizer = Adam([fresh_parameter])

    parameter.grad = np.array([0.7, 0.2])
    fresh_parameter.grad = np.array([0.7])
    optimizer.step()
    fresh_optimizer.step()

    # Its first update must match a genuinely fresh Adam parameter.
    np.testing.assert_allclose(
        parameter.data[0], fresh_parameter.data[0], atol=1e-15
    )
    assert optimizer.state[parameter]["age"][0] == 1
    # The second connection was never pruned, so its history was preserved.
    assert optimizer.state[parameter]["age"][1] == 9


def test_forward_graph_keeps_the_mask_value_used_during_forward():
    parameter = Parameter([2.0, 3.0], prunable=True)

    # This graph was built with both mask entries active.
    output = (parameter * parameter.mask).sum()

    # A later mask change must not rewrite the derivative of the old graph.
    parameter.mask[0] = False
    output.backward()

    np.testing.assert_array_equal(parameter.grad, [1.0, 1.0])

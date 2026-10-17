"""Per-weight scores used to rank connections for pruning."""
import numpy as np


def magnitude(weights):
    """Simple baseline: a larger absolute weight is treated as more important."""
    return np.abs(weights)


def first_order_saliency(weights, gradients):
    """Estimate each weight's loss change when that weight is set to zero.

    First-order Taylor approximation:
      loss(weight + change) - loss(weight) ~= gradient * change

    Removing the weight means ``change = -weight``. The signed estimate is
    therefore ``-weight * gradient``; its absolute value is the importance score.
    A small score predicts a small local loss change, so it is a pruning candidate.
    This is an approximation, not a guarantee about the true loss after removal.

    Weights and gradients must come from the same forward/backward pass.
    """
    return np.abs(weights * gradients)


def update_saliency(previous, weights, gradients, beta=0.9):
    """Smooth batch saliency so one noisy batch has less influence.

    ``beta`` controls how much history is retained. With beta 0.9, the previous
    score contributes 90% and this batch contributes 10%.
    """
    if not 0 <= beta < 1:
        raise ValueError("EMA beta must be in [0,1)")
    batch_score = first_order_saliency(weights, gradients)
    return beta * previous + (1-beta) * batch_score

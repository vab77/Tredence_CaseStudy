"""Finite differences are verification, never the training gradient engine."""
import numpy as np

def numerical_gradient(function, values, epsilon=1e-6):
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    values = np.array(values, dtype=np.float64, copy=True)
    result = np.zeros_like(values)
    for index in np.ndindex(values.shape):
        plus, minus = values.copy(), values.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        result[index] = (function(plus) - function(minus)) / (2 * epsilon)
    return result

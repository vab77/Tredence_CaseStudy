"""Shape helpers used by local derivative rules."""
import numpy as np

def sum_to_shape(gradient, shape):
    """Undo NumPy broadcasting by summing repeated contributions."""
    result = np.asarray(gradient, dtype=np.float64)
    while result.ndim > len(shape):
        result = result.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and result.shape[axis] != 1:
            result = result.sum(axis=axis, keepdims=True)
    return result.reshape(shape)

"""Fused mean cross-entropy with stable log-sum-exp forward."""
import numpy as np
from engine.tensor import Tensor

def cross_entropy(logits, targets):
    logits = Tensor.as_tensor(logits)
    targets = np.asarray(targets)
    if logits.data.ndim != 2 or logits.shape[0] == 0:
        raise ValueError("Expected nonempty (batch, classes) logits")
    n, classes = logits.shape
    if targets.shape != (n,) or not np.issubdtype(targets.dtype, np.integer):
        raise ValueError("Targets must be an integer vector with one label per row")
    if np.any(targets < 0) or np.any(targets >= classes):
        raise ValueError("Target class out of bounds")
    shifted = logits.data - logits.data.max(axis=1, keepdims=True)
    exponential = np.exp(shifted)
    denominator = exponential.sum(axis=1, keepdims=True)
    loss = (np.log(denominator[:, 0]) - shifted[np.arange(n), targets]).mean()
    out = Tensor(loss, logits.requires_grad)
    if logits.requires_grad:
        derivative = exponential / denominator
        derivative[np.arange(n), targets] -= 1
        derivative /= n
        out.parents = (logits,)
        out._backward = lambda: logits._accumulate(out.grad * derivative)
    return out

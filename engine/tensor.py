"""A small eager reverse-mode autodiff engine used by the NumPy network.

Interview summary: each operation saves its inputs and a local derivative.
``backward`` walks those operations in reverse and adds gradient contributions
until they reach trainable leaf tensors.
"""
import numpy as np
from .functional import sum_to_shape

class Tensor:
    __array_priority__ = 1000

    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=np.float64, copy=True)
        self.requires_grad = bool(requires_grad)
        self.grad = None
        self.parents = ()
        self._backward = lambda: None
        self._consumed = False

    @property
    def shape(self):
        return self.data.shape

    def item(self):
        return self.data.item()

    @staticmethod
    def as_tensor(value):
        return value if isinstance(value, Tensor) else Tensor(value)

    def _accumulate(self, gradient):
        """Add a gradient contribution, undoing any NumPy broadcasting first."""
        if self.requires_grad:
            gradient = sum_to_shape(gradient, self.shape)
            self.grad = gradient.copy() if self.grad is None else self.grad + gradient

    def backward(self, gradient=None):
        """Propagate a seed gradient from this result back to its leaves."""
        if not self.requires_grad:
            raise ValueError("Cannot backward through a constant tensor")
        if gradient is None:
            if self.data.size != 1:
                raise ValueError("Non-scalar outputs require an explicit seed gradient")
            gradient = np.ones_like(self.data)
        gradient = np.asarray(gradient, dtype=np.float64)
        if gradient.shape != self.shape:
            raise ValueError("Seed gradient must have the output shape")
        # Record the graph in forward dependency order, without recursion.
        order, visited = [], set()
        stack = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
            elif node not in visited:
                visited.add(node)
                if node._consumed:
                    raise RuntimeError("Build a fresh graph for each backward call")
                stack.append((node, True))
                stack.extend((parent, False) for parent in reversed(node.parents))
        # Clear temporary results; parameter (leaf) gradients accumulate until
        # the optimizer calls zero_grad().
        for node in order:
            if node.parents:
                node.grad = None
        self._accumulate(gradient)
        for node in reversed(order):
            if node.grad is not None:
                node._backward()
            if node.parents:
                node._consumed = True

    def _binary(self, other, forward, da, db):
        """Apply an elementwise operation and record its two local derivatives."""
        other = Tensor.as_tensor(other)
        # Saved forward values protect derivatives from later array mutation.
        a, b = self.data.copy(), other.data.copy()
        out = Tensor(forward(a, b), self.requires_grad or other.requires_grad)
        if out.requires_grad:
            out.parents = (self, other)
            def backward():
                self._accumulate(da(out.grad, a, b))
                other._accumulate(db(out.grad, a, b))
            out._backward = backward
        return out

    def __add__(self, other):
        return self._binary(other, lambda a,b:a+b, lambda g,a,b:g, lambda g,a,b:g)

    __radd__ = __add__

    def __sub__(self, other):
        return self._binary(other, lambda a,b:a-b, lambda g,a,b:g, lambda g,a,b:-g)

    def __rsub__(self, other):
        return Tensor.as_tensor(other) - self

    def __mul__(self, other):
        return self._binary(other, lambda a,b:a*b, lambda g,a,b:g*b, lambda g,a,b:g*a)

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self._binary(other, lambda a,b:a/b, lambda g,a,b:g/b, lambda g,a,b:-g*a/(b*b))

    def __rtruediv__(self, other):
        return Tensor.as_tensor(other) / self

    def __neg__(self):
        return self * -1

    def __matmul__(self, other):
        other = Tensor.as_tensor(other)
        if self.data.ndim != 2 or other.data.ndim != 2:
            raise ValueError("This engine supports 2D matrix multiplication only")
        # If C = A @ B, then dA = dC @ B.T and dB = A.T @ dC.
        return self._binary(other, lambda a,b:a@b, lambda g,a,b:g@b.T, lambda g,a,b:a.T@g)

    def __rmatmul__(self, other):
        return Tensor.as_tensor(other) @ self

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), self.requires_grad)
        if self.requires_grad:
            axes = tuple(range(self.data.ndim)) if axis is None else ((axis,) if isinstance(axis, int) else tuple(axis))
            axes = tuple(a % self.data.ndim for a in axes) if self.data.ndim else ()
            out.parents = (self,)
            def backward():
                gradient = out.grad
                if not keepdims and axes:
                    gradient = np.expand_dims(gradient, axis=tuple(sorted(axes)))
                self._accumulate(np.broadcast_to(gradient, self.shape))
            out._backward = backward
        return out

    def mean(self, axis=None, keepdims=False):
        axes = tuple(range(self.data.ndim)) if axis is None else ((axis,) if isinstance(axis, int) else tuple(axis))
        count = int(np.prod([self.shape[a] for a in axes]))
        if count == 0:
            raise ValueError("Mean of an empty reduction is undefined")
        return self.sum(axis=axis, keepdims=keepdims) / count

    def relu(self):
        out = Tensor(np.maximum(self.data, 0), self.requires_grad)
        if self.requires_grad:
            positive = self.data > 0
            out.parents = (self,)
            out._backward = lambda: self._accumulate(out.grad * positive)
        return out

    def tanh(self):
        """Apply tanh element-wise and propagate its local derivative.

        Since d(tanh(x))/dx = 1 - tanh(x)^2, the forward output is reused
        during backward rather than recomputing tanh.
        """
        value = np.tanh(self.data)
        out = Tensor(value, self.requires_grad)
        if self.requires_grad:
            out.parents = (self,)
            out._backward = lambda: self._accumulate(out.grad * (1 - value*value))
        return out

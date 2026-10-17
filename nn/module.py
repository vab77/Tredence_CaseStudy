"""Parameter leaves persist; batch computation graphs do not."""
import numpy as np
from engine.tensor import Tensor

class Parameter(Tensor):
    def __init__(self, data, prunable=False):
        super().__init__(data, requires_grad=True)
        self.prunable = prunable
        self.mask = np.ones(self.shape, dtype=bool)

class Module:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, inputs):
        raise NotImplementedError

    def parameters(self):
        return []

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.grad = None

"""Composable MLP layers built exclusively from our Tensor operations."""
import numpy as np
from engine.tensor import Tensor
from .module import Module, Parameter

class Linear(Module):
    def __init__(self, in_features, out_features, rng, relu_gain=True):
        if in_features <= 0 or out_features <= 0:
            raise ValueError("Layer dimensions must be positive")
        variance = (2.0 if relu_gain else 1.0) / in_features
        self.weight = Parameter(rng.normal(0, np.sqrt(variance), (in_features, out_features)), prunable=True)
        self.bias = Parameter(np.zeros(out_features))

    def forward(self, inputs):
        inputs = Tensor.as_tensor(inputs)
        # The mask is a constant copied into this graph; dW = mask * dW_eff.
        return inputs @ (self.weight * self.weight.mask) + self.bias

    def parameters(self):
        return [self.weight, self.bias]

class ReLU(Module):
    def forward(self, inputs):
        return Tensor.as_tensor(inputs).relu()

class Tanh(Module):
    """Element-wise tanh activation for models built from Modules."""
    def forward(self, inputs):
        return Tensor.as_tensor(inputs).tanh()

class Sequential(Module):
    def __init__(self, *layers):
        self.layers = layers

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs

    def parameters(self):
        # Deduplicate intentionally shared parameters.
        return list(dict.fromkeys(p for layer in self.layers for p in layer.parameters()))

def make_mlp(dimensions, seed=0, hidden_activations=None):
    hidden_layer_count = len(dimensions) - 2
    if hidden_activations is None:
        hidden_activations = ["relu"] * hidden_layer_count
    if len(hidden_activations) != hidden_layer_count:
        raise ValueError("Provide one activation name per hidden layer")
    if any(name not in ("relu", "tanh") for name in hidden_activations):
        raise ValueError("Hidden activations must be 'relu' or 'tanh'")

    rng = np.random.default_rng(seed)
    layers = []
    for index, (inputs, outputs) in enumerate(zip(dimensions[:-1], dimensions[1:])):
        hidden = index < len(dimensions) - 2
        activation_name = hidden_activations[index] if hidden else None
        # He scaling is used before ReLU; tanh and logits use fan-in scaling.
        layers.append(Linear(inputs, outputs, rng, relu_gain=activation_name == "relu"))
        if hidden:
            layers.append(ReLU() if activation_name == "relu" else Tanh())
    return Sequential(*layers)

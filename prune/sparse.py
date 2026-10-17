"""Simple column-index sparse execution; fewer MACs need not mean lower latency."""
import numpy as np
from nn.layers import Linear, ReLU, Tanh

def export_sparse(model):
    result = []
    for layer in model.layers:
        if isinstance(layer, Linear):
            weight = layer.weight.data * layer.weight.mask
            columns = []
            for j in range(weight.shape[1]):
                indices = np.flatnonzero(layer.weight.mask[:, j])
                columns.append((indices, weight[indices, j].copy()))
            result.append({'kind': 'linear', 'columns': columns, 'bias': layer.bias.data.copy()})
        elif isinstance(layer, ReLU):
            result.append({'kind': 'relu'})
        elif isinstance(layer, Tanh):
            result.append({'kind': 'tanh'})
        else:
            raise TypeError("Unsupported sparse export layer")
    return result

def sparse_forward(exported_model, inputs):
    value = np.asarray(inputs, dtype=np.float64)
    for layer in exported_model:
        if layer['kind'] == 'relu':
            value = np.maximum(value, 0)
        elif layer['kind'] == 'tanh':
            value = np.tanh(value)
        else:
            out = np.broadcast_to(layer['bias'], (len(value), len(layer['columns']))).copy()
            for j, (indices, weights) in enumerate(layer['columns']):
                if len(indices):
                    out[:, j] += value[:, indices] @ weights
            value = out
    return value

def dense_forward(model, inputs):
    """Fair dense inference comparator: no training graph or backward closures."""
    value = np.asarray(inputs, dtype=np.float64)
    for layer in model.layers:
        if isinstance(layer, Linear):
            value = value @ (layer.weight.data * layer.weight.mask) + layer.bias.data
        elif isinstance(layer, ReLU):
            value = np.maximum(value, 0)
        elif isinstance(layer, Tanh):
            value = np.tanh(value)
        else:
            raise TypeError("Unsupported dense inference layer")
    return value

def weight_macs(exported_model):
    return sum(len(indices) for layer in exported_model if layer['kind']=='linear'
               for indices, _ in layer['columns'])

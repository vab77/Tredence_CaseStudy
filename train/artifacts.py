"""Portable inference checkpoints; not resumable optimizer checkpoints."""
import json
import numpy as np
from nn.layers import make_mlp, Linear, ReLU, Tanh

def save_checkpoint(path, model, dimensions):
    arrays = {'dimensions': np.asarray(dimensions, dtype=np.int64)}
    arrays['hidden_activations'] = np.asarray([
        'relu' if isinstance(layer, ReLU) else 'tanh'
        for layer in model.layers if isinstance(layer, (ReLU, Tanh))
    ])
    for i, layer in enumerate(l for l in model.layers if isinstance(l, Linear)):
        arrays[f'w{i}'] = layer.weight.data
        arrays[f'b{i}'] = layer.bias.data
        arrays[f'm{i}'] = layer.weight.mask
    np.savez_compressed(path, **arrays)

def load_checkpoint(path):
    with np.load(path, allow_pickle=False) as arrays:
        dimensions = arrays['dimensions'].tolist()
        hidden_activations = arrays['hidden_activations'].tolist() if 'hidden_activations' in arrays else None
        model = make_mlp(dimensions, hidden_activations=hidden_activations)
        for i, layer in enumerate(l for l in model.layers if isinstance(l, Linear)):
            for target, key in ((layer.weight.data,f'w{i}'),(layer.bias.data,f'b{i}'),(layer.weight.mask,f'm{i}')):
                if arrays[key].shape != target.shape:
                    raise ValueError("Checkpoint shape mismatch")
                target[...] = arrays[key]
    return model

def plot_history(history, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = [row['epoch'] for row in history]
    for prefix, label in [('training','Train'),('validation','Validation')]:
        axes[0].plot(epochs, [r[prefix+'_loss'] for r in history], label=label)
        axes[1].plot(epochs, [r[prefix+'_accuracy'] for r in history], label=label)
    for ax, title in zip(axes, ['Cross-entropy loss','Accuracy']):
        ax.set(xlabel='Epoch', title=title)
        ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)

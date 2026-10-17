"""Only dataset loading uses sklearn; splitting and preprocessing use NumPy."""
import numpy as np

def load_dataset(config):
    if config['dataset'] != 'digits':
        raise ValueError("Only the digits dataset is supported")
    from sklearn.datasets import load_digits
    dataset = load_digits()
    x, y = dataset.data.astype(np.float64) / 16.0, dataset.target
    fractions = [config[k] for k in ('train_fraction','validation_fraction','test_fraction')]
    if any(f <= 0 for f in fractions) or not np.isclose(sum(fractions), 1):
        raise ValueError("Split fractions must be positive and sum to one")
    rng = np.random.default_rng(config['split_seed'])
    groups = [[], [], []]
    for label in np.unique(y):
        indices = rng.permutation(np.flatnonzero(y == label))
        a = int(len(indices)*fractions[0])
        b = a + int(len(indices)*fractions[1])
        for destination, part in zip(groups, (indices[:a], indices[a:b], indices[b:])):
            destination.extend(part.tolist())
    result = {}
    for name, indices in zip(('train','validation','test'), groups):
        indices = rng.permutation(indices)
        result[name] = (x[indices], y[indices])
    return result

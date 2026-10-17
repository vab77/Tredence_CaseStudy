"""One reproducible dense or pruning run."""
import argparse
import copy
import json
import platform
import time
from pathlib import Path
import numpy as np
from nn.layers import make_mlp
from optim.adam import Adam
from prune.controller import PruningController
from .data import load_dataset
from .loop import train, evaluate
from .artifacts import save_checkpoint, plot_history

def run_experiment(config, method='dense', sparsity=0.9, seed=0, output=None):
    if method not in ('dense','magnitude','saliency') or not 0 <= sparsity <= 1:
        raise ValueError("Invalid method or target sparsity")
    config = copy.deepcopy(config)
    config.update(method=method, target_sparsity=0.0 if method=='dense' else sparsity, seed=seed)
    output = Path(output or Path(config['output_dir']) / f'{method}_s{config["target_sparsity"]:g}_seed{seed}')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite existing experiment: {output}')
    output.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(config)
    dimensions = [config['input_dim'], *config['hidden_dims'], config['output_dim']]
    model = make_mlp(dimensions, seed, config.get('hidden_activations'))
    optimizer = Adam(model.parameters(), lr=config['learning_rate'], **config['adam'])
    started = time.perf_counter()
    history = train(model, optimizer, dataset, config,
                    None if method=='dense' else PruningController())
    elapsed = time.perf_counter()-started
    test = evaluate(model, *dataset['test'])
    last = history[-1]
    result = {'method': method, 'seed': seed, 'target_sparsity': config['target_sparsity'],
              'achieved_sparsity': last['achieved_sparsity'], 'active_weights': last['active_weights'],
              'test_accuracy': test['accuracy'], 'test_loss': test['loss'], 'training_seconds': elapsed,
              'epochs': config['epochs'], 'split_sizes': {k:len(v[0]) for k,v in dataset.items()},
              'python': platform.python_version(), 'numpy': np.__version__, 'platform': platform.platform()}
    for name, value in [('config',config), ('history',history), ('summary',result)]:
        (output / f'{name}.json').write_text(json.dumps(value, indent=2)+'\n')
    save_checkpoint(output/'model.npz', model, dimensions)
    plot_history(history, output/'learning_curve.png')
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/default.json')
    parser.add_argument('--method', choices=['dense','magnitude','saliency'], default='dense')
    parser.add_argument('--sparsity', type=float, default=.9)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--output')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.epochs is not None:
        config['epochs'] = args.epochs
    result = run_experiment(config, args.method, args.sparsity, args.seed, args.output)
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()

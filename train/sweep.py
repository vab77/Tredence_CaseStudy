"""Run matched seeds and produce raw summaries and sparsity-accuracy curves."""
import argparse
import json
from pathlib import Path
import numpy as np
from .run import run_experiment

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/default.json')
    parser.add_argument('--output',default='results/sweep')
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text()); output=Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose an empty output folder to preserve earlier evidence')
    output.mkdir(parents=True,exist_ok=True)
    results=[]
    for seed in config['seeds']:
        for method,target in [('dense',0.0)]+[(m,s) for m in ('magnitude','saliency') for s in config['sparsity_targets'] if s>0]:
            record=run_experiment(config,method,target,seed,output/f'{method}_s{target:g}_seed{seed}')
            results.append(record)
            (output/'raw_results.json').write_text(json.dumps(results,indent=2)+'\n')
            print(f'{method} seed={seed} target={target:g} accuracy={record["test_accuracy"]:.4f}',flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4))
    for method in ('magnitude','saliency'):
        points=[]
        for target in sorted(set([0.0,*config['sparsity_targets']])):
            subset=[r for r in results if r['method']==('dense' if target==0 else method) and r['target_sparsity']==target]
            if not subset:continue
            accuracy=[r['test_accuracy'] for r in subset]
            points.append((np.mean([r['achieved_sparsity'] for r in subset]),np.mean(accuracy),np.std(accuracy,ddof=1) if len(accuracy)>1 else 0))
        ax.errorbar([p[0] for p in points],[p[1] for p in points],yerr=[p[2] for p in points],marker='o',capsize=3,label=method)
    ax.set(xlabel='Achieved sparsity',ylabel='Test accuracy',title='Mean accuracy ± sample SD across seeds')
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(output/'pareto.png',dpi=150); plt.close(fig)
    paired=[]
    for target in sorted(set(config['sparsity_targets'])):
        if target==0:continue
        differences=[]
        for seed in config['seeds']:
            values={r['method']:r['test_accuracy'] for r in results if r['seed']==seed and r['target_sparsity']==target}
            differences.append(values['saliency']-values['magnitude'])
        paired.append({'target_sparsity':target,'seed_order':config['seeds'],'paired_accuracy_differences':differences,
                       'mean_difference':float(np.mean(differences)),
                       'sample_sd':float(np.std(differences,ddof=1)) if len(differences)>1 else None})
    (output/'paired_comparison.json').write_text(json.dumps(paired,indent=2)+'\n')

if __name__=='__main__':
    main()

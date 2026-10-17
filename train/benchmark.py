"""Measure sparse and pre-masked dense inference, with no autodiff overhead."""
import argparse
import json
import platform
import time
from pathlib import Path
import numpy as np
from nn.layers import Linear, ReLU, Tanh
from prune.sparse import export_sparse, sparse_forward, weight_macs
from .artifacts import load_checkpoint

def benchmark(model, batch_size=32, repeats=30):
    if batch_size < 1 or repeats < 1:
        raise ValueError("Batch size and repetitions must be positive")
    layers = [layer for layer in model.layers if isinstance(layer, Linear)]
    x = np.random.default_rng(123).normal(size=(batch_size,layers[0].weight.shape[0]))
    started = time.perf_counter()
    sparse = export_sparse(model)
    export_seconds = time.perf_counter()-started
    # Pre-mask once: dense reference does not pay per-call masking or graph costs.
    dense = [(layer.weight.data*layer.weight.mask, layer.bias.data.copy()) for layer in layers]
    def dense_run():
        out = x
        linear_index = 0
        for layer in model.layers:
            if isinstance(layer, Linear):
                weight, bias = dense[linear_index]
                out = out @ weight + bias
                linear_index += 1
            elif isinstance(layer, ReLU):
                out = np.maximum(out, 0)
            elif isinstance(layer, Tanh):
                out = np.tanh(out)
            else:
                raise TypeError("Unsupported dense benchmark layer")
        return out
    def sparse_run():
        return sparse_forward(sparse,x)
    np.testing.assert_allclose(dense_run(), sparse_run(), rtol=1e-10, atol=1e-10)
    for _ in range(5):
        dense_run(); sparse_run()
    timings = {'dense': [], 'sparse': []}
    for repeat in range(repeats):
        order = [('dense',dense_run),('sparse',sparse_run)]
        if repeat%2:
            order.reverse()
        for name, function in order:
            started = time.perf_counter_ns(); function()
            timings[name].append((time.perf_counter_ns()-started)/1e6)
    dense_macs = sum(w.size for w,_ in dense)
    sparse_bytes = sum(layer['bias'].nbytes + sum(i.nbytes+w.nbytes for i,w in layer['columns'])
                       for layer in sparse if layer['kind']=='linear')
    return {'batch_size': batch_size, 'repeats': repeats, 'dense_weight_macs_per_example':dense_macs,
            'sparse_weight_macs_per_example':weight_macs(sparse), 'dense_array_bytes':sum(w.nbytes+b.nbytes for w,b in dense),
            'sparse_array_bytes_including_indices':sparse_bytes, 'export_seconds':export_seconds,
            'median_latency_ms':{k:float(np.median(v)) for k,v in timings.items()},
            'raw_latency_ms':timings, 'platform':platform.platform(),
            'caveat':'Array payload bytes exclude Python object overhead. MACs exclude bias and activations. Latency is hardware dependent.'}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--repeats',type=int,default=30)
    parser.add_argument('--output')
    args = parser.parse_args()
    result = benchmark(load_checkpoint(args.checkpoint), args.batch_size, args.repeats)
    text = json.dumps(result,indent=2)+'\n'
    if args.output:
        Path(args.output).write_text(text)
    print(text)

if __name__ == '__main__':
    main()

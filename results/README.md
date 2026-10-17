# Experiment outputs

`basic_*` folders contain single-seed 30-epoch architecture validation runs.
They are useful for inspecting individual histories and checkpoints, but do not
establish method superiority or seed robustness.

`part2_training/dense_seed0/` contains the reproduced 50-epoch dense baseline
training run on scikit-learn's 8×8 handwritten Digits dataset, including its
train/validation loss and accuracy learning-curve plot, per-epoch history,
summary and inference checkpoint. It reached 96.77% test accuracy on the fixed
test split.

`part3_pruning/` contains the matching 50-epoch 90%-target runs for saliency
pruning and magnitude pruning. Their `learning_curve.png` files plot train and
validation loss and accuracy; both ended with 646 of 6,464 weights active.
Saliency reached 96.42% test accuracy and magnitude reached 95.34% on the same
fixed split. Each run folder includes the raw epoch history, summary, config and
inference checkpoint.

`part4_evidence/` contains the regenerated default matched sweep: 27 runs across
three seeds, a dense baseline, and magnitude/saliency pruning at 50%, 75%, 90%,
and 95% target sparsity. The MLP uses ReLU then tanh in its hidden layers. The
upload-sized evidence keeps the 27 per-run configurations and raw epoch
histories, plus the aggregate Pareto plot, raw results, paired method comparison,
and sparse inference benchmark. Repeated per-run summaries, curves, and model
checkpoints are omitted from this sweep folder; the three representative curves
and their complete run artifacts are in `part2_training/` and `part3_pruning/`.
The seed-0 90% saliency checkpoint is retained alongside its benchmark. The
canonical `part4_evidence/` folder contains 59 files, below GitHub's 100-file
web-upload batch limit.
See [`../VALIDATION.md`](../VALIDATION.md) for the quantitative claim, caveats,
and reproduction commands.

The sparse-aware path reduces the measured number of weight MACs, but the current
Python implementation is slower than dense inference on the measured machine.
The saved benchmark reports both facts so sparse operation counts are not
presented as a latency speedup.

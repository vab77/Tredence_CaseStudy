# Design contracts and trade-offs

## Reverse-mode engine

Tensor stores a float64 array, optional gradient, parents and a local backward
closure. Operations save their forward operand values. Iterative postorder graph
traversal followed by reversed execution applies the chain rule without recursion
limits. Parent contributions accumulate; leaf gradients persist across fresh graphs
until zero_grad clears them. A recorded operation graph supports one backward call;
reusing a consumed graph raises an error. Matmul supports 2D operands only.

`sum_to_shape` reverses broadcasting by summing surplus leading dimensions and
axes whose original size was one. Reduction backward reinserts reduced axes.
The engine implements ReLU and tanh. ReLU chooses derivative zero at zero;
finite differences avoid that kink. For tanh, the forward value is `tanh(x)` and
the local derivative is `1 - tanh(x)^2`; the implementation reuses the forward
value in backward. `tests/test_gradients.py` checks both activations against
finite differences. Cross-entropy uses shifted log-sum-exp; derivative is
(softmax - one_hot)/batch.

## Layers and initialization

Linear computes X @ (W * mask) + b. Mask multiplication is recorded in the graph
as multiplication by a constant. The first hidden layer uses ReLU and weights
with variance 2/fan-in. The second hidden layer uses tanh and fan-in variance
1/fan-in; output weights also use fan-in variance 1/fan-in. Biases start at zero.
Each batch creates a fresh graph, while Parameter leaves and optimizer state
persist. The default activation choices are recorded in `configs/default.json`.

## What gradient does the engine assign to a masked weight, and why?

For fixed M, W_eff=M*W and dL/dW=M*dL/dW_eff. A masked parameter gradient is zero,
although sensitivity to the effective weight can be nonzero. The optimizer uses
explicit masks, skips inactive entries and clears inactive moments/ages. The only
supported lifecycle transition API is Adam.set_mask; direct mask mutation during
training is unsupported. Transitioned weights initialize at zero, including revival.
A revived entry starts with zero moments and age zero, then uses age one for its
first new update. Per-entry ages avoid applying an old global Adam bias correction
to freshly reset moments. Active entries retain their independent histories.

Test evidence covers fixed-mask finite differences, exact zero forward contribution,
zero gradients, frozen values, cleared state and equivalence to a fresh Adam entry
after revival. Mask changes occur only after backward and update complete. Automatic
regrowth scoring is outside this implementation; masked parameter gradients cannot
provide that signal on their own.

## How is saliency derived, and why estimate removal loss change?

Taylor expansion gives L(w+delta) ≈ L(w)+g*delta. Deleting one weight has delta=-w,
so signed delta_L ≈ -w*g. Ranking by abs(w*g) prioritizes small predicted change
magnitude. It does not prove loss will increase or guarantee an advantage over
magnitude pruning. Scores are an EMA of absolute products measured before each
optimizer update. Near stationary points gradients can be small; mini-batch noise,
large deletion perturbations and interactions limit the approximation. No Hessian
or second-order correction is claimed.

## Schedule and budget

Warm-up: first 20% of epochs. Cubic ramp over the next 60%:
s(u)=s_final*(1-(1-u)^3). Remaining epochs fine-tune a fixed mask.
Global stable ranking keeps at most floor((1-s)*P) eligible weights, with deterministic
index ordering for ties. Biases are excluded. Pruning is irreversible in the controller;
requesting a looser budget does not revive entries. No per-layer minimum is imposed,
so extreme global pruning may isolate neurons or degrade a layer.

## Sparse cost

Training computes dense masked matmuls. Sparse export stores input indices and
values for each output neuron; sparse inference evaluates only selected inputs.
The dense benchmark pre-masks weights once and avoids graph construction. Both
paths must agree numerically before latency is measured. Timing uses warm-up,
alternating execution order and repeated samples. MAC counts include active weight
contributions only. Array storage includes sparse indices, excluding Python object
overhead. Latency, MAC reduction and storage are separate facts.

## Experiment design

One fixed stratified train/validation/test split, controlled seed-specific model
initialization and minibatch order. Default sweep: three dense runs plus two methods
× four nonzero sparsities × three seeds = 27 runs. Both methods use the same schedule,
epoch budget, optimizer and architecture. Plot mean accuracy with sample standard
deviation; save paired seed differences. Three seeds provide limited evidence.
The separate one-seed architecture validation alone does not support a superiority claim; see `VALIDATION.md` for the limited matched three-seed comparison.

## Where are the bottlenecks, and how could I optimize them?

Float64, saved operand copies, a Python object per operation, closures, graph
traversal and scalar per-entry Adam ages trade memory/speed for clarity. Sparse
inference loops over output neurons in Python and can lose to optimized dense BLAS.
Possible improvements: float32 with revised tolerances, fewer copies with mutation
version checks, fused operations, compact graph storage and compiled sparse kernels.
2D matmul, sequential Linear/ReLU/Tanh sparse export, no training checkpoint resume,
no automatic regrowth, no GPU and no production inference service are intentional
scope limits.

## How would I serve this in a multi-tenant inference service?

Train offline; publish immutable versioned weights, masks and shape metadata.
Validate export equivalence and latency before rollout. Route tenants under explicit
quotas and admission control, batch compatible model versions, isolate tenant data,
and cache loaded immutable models under a memory budget. Choose dense or supported
sparse kernels from measured workload behavior. Track model version, latency, errors
and drift with privacy-preserving telemetry. Canary releases and retain rollback
artifacts. Do not prune shared serving weights in response to individual requests.


This is a proposed deployment design; this repository does not implement the
registry, gateway, tenancy controls, batching service, telemetry pipeline, or
production rollout machinery.


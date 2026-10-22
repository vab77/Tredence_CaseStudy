# Understanding the implementation

## 1. The complete path


```text
Digits loader → NumPy split / normalization → mini-batch
  → Sequential → Linear → ReLU → Linear → Tanh → Linear
  → logits → stable cross-entropy → scalar loss
  → Tensor.backward → parameter gradients
  → saliency accumulation → Adam.step
  → epoch-end pruning → next batch builds a fresh graph
```

Three separate responsibilities matter: the engine differentiates a mathematical
function; Adam updates parameters; the pruning controller changes which parameters
are allowed to participate. Neither the engine nor the optimizer chooses which
connections are important.

## 2. A Tensor is a value plus its calculation history

`data` holds a NumPy array. `grad` holds dLoss/dTensor. `parents` identifies earlier
inputs. `_backward` implements one local derivative rule. `requires_grad` lets the
engine distinguish constants from trainable values.

The engine has two element-wise activations: ReLU and tanh. For tanh, if the
forward value is `y=tanh(x)`, backward multiplies the incoming gradient by
`1-y*y`. The trained MLP uses ReLU in its first hidden layer and `nn.layers.Tanh`
in its second. `tests/test_gradients.py` checks the engine's tanh derivative
against finite differences.

For `c=a*b`, forward saves a and b. During backward, incoming gradient g contributes
g*b to a and g*a to b. Repeated parents are allowed: x*x contributes twice to x.
A traversal visits each node once but each local rule adds every derivative path.

Example: x=2, a=x*x, loss=a+x. The derivative is 2*x+1=5, not 1 or 4. See the shared
parent test. A fresh forward graph is built for each batch; Parameter leaves persist.

## 3. Why broadcasting requires a sum

A batch has shape (N,D). Adding bias (D,) uses the same bias N times. Therefore
its gradient sums N contributions. `sum_to_shape` generalizes this: remove extra
leading axes and sum axes expanded from size one.

## 4. How a layer works

A Linear layer owns W shaped (in_features,out_features), b shaped (out_features,),
and a binary mask matching W. It computes X@(W*mask)+b. The supplied MLP applies
ReLU after the first hidden linear layer and tanh after the second. The final
layer returns raw class scores.

For 64 → 64 → 32 → 10, the weight counts are 4096 + 2048 + 320 = 6464.
Bias counts are 64 + 32 + 10 = 106. Only weights are included in pruning budgets.

## 5. Why the loss is stable

Large logits can overflow exp. Subtract the row maximum before exponentiating.
For correct class y, loss=log(sum(exp(z-max(z))))-(z_y-max(z)). The common shift
cancels mathematically while keeping exponentials bounded. Backward uses
(softmax(z)-one_hot(y))/batch_size. This derivative is implemented explicitly.

## 6. What backward does

Build postorder iteratively, reverse it, and run each local backward closure.
By the time a node sends gradients to its parents, its downstream contributions
have accumulated. `_accumulate` also reduces broadcast gradients to parent shape.
Constant parents ignore gradients. Reusing a consumed operation graph raises an
error rather than silently double-counting stale intermediates.

## 7. Adam is independent of autodiff

The engine only fills parameter.grad. Adam consumes those arrays to maintain an
EMA of gradients m and squared gradients v. Bias correction compensates for their
zero initialization. An epsilon prevents division by zero. Every active entry
tracks its number of updates, enabling correctly restarted state after revival.

## 8. The masking trap

Forward uses W_eff=M*W. The chain rule gives dL/dW=M*dL/dW_eff. When M=0, the
underlying parameter gradient is zero. But old optimizer moments could still move
stored values. Adam therefore freezes inactive entries and clears their state.

Use `optimizer.set_mask(parameter,new_mask)` between steps. It resets transitioned
values and optimizer moments/ages. A revived entry begins at zero and is treated
as fresh, while other entries keep their history. This is a deliberate policy,
not the only possible policy. The test compares revival against a new Adam instance.

## 9. What one training batch does

1. Clear old parameter gradients; do not clear Adam moments.
2. Forward the input batch through the MLP.
3. Calculate scalar cross-entropy.
4. Backward through the graph into parameter leaves.
5. Update saliency EMA using the current weights and their corresponding gradients.
6. Apply Adam updates to active entries only.
7. At the configured epoch boundary, prune based on the scheduled budget.
8. Measure train/validation loss and accuracy without constructing an inference graph.

Evaluation does not update weights. Final test evaluation happens after training.

## 10. Why saliency is only a heuristic

Removing w means delta=-w. A first-order loss approximation predicts delta_L≈-w*g.
The implemented absolute score captures magnitude of predicted change. EMA reduces
batch noise. The approximation may fail near stationary points or when many weights
are deleted together. Magnitude is a required comparator, not an assumed loser.

## 11. Sparse execution and experiments

A zero in a dense matrix does not tell BLAS to skip multiplication. Export creates
per-output lists of active input indices and weights. The sparse forward path uses
those lists and supports both ReLU and tanh activations. It does fewer weight
MACs, but Python/indexing overhead may make it slower. The benchmark compares
equivalent results, raw latency samples, operation
counts and array payload bytes. It does not infer speedup from sparsity.



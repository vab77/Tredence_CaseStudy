# Correctness tests

These are executable tests, not skipped placeholders. Run `uv run pytest -q`.
They cover finite-difference derivatives for arithmetic, reductions, matrix
multiplication, ReLU and tanh; graph traversal, broadcasting, stable loss, Adam
reference updates, mask/revival state, budgets, sparse equivalence, checkpoint
roundtrip and actual learning on a separable task.

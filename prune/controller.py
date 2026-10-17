"""Remove the lowest-scoring weights until the global sparsity budget is met.

Pruning is irreversible here: the controller does not automatically regrow weights.
"""
import numpy as np

class PruningController:
    def update(self, model, optimizer, scores, sparsity):
        if not 0 <= sparsity <= 1:
            raise ValueError("Sparsity must be in [0,1]")
        parameters = [parameter for parameter in model.parameters() if parameter.prunable]
        if not parameters:
            raise ValueError("No prunable parameters")
        # Flatten all layers so a single global budget is shared between them.
        active_mask = np.concatenate([parameter.mask.ravel() for parameter in parameters])
        all_scores = np.concatenate([
            np.asarray(scores[parameter]).ravel() for parameter in parameters
        ])
        if all_scores.shape != active_mask.shape or not np.isfinite(all_scores).all():
            raise ValueError("Scores must match parameters and be finite")

        total_weights = active_mask.size
        keep_count = int(np.floor((1 - sparsity) * total_weights + 1e-12))
        active_indices = np.flatnonzero(active_mask)

        # Sort active weights from highest score (keep) to lowest (prune).
        # Stable ordering makes equal-score choices repeatable.
        active_scores = all_scores[active_indices]
        order = np.argsort(-active_scores, kind="stable")
        ranked_active_indices = active_indices[order]

        # Start with everything pruned, then turn on only the budgeted survivors.
        new_active_mask = np.zeros_like(active_mask)
        number_to_keep = min(keep_count, len(ranked_active_indices))
        kept_indices = ranked_active_indices[:number_to_keep]
        new_active_mask[kept_indices] = True

        # Restore each flattened part to the original parameter's shape.
        offset = 0
        for parameter in parameters:
            size = parameter.data.size
            parameter_mask = new_active_mask[offset:offset + size].reshape(parameter.shape)
            optimizer.set_mask(parameter, parameter_mask)
            offset += size

        return int(new_active_mask.sum())

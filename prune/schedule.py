"""Cubic target schedule with a warm-up and fixed-mask fine-tuning phase."""


def target_sparsity(step, start_step, end_step, final_sparsity):
    """Return the requested sparsity for this epoch.

    Before ``start_step`` the target is zero. During the ramp it rises smoothly
    to ``final_sparsity``. After ``end_step`` it stays at that final target.
    """
    if end_step <= start_step or not 0 <= final_sparsity <= 1:
        raise ValueError("Invalid pruning interval or target")

    progress = (step - start_step) / (end_step - start_step)
    progress = min(1.0, max(0.0, progress))

    # This cubic reaches much of the target early, then eases into the endpoint.
    return final_sparsity * (1 - (1 - progress) ** 3)

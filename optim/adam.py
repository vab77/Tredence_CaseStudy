"""Adam optimizer with safe per-weight masking and revival."""
import numpy as np


class Adam:
    def __init__(self, parameters, lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8):
        if lr <= 0 or epsilon <= 0 or not (0 <= beta1 < 1 and 0 <= beta2 < 1):
            raise ValueError("Invalid Adam hyperparameters")

        # Keep each parameter once, even if multiple layers share it.
        self.parameters = list(dict.fromkeys(parameters))
        self.lr, self.beta1, self.beta2, self.epsilon = lr, beta1, beta2, epsilon
        self.state = {}
        for parameter in self.parameters:
            self.state[parameter] = {
                "m": np.zeros_like(parameter.data),
                "v": np.zeros_like(parameter.data),
                # Each weight needs its own age because a pruned weight can revive
                # after the other weights have already had many Adam updates.
                "age": np.zeros(parameter.shape, dtype=np.int64),
            }

    def zero_grad(self):
        for p in self.parameters:
            p.grad = None

    def reset_entries(self, parameter, entries):
        """Clear moments and age for selected weights."""
        state = self.state[parameter]
        state["m"][entries] = 0
        state["v"][entries] = 0
        state["age"][entries] = 0

    def set_mask(self, parameter, mask):
        """Call between training steps. Revived entries start at zero with fresh state."""
        if not parameter.prunable:
            raise ValueError("Only prunable parameters may change masks")
        new_mask = np.asarray(mask)
        if (new_mask.shape != parameter.shape
                or not np.all((new_mask == 0) | (new_mask == 1))):
            raise ValueError("Mask must be binary and match parameter shape")

        new_mask = new_mask.astype(bool)
        mask_changed = new_mask != parameter.mask
        inactive = ~new_mask
        # Reset both newly pruned and newly revived weights. A revived weight
        # starts from zero with fresh Adam moments and a fresh bias-correction age.
        reset = mask_changed | inactive
        parameter.data[reset] = 0
        self.reset_entries(parameter, reset)
        if parameter.grad is not None:
            parameter.grad[reset] = 0
        parameter.mask = new_mask.copy()

    def step(self):
        # Validate before mutation to avoid a partial update on nonfinite gradients.
        for parameter in self.parameters:
            if parameter.grad is None:
                continue
            if (parameter.grad.shape != parameter.shape
                    or not np.isfinite(parameter.grad).all()):
                raise FloatingPointError("Invalid gradient; Adam update aborted")

        for parameter in self.parameters:
            active = parameter.mask
            inactive = ~active

            # Keep the pruning invariant true even if a caller changed stored data.
            parameter.data[inactive] = 0
            self.reset_entries(parameter, inactive)
            if parameter.grad is None:
                continue

            state = self.state[parameter]
            m, v, age = state['m'], state['v'], state['age']
            age[active] += 1

            # Adam's first and second gradient averages, updated for active weights only.
            g = parameter.grad[active]
            m[active] = self.beta1 * m[active] + (1-self.beta1) * g
            v[active] = self.beta2 * v[active] + (1-self.beta2) * g*g

            # Correct the early bias in both moving averages, per weight age.
            corrected_m = m[active] / (1-self.beta1**age[active])
            corrected_v = v[active] / (1-self.beta2**age[active])
            parameter.data[active] -= (
                self.lr * corrected_m / (np.sqrt(corrected_v) + self.epsilon)
            )
            if not np.isfinite(parameter.data).all():
                raise FloatingPointError("Nonfinite parameter after Adam update")

"""Schedule indexed by completed optimizer updates, independent of smoke-test stop."""

import math


def learning_rate_factor(step, warmup_steps, total_steps, min_factor=0.1):
    if step < 0 or not 0 <= warmup_steps < total_steps or not 0 <= min_factor <= 1:
        raise ValueError("Invalid cosine schedule configuration")
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = min(1.0, (step - warmup_steps) / (total_steps - warmup_steps))
    return min_factor + (1 - min_factor) * 0.5 * (1 + math.cos(math.pi * progress))

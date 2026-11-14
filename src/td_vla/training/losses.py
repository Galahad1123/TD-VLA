"""Element-normalized masked losses with optional DDP global normalization."""

from dataclasses import dataclass
import math

import torch
import torch.distributed as dist
from torch.nn import functional as F


@dataclass(frozen=True)
class LossWeights:
    action: float = 1.0
    track: float = 0.1
    stage: float = 0.1

    def __post_init__(self):
        if any(not math.isfinite(w) or w < 0 for w in (self.action, self.track, self.stage)):
            raise ValueError("Loss weights must be finite and nonnegative")
        if self.action <= 0:
            raise ValueError("Action must have positive weight")


def normalized_sum(values, *, distributed=False):
    count = values.new_tensor(values.numel())
    world = 1
    if distributed:
        world = dist.get_world_size()
        dist.all_reduce(count)
    # DDP averages gradients; compensate so normalization is over global valid elements.
    return values.sum() * world / count.clamp_min(1)


def masked_regression(prediction, target, valid, *, smooth=False, distributed=False):
    if prediction.shape != target.shape or valid.shape != target.shape[:-1]:
        raise ValueError("Regression target/mask shapes do not match prediction")
    if valid.dtype != torch.bool:
        raise ValueError("Validity masks must be boolean")
    # Select BEFORE arithmetic: masked NaNs must not poison loss or gradients.
    pred, truth = prediction.float()[valid], target.float()[valid]
    if not bool(torch.isfinite(truth).all()):
        raise ValueError("Nonfinite valid regression target")
    error = F.smooth_l1_loss(pred, truth, reduction="none") if smooth else (pred - truth).abs()
    return normalized_sum(error, distributed=distributed)


def multitask_loss(outputs, targets, weights=LossWeights(), *, distributed=False):
    result = {
        "action": masked_regression(
            outputs["actions"], targets["actions"], targets["action_valid"], distributed=distributed
        )
    }
    if "track" in outputs:
        result["track"] = masked_regression(
            outputs["track"],
            targets["track"],
            targets["track_valid"],
            smooth=True,
            distributed=distributed,
        )
        label = targets["stage"]
        valid = targets["stage_valid"] & (label != -1)
        if label.shape != outputs["stage_logits"].shape[:1] or valid.dtype != torch.bool:
            raise ValueError("Invalid stage target shape/mask")
        if bool(((label[valid] < 0) | (label[valid] >= 5)).any()):
            raise ValueError("Valid stage label outside five classes")
        ce = F.cross_entropy(outputs["stage_logits"].float()[valid], label[valid], reduction="none")
        result["stage"] = normalized_sum(ce, distributed=distributed)
    result["total"] = sum(getattr(weights, name) * loss for name, loss in result.items())
    return result

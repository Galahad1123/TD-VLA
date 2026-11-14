"""Deterministic episode-stratified validation; explicitly not full-set/closed-loop evaluation."""

from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import default_collate

from td_vla.models.multistream import decode_stage


@torch.inference_mode()
def evaluate_fixed_subset(model, dataset, device, batch_size, frames_per_episode=8):
    if batch_size < 1 or frames_per_episode < 1:
        raise ValueError("Positive validation sizes required")
    indices, start = [], 0
    for episode in dataset.episodes:
        local = np.linspace(
            0, episode["frames"] - 1, min(frames_per_episode, episode["frames"]), dtype=int
        )
        indices.extend((start + local).tolist())
        start += episode["frames"]
    totals = defaultdict(lambda: defaultdict(float))
    was_training = model.training
    model.eval()
    try:
        for offset in range(0, len(indices), batch_size):
            item = default_collate([dataset[i] for i in indices[offset : offset + batch_size]])
            output = model(item["instruction"], item["images"].to(device), item["state"].to(device))
            if isinstance(output, torch.Tensor):
                output = {"actions": output}
            output = {k: v.float().cpu() for k, v in output.items()}
            if "stage_logits" in output:
                stages = decode_stage(output["stage_logits"], item["stage_count"])
            targets = item["targets"]
            for j, task in enumerate(item["task"]):
                record = totals[task]
                record["samples"] += 1
                for name, mask in [("actions", "action_valid"), ("track", "track_valid")]:
                    if name not in output:
                        continue
                    valid = targets[mask][j]
                    error = (output[name][j][valid] - targets[name][j][valid]).abs()
                    record[name + "_sum"] += error.sum().item()
                    record[name + "_count"] += error.numel()
                if (
                    "stage_logits" in output
                    and targets["stage_valid"][j]
                    and targets["stage"][j] >= 0
                ):
                    label = targets["stage"][j]
                    record["stage_count"] += 1
                    record["stage_correct"] += int(stages["stage"][j] == label)
                    record["raw_correct"] += int(stages["raw_stage"][j] == label)
                    record["out_of_range"] += int(stages["out_of_range"][j])
    finally:
        model.train(was_training)
    tasks = {}
    for task, values in totals.items():

        def ratio(numerator, denominator):
            return values[numerator] / values[denominator] if values[denominator] else None

        tasks[task] = dict(
            samples=int(values["samples"]),
            action_l1=ratio("actions_sum", "actions_count"),
            track_mae=ratio("track_sum", "track_count"),
            stage_accuracy=ratio("stage_correct", "stage_count"),
            stage_raw_accuracy=ratio("raw_correct", "stage_count"),
            stage_out_of_range=ratio("out_of_range", "stage_count"),
        )
    return dict(
        scope="fixed evenly spaced frames per validation episode; online weights",
        samples=len(indices),
        frames_per_episode=frames_per_episode,
        tasks=tasks,
    )

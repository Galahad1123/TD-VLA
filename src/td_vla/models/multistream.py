"""Parallel continuous multistream extension of the pinned TurboVLA ACT decoder."""

import torch
from torch import nn
from turbovla.models.action_head import ACTDecoder
from turbovla.models.turbovla import TurboVLA


class MultiStreamDecoder(ACTDecoder):
    def __init__(
        self, config, hidden_dim, nheads, dim_feedforward, *, track_points=10, attention="full"
    ):
        super().__init__(config, hidden_dim, nheads, dim_feedforward)
        if track_points < 1 or attention not in {"full", "block"}:
            raise ValueError("Positive track_points and attention full/block required")
        self.track_points = track_points
        self.attention = attention
        # Separate embedding tables encode stream identity; track index = time * 2 + arm.
        self.track_queries = nn.Embedding(track_points * 2, hidden_dim)
        self.stage_query = nn.Embedding(1, hidden_dim)
        self.track_projection = nn.Linear(hidden_dim, 2)  # deliberately unbounded
        self.stage_projection = nn.Linear(hidden_dim, 5)
        streams = torch.tensor([0] * self.horizon + [1] * (2 * track_points) + [2])
        self.register_buffer("block_mask", streams[:, None] != streams[None, :], persistent=False)

    def forward(self, memory):
        queries = torch.cat(
            [self.action_queries.weight, self.track_queries.weight, self.stage_query.weight]
        )
        queries = queries.unsqueeze(0).expand(memory.shape[0], -1, -1)
        hidden = self.decoder(
            queries, memory, tgt_mask=self.block_mask if self.attention == "block" else None
        )
        action, track, stage = torch.split(hidden, [self.horizon, 2 * self.track_points, 1], dim=1)
        return {
            "actions": torch.tanh(self.action_projection(action)),
            "track": self.track_projection(track).reshape(-1, self.track_points, 2, 2),
            "stage_logits": self.stage_projection(stage[:, 0]),
        }


class TDVLA(TurboVLA):
    """Keep all upstream condition/state paths and parameter names; extend only ACT."""

    def __init__(self, config, *, track_points=10, attention="full"):
        if not config.text.frozen or config.vision.frozen:
            raise ValueError("v1 freezes BERT only; vision must be trainable")
        super().__init__(config)
        original = self.action_head.decoder
        decoder = MultiStreamDecoder(
            config.action,
            config.interaction.hidden_dim,
            config.interaction.nheads,
            config.interaction.dim_feedforward,
            track_points=track_points,
            attention=attention,
        )
        # Preserve identical common initialization across baseline/full/block given the same seed.
        decoder.action_queries = original.action_queries
        decoder.decoder = original.decoder
        decoder.action_projection = original.action_projection
        self.action_head.decoder = decoder


def decode_stage(logits, stage_counts):
    counts = torch.as_tensor(stage_counts, device=logits.device)
    if logits.ndim != 2 or logits.shape[-1] != 5 or counts.shape != logits.shape[:1]:
        raise ValueError("Expected logits [B,5] and stage_counts [B]")
    if counts.is_floating_point() or bool(((counts < 1) | (counts > 5)).any()):
        raise ValueError("stage_counts must be integer counts in [1,5]")
    raw = logits.argmax(-1)
    return {
        "raw_stage": raw,
        "stage": torch.minimum(raw, counts - 1),
        "out_of_range": raw >= counts,
    }

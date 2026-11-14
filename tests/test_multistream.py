# ruff: noqa: E402 -- optional policy dependencies are checked before imports
"""Run with the policy environment; lightweight development env skips torch tests."""

import pytest

torch = pytest.importorskip("torch")
from turbovla.models.action_head import ACTDecoder
from turbovla.models.configuration import ActionHeadConfig
from td_vla.models.multistream import MultiStreamDecoder, decode_stage
from td_vla.training.losses import LossWeights, multitask_loss


def decoder(mode="full"):
    cfg = ActionHeadConfig(horizon=4, action_dim=14, num_layers=2, dropout=0, mlp_hidden_dim=32)
    return MultiStreamDecoder(cfg, 16, 4, 32, track_points=10, attention=mode)


def test_shapes_and_mask_dependency():
    memory = torch.randn(2, 6, 16)
    full = decoder().eval()
    block = decoder("block").eval()
    block.load_state_dict(full.state_dict())
    for model, interacts in [(full, True), (block, False)]:
        out = model(memory)
        assert out["actions"].shape == (2, 4, 14)
        assert out["track"].shape == (2, 10, 2, 2)
        assert out["stage_logits"].shape == (2, 5)
        grad = torch.autograd.grad(out["actions"].square().sum(), model.track_queries.weight)[0]
        assert (grad.abs().sum().item() > 0) == interacts
        before = out["actions"].detach()
        with torch.no_grad():
            model.track_queries.weight.add_(torch.randn_like(model.track_queries.weight))
        assert torch.equal(before, model(memory)["actions"]) == (not interacts)


def test_block_actions_equal_upstream():
    model = decoder("block").eval()
    upstream = ACTDecoder(
        ActionHeadConfig(horizon=4, action_dim=14, num_layers=2, dropout=0, mlp_hidden_dim=32),
        16,
        4,
        32,
    ).eval()
    upstream.load_state_dict(
        {k: v for k, v in model.state_dict().items() if k in upstream.state_dict()}
    )
    memory = torch.randn(2, 6, 16)
    torch.testing.assert_close(model(memory)["actions"], upstream(memory), atol=1e-6, rtol=1e-5)


def test_stage_clamp_and_raw_prediction():
    logits = torch.tensor([[0.0, 0.0, 0.0, 0.0, 5.0], [0.0, 0.0, 0.0, 5.0, 0.0]])
    result = decode_stage(logits, torch.tensor([3, 5]))
    assert result["raw_stage"].tolist() == [4, 3]
    assert result["stage"].tolist() == [2, 3]
    assert result["out_of_range"].tolist() == [True, False]


def test_weighted_masked_losses():
    out = {
        "actions": torch.zeros(1, 2, 14, requires_grad=True),
        "track": torch.zeros(1, 2, 2, 2, requires_grad=True),
        "stage_logits": torch.zeros(1, 5, requires_grad=True),
    }
    target = {
        "actions": torch.ones(1, 2, 14),
        "action_valid": torch.tensor([[True, False]]),
        "track": torch.ones(1, 2, 2, 2),
        "track_valid": torch.tensor([[[True, True], [False, False]]]),
        "stage": torch.tensor([2]),
        "stage_valid": torch.tensor([True]),
    }
    target["actions"][:, 1] = float("nan")
    target["track"][:, 1] = float("nan")
    losses = multitask_loss(out, target, LossWeights(track=0.1, stage=0.2))
    assert losses["action"].item() == 1
    assert losses["track"].item() == 0.5  # Smooth L1 beta=1, mean over valid coordinates
    torch.testing.assert_close(losses["total"], 1.05 + 0.2 * torch.log(torch.tensor(5.0)))
    losses["total"].backward()
    assert out["track"].grad[:, 1].abs().sum() == 0
    for value in out.values():
        assert torch.isfinite(value.grad).all()


def test_all_invalid_losses_are_finite_and_differentiable():
    out = decoder()(torch.randn(1, 3, 16))
    target = {
        "actions": torch.full_like(out["actions"], float("nan")),
        "action_valid": torch.zeros(1, 4, dtype=torch.bool),
        "track": torch.full_like(out["track"], float("nan")),
        "track_valid": torch.zeros(1, 10, 2, dtype=torch.bool),
        "stage": torch.tensor([-1]),
        "stage_valid": torch.tensor([False]),
    }
    losses = multitask_loss(out, target, LossWeights())
    assert losses["total"].item() == 0
    losses["total"].backward()


def test_reject_invalid_weights_and_labels():
    with pytest.raises(ValueError):
        LossWeights(track=-1)
    with pytest.raises(ValueError):
        LossWeights(stage=float("nan"))
    with pytest.raises(ValueError):
        decode_stage(torch.zeros(1, 5), torch.tensor([6]))

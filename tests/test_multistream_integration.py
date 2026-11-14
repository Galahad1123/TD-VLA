# ruff: noqa: E402 -- optional policy dependencies are checked before imports
"""Real encoder integration; TD_VLA_TEST_FIXTURES points to local random BERT/DINO."""

import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from turbovla.models.configuration import TurboVLAConfig
from td_vla.models.multistream import TDVLA
from td_vla.training.losses import multitask_loss
from td_vla.training.checkpoint import save_checkpoint, load_checkpoint


def test_checkpoint_restores_training_and_rng(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1 / (s + 1))
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    scheduler.step()
    ema = {n: p.detach().clone() for n, p in model.named_parameters()}
    save_checkpoint(tmp_path / "state.pt", model, optimizer, scheduler, ema, 1, "config-id")
    expected_rng = torch.rand(3)
    with torch.no_grad():
        model.weight.zero_()
    step, restored = load_checkpoint(
        tmp_path / "state.pt", model, optimizer, scheduler, "config-id"
    )
    assert step == 1 and scheduler.last_epoch == 1
    torch.testing.assert_close(model.weight, ema["weight"])
    torch.testing.assert_close(restored["weight"], ema["weight"])
    torch.testing.assert_close(torch.rand(3), expected_rng)
    with pytest.raises(ValueError, match="signature"):
        load_checkpoint(tmp_path / "state.pt", model, optimizer, scheduler, "wrong")


@pytest.mark.skipif(not os.environ.get("TD_VLA_TEST_FIXTURES"), reason="Requires encoder fixtures")
def test_real_encoders_backward_and_roundtrip():
    root = Path(os.environ["TD_VLA_TEST_FIXTURES"])
    config = TurboVLAConfig.from_mapping(
        {
            "compatibility_profile": "robotwin",
            "text": {
                "model_name_or_path": str(root / "random-bert"),
                "attention_implementation": "eager",
            },
            "vision": {
                "model_name_or_path": str(root / "random-dino"),
                "image_size": 224,
                "num_views": 3,
                "compute_precision": "fp32",
            },
            "interaction": {
                "hidden_dim": 32,
                "nheads": 4,
                "num_layers": 1,
                "dim_feedforward": 64,
                "enhancer_inner_dim": 64,
                "fusion_droppath": 0,
                "residual_style": "pre_norm",
            },
            "action": {
                "action_dim": 14,
                "state_dim": 14,
                "horizon": 4,
                "num_layers": 2,
                "dropout": 0,
                "mlp_hidden_dim": 32,
                "state_hidden_dim": 32,
            },
        }
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TDVLA(config).to(device)
    model.train()
    images, state = torch.randn(1, 3, 3, 224, 224, device=device), torch.randn(1, 14, device=device)
    out = model(["lift the pot."], images, state)
    targets = {
        "actions": torch.zeros_like(out["actions"]),
        "action_valid": torch.ones(1, 4, dtype=torch.bool),
        "track": torch.ones_like(out["track"]),
        "track_valid": torch.ones(1, 10, 2, dtype=torch.bool),
        "stage": torch.tensor([1]),
        "stage_valid": torch.tensor([True]),
    }
    targets = {k: v.to(device) for k, v in targets.items()}
    multitask_loss(out, targets)["total"].backward()
    assert all(p.grad is None and not p.requires_grad for p in model.text_encoder.bert.parameters())
    for module in [
        model.text_encoder.text_projection,
        model.vision_encoder,
        model.vision_projection,
        model.vision_language_interaction,
        model.action_head.state_projection,
        model.action_head.decoder,
    ]:
        assert sum(p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None) > 0
    clone = TDVLA(config).to(device)
    clone.load_state_dict(model.state_dict())
    model.eval()
    clone.eval()
    with torch.no_grad():
        for key, value in model(["lift the pot."], images, state).items():
            torch.testing.assert_close(value, clone(["lift the pot."], images, state)[key])

# ruff: noqa: E402 -- optional policy dependencies
import math
import pytest

torch = pytest.importorskip("torch")
from td_vla.training.initialization import map_grounding_state
from td_vla.training.schedule import learning_rate_factor


def test_initialization_mapping_is_complete_and_excludes_policy():
    target = {
        "text_encoder.bert.weight": torch.zeros(2),
        "text_encoder.text_projection.weight": torch.zeros(2),
        "vision_language_interaction.text_layers.0.weight": torch.zeros(2),
        "vision_language_interaction.fusion_layers.0.weight": torch.zeros(2),
        "action_head.decoder.weight": torch.zeros(2),
    }
    source = {
        "module.bert.weight": torch.ones(2),
        "module.feat_map.weight": torch.ones(2),
        "module.transformer.encoder.text_layers.0.weight": torch.ones(2),
        "module.transformer.encoder.fusion_layers.0.weight": torch.ones(2),
        "module.action_head.decoder.weight": torch.ones(2),
    }
    loaded, report = map_grounding_state(source, target)
    assert len(loaded) == 4
    assert "action_head.decoder.weight" not in loaded
    assert report["loaded_tensors"] == 4
    del source["module.feat_map.weight"]
    with pytest.raises(ValueError, match="Incomplete"):
        map_grounding_state(source, target)


def test_initialization_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        map_grounding_state(
            {"bert.weight": torch.zeros(3)}, {"text_encoder.bert.weight": torch.zeros(2)}
        )


def test_cosine_schedule_uses_fixed_total_not_smoke_stop():
    assert learning_rate_factor(0, 1000, 45000, 0.1) == 0.001
    assert learning_rate_factor(999, 1000, 45000, 0.1) == 1.0
    assert learning_rate_factor(1000, 1000, 45000, 0.1) == 1.0
    assert math.isclose(learning_rate_factor(45000, 1000, 45000, 0.1), 0.1)
    assert learning_rate_factor(90000, 1000, 45000, 0.1) == 0.1
    with pytest.raises(ValueError):
        learning_rate_factor(0, 1000, 500, 0.1)


def test_fixed_validation_reports_task_local_clamp_and_out_of_range(tmp_path):
    from test_multistream_data import fixture_dataset
    from td_vla.data.multistream import MultiStreamDataset, prepare_split
    from td_vla.evaluation.offline import evaluate_fixed_subset

    class ConstantModel(torch.nn.Module):
        def forward(self, instructions, images, state):
            n = len(instructions)
            return dict(
                actions=torch.zeros(n, 50, 14),
                track=torch.zeros(n, 10, 2, 2),
                stage_logits=torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]).expand(n, -1),
            )

    root = fixture_dataset(tmp_path)
    split = prepare_split(root, validation_per_task=1)
    data = MultiStreamDataset(root, split, "validation", image_size=16)
    result = evaluate_fixed_subset(ConstantModel(), data, torch.device("cpu"), 2, 8)
    task = result["tasks"]["lift_pot"]
    assert task["samples"] == 3
    assert task["stage_accuracy"] == pytest.approx(1 / 3)
    assert task["stage_raw_accuracy"] == 0
    assert task["stage_out_of_range"] == 1
    assert task["track_mae"] == 1.5


def test_pretrained_manifest_checks_origin_and_file_content(tmp_path):
    import json
    import hashlib
    from td_vla.training.initialization import verify_pretrained

    root = tmp_path / "bert"
    root.mkdir()
    (root / "model.safetensors").write_bytes(b"original")
    spec = dict(repo_id="official/base", revision="locked", files=["model.safetensors"])
    manifest = dict(
        schema="td-vla-pretrained-v1",
        **spec,
        artifacts=[
            dict(file="model.safetensors", sha256=hashlib.sha256(b"original").hexdigest(), bytes=8)
        ],
    )
    (root / "manifest.json").write_text(json.dumps(manifest))
    assert verify_pretrained(tmp_path, {"bert": spec})["bert"]["revision"] == "locked"
    (root / "model.safetensors").write_bytes(b"modified")
    with pytest.raises(ValueError, match="checksum"):
        verify_pretrained(tmp_path, {"bert": spec})
    manifest["revision"] = "different"
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="origin"):
        verify_pretrained(tmp_path, {"bert": spec})


def test_pinned_distribution_hash_cannot_be_replaced_with_self_signed_manifest(tmp_path):
    import json
    import hashlib
    from td_vla.training.initialization import verify_pretrained

    root = tmp_path / "vision"
    root.mkdir()
    (root / "weights").write_bytes(b"wrong")
    spec = dict(
        repo_id="official/vision",
        revision="locked",
        files=["weights"],
        public_distribution={"files": {"weights": hashlib.sha256(b"right").hexdigest()}},
    )
    manifest = dict(
        schema="td-vla-pretrained-v1",
        **spec,
        artifacts=[dict(file="weights", sha256=hashlib.sha256(b"wrong").hexdigest(), bytes=5)],
    )
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Pinned"):
        verify_pretrained(tmp_path, {"vision": spec})

"""Run the real upstream TurboVLA graph with random fixture weights (NOT a policy eval)."""
import argparse
import json
from pathlib import Path

import torch
from transformers import BertConfig, BertModel, BertTokenizerFast, DINOv3ViTConfig, DINOv3ViTModel
from turbovla.models.configuration import TurboVLAConfig
from turbovla.models.turbovla import TurboVLA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(0)
    args.fixtures.mkdir(parents=True, exist_ok=True)
    bert_path = args.fixtures / "random-bert"
    dino_path = args.fixtures / "random-dino"
    # Full encoder sizes; these fixtures verify computation, never task competence.
    if not (bert_path / "config.json").exists():
        bert_path.mkdir(exist_ok=True)
        vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", ".", "?", "move",
                 "the", "stapler", "onto", "pad"]
        (bert_path / "vocab.txt").write_text("\n".join(vocab) + "\n")
        BertTokenizerFast(vocab_file=str(bert_path / "vocab.txt")).save_pretrained(bert_path)
        BertModel(BertConfig(vocab_size=len(vocab))).save_pretrained(bert_path)
    if not (dino_path / "config.json").exists():
        DINOv3ViTModel(DINOv3ViTConfig(hidden_size=1024, intermediate_size=4096,
                       num_hidden_layers=24, num_attention_heads=16)).save_pretrained(dino_path)
    cfg = TurboVLAConfig.from_mapping({
        "compatibility_profile": "robotwin",
        "text": {"model_name_or_path": str(bert_path), "attention_implementation": "eager"},
        "vision": {"model_name_or_path": str(dino_path), "image_size": 224, "num_views": 3,
                   "attention_implementation": "sdpa", "position_embedding": "learned_patch"},
        "interaction": {"padding_strategy": "zero_fill", "residual_style": "pre_norm",
                        "attention_backend": "sdpa", "compute_precision": "bf16_autocast"},
        "action": {"action_dim": 14, "state_dim": 14, "horizon": 50},
    })
    model = TurboVLA(cfg).cuda().train()
    image = torch.rand(1, 3, 3, 224, 224, device="cuda")
    state = torch.zeros(1, 14, device="cuda")
    action = model(["move the stapler onto the pad."], image, state)
    assert action.shape == (1, 50, 14) and torch.isfinite(action).all()
    action.float().square().mean().backward()
    assert all(p.grad is None for p in model.text_encoder.bert.parameters())
    assert any(p.grad is not None for p in model.vision_encoder.parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    print(json.dumps({"upstream_graph": "pass", "weights": "random-fixture-not-pretrained",
                      "parameters": sum(p.numel() for p in model.parameters()),
                      "action_shape": list(action.shape), "frozen_bert": True,
                      "peak_cuda_bytes": torch.cuda.max_memory_allocated()}))


if __name__ == "__main__":
    main()

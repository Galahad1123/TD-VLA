import numpy as np
from td_vla.integrations.robotwin_policy import prepare_images, ChunkPolicy


def test_rgb_preprocessing():
    rgb = np.zeros((3, 8, 8, 3), dtype=np.uint8)
    result = prepare_images(rgb, 224)
    assert result.shape == (3, 3, 224, 224)
    np.testing.assert_allclose(
        result[0, :, 0, 0], -np.array([0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225], rtol=1e-6
    )


def test_chunk_reset_and_action_order():
    class Client:
        calls = 0

        def predict_action(self, request):
            self.calls += 1
            return {
                "ok": True,
                "data": {
                    "actions": np.tile(np.arange(14), (50, 1)),
                    "track": np.zeros((10, 2, 2)),
                    "stage": 1,
                },
            }

    client = Client()
    policy = ChunkPolicy(client, "lift_pot", 50)
    obs = {
        "joint_action": {"vector": np.zeros(14)},
        "observation": {
            k: {"rgb": np.zeros((8, 8, 3), dtype=np.uint8)}
            for k in ["head_camera", "left_camera", "right_camera"]
        },
    }
    for _ in range(51):
        np.testing.assert_array_equal(policy.action(obs), np.arange(14))
    assert client.calls == 2
    policy.reset()
    policy.action(obs)
    assert client.calls == 3
    assert len(policy.predictions) == 1


def test_action_only_baseline_response():
    class Client:
        def predict_action(self, request):
            return {"ok": True, "data": {"actions": np.zeros((50, 14))}}

    policy = ChunkPolicy(Client(), "lift_pot")
    obs = {
        "joint_action": {"vector": np.zeros(14)},
        "observation": {
            k: {"rgb": np.zeros((8, 8, 3), dtype=np.uint8)}
            for k in ["head_camera", "left_camera", "right_camera"]
        },
    }
    np.testing.assert_array_equal(policy.action(obs), np.zeros(14))
    assert "track" not in policy.predictions[0] and "stage" not in policy.predictions[0]

"""TD-VLA inference contract; only RGB, measured joint state and task text enter policy."""

import time
import numpy as np
from PIL import Image
from td_vla.data.pilot import INSTRUCTIONS

CAMERAS = ("head_camera", "left_camera", "right_camera")


def prepare_images(images, size):
    images = np.asarray(
        [
            np.asarray(
                Image.fromarray(im).resize((size, size), Image.Resampling.BILINEAR)
            ).transpose(2, 0, 1)
            / 255.0
            for im in images
        ]
    )
    images -= np.array([0.485, 0.456, 0.406])[None, :, None, None]
    images /= np.array([0.229, 0.224, 0.225])[None, :, None, None]
    return images.astype(np.float32)


class ChunkPolicy:
    def __init__(self, client, task, exec_horizon=50):
        if not 1 <= exec_horizon <= 50:
            raise ValueError("Execution horizon must be 1..50")
        self.client, self.task, self.exec_horizon = client, task, exec_horizon
        self.reset()

    def reset(self):
        self.step = 0
        self.predictions = []
        self.actions = None

    def action(self, observation):
        if self.step % self.exec_horizon == 0:
            start = time.monotonic()
            reply = self.client.predict_action(
                {
                    "type": "infer",
                    "payload": {
                        "task": self.task,
                        "lang": INSTRUCTIONS[self.task],
                        "state": observation["joint_action"]["vector"],
                        "image": np.stack([observation["observation"][c]["rgb"] for c in CAMERAS]),
                    },
                }
            )
            if not reply.get("ok"):
                raise RuntimeError(str(reply))
            result = reply["data"]
            self.actions = np.asarray(result["actions"])
            if self.actions.shape != (50, 14) or not np.isfinite(self.actions).all():
                raise ValueError("Invalid action chunk")
            record = dict(step=self.step, seconds=time.monotonic() - start)
            if "track" in result:
                record["track"] = np.asarray(result["track"]).tolist()
                record["stage"] = int(result["stage"])
            self.predictions.append(record)
        action = self.actions[self.step % self.exec_horizon]
        self.step += 1
        return action

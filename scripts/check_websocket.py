"""Cross-environment test using the upstream server/client and ndarray codec."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external/TurboVLA/third_party/starvla_runtime"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["server", "client"])
    parser.add_argument("--port", type=int, default=5694)
    args = parser.parse_args()
    if args.mode == "server":
        from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer

        class EchoPolicy:
            def predict_action(self, **payload):
                state = np.asarray(payload["state"], dtype=np.float32).reshape(14)
                assert np.asarray(payload["image"]).shape == (3, 224, 224, 3)
                return {"action": np.repeat(state[None, :], 50, axis=0)}

        WebsocketPolicyServer(EchoPolicy(), host="127.0.0.1", port=args.port,
                              idle_timeout=60, metadata={"test": "echo-only"}).serve_forever()
    else:
        from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

        client = WebsocketClientPolicy("127.0.0.1", args.port)
        try:
            state = np.arange(14, dtype=np.float32) / 14
            reply = client.predict_action({"request_id": "p1", "type": "infer", "payload": {
                "state": state, "image": np.zeros((3, 224, 224, 3), dtype=np.uint8),
                "lang": "move the stapler onto the pad"}})
            assert reply["ok"] and reply["request_id"] == "p1", reply
            action = reply["data"]["action"]
            assert action.shape == (50, 14) and np.array_equal(action[0], state)
            print(json.dumps({"websocket": "pass", "policy": "echo-only", "shape": list(action.shape),
                              "python": sys.executable, "pid": os.getpid()}))
        finally:
            client.close()


if __name__ == "__main__":
    main()

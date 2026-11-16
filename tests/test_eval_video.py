import json
import subprocess
import numpy as np
from td_vla.integrations.eval_video import EpisodeVideo


def test_video_preserves_all_observation_frames(tmp_path):
    path = tmp_path / "episode.mp4"
    with EpisodeVideo(path, fps=10) as video:
        for i in range(3):
            video.append(np.full((48, 64, 3), i * 100, dtype=np.uint8))
    result = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,nb_read_frames",
                "-of",
                "json",
                str(path),
            ]
        )
    )
    assert result["streams"][0] == {"width": 64, "height": 48, "nb_read_frames": "3"}

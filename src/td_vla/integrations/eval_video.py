"""Observation-frame videos; FPS is playback speed, not a physics clock."""

import subprocess
import numpy as np


class EpisodeVideo:
    def __init__(self, path, fps=10):
        self.path, self.fps, self.process = path, fps, None
        self.shape = None
        self.frames = 0

    def append(self, rgb):
        rgb = np.asarray(rgb)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError("Video expects uint8 RGB")
        if self.process is None:
            self.shape = rgb.shape
            h, w, _ = rgb.shape
            self.process = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    f"{w}x{h}",
                    "-framerate",
                    str(self.fps),
                    "-i",
                    "-",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-crf",
                    "20",
                    str(self.path),
                ],
                stdin=subprocess.PIPE,
            )
        if rgb.shape != self.shape:
            raise ValueError("Frame dimensions changed")
        self.process.stdin.write(rgb.tobytes())
        self.frames += 1

    def close(self):
        if self.process is not None:
            self.process.stdin.close()
            code = self.process.wait(timeout=30)
            self.process = None
            if code:
                raise RuntimeError(f"ffmpeg exited {code}")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

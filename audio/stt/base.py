from __future__ import annotations

import importlib.util
import platform

import numpy as np

SAMPLE_RATE = 16000


def apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


class Transcriber:
    name = "?"
    model = ""
    streaming = False

    def transcribe(self, audio: np.ndarray) -> str:
        raise NotImplementedError

    def start(self) -> None:
        self._frames: list[np.ndarray] = []

    def feed(self, frame: np.ndarray) -> None:
        self._frames.append(np.asarray(frame, dtype=np.float32))

    def finish(self) -> str:
        frames = getattr(self, "_frames", [])
        self._frames = []

        return self.transcribe(np.concatenate(frames)) if frames else ""

    def close(self) -> None:
        pass

    def __repr__(self) -> str:
        return f"{self.name} ({self.model})" if self.model else self.name

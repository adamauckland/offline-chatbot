"""Whisper on MLX (Apple silicon), the original backend.

The default model is ``mlx-community/whisper-small.en-mlx`` (English only,
about 500 MB, a quarter of a second per utterance). Use
``mlx-community/whisper-base.en-mlx`` for something smaller and faster, or
``mlx-community/whisper-large-v3-turbo`` for other languages (then change
``LANGUAGE``).
"""

from __future__ import annotations

import platform

import numpy as np

from audio.stt.base import SAMPLE_RATE, Transcriber

DEFAULT_MODEL = "mlx-community/whisper-small.en-mlx"
LANGUAGE = "en"
LEAD_IN_SAMPLES = int(0.3 * SAMPLE_RATE)  # silence put in front of every clip before Whisper sees it


def available() -> tuple[bool, str]:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False, "needs Apple silicon (MLX)"
    try:
        import mlx_whisper  # noqa: F401
    except ImportError:
        return False, "mlx-whisper is not installed"
    return True, ""


class WhisperMLX(Transcriber):
    name = "whisper"

    def __init__(self, model: str | None = None) -> None:
        import mlx_whisper  # slow import, keep it local

        self._whisper = mlx_whisper
        self.model = model or DEFAULT_MODEL
        self.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32))  # downloads and loads the model now

    def transcribe(self, audio: np.ndarray) -> str:
        # Whisper drops the first consonant of a clip that starts on speech ("set a timer"
        # came back as "at a timer" from wake-word clips); a little leading silence fixes it.
        audio = np.concatenate([np.zeros(LEAD_IN_SAMPLES, dtype=np.float32), np.asarray(audio, dtype=np.float32)])
        result = self._whisper.transcribe(
            audio, path_or_hf_repo=self.model, language=LANGUAGE, fp16=True, temperature=0.0
        )
        return result["text"].strip()

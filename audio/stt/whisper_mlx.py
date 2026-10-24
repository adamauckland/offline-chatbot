from __future__ import annotations

import numpy as np

from audio.stt.base import SAMPLE_RATE, Transcriber, apple_silicon, installed

DEFAULT_MODEL = "mlx-community/whisper-small.en-mlx"
LANGUAGE = "en"
LEAD_IN_SILENCE = np.zeros(int(0.3 * SAMPLE_RATE), dtype=np.float32)


def available() -> tuple[bool, str]:
    if not apple_silicon():
        return False, "needs Apple silicon (MLX)"

    if not installed("mlx_whisper"):
        return False, "mlx-whisper is not installed"

    return True, ""


def with_lead_in_silence(audio: np.ndarray) -> np.ndarray:
    return np.concatenate([LEAD_IN_SILENCE, np.asarray(audio, dtype=np.float32)])


class WhisperMLX(Transcriber):
    name = "whisper"

    def __init__(self, model: str | None = None) -> None:
        import mlx_whisper

        self._whisper = mlx_whisper
        self.model = model or DEFAULT_MODEL
        self._load_model_now()

    def _load_model_now(self) -> None:
        self.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32))

    def transcribe(self, audio: np.ndarray) -> str:
        result = self._whisper.transcribe(
            with_lead_in_silence(audio), path_or_hf_repo=self.model, language=LANGUAGE, fp16=True, temperature=0.0
        )

        return result["text"].strip()

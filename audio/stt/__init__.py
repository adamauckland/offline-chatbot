from __future__ import annotations

import importlib
from typing import NamedTuple

from audio.stt.base import SAMPLE_RATE, Transcriber


class Backend(NamedTuple):
    module: str
    class_name: str
    description: str


BACKENDS: dict[str, Backend] = {
    "whisper": Backend("audio.stt.whisper_mlx", "WhisperMLX", "Whisper on MLX (Apple silicon)"),
    "apple": Backend("audio.stt.apple", "AppleSpeech", "macOS Speech framework (macOS 26 or later)"),
    "kyutai": Backend("audio.stt.kyutai", "KyutaiSTT", "Kyutai STT 1B on MLX, in a helper process (Apple silicon)"),
    "sherpa": Backend("audio.stt.sherpa", "SherpaMoonshine", "Moonshine via sherpa-onnx (any CPU, e.g. an Intel NUC)"),
}
AUTO_ORDER = ("whisper", "sherpa")
CHOICES = ("auto", *BACKENDS)


def available(name: str) -> tuple[bool, str]:
    return importlib.import_module(BACKENDS[name].module).available()


def load_transcriber(name: str = "auto", model: str | None = None) -> Transcriber:
    if name == "auto":
        name = first_available_backend()

    if name not in BACKENDS:
        raise ValueError(f"unknown speech-to-text backend {name!r}; choose from {', '.join(BACKENDS)}")

    ok, why = available(name)

    if not ok:
        raise RuntimeError(f"speech-to-text backend {name!r} cannot run here: {why}")

    backend = BACKENDS[name]

    return getattr(importlib.import_module(backend.module), backend.class_name)(model)


def first_available_backend() -> str:
    reasons = []

    for candidate in AUTO_ORDER:
        ok, why = available(candidate)

        if ok:
            return candidate

        reasons.append(f"{candidate}: {why}")

    raise RuntimeError("no speech-to-text backend can run here (" + "; ".join(reasons) + ")")


def describe() -> str:
    lines = []

    for name, backend in BACKENDS.items():
        ok, why = available(name)
        lines.append(f"  {name:8s} {backend.description}: {'available' if ok else why}")

    return "\n".join(lines)


__all__ = ["BACKENDS", "CHOICES", "SAMPLE_RATE", "Transcriber", "available", "describe", "load_transcriber"]

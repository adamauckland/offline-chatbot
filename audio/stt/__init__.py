"""Speech to text, with the engine chosen at start-up.

Every backend is one module here with the same two things: ``available()``,
which says whether it can run on this machine and why not, and a
``Transcriber`` class (see ``base.py``) whose ``transcribe`` turns 16 kHz
audio into text. ``load_transcriber`` picks one by name, or the first that
works for ``"auto"``, so the same code runs on an Apple silicon Mac and on
an Intel NUC with just a different backend underneath.

Backends:

``whisper``  Whisper on MLX. Apple silicon. The original; a quarter of a second per utterance.
``apple``    The macOS Speech framework (macOS 26+). No download, nothing on the GPU, about 0.1 s.
``kyutai``   Kyutai STT 1B on MLX, in a helper process with its own environment. Apple silicon.
``sherpa``   Moonshine via sherpa-onnx. Any CPU, so the one for an Intel NUC; 0.04 s per utterance.
"""

from __future__ import annotations

import importlib

from audio.stt.base import SAMPLE_RATE, Transcriber

BACKENDS: dict[str, tuple[str, str, str]] = {  # name -> (module, class, description)
    "whisper": ("audio.stt.whisper_mlx", "WhisperMLX", "Whisper on MLX (Apple silicon)"),
    "apple": ("audio.stt.apple", "AppleSpeech", "macOS Speech framework (macOS 26 or later)"),
    "kyutai": ("audio.stt.kyutai", "KyutaiSTT", "Kyutai STT 1B on MLX, in a helper process (Apple silicon)"),
    "sherpa": ("audio.stt.sherpa", "SherpaMoonshine", "Moonshine via sherpa-onnx (any CPU, e.g. an Intel NUC)"),
}
# What "auto" tries, in order: the original backend where it runs, else the one that runs anywhere.
AUTO_ORDER = ("whisper", "sherpa")
CHOICES = ("auto", *BACKENDS)


def available(name: str) -> tuple[bool, str]:
    """Whether backend ``name`` can run here, and the reason when it cannot."""
    module = importlib.import_module(BACKENDS[name][0])
    return module.available()


def load_transcriber(name: str = "auto", model: str | None = None) -> Transcriber:
    """Load backend ``name`` (its default model unless ``model`` is given)."""
    if name == "auto":
        reasons = []
        for candidate in AUTO_ORDER:
            ok, why = available(candidate)
            if ok:
                name = candidate
                break
            reasons.append(f"{candidate}: {why}")
        else:
            raise RuntimeError("no speech-to-text backend can run here (" + "; ".join(reasons) + ")")
    if name not in BACKENDS:
        raise ValueError(f"unknown speech-to-text backend {name!r}; choose from {', '.join(BACKENDS)}")
    ok, why = available(name)
    if not ok:
        raise RuntimeError(f"speech-to-text backend {name!r} cannot run here: {why}")
    module_name, class_name, _ = BACKENDS[name]
    return getattr(importlib.import_module(module_name), class_name)(model)


def describe() -> str:
    """One line per backend with whether it can run here, for ``--help`` and diagnostics."""
    lines = []
    for name, (_, _, description) in BACKENDS.items():
        ok, why = available(name)
        lines.append(f"  {name:8s} {description}: {'available' if ok else why}")
    return "\n".join(lines)


__all__ = ["BACKENDS", "CHOICES", "SAMPLE_RATE", "Transcriber", "available", "describe", "load_transcriber"]

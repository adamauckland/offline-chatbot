"""Moonshine through sherpa-onnx: runs on any CPU, so this is the backend for an Intel NUC.

Moonshine is a small English model made for on-device use; sherpa-onnx is
already a dependency (the wake word and VAD use it). The tiny model is about
100 MB, loads in a third of a second and transcribes a four-second utterance
in 0.04 s on a laptop CPU, with accuracy close to Whisper small. The models
are downloaded once into ``models/``, like the wake-word models.

``model`` is the name of a sherpa-onnx Moonshine release, for example
``sherpa-onnx-moonshine-base-en-int8`` for the larger one.
"""

from __future__ import annotations

import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

from audio.stt.base import SAMPLE_RATE, Transcriber

DEFAULT_MODEL = "sherpa-onnx-moonshine-tiny-en-int8"
MODELS_DIR = Path("models")
RELEASE_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{name}.tar.bz2"


def available() -> tuple[bool, str]:
    try:
        import sherpa_onnx  # noqa: F401
    except ImportError:
        return False, "sherpa-onnx is not installed"
    return True, ""


def ensure_model(name: str, models_dir: Path = MODELS_DIR) -> Path:
    model_dir = models_dir / name
    if not model_dir.is_dir():
        archive = models_dir / f"{name}.tar.bz2"
        print(f"Downloading {name}...", file=sys.stderr)
        models_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(RELEASE_URL.format(name=name), archive)
        with tarfile.open(archive) as tar:
            tar.extractall(models_dir, filter="data")
        archive.unlink()
    return model_dir


class SherpaMoonshine(Transcriber):
    name = "sherpa"

    def __init__(self, model: str | None = None, threads: int = 4) -> None:
        import sherpa_onnx

        self.model = model or DEFAULT_MODEL
        d = ensure_model(self.model)
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_moonshine(
            preprocessor=str(d / "preprocess.onnx"),
            encoder=str(d / "encode.int8.onnx"),
            uncached_decoder=str(d / "uncached_decode.int8.onnx"),
            cached_decoder=str(d / "cached_decode.int8.onnx"),
            tokens=str(d / "tokens.txt"),
            num_threads=threads,
        )

    def transcribe(self, audio: np.ndarray) -> str:
        stream = self._recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, np.asarray(audio, dtype=np.float32))
        self._recognizer.decode_stream(stream)
        return stream.result.text.strip()

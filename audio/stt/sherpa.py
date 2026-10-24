from __future__ import annotations

import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

from audio.stt.base import SAMPLE_RATE, Transcriber, installed

DEFAULT_MODEL = "sherpa-onnx-moonshine-tiny-en-int8"
MODELS_DIR = Path("models")
RELEASE_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{name}.tar.bz2"


def available() -> tuple[bool, str]:
    if not installed("sherpa_onnx"):
        return False, "sherpa-onnx is not installed"

    return True, ""


def ensure_model(name: str, models_dir: Path = MODELS_DIR) -> Path:
    model_dir = models_dir / name

    if not model_dir.is_dir():
        download_and_unpack(RELEASE_URL.format(name=name), models_dir, name)

    return model_dir


def download_and_unpack(url: str, models_dir: Path, name: str) -> None:
    archive = models_dir / f"{name}.tar.bz2"
    print(f"Downloading {name}...", file=sys.stderr)
    models_dir.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, archive)

    with tarfile.open(archive) as tar:
        tar.extractall(models_dir, filter="data")

    archive.unlink()


class SherpaMoonshine(Transcriber):
    name = "sherpa"

    def __init__(self, model: str | None = None, threads: int = 4) -> None:
        import sherpa_onnx

        self.model = model or DEFAULT_MODEL
        files = ensure_model(self.model)
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_moonshine(
            preprocessor=str(files / "preprocess.onnx"),
            encoder=str(files / "encode.int8.onnx"),
            uncached_decoder=str(files / "uncached_decode.int8.onnx"),
            cached_decoder=str(files / "cached_decode.int8.onnx"),
            tokens=str(files / "tokens.txt"),
            num_threads=threads,
        )

    def transcribe(self, audio: np.ndarray) -> str:
        stream = self._recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, np.asarray(audio, dtype=np.float32))
        self._recognizer.decode_stream(stream)

        return stream.result.text.strip()

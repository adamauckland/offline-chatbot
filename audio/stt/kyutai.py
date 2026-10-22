"""Kyutai STT 1B on MLX: a streaming recogniser, run as a helper in its own environment.

Kyutai's ``moshi-mlx`` package pins a huggingface-hub this project cannot use
(memory's model2vec needs a newer one), so ``kyutai_helper.py`` runs under
``uv run --no-project --with moshi-mlx``: uv builds that environment once and
caches it, and the model (about 2.5 GB, ``kyutai/stt-1b-en_fr-mlx``) is
downloaded once by huggingface-hub.

The model transcribes at about 1.8 times real time here, so as a batch step it
would add seconds; it earns its place by streaming. Frames are sent while you
speak, and the text is ready about half a second (the model's delay) after you
stop. English and French. Needs Apple silicon and ``uv`` on the PATH.
"""

from __future__ import annotations

import platform
import shutil
from pathlib import Path

from audio.stt.helper import PipeTranscriber

DEFAULT_MODEL = "kyutai/stt-1b-en_fr-mlx"
HELPER = Path(__file__).with_name("kyutai_helper.py")
MOSHI_MLX = "moshi-mlx>=0.2.6"


def available() -> tuple[bool, str]:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False, "needs Apple silicon (MLX)"
    if shutil.which("uv") is None:
        return False, "needs uv on the PATH to run the helper in its own environment"
    return True, ""


class KyutaiSTT(PipeTranscriber):
    name = "kyutai"
    streaming = True
    ready_timeout = 1800.0  # the first start builds the environment and downloads the model

    def __init__(self, model: str | None = None) -> None:
        self.model = model or DEFAULT_MODEL
        command = [
            "uv", "run", "--no-project", "--python", "3.12", "--with", MOSHI_MLX,
            "python", str(HELPER), "--hf-repo", self.model,
        ]  # fmt: skip
        super().__init__(command)

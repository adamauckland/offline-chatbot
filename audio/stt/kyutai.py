from __future__ import annotations

import shutil
from pathlib import Path

from audio.stt.base import apple_silicon
from audio.stt.helper import PipeTranscriber

DEFAULT_MODEL = "kyutai/stt-1b-en_fr-mlx"
HELPER = Path(__file__).with_name("kyutai_helper.py")
MOSHI_MLX = "moshi-mlx>=0.2.6"


def available() -> tuple[bool, str]:
    if not apple_silicon():
        return False, "needs Apple silicon (MLX)"

    if shutil.which("uv") is None:
        return False, "needs uv on the PATH to run the helper in its own environment"

    return True, ""


def helper_command(model: str) -> list[str]:
    isolated_python = ["uv", "run", "--no-project", "--python", "3.12", "--with", MOSHI_MLX, "python"]

    return isolated_python + [str(HELPER), "--hf-repo", model]


class KyutaiSTT(PipeTranscriber):
    name = "kyutai"
    streaming = True
    ready_timeout = 1800.0

    def __init__(self, model: str | None = None) -> None:
        self.model = model or DEFAULT_MODEL
        super().__init__(helper_command(self.model))

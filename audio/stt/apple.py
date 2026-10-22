"""The macOS Speech framework (macOS 26 and later): on-device, no model download, nothing on the GPU.

Measured on an M-series Mac it transcribes a four-second utterance in about
0.1 s and is word-perfect on clear speech. It is only reachable from Swift,
so a small helper program (``apple_helper.swift``, next to this file) is
compiled with ``swiftc`` on first use into ``models/`` and run as a child
process; see ``helper.py`` for the pipe protocol. Needs the Xcode command
line tools for the compiler.
"""

from __future__ import annotations

import hashlib
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from audio.stt.helper import PipeTranscriber

MODELS_DIR = Path("models")
SOURCE = Path(__file__).with_name("apple_helper.swift")
DEFAULT_MODEL = "SpeechTranscriber en_US"


def available() -> tuple[bool, str]:
    if platform.system() != "Darwin":
        return False, "needs macOS"
    major = int(platform.mac_ver()[0].split(".")[0] or 0)
    if major < 26:
        return False, "needs macOS 26 or later (SpeechAnalyzer)"
    if shutil.which("swiftc") is None and shutil.which("xcrun") is None:
        return False, "needs the Xcode command line tools (swiftc) to build the helper"
    return True, ""


def helper_binary(models_dir: Path = MODELS_DIR) -> Path:
    """The compiled helper for the current source, building it if it is missing."""
    source = SOURCE.read_bytes()
    stamp = hashlib.sha1(source).hexdigest()[:10]
    binary = models_dir / f"apple-stt-helper-{stamp}"
    if binary.is_file():
        return binary
    models_dir.mkdir(parents=True, exist_ok=True)
    print("Building the Speech framework helper (once)...", file=sys.stderr)
    compiler = ["xcrun", "swiftc"] if shutil.which("xcrun") else ["swiftc"]
    subprocess.run(compiler + ["-O", "-parse-as-library", "-o", str(binary), str(SOURCE)], check=True)
    return binary


class AppleSpeech(PipeTranscriber):
    name = "apple"

    def __init__(self, model: str | None = None) -> None:
        self.model = model or DEFAULT_MODEL
        super().__init__([str(helper_binary())])

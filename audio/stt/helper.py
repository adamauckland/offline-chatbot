"""A transcriber that lives in another process, talked to over a pipe.

Two backends need this: the macOS Speech framework can only be reached from
Swift, and Kyutai's package pins dependencies this project cannot share. The
protocol is as small as it gets. The helper prints one JSON line
``{"ready": true}`` when it can take audio. Each message from Python is a
one-byte command, a four-byte little-endian length and that many bytes of
16-bit mono PCM at 16 kHz:

``T``  transcribe this whole utterance; the helper answers ``{"text": ...}``
``S``  an utterance starts (no payload, no answer)
``F``  a frame of it (no answer)
``E``  it has ended (no payload); the helper answers ``{"text": ...}``

Any answer may instead be ``{"error": ...}``. A batch helper simply collects
``F`` frames and transcribes them at ``E``.
"""

from __future__ import annotations

import json
import struct
import subprocess
import threading

import numpy as np

from audio.stt.base import Transcriber


def pcm_bytes(audio: np.ndarray) -> bytes:
    return (np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0) * 32767).astype("<i2").tobytes()


class PipeTranscriber(Transcriber):
    """Runs ``command`` as a helper and sends it each utterance."""

    ready_timeout = 600.0  # a helper may be downloading or compiling on first use
    reply_timeout = 60.0

    def __init__(self, command: list[str], cwd: str | None = None) -> None:
        self.command = command
        self.cwd = cwd
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._start()

    # -- the process ---------------------------------------------------------------
    def _start(self) -> None:
        self._proc = subprocess.Popen(
            self.command, cwd=self.cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        ready = self._read_line(self.ready_timeout)
        if not ready.get("ready"):
            raise RuntimeError(f"{self.name} helper did not start: {ready.get('error', 'no ready line')}")

    def _alive(self) -> None:
        if self._proc is None or self._proc.poll() is not None:
            self._start()  # the helper died: bring it back

    def _send(self, command: bytes, payload: bytes = b"") -> None:
        assert self._proc is not None and self._proc.stdin is not None
        try:
            self._proc.stdin.write(command + struct.pack("<I", len(payload)) + payload)
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError(f"{self.name} helper is gone: {exc}") from exc

    def _read_line(self, timeout: float) -> dict:
        assert self._proc is not None and self._proc.stdout is not None
        result: dict = {}

        def read() -> None:
            line = self._proc.stdout.readline()  # type: ignore[union-attr]
            try:
                result.update(json.loads(line) if line else {"error": "helper exited"})
            except json.JSONDecodeError:
                result["error"] = f"bad reply from helper: {line[:80]!r}"

        thread = threading.Thread(target=read, daemon=True)
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            return {"error": "helper timed out"}
        return result

    def _answer(self) -> str:
        reply = self._read_line(self.reply_timeout)
        if "error" in reply:
            raise RuntimeError(f"{self.name}: {reply['error']}")
        return str(reply.get("text", "")).strip()

    # -- the transcriber interface -----------------------------------------------------
    def transcribe(self, audio: np.ndarray) -> str:
        with self._lock:
            self._alive()
            self._send(b"T", pcm_bytes(audio))
            return self._answer()

    def start(self) -> None:
        with self._lock:
            self._alive()
            self._send(b"S")

    def feed(self, frame: np.ndarray) -> None:
        with self._lock:
            self._send(b"F", pcm_bytes(frame))

    def finish(self) -> str:
        with self._lock:
            self._send(b"E")
            return self._answer()

    def close(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None

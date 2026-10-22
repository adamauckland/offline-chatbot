"""What every speech-to-text backend provides."""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000  # every backend gets mono float32 audio at this rate


class Transcriber:
    """One speech-to-text engine. Subclasses load their model in ``__init__``.

    ``transcribe`` takes a whole utterance. A backend that can work while the
    user is still speaking sets ``streaming`` and implements ``start``,
    ``feed`` and ``finish``; for the others those three buffer the frames and
    call ``transcribe`` at the end, so callers can always use them.
    """

    name = "?"  # the backend's name in the registry
    model = ""  # which model it loaded, for the start-up message
    streaming = False

    def transcribe(self, audio: np.ndarray) -> str:
        """Text for one utterance of 16 kHz mono float32 audio ("" if nothing was said)."""
        raise NotImplementedError

    def start(self) -> None:
        self._frames: list[np.ndarray] = []

    def feed(self, frame: np.ndarray) -> None:
        self._frames.append(np.asarray(frame, dtype=np.float32))

    def finish(self) -> str:
        frames = getattr(self, "_frames", [])
        self._frames = []
        return self.transcribe(np.concatenate(frames)) if frames else ""

    def close(self) -> None:
        """Release anything held (a helper process, for instance)."""

    def __repr__(self) -> str:
        return f"{self.name} ({self.model})" if self.model else self.name

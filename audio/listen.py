"""Talk to the chatbot: record from the microphone and transcribe.

Recording is hands-free once started: it waits for you to begin speaking, then
stops after a short silence (or at ``max_seconds``). Transcription is done by
whichever speech-to-text backend was chosen at start-up (see ``audio/stt``):
Whisper on MLX, the macOS Speech framework, Kyutai STT or Moonshine via
sherpa-onnx, behind one ``transcribe`` call.
"""

from __future__ import annotations

import queue
import sys

import numpy as np

import events
from audio.input import AudioIO, cue
from audio.stt import SAMPLE_RATE, Transcriber, load_transcriber

BLOCK_SECONDS = 0.1  # the shared stream delivers 100 ms frames

# Whisper's favourite things to "hear" in silence or noise; cheap to filter for every backend.
_HALLUCINATIONS = {"you", "thank you", "thanks", "bye", "thank you for watching", "."}


class VoiceGate:
    """Decide when an utterance starts and ends from the loudness of audio blocks.

    The room's noise level is the quietest block heard so far, so the gate
    adapts to a quiet study or a noisy kitchen. Speech begins when a block is
    clearly louder than that; it ends after ``silence_seconds`` of quiet.

    An earlier version took the first 0.6 s as room noise, which meant that
    someone who started talking the moment the bot listened set the threshold
    with their own voice and was never heard. Now the floor is a running
    minimum, and it is capped at ``max_floor`` (the auto gain never lifts
    silence above 0.01, so anything louder is not the room), so speech from
    the very first block still counts once a second block has been seen.
    """

    def __init__(
        self,
        silence_seconds: float = 1.0,
        min_blocks: int = 2,
        min_threshold: float = 0.01,
        boost: float = 4.0,
        max_floor: float = 0.03,
    ) -> None:
        self.silence_blocks = int(silence_seconds / BLOCK_SECONDS)
        self.min_blocks = min_blocks
        self.min_threshold = min_threshold
        self.boost = boost
        self.max_floor = max_floor
        self.floor: float | None = None
        self.seen = 0
        self.speaking = False
        self.quiet_run = 0

    @property
    def threshold(self) -> float | None:
        if self.floor is None:
            return None
        return max(self.min_threshold, min(self.floor, self.max_floor) * self.boost)

    def feed(self, block: np.ndarray) -> str:
        """Return "wait", "speech" or "done" for this block."""
        rms = float(np.sqrt(np.mean(block.astype(np.float32) ** 2)))
        self.seen += 1
        if not self.speaking:
            self.floor = rms if self.floor is None else min(self.floor, rms)
        if self.seen < self.min_blocks:
            return "wait"
        loud = rms > self.threshold
        if not self.speaking:
            if loud:
                self.speaking = True
                self.quiet_run = 0
                return "speech"
            return "wait"
        self.quiet_run = 0 if loud else self.quiet_run + 1
        return "done" if self.quiet_run >= self.silence_blocks else "speech"


class Listener:
    """Records utterances from the shared microphone stream and turns them into text."""

    def __init__(self, audio: AudioIO, transcriber: Transcriber | str = "auto", model: str | None = None) -> None:
        self.audio = audio
        # A backend name loads that backend (its default model unless ``model`` says otherwise).
        self.transcriber = transcriber if isinstance(transcriber, Transcriber) else load_transcriber(transcriber, model)

    def record(
        self, max_seconds: float = 30.0, wait_seconds: float = 8.0, silence_seconds: float = 1.0
    ) -> np.ndarray | None:
        """Record one utterance from the default microphone. None if nobody spoke.

        Every frame kept is also passed to ``feed``, so a streaming backend
        hears the utterance as it is recorded.
        """
        gate = VoiceGate(silence_seconds=silence_seconds)
        blocks: queue.Queue[np.ndarray] = queue.Queue()
        kept: list[np.ndarray] = []
        pre_roll: list[np.ndarray] = []  # a little audio from just before speech was detected

        print("(listening... speak now)", flush=True)
        events.state("listening")
        self.audio.subscribe(blocks.put)
        try:
            waited = 0.0
            while True:
                block = blocks.get()
                state = gate.feed(block)
                if state == "wait":
                    pre_roll = (pre_roll + [block])[-6:]
                    waited += BLOCK_SECONDS
                    if waited >= wait_seconds:
                        print("(heard nothing)", flush=True)
                        return None
                    continue
                if not kept:
                    kept.extend(pre_roll)
                    for frame in pre_roll:
                        self.feed(frame)
                    events.state("recording")
                kept.append(block)
                self.feed(block)
                if state == "done" or len(kept) * BLOCK_SECONDS >= max_seconds:
                    break
        finally:
            self.audio.unsubscribe(blocks.put)
        cue(self.audio, "parse", wait=False)
        print("(transcribing)", flush=True)
        events.state("transcribing")
        return np.concatenate(kept)

    def transcribe(self, audio: np.ndarray) -> str:
        """Text for a whole utterance (the batch path)."""
        return self._clean(self.transcriber.transcribe(np.asarray(audio, dtype=np.float32)))

    # An utterance fed frame by frame while it is being recorded: a streaming backend works
    # on it meanwhile and has the text ready when it ends; the others transcribe at ``finish``.
    def start(self) -> None:
        self.transcriber.start()

    def feed(self, frame: np.ndarray) -> None:
        self.transcriber.feed(frame)

    def finish(self) -> str:
        return self._clean(self.transcriber.finish())

    @staticmethod
    def _clean(text: str) -> str:
        text = text.strip()
        return "" if text.lower().strip(".!? ") in _HALLUCINATIONS else text

    def close(self) -> None:
        self.transcriber.close()

    def listen(self) -> str:
        """Record one utterance and return its text ("" if nothing usable was said)."""
        self.start()
        audio = self.record()
        if audio is None:
            self.finish()  # nothing was fed; just closes the utterance
            return ""
        return self.finish()

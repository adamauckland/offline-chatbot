from __future__ import annotations

import random
import threading

import events
from audio.input import AudioIO, tone

GAIN = 0.01
INTERVAL_SECONDS = 0.25
DEBOUNCE_SECONDS = 0.15
NOTE_SECONDS = 0.06
C_MAJOR_PENTATONIC_C5_TO_C6 = (523.25, 587.33, 659.25, 783.99, 880.0, 1046.5)
BUSY_STATES = {"transcribing", "thinking", "tool"}


class Turnaround:
    def __init__(self, audio: AudioIO, gain: float = GAIN, interval: float = INTERVAL_SECONDS) -> None:
        self.audio = audio
        self.gain = gain
        self.interval = interval
        self._rng = random.Random()
        self._busy = threading.Event()
        self._closed = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="turnaround")

        events.subscribe(self._on_event)
        self._thread.start()

    def _on_event(self, name: str, data: dict) -> None:
        if name == "state":
            self._set_busy(data.get("state") in BUSY_STATES)
        elif name == "cancel":
            self._busy.clear()

    def _set_busy(self, busy: bool) -> None:
        if busy:
            self._busy.set()
        else:
            self._busy.clear()

    def _run(self) -> None:
        while not self._closed.is_set():
            self._busy.wait()

            if self._closed.is_set():
                return

            if self._still_busy_after(DEBOUNCE_SECONDS):
                self._play_a_note_if_quiet()
                self._closed.wait(self.interval - DEBOUNCE_SECONDS)

    def _still_busy_after(self, seconds: float) -> bool:
        self._closed.wait(seconds)

        return self._busy.is_set()

    def _play_a_note_if_quiet(self) -> None:
        if self.audio.is_idle():
            note = self._rng.choice(C_MAJOR_PENTATONIC_C5_TO_C6)
            self.audio.play(tone(note, NOTE_SECONDS, volume=self.gain))

    @property
    def busy(self) -> bool:
        return self._busy.is_set()

    def close(self) -> None:
        events.unsubscribe(self._on_event)
        self._busy.set()
        self._closed.set()

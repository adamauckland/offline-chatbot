"""Random notes while the bot is working on a spoken answer, so the wait is not silence.

The turnaround phase runs from the moment the bot starts making sense of what
you said until it begins to speak: transcribing, thinking, calling tools.
During it a short random note from a pentatonic scale is played every
``interval`` seconds at a low gain, so you can hear the bot is busy.

It is driven by the state events on the bus (see ``events.bus``), so nothing
in the chat loop calls it: it starts on ``transcribing`` or ``thinking`` and
stops on ``speaking`` and on every state that means the turn is over. A note
is only queued when nothing else is playing, so notes never pile up behind a
cue or slip in between chunks of speech.
"""

from __future__ import annotations

import random
import threading

import events
from audio.input import AudioIO, tone

GAIN = 0.01
INTERVAL = 0.25  # seconds between notes
DEBOUNCE = 0.15  # the bot must have been busy this long before the first note; a flicker stays silent
NOTE_SECONDS = 0.06
# C major pentatonic, C5 to C6: any two of these sound fine in a row.
NOTES = (523.25, 587.33, 659.25, 783.99, 880.0, 1046.5)
# Playing while the bot is in one of these; silent in any other state.
BUSY_STATES = {"transcribing", "thinking", "tool"}


class Turnaround:
    """Plays the notes while the bot is between hearing you and answering."""

    def __init__(self, audio: AudioIO, gain: float = GAIN, interval: float = INTERVAL) -> None:
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
            if data.get("state") in BUSY_STATES:
                self._busy.set()
            else:
                self._busy.clear()
        elif name == "cancel":
            self._busy.clear()

    def _run(self) -> None:
        while not self._closed.is_set():
            self._busy.wait()
            if self._closed.is_set():
                return
            self._closed.wait(DEBOUNCE)
            if not self._busy.is_set():
                continue  # it was only a flicker between two states
            if self.audio.is_idle():  # otherwise a cue or speech is still playing: skip this beat
                self.audio.play(tone(self._rng.choice(NOTES), NOTE_SECONDS, volume=self.gain))
            self._closed.wait(self.interval - DEBOUNCE)

    @property
    def busy(self) -> bool:
        return self._busy.is_set()

    def close(self) -> None:
        events.unsubscribe(self._on_event)
        self._busy.set()  # release the wait so the thread sees it is closed
        self._closed.set()

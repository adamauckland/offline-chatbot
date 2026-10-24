"""Melodic notes from a General MIDI SoundFont: the piano behind ``play_note`` and ``keep_playing_note``.

Lives in the audio engine's process (see ``audio/server.py``) as a ``Layer``.
The SoundFont is TimGM6mb, a complete 6 MB General MIDI bank, fetched once
from Debian's archive into ``assets/`` (any GM bank works: drop one in as
``assets/gm-soundfont.sf2``). Preset 0 of bank 0 is the acoustic piano.

A one-shot note is held for ``seconds`` and then released, and the layer
ends once the release tail has died away. A held note is re-struck every
``repeat`` seconds, since a piano note decays; it carries on until "stop".
"""

from __future__ import annotations

import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

from audio.input import STREAM_RATE, Layer

SOUNDFONT = Path("assets/gm-soundfont.sf2")
SOUNDFONT_URL = "http://deb.debian.org/debian/pool/main/t/timgm6mb-soundfont/timgm6mb-soundfont_1.3.orig.tar.gz"
PIANO = 0  # General MIDI program: acoustic grand
VELOCITY = 110
GAIN = 2.5  # the bank's piano is quiet
RELEASE_SECONDS = 1.5  # how long a note rings after release before the layer ends
REPEAT_SECONDS = 1.6  # a held note is struck again this often


def ensure_soundfont(path: Path = SOUNDFONT) -> Path:
    """The GM bank, downloaded on first use."""
    if path.is_file():
        return path
    print("Downloading the General MIDI SoundFont (6 MB, once)...", file=sys.stderr)
    path.parent.mkdir(parents=True, exist_ok=True)
    archive = path.with_suffix(".tar.gz")
    urllib.request.urlretrieve(SOUNDFONT_URL, archive)
    with tarfile.open(archive) as tar:
        member = next(m for m in tar.getmembers() if m.name.lower().endswith(".sf2"))
        with tar.extractfile(member) as src, open(path, "wb") as dst:  # type: ignore[union-attr]
            dst.write(src.read())
    archive.unlink()
    return path


class SynthNote(Layer):
    """One note on a GM instrument: struck once, or re-struck every ``repeat`` seconds until removed."""

    gain = GAIN

    def __init__(
        self,
        midi: int,
        seconds: float | None = None,
        repeat: float | None = None,
        velocity: int = VELOCITY,
        program: int = PIANO,
        soundfont: Path = SOUNDFONT,
    ) -> None:
        import tinysoundfont

        self.midi, self.velocity = int(midi), int(velocity)
        self.seconds, self.repeat = seconds, repeat
        self.finished = False
        self._synth = tinysoundfont.Synth(samplerate=STREAM_RATE)
        sfid = self._synth.sfload(str(ensure_soundfont(soundfont)))
        self._synth.program_select(0, sfid, 0, program)
        self._rendered = 0  # frames rendered so far
        self._released_at: int | None = None
        self._strike()

    def _strike(self) -> None:
        self._synth.noteon(0, self.midi, self.velocity)
        self._struck_at = self._rendered

    def retune(self, midi: int) -> None:
        """Change the pitch of a held note: the old one is released, the new one struck."""
        self._synth.noteoff(0, self.midi)
        self.midi = int(midi)
        self._strike()

    def render(self, frames: int) -> np.ndarray:
        out = np.frombuffer(self._synth.generate(frames), dtype=np.float32).reshape(-1, 2).copy()
        self._rendered += frames
        held_for = (self._rendered - self._struck_at) / STREAM_RATE
        if self.repeat is not None:
            if held_for >= self.repeat:
                self._strike()
        elif self._released_at is None:
            if self.seconds is not None and held_for >= self.seconds:
                self._synth.noteoff(0, self.midi)
                self._released_at = self._rendered
        elif (self._rendered - self._released_at) / STREAM_RATE >= RELEASE_SECONDS:
            self.finished = True
        return out

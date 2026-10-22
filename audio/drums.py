"""The ``play_drums`` and ``set_tempo`` tools: a drum loop from a SoundFont, mixed into the output.

The loop is a ``Layer`` on the shared audio stream (see ``audio/input.py``), so
it plays under speech and cues without interrupting them, keeps time
sample-accurately, and stops the instant the user says "stop". The sounds
come from a General MIDI drum SoundFont rendered by TinySoundFont (the
``tinysoundfont`` package), loaded on first use because the bank is large.

Patterns (``audio/patterns.py``) are a bar of steps; each step names the
drums that hit on it. The machine renders one step at a time, so a tempo, pattern or kit change asked
for mid-bar takes effect at the start of the next bar, as a drum machine
would do it.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from audio.input import STREAM_RATE, AudioIO, Layer
from audio.patterns import DEFAULT_VELOCITY, PATTERNS, describe, resolve
from tools import Tool

SOUNDFONT = Path("assets/drums-soundfont.sf2")
DRUM_BANK = 128  # General MIDI percussion
DEFAULT_BPM = 120
MIN_BPM, MAX_BPM = 40, 220
TEMPO_STEP = 10  # "faster" / "slower"
VOLUME = 0.6

DEFAULT_PATTERN = "rock"


def parse_tempo(value, current: int) -> int:
    """A BPM from what the model sends: a number ("130", "130 bpm") or a word ("faster", "much slower")."""
    text = str(value).lower().strip()
    match = re.search(r"\d+(\.\d+)?", text)
    if match:
        bpm = float(match.group())
    else:
        step = TEMPO_STEP * (2 if re.search(r"\b(much|lot|way)\b", text) else 1)
        if re.search(r"\b(fast|quick|up|speed)", text):
            bpm = current + step
        elif re.search(r"\b(slow|down)", text):
            bpm = current - step
        else:
            raise ValueError(f"not a tempo: {value!r}; say a number of beats per minute, or faster or slower")
    return int(max(MIN_BPM, min(MAX_BPM, round(bpm))))


class DrumMachine(Layer):
    """Renders the loop step by step from the SoundFont; changes apply at the bar line."""

    def __init__(
        self,
        soundfont: Path = SOUNDFONT,
        bpm: int = DEFAULT_BPM,
        pattern: str = DEFAULT_PATTERN,
        kit: int = 1,
        volume: float = VOLUME,
    ) -> None:
        self.soundfont = Path(soundfont)
        self.gain = volume
        self.bpm, self.pattern, self.kit = bpm, pattern, kit
        self._pending: dict = {}  # changes waiting for the next bar
        self._synth = None
        self._sfid = 0
        self._step = 0
        self._buffer = np.zeros((0, 2), dtype=np.float32)

    # -- set-up ----------------------------------------------------------------
    def load(self) -> None:
        """Load the SoundFont (a few hundred milliseconds); done before the loop is attached."""
        if self._synth is not None:
            return
        import tinysoundfont  # the C synth; renders many times faster than real time

        if not self.soundfont.is_file():
            raise FileNotFoundError(f"no drum SoundFont at {self.soundfont}")
        self._synth = tinysoundfont.Synth(samplerate=STREAM_RATE)
        self._sfid = self._synth.sfload(str(self.soundfont))
        self._select_kit(self.kit)

    def kit_name(self, kit: int) -> str | None:
        return self._synth.sfpreset_name(self._sfid, DRUM_BANK, kit - 1) if self._synth is not None else None

    def _select_kit(self, kit: int) -> None:
        if self.kit_name(kit) is None:
            raise ValueError(f"no drum kit {kit} in {self.soundfont.name}")
        self._synth.program_select(0, self._sfid, DRUM_BANK, kit - 1, is_drums=True)
        self.kit = kit

    def unload(self) -> None:
        self._synth = None
        self._buffer = np.zeros((0, 2), dtype=np.float32)
        self._step = 0

    # -- live changes, applied at the next bar line ----------------------------------
    def set_tempo(self, bpm: int) -> None:
        self._pending["bpm"] = int(bpm)

    def set_pattern(self, name: str) -> None:
        if name not in PATTERNS:
            raise ValueError(f"unknown pattern {name!r}; {describe()}")
        self._pending["pattern"] = name

    def set_kit(self, kit: int) -> None:
        if self._synth is not None and self.kit_name(kit) is None:
            raise ValueError(f"no drum kit {kit} in {self.soundfont.name}")
        self._pending["kit"] = int(kit)

    @property
    def tempo(self) -> int:
        """The tempo the loop is heading for: the pending one if a change is waiting."""
        return int(self._pending.get("bpm", self.bpm))

    def pending_or_current(self, field: str):
        return self._pending.get(field, getattr(self, field))

    @staticmethod
    def soundfont_default() -> str:
        return str(SOUNDFONT.resolve())

    def request(self, params: dict, playing: bool) -> None:
        """Take a tool's request (pattern, bpm, kit, any of them) as pending changes; raises on a bad one."""
        if params.get("pattern"):
            name = resolve(params["pattern"])
            if name is None:
                raise ValueError(f"unknown pattern {params['pattern']!r}; {describe()}")
            self.set_pattern(name)
            if params.get("bpm") in (None, "") and not playing and "bpm" in PATTERNS[name]:
                self.set_tempo(PATTERNS[name]["bpm"])  # the style's usual tempo, when starting fresh without one
        if params.get("bpm") not in (None, ""):
            self.set_tempo(parse_tempo(params["bpm"], self.tempo))
        if params.get("kit") not in (None, ""):
            self.set_kit(int(str(params["kit"]).strip().split()[-1]))
        self._requested_while_playing = playing

    def state(self, playing: bool) -> dict:
        """What the tools report: the settings the loop is heading for."""
        return {
            "playing": playing,
            "bpm": self.tempo,
            "pattern": self.pending_or_current("pattern"),
            "kit": self.pending_or_current("kit"),
            "changed_at": "next bar" if getattr(self, "_requested_while_playing", False) else "now",
        }

    def apply_pending_now(self) -> None:
        """Take the pending changes immediately (used when the loop is not playing)."""
        self._apply_pending()
        self._step = 0
        self._buffer = np.zeros((0, 2), dtype=np.float32)

    def _apply_pending(self) -> None:
        if "bpm" in self._pending:
            self.bpm = self._pending["bpm"]
        if "pattern" in self._pending:
            self.pattern = self._pending["pattern"]
        if "kit" in self._pending and self._synth is not None:
            self._select_kit(self._pending["kit"])
        self._pending.clear()

    # -- rendering, called from the audio callback ---------------------------------------
    def render(self, frames: int) -> np.ndarray:
        while len(self._buffer) < frames:
            self._render_step()
        out, self._buffer = self._buffer[:frames], self._buffer[frames:]
        return out

    def _render_step(self) -> None:
        if self._step == 0:
            self._apply_pending()  # the bar line: the only place tempo, pattern and kit change
        pattern = PATTERNS[self.pattern]
        step_frames = int(round(60.0 / self.bpm * pattern.get("beats", 4) / pattern["steps"] * STREAM_RATE))
        for hit in pattern["hits"][self._step]:
            key, velocity = hit if isinstance(hit, tuple) else (hit, DEFAULT_VELOCITY)
            self._synth.noteon(0, key, velocity)
        samples = np.frombuffer(self._synth.generate(step_frames), dtype=np.float32).reshape(-1, 2)
        self._buffer = np.concatenate([self._buffer, samples])
        self._step = (self._step + 1) % pattern["steps"]


# -- the tools ------------------------------------------------------------------------
# The machine itself lives in the audio engine's process (see audio/server.py); these send it
# requests through ``AudioIO.drums`` and word the replies.


def play_drums(audio: AudioIO, pattern=DEFAULT_PATTERN, bpm=None, kit=None) -> dict:
    """Start (or change) the loop; it plays until the user says stop."""
    state = audio.drums("start", soundfont=DrumMachine.soundfont_default(), pattern=pattern, bpm=bpm, kit=kit)
    if "error" in state:
        return {"ok": False, "error": state["error"]}
    return {"ok": True, **state}


def set_tempo(audio: AudioIO, bpm) -> dict:
    state = audio.drums("set", soundfont=DrumMachine.soundfont_default(), bpm=bpm)
    if "error" in state:
        return {"ok": False, "error": state["error"]}
    return {"ok": True, **state}


def drum_tools(audio: AudioIO) -> list[Tool]:
    """The tool entries for the registry, bound to the running audio stream."""
    return [
        Tool(
            name="play_drums",
            description="Play a drum loop through the speakers until the user says stop; changes to a loop that "
            "is already playing take effect at the next bar.",
            arguments={
                "pattern": f"{describe()} (optional, default {DEFAULT_PATTERN})",
                "bpm": f"tempo in beats per minute, {MIN_BPM} to {MAX_BPM} (optional, default {DEFAULT_BPM})",
                "kit": "drum kit number 1 to 12 (optional)",
            },
            func=lambda pattern=DEFAULT_PATTERN, bpm=None, kit=None: play_drums(audio, pattern, bpm, kit),
            example=("give me a rock beat at 100", {"pattern": "rock", "bpm": 100}),
            reply=lambda r: (
                f"Drums, {r['pattern']} at {r['bpm']} beats per minute"
                + (", from the next bar." if r["changed_at"] == "next bar" else ". Say stop to end them.")
            ),
        ),
        Tool(
            name="set_tempo",
            description="Change the tempo of the drums: a number of beats per minute, or faster or slower. Takes "
            "effect at the next bar.",
            arguments={"bpm": "the new tempo as a number, or 'faster' / 'slower' (optionally 'much')"},
            func=lambda bpm="faster": set_tempo(audio, bpm),
            example=("a bit faster", {"bpm": "faster"}),
            reply=lambda r: (
                f"Tempo {r['bpm']} from the next bar." if r["playing"] else f"Tempo set to {r['bpm']} for next time."
            ),
        ),
    ]

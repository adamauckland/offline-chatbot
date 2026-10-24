"""The band: drums, piano chords and bass on one clock, for playing along to.

A ``Band`` is one ``Layer`` on the audio stream (see ``audio/server.py``) with
a bar clock of ``TICKS`` ticks per bar at one tempo. Each track has its own
SoundFont synth and answers "what sounds on this tick"; the band renders a
tick from every enabled track, applies each track's mute and volume, and
mixes. Because they share the clock, chords and bass stay locked to the
drums however they are started, stopped and muted, and a tempo change
lands on the bar line for all of them.

Tracks: ``drums`` (the pattern library, see ``audio/drums.py``), ``piano``
(diatonic chords of the chosen key, struck on beats 1 and 3) and ``bass``
(roots and fifths of the same chords). The harmony comes from
``audio/theory.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from audio import theory
from audio.drums import DrumMachine, parse_tempo
from audio.input import STREAM_RATE, Layer
from audio.patterns import PATTERNS
from audio.synth import SOUNDFONT as GM_SOUNDFONT, ensure_soundfont

TICKS = 96  # per bar: divisible by 8, 12, 16, 24, 32 and 48, so every pattern grid lands on a tick
PIANO_PROGRAM, BASS_PROGRAM = 0, 33  # General MIDI: acoustic grand, fingered bass
VOLUMES = {"drums": 0.6, "piano": 2.0, "bass": 2.5}  # the GM bank's instruments are quiet
TRACK_ALIASES = {
    "drum": "drums", "percussion": "drums", "beat": "drums", "kit": "drums",
    "chords": "piano", "keys": "piano", "notes": "piano", "keyboard": "piano", "harmony": "piano",
    "bass guitar": "bass", "bassline": "bass", "bass line": "bass", "low end": "bass",
}  # fmt: skip


class Track:
    """One instrument: its synth, whether it is enabled and muted, and its events per tick."""

    name = "?"
    program = 0
    soundfont: Path = GM_SOUNDFONT

    def __init__(self) -> None:
        self.enabled = False
        self.muted = False
        self.volume = VOLUMES.get(self.name, 1.0)
        self._synth = None

    def load(self) -> None:
        if self._synth is not None:
            return
        import tinysoundfont

        self._synth = tinysoundfont.Synth(samplerate=STREAM_RATE)
        sfid = self._synth.sfload(str(ensure_soundfont(self.soundfont)))
        self._synth.program_select(0, sfid, 0, self.program)

    def bar_start(self) -> None:
        """Called on tick 0; the place to take pending changes."""

    def events(self, tick: int) -> list[tuple[str, int, int]]:
        """("on", key, velocity) and ("off", key, 0) for this tick."""
        return []

    def render(self, tick: int, frames: int) -> np.ndarray:
        for kind, key, velocity in self.events(tick):
            if kind == "on":
                self._synth.noteon(0, key, velocity)
            else:
                self._synth.noteoff(0, key)
        return np.frombuffer(self._synth.generate(frames), dtype=np.float32).reshape(-1, 2)

    def silence(self) -> None:
        if self._synth is not None:
            self._synth.notes_off(0)

    def state(self) -> dict:
        return {"enabled": self.enabled, "muted": self.muted}


class DrumTrack(Track):
    """The drum pattern library on the band's clock; the pattern state is a ``DrumMachine``."""

    name = "drums"

    def __init__(self, soundfont: Path) -> None:
        super().__init__()
        self.machine = DrumMachine(soundfont=soundfont)
        self.soundfont = Path(soundfont)

    def load(self) -> None:
        self.machine.load()
        self._synth = self.machine._synth  # the machine owns the drum synth and its kit selection

    def bar_start(self) -> None:
        self.machine._apply_pending()

    @property
    def beats(self) -> int:
        return PATTERNS[self.machine.pattern].get("beats", 4)

    def events(self, tick: int) -> list[tuple[str, int, int]]:
        pattern = PATTERNS[self.machine.pattern]
        steps = pattern["steps"]
        if tick * steps % TICKS:
            return []  # not on this pattern's grid
        step = tick * steps // TICKS
        out = []
        for hit in pattern["hits"][step]:
            key, velocity = hit if isinstance(hit, tuple) else (hit, 100)
            out.append(("on", key, velocity))
        return out


class ChordTrack(Track):
    """Diatonic chords of the key, struck on beats 1 and 3 and held between."""

    name = "piano"
    program = PIANO_PROGRAM
    STRIKES = (0, TICKS // 2)

    def __init__(self) -> None:
        super().__init__()
        self.key = theory.Key(0)
        self.progression = "pop"
        self.chords: list[list[int]] = theory.progression_chords(self.key, self.progression)
        self.bar = 0
        self._pending: tuple[theory.Key, str] | None = None
        self._sounding: list[int] = []

    def set_harmony(self, key: theory.Key, progression: str) -> None:
        self._pending = (key, progression)

    def bar_start(self) -> None:
        if self._pending is not None:
            self.key, self.progression = self._pending
            self.chords = theory.progression_chords(self.key, self.progression)
            self.bar = 0
            self._pending = None
        else:
            self.bar += 1

    def current_chord(self) -> list[int]:
        return self.chords[self.bar % len(self.chords)]

    def notes_for(self, chord: list[int]) -> list[int]:
        return theory.voice(chord)

    def events(self, tick: int) -> list[tuple[str, int, int]]:
        if tick not in self.STRIKES:
            return []
        out = [("off", key, 0) for key in self._sounding]
        self._sounding = self.notes_for(self.current_chord())
        out += [("on", key, 95) for key in self._sounding]
        return out

    def state(self) -> dict:
        key, progression = self._pending or (self.key, self.progression)
        chords = theory.progression_chords(key, progression)
        flats = key.use_flats
        return {
            **super().state(),
            "key": key.name,
            "progression": progression,
            "chords": [theory.chord_name(c, flats) for c in chords],
        }


class BassTrack(ChordTrack):
    """Root and fifth of the current chord, low: beats 1 and 2 root, 3 fifth, 4 root, and a pickup."""

    name = "bass"
    program = BASS_PROGRAM
    LINE = (
        (0, "root"),
        (TICKS // 4, "root"),
        (TICKS // 2, "fifth"),
        (3 * TICKS // 4, "root"),
        (TICKS - TICKS // 8, "fifth_below"),
    )

    def events(self, tick: int) -> list[tuple[str, int, int]]:
        hit = dict(self.LINE).get(tick)
        if hit is None:
            return []
        chord = self.current_chord()
        root = theory.bass_note(chord[0])
        key = {"root": root, "fifth": root + 7 if root + 7 <= 52 else root - 5, "fifth_below": root - 5}[hit]
        out = [("off", k, 0) for k in self._sounding]
        self._sounding = [key]
        return out + [("on", key, 105)]


class Band(Layer):
    """The tracks on one clock. Attached to the stream while any track is enabled."""

    gain = 1.0

    def __init__(self, drum_soundfont: Path) -> None:
        self.tracks: dict[str, Track] = {"drums": DrumTrack(drum_soundfont), "piano": ChordTrack(), "bass": BassTrack()}
        self.bpm = 120
        self._pending_bpm: int | None = None
        self.tick = 0
        self._buffer = np.zeros((0, 2), dtype=np.float32)

    # -- control ------------------------------------------------------------------
    @property
    def active(self) -> bool:
        return any(t.enabled for t in self.tracks.values())

    @property
    def tempo(self) -> int:
        return self._pending_bpm if self._pending_bpm is not None else self.bpm

    def set_tempo(self, bpm: int) -> None:
        self._pending_bpm = int(bpm)

    def enable(self, name: str) -> None:
        track = self.tracks[name]
        track.load()
        if not track.enabled and not self.active:
            self.tick = 0  # starting from silence: the bar begins now
            self._buffer = np.zeros((0, 2), dtype=np.float32)
            self._apply_pending()
        track.enabled = True

    def disable(self, name: str) -> None:
        track = self.tracks[name]
        track.enabled = False
        track.silence()

    def silence_all(self) -> None:
        """Stop everything; mutes are cleared too, so the next start is heard."""
        for name, track in self.tracks.items():
            self.disable(name)
            track.muted = False

    def drums(self) -> DrumTrack:
        return self.tracks["drums"]  # type: ignore[return-value]

    def state(self) -> dict:
        drums = self.drums()
        playing = drums.enabled
        return {
            "playing": self.active,
            "bpm": self.tempo,
            "tracks": {name: t.state() for name, t in self.tracks.items()},
            "drums": {**drums.machine.state(playing), "bpm": self.tempo},
            "backing": self.tracks["piano"].state(),
        }

    # -- rendering ----------------------------------------------------------------------
    def _apply_pending(self) -> None:
        if self._pending_bpm is not None:
            self.bpm, self._pending_bpm = self._pending_bpm, None
        for track in self.tracks.values():
            track.bar_start()

    def render(self, frames: int) -> np.ndarray:
        while len(self._buffer) < frames:
            self._render_tick()
        out, self._buffer = self._buffer[:frames], self._buffer[frames:]
        return out

    def _render_tick(self) -> None:
        if self.tick == 0:
            self._apply_pending()  # the bar line: tempo, pattern, kit, key and progression change here
        beats = self.drums().beats if self.drums().enabled else 4
        tick_frames = max(1, int(round(60.0 / self.bpm * beats / TICKS * STREAM_RATE)))
        mix = np.zeros((tick_frames, 2), dtype=np.float32)
        for track in self.tracks.values():
            if not track.enabled:
                continue
            samples = track.render(self.tick, tick_frames)
            if not track.muted:
                mix += samples[:tick_frames] * track.volume
        self._buffer = np.concatenate([self._buffer, mix])
        self.tick = (self.tick + 1) % TICKS

    # -- requests from the tools --------------------------------------------------------
    def request(self, action: str, params: dict) -> dict:
        """Apply a tool's request; see the tools in ``audio/drums.py`` and ``audio/backing.py``."""
        drums = self.drums()
        if action == "drums_start" or action == "drums_set":
            drums.machine.load()
            drums.machine.request(params, drums.enabled)
            if params.get("bpm") not in (None, ""):
                self.set_tempo(drums.machine.tempo)
            if (
                not drums.enabled
                and "bpm" not in params
                and "bpm" in PATTERNS.get(drums.machine.pending_or_current("pattern"), {})
            ):
                self.set_tempo(drums.machine.tempo)
            if action == "drums_start":
                self.enable("drums")
                drums.muted = False  # asked for by name: heard, whatever was muted before
            elif not self.active:
                self._apply_pending()
        elif action == "drums_stop":
            self.disable("drums")
        elif action in ("backing_start", "backing_set"):
            key = theory.parse_key(params.get("key") or self.tracks["piano"].state()["key"])
            progression = theory.resolve_progression(params.get("progression"), key)
            for name in ("piano", "bass"):
                self.tracks[name].set_harmony(key, progression)  # type: ignore[attr-defined]
            if params.get("bpm") not in (None, ""):
                self.set_tempo(parse_tempo(params["bpm"], self.tempo))
            if action == "backing_start":
                for name in ("drums", "piano", "bass"):
                    self.enable(name)
                    self.tracks[name].muted = False  # a fresh backing track is heard in full
            elif not self.active:
                self._apply_pending()
        elif action == "backing_stop":
            self.disable("piano")
            self.disable("bass")
        elif action == "tempo":
            self.set_tempo(parse_tempo(params["bpm"], self.tempo))
            if not self.active:
                self._apply_pending()
        elif action in ("mute", "unmute", "solo"):
            name = resolve_track(params.get("track", ""))
            if name is None:
                raise ValueError(f"which track? drums, piano or bass (not {params.get('track')!r})")
            if action == "solo":
                for other, track in self.tracks.items():
                    track.muted = other != name
            else:
                self.tracks[name].muted = action == "mute"
        elif action == "stop":
            self.silence_all()
        else:
            raise ValueError(f"unknown band action {action!r}")
        return self.state()


def resolve_track(text: str) -> str | None:
    t = str(text).lower().strip()
    for name in ("drums", "piano", "bass"):
        if name in t:
            return name
    for alias in sorted(TRACK_ALIASES, key=len, reverse=True):
        if alias in t:
            return TRACK_ALIASES[alias]
    return None

"""The ``play_note``, ``keep_playing_note`` and ``stop_sound`` tools: musical notes through the speakers.

A note is named the usual way, a letter A to G, an optional sharp (``#``) or
flat (``b``), and an octave (``C4`` is middle C, ``A4`` is 440 Hz); the octave
defaults to 4 so "A" or "F#" work too. The notes are played on the piano of a
General MIDI SoundFont (see ``audio/synth.py``), falling back to a plain sine
tone when that cannot be loaded.
"""

from __future__ import annotations

import re

from audio.input import AudioIO, tone
from tools import Tool

DEFAULT_SECONDS = 0.6
MAX_SECONDS = 5.0
VOLUME = 0.3
# Said on its own while a note is held, this stops it (typed, or spoken in any listening mode).
STOP_WORDS = {"stop", "stop it", "stop that", "stop playing", "stop the note", "enough", "that's enough", "ok stop"}

_SEMITONES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_NOTE = re.compile(r"^\s*([A-Ga-g])\s*([#♯]|[b♭]|sharp|flat)?\s*(-?\d)?\s*$")


def note_frequency(name: str) -> float:
    """Frequency in Hz of a note name like ``A4``, ``C#5``, ``Bb3`` or ``G`` (octave 4 assumed)."""
    match = _NOTE.match(name.replace("-sharp", "#").replace("-flat", "b").replace(" sharp", "#").replace(" flat", "b"))
    if match is None:
        raise ValueError(f"not a note: {name!r}; say something like C4, A, F#5 or Bb3")
    letter, accidental, octave = match.groups()
    semitone = _SEMITONES[letter.upper()]
    if accidental in ("#", "♯", "sharp"):
        semitone += 1
    elif accidental in ("b", "♭", "flat"):
        semitone -= 1
    octave_number = int(octave) if octave is not None else 4
    midi = 12 * (octave_number + 1) + semitone
    return 440.0 * 2 ** ((midi - 69) / 12)


def note_midi(frequency: float) -> int:
    """The MIDI key nearest ``frequency``."""
    import math

    return int(round(69 + 12 * math.log2(frequency / 440.0)))


def play_note(audio: AudioIO, note: str = "A4", seconds: float | str = DEFAULT_SECONDS) -> dict:
    """Play ``note`` for ``seconds`` on the piano (a plain tone if the SoundFont is not available)."""
    try:
        frequency = note_frequency(str(note))
        length = max(0.05, min(MAX_SECONDS, float(seconds)))
    except (TypeError, ValueError) as exc:
        return {"ok": False, "error": str(exc)}
    state = audio.note("play", midi=note_midi(frequency), seconds=length) if hasattr(audio, "note") else {"error": "-"}
    instrument = "piano"
    if "error" in state:
        audio.play(tone(frequency, length, volume=VOLUME))
        instrument = "tone"
    return {
        "ok": True,
        "note": str(note).strip(),
        "frequency_hz": round(frequency, 2),
        "seconds": length,
        "instrument": instrument,
    }


def keep_playing_note(audio: AudioIO, note: str = "A4") -> dict:
    """Hold ``note`` on the piano, struck again every so often, until the user says stop."""
    try:
        frequency = note_frequency(str(note))
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    state = audio.note("hold", midi=note_midi(frequency)) if hasattr(audio, "note") else {"error": "-"}
    if "error" in state:
        audio.hold_note(frequency)
    return {
        "ok": True,
        "note": str(note).strip(),
        "frequency_hz": round(frequency, 2),
        "message": "Playing until the user says stop.",
    }


def stop_sound(audio: AudioIO) -> dict:
    """Stop whatever is playing on, a held note or the drums; fine to call when nothing is."""
    if not audio.has_layers:
        return {"ok": True, "message": "Nothing was playing."}
    audio.stop_layers()
    return {"ok": True, "message": "Stopped."}


def spoken_note(name: str) -> str:
    """A note name as words the speaker will read right: "C#5" -> "C sharp 5", "Bb3" -> "B flat 3"."""
    match = _NOTE.match(name.replace("-sharp", "#").replace("-flat", "b").replace(" sharp", "#").replace(" flat", "b"))
    if match is None:
        return name
    letter, accidental, octave = match.groups()
    words = [letter.upper()]
    if accidental in ("#", "♯", "sharp"):
        words.append("sharp")
    elif accidental in ("b", "♭", "flat"):
        words.append("flat")
    if octave is not None:
        words.append(octave)
    return " ".join(words)


def asked_to_stop(line: str) -> bool:
    """True when ``line`` is one of the stop words, on its own."""
    return line.lower().strip(" .,!?") in STOP_WORDS


def note_tools(audio: AudioIO) -> list[Tool]:
    """The tool entries for the registry, bound to the running audio stream."""
    note_arg = "the note name, e.g. 'A4', 'C#5', 'Bb3' or just 'G' (octave 4 if not given)"
    return [
        Tool(
            name="play_note",
            description="Play a musical note through the speakers.",
            arguments={"note": note_arg, "seconds": "how long to hold it, 0.05 to 5 (optional, default 0.6)"},
            func=lambda note="A4", seconds=DEFAULT_SECONDS: play_note(audio, note, seconds),
            example=("play me a C sharp", {"note": "C#4", "seconds": 0.6}),
            reply=lambda r: f"Played {spoken_note(r['note'])} for {r['seconds']:g} seconds.",
        ),
        Tool(
            name="keep_playing_note",
            description="Keep playing a musical note until the user says stop.",
            arguments={"note": note_arg},
            func=lambda note="A4": keep_playing_note(audio, note),
            example=("keep playing a G", {"note": "G4"}),
            reply=lambda r: f"Playing {spoken_note(r['note'])} until you say stop.",
        ),
        Tool(
            name="stop_sound",
            description="Stop the note or the drums that are playing.",
            func=lambda: stop_sound(audio),
            example=("stop the drums", {}),
            reply=lambda r: r["message"],
        ),
    ]


def note_tool(audio: AudioIO) -> Tool:
    """The ``play_note`` entry alone (see ``note_tools`` for all three)."""
    return note_tools(audio)[0]

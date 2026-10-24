"""Just enough music theory for a backing loop: keys, diatonic chords, progressions, voicings.

Written here rather than taken from music21 because the engine needs about
thirty lines of scale arithmetic and music21 is a large library that is slow
to import; if chord symbols or MIDI files ever need reading, that is where
it would come in.

Pitch classes are 0..11 with C = 0; MIDI keys are integers with middle C = 60.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

NOTE_PCS = {
    "c": 0, "c#": 1, "db": 1, "d": 2, "d#": 3, "eb": 3, "e": 4, "f": 5, "f#": 6, "gb": 6,
    "g": 7, "g#": 8, "ab": 8, "a": 9, "a#": 10, "bb": 10, "b": 11,
}  # fmt: skip
SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
FLAT_KEYS = {5, 10, 3, 8, 1, 6}  # F, Bb, Eb, Ab, Db, Gb are spelt with flats
MAJOR = [0, 2, 4, 5, 7, 9, 11]
MINOR = [0, 2, 3, 5, 7, 8, 10]  # natural minor


@dataclass(frozen=True)
class Key:
    tonic: int  # pitch class
    minor: bool = False
    flats: bool | None = None  # how the user spelt it; None picks the usual spelling for the key

    @property
    def use_flats(self) -> bool:
        if self.flats is not None:
            return self.flats
        return self.tonic in FLAT_KEYS and not self.minor

    @property
    def name(self) -> str:
        return f"{pitch_name(self.tonic, flats=self.use_flats)} {'minor' if self.minor else 'major'}"

    def scale(self) -> list[int]:
        return [(self.tonic + step) % 12 for step in (MINOR if self.minor else MAJOR)]


def pitch_name(pc: int, flats: bool = False) -> str:
    return (FLAT_NAMES if flats else SHARP_NAMES)[pc % 12]


_KEY = re.compile(r"\b([a-g])\s*(#|♯|sharp|b|♭|flat)?\s*(major|maj|minor|min|m)?(?![a-z])", re.I)


def parse_key(text: str) -> Key:
    """The key named in ``text``: "A minor", "F#", "key of Bb major", "Dm", "d min". Major unless said."""
    cleaned = str(text).strip()
    cleaned = re.sub(r"\b(the )?key of\b", "", cleaned, flags=re.I)
    match = _KEY.search(cleaned)
    if match is None:
        raise ValueError(f"not a key: {text!r}; say something like A minor, F sharp major or Bb")
    letter, accidental, mode = match.groups()
    name = letter.lower()
    if accidental in ("#", "♯", "sharp"):
        name += "#"
    elif accidental in ("b", "♭", "flat"):
        name += "b"
    minor = bool(mode) and mode.lower().startswith("m") and mode.lower() not in ("maj", "major")
    if not mode and re.search(r"\bminor\b", cleaned, re.I):
        minor = True
    flats = True if name.endswith("b") else False if name.endswith("#") else None
    return Key(NOTE_PCS[name], minor, flats)


def diatonic_chord(key: Key, degree: int, seventh: bool = False) -> list[int]:
    """The chord built on scale ``degree`` (1..7): root, third, fifth and optionally the seventh, as pitch classes."""
    scale = key.scale()
    steps = (0, 2, 4, 6) if seventh else (0, 2, 4)
    return [scale[(degree - 1 + step) % 7] for step in steps]


def chord_name(pcs: list[int], flats: bool = False) -> str:
    """ "A minor", "F", "B dim", "G7" from the pitch classes of a chord."""
    root = pcs[0]
    third = (pcs[1] - root) % 12
    fifth = (pcs[2] - root) % 12
    name = pitch_name(root, flats)
    if fifth == 6:
        name += " dim"
    elif third == 3:
        name += " minor"
    elif fifth == 8:
        name += " aug"
    if len(pcs) > 3:
        name += " maj7" if (pcs[3] - root) % 12 == 11 and third == 4 else "7"
    return name


# Chord progressions as scale degrees, one chord per bar; ``sevenths`` adds the seventh to each chord.
PROGRESSIONS: dict[str, dict] = {
    "pop": {"degrees": [1, 5, 6, 4]},
    "rock": {"degrees": [1, 4, 5, 4]},
    "folk": {"degrees": [1, 4, 5, 1]},
    "ballad": {"degrees": [1, 6, 4, 5]},
    "blues": {"degrees": [1, 1, 1, 1, 4, 4, 1, 1, 5, 4, 1, 5], "sevenths": True, "dominant": True},
    "jazz": {"degrees": [2, 5, 1, 1], "sevenths": True},
    "minor": {"degrees": [1, 6, 3, 7]},
    "andalusian": {"degrees": [1, 7, 6, 5]},
    "one chord": {"degrees": [1]},
}
PROGRESSION_ALIASES = {"four chord": "pop", "four chords": "pop", "axis": "pop", "twelve bar": "blues", "12 bar": "blues",
                       "two five one": "jazz", "251": "jazz", "ii v i": "jazz", "drone": "one chord", "single": "one chord"}  # fmt: skip


def default_progression(key: Key) -> str:
    return "minor" if key.minor else "pop"


def resolve_progression(text: str | None, key: Key) -> str:
    if not text:
        return default_progression(key)
    t = str(text).lower().strip()
    if t in PROGRESSIONS:
        return t
    if t in PROGRESSION_ALIASES:
        return PROGRESSION_ALIASES[t]
    for name in sorted(list(PROGRESSIONS) + list(PROGRESSION_ALIASES), key=len, reverse=True):
        if name in t:
            return PROGRESSIONS.get(name) and name or PROGRESSION_ALIASES[name]
    raise ValueError(f"unknown progression {text!r}; one of {', '.join(PROGRESSIONS)}")


def progression_chords(key: Key, name: str) -> list[list[int]]:
    """The chords of ``name`` in ``key``, one list of pitch classes per bar."""
    spec = PROGRESSIONS[name]
    chords = [diatonic_chord(key, d, spec.get("sevenths", False)) for d in spec["degrees"]]
    if spec.get("dominant"):  # the blues: every chord takes a flat seventh, whatever the scale says
        chords = [c[:3] + [(c[0] + 10) % 12] for c in chords]
    return chords


def voice(pcs: list[int], centre: int = 60) -> list[int]:
    """MIDI keys for a chord in close position near ``centre`` (middle C)."""
    root = centre - 6 + ((pcs[0] - (centre - 6)) % 12)
    notes = [root]
    for pc in pcs[1:]:
        candidate = notes[-1] + ((pc - notes[-1]) % 12 or 12)
        notes.append(candidate)
    while sum(notes) / len(notes) > centre + 7:  # drop an octave if the stack sits high
        notes = [n - 12 for n in notes]
    return notes


def bass_note(pc: int, low: int = 40) -> int:
    """A MIDI key for ``pc`` in the bass guitar's lower octave (E1 = 28 is open E; 40 is E2)."""
    return low + ((pc - low) % 12)

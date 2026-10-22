"""Drum patterns for the drum machine: grooves in common styles, and the standard rudiments.

A pattern is one bar. ``steps`` is how many equal steps the bar has (8 for
eighth notes, 12 for triplet eighths, 16 for sixteenths, 32 for thirty-seconds,
which flams need), ``beats`` how many quarter-note beats it spans (4 unless
said otherwise; the waltz has 3), and ``hits`` one tuple per step naming the
drums that sound on it. A hit is a General MIDI drum key, or ``(key, velocity)``
when it should be louder or softer than the default: ``ACCENT`` for the strong
strokes a rudiment is built on, ``GHOST`` for the barely-there snare notes that
make funk and hip-hop breathe.

Rudiments are the stickings every drummer learns (single and double stroke
rolls, paradiddles, flams). They are played here on the snare: the hands are
not modelled, only where the accents fall, which is what you hear. A flam is
a soft grace note a thirty-second before the main stroke.

Adding a pattern is one entry in ``GROOVES`` or ``RUDIMENTS``; ``ALIASES`` maps
other names people use onto them ("swing" is jazz, "four on the floor" is
dance).
"""

from __future__ import annotations

import re

# General MIDI percussion keys.
KICK, SIDE_STICK, SNARE, CLAP = 36, 37, 38, 39
HAT, PEDAL_HAT, OPEN_HAT = 42, 44, 46
LOW_TOM, MID_TOM, HIGH_TOM = 43, 47, 50
CRASH, RIDE, RIDE_BELL = 49, 51, 53
TAMBOURINE, COWBELL = 54, 56

# Velocities.
ACCENT, NORMAL, SOFT, GHOST = 120, 100, 70, 45
DEFAULT_VELOCITY = NORMAL

Hit = int | tuple[int, int]
Pattern = dict  # {"steps": int, "beats": int, "hits": list[tuple[Hit, ...]]}


def _bar(steps: int, beats: int = 4, bpm: int | None = None, **lanes: str) -> Pattern:
    """Build a pattern from drum lanes written as strings, one character per step.

    ``x`` is a normal stroke, ``X`` an accent, ``o`` soft, ``g`` a ghost note,
    ``.`` nothing. The lane name is the drum (lower-case key name). ``bpm`` is
    the tempo the style is usually played at, used when the loop starts with
    this pattern and no tempo was asked for.
    """
    keys = {
        "kick": KICK, "snare": SNARE, "side_stick": SIDE_STICK, "clap": CLAP, "hat": HAT, "pedal_hat": PEDAL_HAT,
        "open_hat": OPEN_HAT, "ride": RIDE, "ride_bell": RIDE_BELL, "crash": CRASH, "low_tom": LOW_TOM,
        "mid_tom": MID_TOM, "high_tom": HIGH_TOM, "tambourine": TAMBOURINE, "cowbell": COWBELL,
    }  # fmt: skip
    velocity = {"x": NORMAL, "X": ACCENT, "o": SOFT, "g": GHOST}
    hits: list[list[Hit]] = [[] for _ in range(steps)]
    for lane, line in lanes.items():
        line = line.replace(" ", "").replace("|", "")
        if len(line) != steps:
            raise ValueError(f"lane {lane!r} has {len(line)} steps, pattern has {steps}")
        for i, char in enumerate(line):
            if char != ".":
                hits[i].append((keys[lane], velocity[char]))
    pattern: Pattern = {"steps": steps, "beats": beats, "hits": [tuple(h) for h in hits]}
    if bpm is not None:
        pattern["bpm"] = bpm
    return pattern


# Grooves, in common styles. Lanes read left to right across the bar; a space every beat.
GROOVES: dict[str, Pattern] = {
    "rock": _bar(8, kick="x... xx..", snare="..x. ..x.", hat="xxxx xxxx"),
    "pop": _bar(
        16,
        kick="x... ..x. x... ....",
        snare=".... x... .... x...",
        hat="x.x. x.x. x.x. x.x.",
        open_hat=".... .... .... ...x",
    ),
    "ballad": _bar(8, bpm=70, kick="x... ....", snare=".... x...", ride="xxxx xxxx", crash="x... ...."),
    "punk": _bar(8, bpm=180, kick="x.x. x.x.", snare="..X. ..X.", hat="xxxx xxxx", crash="x... ...."),
    "dance": _bar(8, kick="x.x. x.x.", clap="..x. ..x.", hat="x.x. x.x.", open_hat=".x.x .x.x"),
    "shuffle": _bar(12, kick="x.. ... x.. x..", snare="... x.. ... x..", hat="x.x x.x x.x x.x"),
    "jazz": _bar(
        12,
        ride="X.. X.x X.. X.x",
        pedal_hat="... x.. ... x..",
        kick="o.. o.. o.. o..",
        snare="... ... ..g ...",
    ),
    "funk": _bar(
        16,
        kick="x..x ..x. ..x. x...",
        snare=".... x..g .g.. x...",
        hat="xxxx xxxx xxxx xxxx",
        open_hat=".... .... .... ..x.",
    ),
    "hiphop": _bar(16, kick="x... ...x ..x. ....", snare=".... x... .... x..g", hat="x.x. x.x. x.x. x.x."),
    "reggae": _bar(8, kick=".... x...", side_stick=".... x...", hat="xxxx xxxx", open_hat="...x ...."),
    "bossa": _bar(
        16,
        kick="x..x x..x x..x x..x",
        side_stick="x..x ..x. ..x. .x..",
        hat="x.x. x.x. x.x. x.x.",
    ),
    "waltz": _bar(6, beats=3, kick="x. .. ..", snare=".. x. x.", hat="xx xx xx"),
    "march": _bar(8, kick="x... x...", snare="Xxxx Xxxx", crash="x... ...."),
    "metal": _bar(16, bpm=160, kick="xxxx xxxx xxxx xxxx", snare=".... x... .... x...", ride="x.x. x.x. x.x. x.x."),
    "train": _bar(16, kick="x... x... x... x...", snare="XxXx XxXx XxXx XxXx", hat="x... x... x... x..."),
    # Latin and world. Clave and bell patterns go on the side stick and cowbell, congas and
    # dhol on the toms: what a kit drummer does with them.
    "salsa": _bar(
        16,
        side_stick="..x. x... x..x ..x.",  # 2-3 son clave
        cowbell="x.xx .x.x x.xx .x.x",  # cascara
        kick=".... ..x. .... ..x.",  # tumbao: the "and" of 2 and 4
        low_tom=".... .... .... x.x.",  # conga open tones on 4
    ),
    "samba": _bar(
        16,
        kick="o... X... o... X...",  # surdo: the big hit on 2 and 4
        hat="xxXx xxXx xxXx xxXx",  # shaker
        side_stick="x..x ..x. x..x .x..",  # tamborim
    ),
    "afrobeat": _bar(
        16,
        kick="x... ..x. ..x. ....",
        snare=".... x..g ..g. x...",
        hat="x.xx x.xx x.xx x.xx",
        open_hat="..x. .... ..x. ....",
    ),
    "bembe": _bar(  # Afro-Cuban 6/8: the bell pattern over a 12/8 bar
        12,
        cowbell="x.x.xx.x.x.x",
        kick="x.....x.....",
        low_tom="...x.....x..",
        pedal_hat="...x.....x..",
    ),
    "cumbia": _bar(
        16,
        kick="x... x... x... x...",
        hat=".x.x .x.x .x.x .x.x",  # guiro's offbeats
        side_stick=".... x..x .... x..x",
        high_tom="..x. .... ..x. ....",
    ),
    "reggaeton": _bar(16, kick="x... x... x... x...", snare="...x ..x. ...x ..x.", hat="x.x. x.x. x.x. x.x."),  # dembow
    "ska": _bar(8, kick="x... x...", snare="..x. ..x.", hat=".x.x .x.x"),
    "bhangra": _bar(
        16,
        kick="x..x ..x. x..x ..x.",  # dhol bass
        high_tom=".x.. x..x .x.. x..x",  # dhol treble
        tambourine="x.x. x.x. x.x. x.x.",
    ),
    "tango": _bar(8, kick="x..x x.x.", snare="..x. ..x.", hat="xxxx xxxx"),  # habanera
}

# Rudiments on the snare, accents where the sticking puts them. A flam's grace note sits a
# thirty-second before the stroke, so those patterns use 32 steps.
RUDIMENTS: dict[str, Pattern] = {
    "single_stroke_roll": _bar(16, snare="Xxxx Xxxx Xxxx Xxxx"),
    "double_stroke_roll": _bar(16, snare="XxXx XxXx XxXx XxXx"),  # RRLL: accent on each hand's first
    "single_paradiddle": _bar(16, snare="Xxxx Xxxx Xxxx Xxxx"),  # RLRR LRLL: accent on each group's first
    "double_paradiddle": _bar(12, snare="Xxxxxx Xxxxxx"),  # RLRLRR LRLRLL
    "triple_paradiddle": _bar(16, snare="Xxxxxxxx Xxxxxxxx"),
    "paradiddle_diddle": _bar(12, snare="Xxxxxx Xxxxxx"),  # RLRRLL
    "five_stroke_roll": _bar(16, snare="xxxxX... xxxxX..."),
    "seven_stroke_roll": _bar(16, snare="xxxxxxX. xxxxxxX."),
    "flam": _bar(32, snare="oX...... oX...... oX...... oX......"),
    "flam_tap": _bar(32, snare="oX..x... oX..x... oX..x... oX..x..."),
    "flam_accent": _bar(24, snare="oX.x.x oX.x.x oX.x.x oX.x.x"),
    "drag": _bar(32, snare="ooX..... ooX..... ooX..... ooX....."),
    "single_stroke_four": _bar(16, snare="xxxX xxxX xxxX xxxX"),
}

PATTERNS: dict[str, Pattern] = {**GROOVES, **RUDIMENTS}

# Other names for the same thing, as people say them.
ALIASES: dict[str, str] = {
    "swing": "jazz",
    "ride": "jazz",
    "four on the floor": "dance",
    "disco": "dance",
    "house": "dance",
    "techno": "dance",
    "boom bap": "hiphop",
    "hip hop": "hiphop",
    "hip-hop": "hiphop",
    "one drop": "reggae",
    "bossa nova": "bossa",
    "latin": "bossa",
    "slow": "ballad",
    "blues": "shuffle",
    "basic": "rock",
    "simple": "rock",
    "double kick": "metal",
    "hardcore": "punk",
    "d-beat": "punk",
    "mambo": "salsa",
    "clave": "salsa",
    "cuban": "salsa",
    "brazil": "samba",
    "batucada": "samba",
    "afro": "afrobeat",
    "african": "afrobeat",
    "six eight": "bembe",
    "6/8": "bembe",
    "dembow": "reggaeton",
    "dhol": "bhangra",
    "punjabi": "bhangra",
    "habanera": "tango",
    "world": "afrobeat",
    "paradiddle": "single_paradiddle",
    "single stroke": "single_stroke_roll",
    "double stroke": "double_stroke_roll",
    "roll": "single_stroke_roll",
    "flams": "flam",
}


DEFAULT = "rock"
# Words that ask for drums without naming a style; they get the default rather than an error.
GENERIC_WORDS = {
    "music",
    "song",
    "tune",
    "beat",
    "beats",
    "drums",
    "drum",
    "groove",
    "rhythm",
    "something",
    "anything",
    "any",
    "default",
    "normal",
    "loop",
}


def resolve(name: str) -> str | None:
    """The pattern a request names: exact, an alias, or the first pattern mentioned in the words."""
    text = str(name).lower().strip().replace("_", " ")
    for candidate in PATTERNS:
        if text == candidate.replace("_", " "):
            return candidate
    if text in ALIASES:
        return ALIASES[text]
    # A pattern named anywhere in the words wins over an alias ("afro beat" is afrobeat, not
    # whatever "beat" might mean); longer names first so "double stroke roll" beats "roll".
    for candidate in sorted(PATTERNS, key=len, reverse=True):
        if candidate.replace("_", " ") in text:
            return candidate
    for alias in sorted(ALIASES, key=len, reverse=True):
        if alias in text:
            return ALIASES[alias]
    nearest = nearest_name(text)
    if nearest is not None:
        return nearest
    if any(word in text.split() for word in GENERIC_WORDS):
        return DEFAULT  # "give me a beat": nothing in particular was asked for
    return None


FUZZY_CUTOFF = 0.75  # how alike a heard word and a name must sound or be spelt
# Words around a style name in a request; never candidates for a misheard name ("some" is not "samba").
FILLER_WORDS = {
    "some", "play", "please", "the", "and", "with", "bit", "little", "nice", "good", "give", "start", "kit",
    "mode", "music", "song", "want", "like", "could", "would", "can", "you", "lets", "let's", "now", "that",
    "this", "more", "again", "fast", "slow", "quick", "loud", "quiet",
}  # fmt: skip


def sound_key(word: str) -> str:
    """A rough "sounds like" form: c as k, vowels collapsed, doubles squashed ("scar" and "ska" both give "sk*")."""
    w = word.lower()
    for a, b in (("ph", "f"), ("ck", "k"), ("qu", "kw"), ("x", "ks"), ("c", "k"), ("z", "s"), ("y", "i"), ("gh", "")):
        w = w.replace(a, b)
    w = re.sub(r"[aeiouh]+", "*", w)
    w = re.sub(r"(.)\1+", r"\1", w)
    w = re.sub(r"\*$", "", w) if len(w) > 2 else w  # a trailing vowel sound is the first thing speech loses
    w = re.sub(r"r$", "", w) if len(w) > 2 else w  # ...and a trailing r is the first thing it adds ("scar")
    return w


def nearest_name(text: str) -> str | None:
    """The pattern a misheard word was probably meant to be ("scar drums" -> ska), or None.

    Speech recognition hears "ska" as "scar", "punk" as "panic", "salsa" as
    "sulsa". Each word and word pair of the request is compared with every
    pattern name and alias both by spelling and by ``sound_key``, and the
    best score above the cutoff wins, provided the first letters agree.
    Generic words are skipped so "drums" cannot match anything.
    """
    import difflib

    words = [w for w in text.split() if w not in GENERIC_WORDS and w not in FILLER_WORDS and len(w) >= 3]
    candidates = words + [f"{a} {b}" for a, b in zip(words, words[1:])]
    names = {name.replace("_", " "): name for name in PATTERNS}
    names.update(ALIASES)
    best, best_score = None, FUZZY_CUTOFF
    for candidate in candidates:
        for spelt, name in names.items():
            if candidate[0] != spelt[0]:
                continue
            score = max(
                difflib.SequenceMatcher(None, candidate, spelt).ratio(),
                difflib.SequenceMatcher(None, sound_key(candidate), sound_key(spelt)).ratio(),
            )
            if score > best_score:
                best, best_score = name, score
    return best


def describe() -> str:
    """The names, for the tool description."""
    return "grooves: " + ", ".join(GROOVES) + "; rudiments: " + ", ".join(RUDIMENTS)

"""The ``play_backing`` and ``set_mix`` tools: a loop in a chosen key to practise along to.

``play_backing`` starts the band (see ``audio/band.py``): drums, piano chords
and bass on one clock, in the key and chord progression asked for, at the
drums' tempo. ``set_mix`` mutes, unmutes or solos a track, so you can play
along with just the drums, just the piano, or the bass by itself. The band
lives in the audio engine's process; these tools send it requests through
``AudioIO.band`` and word the replies.
"""

from __future__ import annotations

from audio.input import AudioIO
from audio.theory import PROGRESSIONS
from tools import Tool


def play_backing(audio: AudioIO, key="C major", progression=None, bpm=None) -> dict:
    state = audio.band("backing_start", key=key, progression=progression, bpm=bpm)
    if "error" in state:
        return {"ok": False, "error": state["error"]}
    backing = state["backing"]
    return {
        "ok": True,
        "key": backing["key"],
        "progression": backing["progression"],
        "chords": backing["chords"],
        "bpm": state["bpm"],
    }


def set_mix(audio: AudioIO, track="drums", state="mute") -> dict:
    action = str(state).lower().strip()
    action = {
        "off": "mute",
        "muted": "mute",
        "silence": "mute",
        "on": "unmute",
        "unmuted": "unmute",
        "only": "solo",
        "alone": "solo",
        "just": "solo",
    }.get(action, action)
    if action not in ("mute", "unmute", "solo"):
        return {"ok": False, "error": f"say mute, unmute or solo (not {state!r})"}
    result = audio.band(action, track=track)
    if "error" in result:
        return {"ok": False, "error": result["error"]}
    muted = [name for name, t in result["tracks"].items() if t["muted"]]
    return {"ok": True, "action": action, "track": track, "muted": muted}


def backing_tools(audio: AudioIO) -> list[Tool]:
    return [
        Tool(
            name="play_backing",
            description="Play a backing loop to practise along to: drums, piano chords and bass in a chosen key "
            "and chord progression, at the drums' tempo, until the user says stop.",
            arguments={
                "key": "the key, e.g. 'A minor', 'G major', 'F#' (optional, default C major)",
                "progression": f"one of {', '.join(PROGRESSIONS)} (optional: pop for major keys, minor for minor)",
                "bpm": "tempo in beats per minute (optional, keeps the current one)",
            },
            func=lambda key="C major", progression=None, bpm=None: play_backing(audio, key, progression, bpm),
            example=("give me a backing track in A minor", {"key": "A minor"}),
            reply=lambda r: (
                f"Backing in {r['key']}, {r['progression']} progression: {', '.join(r['chords'])}, "
                f"at {r['bpm']} beats per minute. Say stop to end it."
            ),
        ),
        Tool(
            name="set_mix",
            description="Mute, unmute or solo one part of the backing loop (the drums, the piano or the bass), so "
            "the user can practise with just some of them.",
            arguments={"track": "drums, piano or bass", "state": "mute, unmute or solo"},
            func=lambda track="drums", state="mute": set_mix(audio, track, state),
            example=("mute the drums", {"track": "drums", "state": "mute"}),
            reply=lambda r: (
                f"{r['track'].capitalize()} {dict(mute='muted', unmute='unmuted', solo='solo')[r['action']]}."
                + (f" Muted: {', '.join(r['muted'])}." if r["muted"] else " Nothing muted.")
            ),
        ),
    ]

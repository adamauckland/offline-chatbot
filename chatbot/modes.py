from __future__ import annotations

import re
from dataclasses import dataclass, field

from tools import TOOLS, Tool

DEFAULT_MODE = "chat"
ALWAYS = ("exit", "clear_context")
LETS_DO = re.compile(r"lets (?:do|go|try|have) (?:some |the )?(\w+)")
APOSTROPHE_S = re.compile(r"\blet['’]s\b")
SPACES = re.compile(r"\s+")


@dataclass(frozen=True)
class Mode:
    name: str
    tools: tuple[str, ...]
    phrases: tuple[str, ...]
    greeting: str
    extra: tuple[str, ...] = field(default_factory=tuple)
    quiet: bool = False  # tool results are printed, not spoken, and no announcement over something playing


MODES: dict[str, Mode] = {
    "chat": Mode(
        "chat",
        (),
        ("lets chat", "lets just chat", "chat mode", "back to chat", "just chat"),
        "Chat mode. No tools, just talk.",
        ("chat", "chatting", "talk", "talking"),
    ),
    "photographer": Mode(
        "photographer",
        ("take_picture",),
        ("lets do photos", "lets take photos", "photo mode", "photographer mode", "camera mode"),
        "Photographer mode. I can take pictures and look at them.",
        ("photos", "photo", "pictures", "photography", "camera"),
    ),
    "music": Mode(
        "music",
        ("play_drums", "play_backing", "set_mix", "set_tempo", "play_note", "keep_playing_note", "stop_sound"),
        ("lets do music", "lets play music", "music mode", "lets make music"),
        "Music mode. Drums, backing loops, tempo and notes.",
        ("music", "drums", "beats"),
        quiet=True,  # speech over the notes and drums is what the music is not for
    ),
    "home": Mode(
        "home",
        ("set_lights", "get_lights", "get_time"),
        ("lets do home", "home mode", "lets do the house"),
        "Home mode. Lights and the time.",
        ("home", "house", "lights"),
    ),
}

YES_WORDS = {
    "yes",
    "yeah",
    "yep",
    "yup",
    "sure",
    "ok",
    "okay",
    "please",
    "yes please",
    "do it",
    "go on",
    "go ahead",
    "switch",
    "fine",
    "alright",
    "all right",
}
NO_WORDS = {
    "no",
    "nope",
    "no thanks",
    "don't",
    "dont",
    "never mind",
    "nevermind",
    "cancel",
    "stop",
    "not now",
    "leave it",
}


def normalise(line: str) -> str:
    text = line.lower().strip().strip(" .,!?")
    text = APOSTROPHE_S.sub("lets", text)

    return SPACES.sub(" ", text)


def detect_switch(line: str) -> str | None:
    text = normalise(line)

    if not text:
        return None

    return mode_by_phrase(text) or mode_by_lets_do(text)


def mode_by_phrase(text: str) -> str | None:
    for mode in MODES.values():
        for phrase in mode.phrases:
            if text == phrase or text.startswith(phrase + " "):
                return mode.name

    return None


def mode_by_lets_do(text: str) -> str | None:
    match = LETS_DO.match(text)

    if not match:
        return None

    word = match.group(1)

    for mode in MODES.values():
        if word in mode.extra:
            return mode.name

    return None


def allowed_tools(mode: str) -> list[Tool]:
    names = set(MODES[mode].tools) | set(ALWAYS)

    return [tool for tool in TOOLS if tool.name in names]


def is_quiet(mode: str) -> bool:
    """Whether tool results in ``mode`` go unspoken (see ``Mode.quiet``)."""
    return mode in MODES and MODES[mode].quiet


def mode_for_tool(name: str) -> str | None:
    for mode in MODES.values():
        if name in mode.tools:
            return mode.name

    return None


def answer_to_offer(line: str) -> str | None:
    text = normalise(line)

    if text in YES_WORDS or text.startswith("yes"):
        return "yes"

    if text in NO_WORDS or text.startswith("no"):
        return "no"

    return None


def describe() -> str:
    return "; ".join(f"{m.name}: {', '.join(m.tools) if m.tools else 'no tools'}" for m in MODES.values())

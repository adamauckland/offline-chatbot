"""Modes: which tools the bot can use right now, switched by saying so.

The bot is in one mode at a time. ``chat`` (the default) has no tools; the
others each open a few: ``photographer`` the webcam, ``music`` the notes and
drums, ``home`` the example server's lights and clock. "Let's chat", "let's
do photos", "let's do music", "let's do home" switch, handled in the chat loop
without a model call, like "stop".

Why modes: the routing question ("which tool, if any?") lists the available
tools with an example each, so fewer tools means a shorter prompt and a
quicker answer, and in chat mode the question is not asked at all. They
also keep the small model from reaching for a tool nobody wanted. The
control tools, ``exit`` and ``clear_context``, are available in every mode.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from tools import TOOLS, Tool

DEFAULT_MODE = "chat"
ALWAYS = ("exit", "clear_context")  # available whatever the mode


@dataclass(frozen=True)
class Mode:
    name: str
    tools: tuple[str, ...]  # tool names this mode opens, besides ALWAYS
    phrases: tuple[str, ...]  # said on their own (or at the start of a line), these switch to it
    greeting: str  # spoken on switching
    extra: tuple[str, ...] = field(default_factory=tuple)  # words that, after "let's do", mean this mode


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
        ("play_drums", "set_tempo", "play_note", "keep_playing_note", "stop_sound"),
        ("lets do music", "lets play music", "music mode", "lets make music"),
        "Music mode. Drums, tempo and notes.",
        ("music", "drums", "beats"),
    ),
    "home": Mode(
        "home",
        ("set_lights", "get_lights", "get_time"),
        ("lets do home", "home mode", "lets do the house"),
        "Home mode. Lights and the time.",
        ("home", "house", "lights"),
    ),
}


def _normalise(line: str) -> str:
    text = line.lower().strip().strip(" .,!?")
    text = re.sub(r"\blet['’]s\b", "lets", text)
    return re.sub(r"\s+", " ", text)


def detect_switch(line: str) -> str | None:
    """The mode ``line`` asks for, or None. Matches a phrase on its own or opening the line."""
    text = _normalise(line)
    if not text:
        return None
    for mode in MODES.values():
        for phrase in mode.phrases:
            if text == phrase or text.startswith(phrase + " "):
                return mode.name
    match = re.match(r"lets (?:do|go|try|have) (?:some |the )?(\w+)", text)
    if match:
        word = match.group(1)
        for mode in MODES.values():
            if word in mode.extra:
                return mode.name
    return None


def allowed_tools(mode: str) -> list[Tool]:
    """The registered tools the mode opens, in registry order, plus the control tools."""
    names = set(MODES[mode].tools) | set(ALWAYS)
    return [tool for tool in TOOLS if tool.name in names]


def mode_for_tool(name: str) -> str | None:
    """The mode that opens tool ``name`` (None for the control tools and unknown names)."""
    for mode in MODES.values():
        if name in mode.tools:
            return mode.name
    return None


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


def answer_to_offer(line: str) -> str | None:
    """Whether ``line`` accepts ("yes") or declines ("no") an offer to switch mode; None if it is something else."""
    text = _normalise(line)
    if text in YES_WORDS or text.startswith("yes"):
        return "yes"
    if text in NO_WORDS or text.startswith("no"):
        return "no"
    return None


def describe() -> str:
    return "; ".join(f"{m.name}: {', '.join(m.tools) if m.tools else 'no tools'}" for m in MODES.values())

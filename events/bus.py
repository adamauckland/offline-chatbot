"""A tiny event bus: what the chatbot is doing, for anything that wants to show it.

The chat loop, the microphone code and the speaker report what they are up to
by calling ``emit``; a display (see tui/app.py) subscribes and reacts. Nothing in
the chatbot depends on anyone listening, and a listener that raises never
breaks the chatbot: exceptions are swallowed here.

Events and their data:

``state``       ``state=<name>``, the bot's current activity. One of ``STATES``:
                asleep (waiting for the wake phrase; ``phrase``), idle (waiting
                for typed input), listening (microphone open, nobody speaking
                yet), recording (speech heard), transcribing, thinking (the
                model is generating), tool (calling a tool; ``name``, ``args``),
                speaking (reading a reply aloud).
``level``       ``mic`` and ``out``: RMS of the last 100 ms of microphone input
                and speaker output, 0..1. Sent from the audio callback ten times
                a second, so handlers must be quick and never block.
``user``        ``text``, ``spoken``: a message from the user was accepted.
``reply``       ``text``: the finished reply.
``tool``        ``name``, ``args``, ``result``: a tool was called.
``remembered``  ``fact``: a fact was stored in long-term memory.
``cancel``      a display asked for the current reply or speech to stop.

``CANCEL`` is the same request as a flag: a display sets it (and emits
``cancel``), the generation loop checks it between tokens and the chat loop
clears it before each reply, so it behaves like Ctrl-C in the terminal.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

STATES = ("asleep", "idle", "listening", "recording", "transcribing", "thinking", "tool", "speaking")

Handler = Callable[[str, dict], None]

CANCEL = threading.Event()

_handlers: list[Handler] = []
_lock = threading.Lock()


def subscribe(handler: Handler) -> None:
    """``handler(name, data)`` is called for every event, on whichever thread emitted it."""
    with _lock:
        if handler not in _handlers:
            _handlers.append(handler)


def unsubscribe(handler: Handler) -> None:
    with _lock:
        if handler in _handlers:
            _handlers.remove(handler)


def emit(name: str, /, **data) -> None:
    """Deliver an event to every handler; a failing handler is ignored."""
    with _lock:
        handlers = list(_handlers)
    for handler in handlers:
        try:
            handler(name, data)
        except Exception:  # noqa: BLE001 - a display must never take the chatbot down
            pass


def state(name: str, /, **data) -> None:
    """Shorthand for ``emit("state", state=name, ...)``.

    ``name`` is positional-only so that event data may also carry a ``name``
    (the tool state does: ``state("tool", name="set_lights", args=...)``).
    """
    if name not in STATES:
        raise ValueError(f"unknown state {name!r}; expected one of {STATES}")
    emit("state", state=name, **data)


def cancel() -> None:
    """Ask for the current reply or speech to stop (what Ctrl-C does in the terminal)."""
    CANCEL.set()
    emit("cancel")

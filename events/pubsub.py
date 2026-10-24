"""Events on the bus and the data each carries.

``state``       ``state``, one of ``STATES``; ``asleep`` adds ``phrase``, ``tool`` adds ``name`` and ``args``
``level``       ``mic``, ``out``: RMS of the last 100 ms, 0..1, ten times a second from the audio thread
``mode``        ``mode``
``user``        ``text``, ``spoken``
``reply``       ``text``
``tool``        ``name``, ``args``, ``result``
``remembered``  ``fact``
``cancel``      a display asked for the current reply or speech to stop; ``CANCEL`` is the same request as a flag
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
    with _lock:
        if handler not in _handlers:
            _handlers.append(handler)


def unsubscribe(handler: Handler) -> None:
    with _lock:
        if handler in _handlers:
            _handlers.remove(handler)


def emit(name: str, /, **data) -> None:
    for handler in _current_handlers():
        _deliver(handler, name, data)


def _current_handlers() -> list[Handler]:
    with _lock:
        return list(_handlers)


def _deliver(handler: Handler, name: str, data: dict) -> None:
    try:
        handler(name, data)
    except Exception:
        pass


def state(name: str, /, **data) -> None:
    if name not in STATES:
        raise ValueError(f"unknown state {name!r}; expected one of {STATES}")

    emit("state", state=name, **data)


def cancel() -> None:
    CANCEL.set()
    emit("cancel")

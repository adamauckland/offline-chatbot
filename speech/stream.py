"""Turn text that arrives a few words at a time into whole sentences for the speaker."""

from __future__ import annotations

import re
from collections.abc import Callable

# A sentence ends at . ! or ? followed by whitespace, or at a line break (lists, paragraphs).
BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")


class SentenceStream:
    """Collect streamed text and hand each complete sentence to ``say`` as soon as it is.

    Speech can then start after the first sentence of a reply instead of
    after the whole of it. A fragment shorter than ``min_words`` ("1." in a
    numbered list, "e.g.") is not spoken on its own but joined to what follows.
    """

    def __init__(self, say: Callable[[str], None], min_words: int = 2) -> None:
        self.say = say
        self.min_words = min_words
        self.buffer = ""

    def feed(self, text: str) -> None:
        """Add text as it is generated; complete sentences are spoken at once."""
        self.buffer += text
        start = 0
        for match in BOUNDARY.finditer(self.buffer):
            candidate = self.buffer[start : match.start()]
            if len(candidate.split()) >= self.min_words:
                self.say(candidate.strip())
                start = match.end()
        self.buffer = self.buffer[start:]

    def flush(self) -> None:
        """Speak whatever is left (the end of the reply)."""
        if self.buffer.strip():
            self.say(self.buffer.strip())
        self.buffer = ""

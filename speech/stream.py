from __future__ import annotations

import re
from collections.abc import Callable

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")


class SentenceStream:
    def __init__(self, say: Callable[[str], None], min_words: int = 2) -> None:
        self.say = say
        self.min_words = min_words
        self.buffer = ""

    def feed(self, text: str) -> None:
        self.buffer += text
        start = 0

        for boundary in SENTENCE_BOUNDARY.finditer(self.buffer):
            sentence = self.buffer[start : boundary.start()]

            if self._long_enough(sentence):
                self.say(sentence.strip())
                start = boundary.end()

        self.buffer = self.buffer[start:]

    def flush(self) -> None:
        if self.buffer.strip():
            self.say(self.buffer.strip())

        self.buffer = ""

    def _long_enough(self, sentence: str) -> bool:
        return len(sentence.split()) >= self.min_words

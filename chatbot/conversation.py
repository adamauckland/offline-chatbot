from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

END = "<|end|>"
IMAGE_TAG = re.compile(r"<\|image_(\d+)\|>")
IMAGE_TAG_LINE = re.compile(r"<\|image_(\d+)\|>\n?")
DOWNLOAD_TIMEOUT_SECONDS = 30


def load_image(source: str) -> Image.Image:
    if source.startswith(("http://", "https://")):
        return download_image(source)

    return open_local_image(source)


def download_image(url: str) -> Image.Image:
    response = requests.get(url, timeout=DOWNLOAD_TIMEOUT_SECONDS)
    response.raise_for_status()

    return Image.open(BytesIO(response.content))


def open_local_image(source: str) -> Image.Image:
    path = Path(source).expanduser()

    if not path.is_file():
        raise FileNotFoundError(f"No such image: {source}")

    return Image.open(path)


def clean_reply(text: str) -> str:
    if END in text:
        text = text[: text.index(END)]

    return text.strip()


def image_numbers_in(turns: list[tuple[str, str]]) -> set[int]:
    return {int(n) for _, content in turns for n in IMAGE_TAG.findall(content)}


def renumber_after_dropping(content: str, dropped: int) -> str:
    def renumber(match: re.Match) -> str:
        n = int(match.group(1))

        if n == dropped:
            return ""

        return f"<|image_{n - 1}|>\n" if n > dropped else match.group(0)

    return IMAGE_TAG_LINE.sub(renumber, content)


def shift_after_dropping(number: int | None, dropped: int) -> int | None:
    if number is None or number == dropped:
        return None

    return number - 1 if number > dropped else number


class Conversation:
    def __init__(self, system: str | None = None) -> None:
        self.system = system
        self.turns: list[tuple[str, str]] = []
        self.images: list[Image.Image] = []
        self.tool_image: int | None = None

    def add_user(self, text: str, images: list[Image.Image]) -> None:
        tags = []

        for img in images:
            self.images.append(img)
            tags.append(f"<|image_{len(self.images)}|>")

        content = "\n".join(tags + [text.strip()]) if tags else text.strip()
        self.turns.append(("user", content))

    def add_user_and_prompt(self, text: str) -> str:
        self.add_user(text, [])

        return self.prompt()

    def add_assistant(self, text: str) -> None:
        self.turns.append(("assistant", clean_reply(text)))

    def drop_image(self, index: int) -> None:
        del self.images[index - 1]
        self.turns = [(role, renumber_after_dropping(content, index)) for role, content in self.turns]
        self.tool_image = shift_after_dropping(self.tool_image, index)

    def truncate(self, n_turns: int) -> None:
        del self.turns[n_turns:]
        used = max(image_numbers_in(self.turns), default=0)
        del self.images[used:]

        if self.tool_image is not None and self.tool_image > used:
            self.tool_image = None

    def prompt(self, extra_system: str = "", prefill: str = "", note: str = "") -> str:
        parts = []
        system = "\n\n".join(p for p in (extra_system, self.system) if p)
        last_user = self._last_user_turn()

        if system:
            parts.append(f"<|system|>\n{system.strip()}{END}\n")

        for i, (role, content) in enumerate(self.turns):
            if note and i == last_user:
                parts.append(f"<|user|>\n{note.strip()}{END}\n")

            parts.append(f"<|{role}|>\n{content}{END}\n")

        parts.append("<|assistant|>\n" + prefill)

        return "".join(parts)

    def _last_user_turn(self) -> int | None:
        return max((i for i, (role, _) in enumerate(self.turns) if role == "user"), default=None)

    def _second_user_turn(self) -> int | None:
        return next((i for i in range(1, len(self.turns)) if self.turns[i][0] == "user"), None)

    def drop_oldest_exchange(self) -> bool:
        end = self._second_user_turn()

        if end is None:
            return False

        removed, self.turns = self.turns[:end], self.turns[end:]

        for number in sorted(image_numbers_in(removed), reverse=True):
            self.drop_image(number)

        return True

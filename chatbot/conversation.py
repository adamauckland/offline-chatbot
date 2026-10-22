"""The chat history, and how it is rendered for Phi-3."""

from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

END = "<|end|>"


def load_image(source: str) -> Image.Image:
    """Open a local file or an http(s) URL as a PIL image."""
    if source.startswith(("http://", "https://")):
        response = requests.get(source, timeout=30)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    path = Path(source).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"No such image: {source}")
    return Image.open(path)


def clean_reply(text: str) -> str:
    """Strip the end-of-turn token and surrounding whitespace from a reply."""
    if END in text:
        text = text[: text.index(END)]
    return text.strip()


class Conversation:
    """Keeps the chat history and renders it in Phi-3's chat format.

    Phi-3 expects:
        <|system|>\\n...<|end|>\\n
        <|user|>\\n<|image_1|>\\n...<|end|>\\n
        <|assistant|>\\n...<|end|>\\n
    Images are numbered globally across the conversation, so the list of PIL
    images passed to the model must stay in the order the tags were issued.
    """

    def __init__(self, system: str | None = None) -> None:
        self.system = system
        self.turns: list[tuple[str, str]] = []  # (role, content)
        self.images: list[Image.Image] = []
        self.tool_image: int | None = None  # 1-based index of the picture a tool last attached

    def add_user(self, text: str, images: list[Image.Image]) -> None:
        tags = []
        for img in images:
            self.images.append(img)
            tags.append(f"<|image_{len(self.images)}|>")
        content = "\n".join(tags + [text.strip()]) if tags else text.strip()
        self.turns.append(("user", content))

    def add_user_and_prompt(self, text: str) -> str:
        """Convenience for one-off questions: add ``text`` as the user turn and render the prompt."""
        self.add_user(text, [])
        return self.prompt()

    def add_assistant(self, text: str) -> None:
        self.turns.append(("assistant", clean_reply(text)))

    def drop_image(self, index: int) -> None:
        """Remove image ``index`` (1-based) from the history; later tags are renumbered."""
        del self.images[index - 1]

        def renumber(match: re.Match) -> str:
            n = int(match.group(1))
            if n == index:
                return ""
            return f"<|image_{n - 1}|>\n" if n > index else match.group(0)

        self.turns = [(role, re.sub(r"<\|image_(\d+)\|>\n?", renumber, content)) for role, content in self.turns]
        if self.tool_image is not None:
            self.tool_image = (
                None
                if self.tool_image == index
                else (self.tool_image - 1 if self.tool_image > index else self.tool_image)
            )

    def truncate(self, n_turns: int) -> None:
        """Drop turns from ``n_turns`` on, and the images only they referred to."""
        del self.turns[n_turns:]
        used = max((int(n) for _, content in self.turns for n in re.findall(r"<\|image_(\d+)\|>", content)), default=0)
        del self.images[used:]
        if self.tool_image is not None and self.tool_image > used:
            self.tool_image = None

    def prompt(self, extra_system: str = "", prefill: str = "", note: str = "") -> str:
        """Render the chat; ``prefill`` is text the assistant turn is forced to start with.

        ``note`` (the facts about the user and the recalled conversations) is
        rendered as a user turn of its own just before the last message, for
        this call only. Placed there it never changes the start of the
        prompt, so a new fact or a different recall costs only its own tokens
        and the model's cache of everything before stays valid; and the real
        question is still the last thing the model reads. (Placed after the
        question, the model answered the note instead.) ``extra_system`` is
        still accepted for text that genuinely belongs in the system prompt.
        """
        parts = []
        system = "\n\n".join(p for p in (extra_system, self.system) if p)
        if system:
            parts.append(f"<|system|>\n{system.strip()}{END}\n")
        last_user = max((i for i, (role, _) in enumerate(self.turns) if role == "user"), default=None)
        for i, (role, content) in enumerate(self.turns):
            if note and i == last_user:
                parts.append(f"<|user|>\n{note.strip()}{END}\n")
            parts.append(f"<|{role}|>\n{content}{END}\n")
        parts.append("<|assistant|>\n" + prefill)
        return "".join(parts)

    def drop_oldest_exchange(self) -> bool:
        """Forget the first user message and the replies up to the next one; False if there is nothing to drop.

        Used to keep a long conversation inside the model's cache: the oldest
        part goes, the system prompt and the recent turns stay. Pictures only
        the dropped turns referred to are dropped too.
        """
        if len(self.turns) < 2:
            return False
        end = 1
        while end < len(self.turns) and self.turns[end][0] != "user":
            end += 1
        if end >= len(self.turns):
            return False  # only one exchange: keep it
        removed, self.turns = self.turns[:end], self.turns[end:]
        for number in sorted({int(n) for _, c in removed for n in re.findall(r"<\|image_(\d+)\|>", c)}, reverse=True):
            self.drop_image(number)
        return True

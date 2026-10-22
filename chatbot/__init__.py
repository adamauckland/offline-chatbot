"""The chatbot itself: the chat loop, the model calls and the command line.

Run it with ``uv run -m chatbot`` (or ``uv run scrudley``); see ``chatbot.cli``.
"""

from chatbot.cli import main

__all__ = ["main"]

"""The two built-in tools that steer the chat loop itself: "exit" and "clear_context".

Both only raise a flag; the loop acts on it once the current reply is out.
One ``Control`` object is shared by the tools, ``answer`` and ``run_repl``.
"""

from __future__ import annotations

from dataclasses import dataclass

import requests

from tools import BASE_URL, Tool, register


@dataclass
class Control:
    exit_requested: bool = False  # the chat loop ends after the goodbye
    reset_requested: bool = False  # the chat loop starts a fresh conversation after the reply
    wanted_mode: str | None = None  # the turn needed a tool from this mode: the loop offers to switch

    def request_exit(self) -> dict:
        """The "exit" tool: stop the tools server (even one we did not start) and flag the loop to end."""
        self.exit_requested = True
        try:
            requests.post(f"{BASE_URL}/shutdown", timeout=2)
            server = "tools server stopping"
        except requests.RequestException:
            server = "no tools server to stop"
        return {"ok": True, "message": f"Shutting down; {server}."}

    def request_reset(self) -> dict:
        """The "clear_context" tool: forget the conversation so far (text and images), as /reset does.

        Long-term memory is untouched; the exchanges so far simply become
        recallable history.
        """
        self.reset_requested = True
        return {"ok": True, "message": "The conversation will be cleared."}

    def register_tools(self) -> None:
        """Offer both tools to the model."""
        register(
            Tool(
                name="exit",
                description="Shut the assistant down and stop the tools server. Use when the user says exit, "
                "quit, shut down, or goodbye.",
                func=self.request_exit,
                example=("shut down now", {}),
            )
        )
        register(
            Tool(
                name="clear_context",
                description="Forget the conversation so far, including any pictures, and start fresh (what the user "
                "told you to remember is kept). Use when the user says start over, clear the context, "
                "new conversation, or forget what we talked about.",
                func=self.request_reset,
                example=("start over", {}),
            )
        )

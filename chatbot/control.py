from __future__ import annotations

from dataclasses import dataclass

import requests

from tools import BASE_URL, Tool, register

SHUTDOWN_TIMEOUT_SECONDS = 2


@dataclass
class Control:
    exit_requested: bool = False
    reset_requested: bool = False
    wanted_mode: str | None = None

    def request_exit(self) -> dict:
        self.exit_requested = True
        server = stop_tools_server()

        return {"ok": True, "message": f"Shutting down; {server}."}

    def request_reset(self) -> dict:
        self.reset_requested = True

        return {"ok": True, "message": "The conversation will be cleared."}

    def register_tools(self) -> None:
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


def stop_tools_server() -> str:
    try:
        requests.post(f"{BASE_URL}/shutdown", timeout=SHUTDOWN_TIMEOUT_SECONDS)

        return "tools server stopping"
    except requests.RequestException:
        return "no tools server to stop"

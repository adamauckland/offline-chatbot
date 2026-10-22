"""Tool registry: the endpoints the chatbot may call, and how to describe them to the model.

To add a tool, append a ``Tool`` to ``TOOLS`` (or call ``register`` at runtime).
The chatbot turns the list into instructions in the system prompt and, when the
model asks for a tool, either sends the arguments to ``url`` (JSON body for
POST, query string for GET) or calls ``func(**arguments)`` for a Python tool,
and gives the result back to the model.

A result may include ``"image": <local path or URL>``: the chatbot then shows
that image to the model along with the result, so a tool can hand the model
something to look at (``take_picture`` does this with the webcam shot).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field

import requests

from images.webcam import take_picture

BASE_URL = "http://127.0.0.1:8000"


@dataclass
class Tool:
    name: str
    description: str
    url: str = ""  # HTTP tool: endpoint to call
    method: str = "GET"
    # Argument name -> short description shown to the model, e.g. {"state": "'on' or 'off'"}
    arguments: dict[str, str] = field(default_factory=dict)
    func: Callable[..., dict] | None = None  # Python tool: called instead of the URL
    # Optional worked example shown to the model: (what the user says, the arguments to send)
    example: tuple[str, dict] | None = None
    # Optional: the reply to give from the result, instead of asking the model to word one. For a
    # tool whose outcome is fully known from its result this is exact and saves a model pass; a
    # small model asked to summarise "played E3 for 0.6 s" may well say "E4 for one second".
    # Only used when the result has no "error"; errors are left to the model to explain.
    reply: Callable[[dict], str] | None = None

    def call(self, args: dict) -> dict:
        if self.func is not None:
            try:
                return self.func(**{k: v for k, v in args.items() if k in self.arguments})
            except Exception as exc:  # noqa: BLE001 - report to the model, do not crash
                return {"error": f"{self.name} failed: {exc}"}
        try:
            if self.method.upper() == "GET":
                resp = requests.get(self.url, params=args, timeout=10)
            else:
                resp = requests.request(self.method, self.url, json=args, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            return {"error": f"request to {self.url} failed: {exc}"}

    def describe(self, brief: bool = False) -> str:
        """The line the model sees. ``brief`` is the name and first sentence only, for the routing question,
        which must stay short: its prompt is re-encoded in full if it outgrows the router's cache."""
        if brief:
            first = re.split(r"(?<=[.;])\s", self.description.strip(), maxsplit=1)[0]
            return f"- {self.name}: {first}"
        if self.arguments:
            args = ", ".join(f'"{k}": {v}' for k, v in self.arguments.items())
            return f"- {self.name}: {self.description} Arguments: {{{args}}}"
        return f"- {self.name}: {self.description} Arguments: {{}}"


TOOLS: list[Tool] = [
    Tool(
        name="set_lights",
        description="Turn the lights on or off in a room.",
        url=f"{BASE_URL}/lights",
        method="POST",
        arguments={"state": "'on' or 'off'", "room": "'living room', 'kitchen', 'bedroom' or 'all' (optional)"},
        example=("switch off the bedroom light", {"state": "off", "room": "bedroom"}),
    ),
    Tool(
        name="get_lights",
        description="Report whether the lights are on or off in every room.",
        url=f"{BASE_URL}/lights",
        example=("are the lights on", {}),
    ),
    Tool(
        name="get_time",
        description="Get the current time and date.",
        url=f"{BASE_URL}/time",
        example=("what time is it", {}),
    ),
    Tool(
        name="take_picture",
        description="Take a picture (photo) with the webcam, save it to the webcam-pictures folder and look "
        "at it. Use this to see the user, what they are holding or showing, or the room.",
        func=take_picture,
        example=("what can you see", {}),
    ),
]

TOOLS_BY_NAME = {t.name: t for t in TOOLS}


def register(tool: Tool) -> None:
    """Add a tool at runtime (used for tools that need live objects, like memory)."""
    TOOLS.append(tool)
    TOOLS_BY_NAME[tool.name] = tool


def tools_prompt() -> str:
    """System-prompt section that teaches the model how to request a tool."""
    lines = "\n".join(t.describe() for t in TOOLS)
    return (
        "You can use these tools:\n"
        f"{lines}\n"
        "To use a tool, reply with ONLY a single-line JSON object and nothing else, "
        'in the form {"tool": "<name>", "arguments": {...}}. '
        "You will then receive the tool's result and must answer the user in plain language. "
        "If no tool is needed, answer normally."
    )


def _balanced_json(text: str, start: int) -> str | None:
    """Return the substring from ``start`` to its matching closing brace, if complete."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_tool_call(text: str) -> tuple[str, dict] | None:
    """Return (name, arguments) if ``text`` contains a tool request, else None.

    Small models often wrap the JSON in a sentence ("Sure, turning them on.
    {...}"), so any complete JSON object with a "tool" key counts.
    """
    for match in re.finditer(r"\{", text):
        candidate = _balanced_json(text, match.start())
        if candidate is None:
            return None
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "tool" in data:
            args = data.get("arguments") or {}
            return str(data["tool"]), (args if isinstance(args, dict) else {})
    return None


def run_tool(name: str, args: dict) -> dict:
    tool = TOOLS_BY_NAME.get(name)
    if tool is None:
        return {"error": f"unknown tool '{name}'", "available": list(TOOLS_BY_NAME)}
    return tool.call(args)

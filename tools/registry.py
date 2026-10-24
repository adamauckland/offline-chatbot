from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field

import requests

from images.webcam import take_picture

BASE_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT_SECONDS = 10
SENTENCE_END = re.compile(r"(?<=[.;])\s")
OPENING_BRACE = re.compile(r"\{")


@dataclass
class Tool:
    name: str
    description: str
    url: str = ""
    method: str = "GET"
    arguments: dict[str, str] = field(default_factory=dict)
    func: Callable[..., dict] | None = None
    example: tuple[str, dict] | None = None
    reply: Callable[[dict], str] | None = None

    def call(self, args: dict) -> dict:
        if self.func is not None:
            return self._call_function(args)

        return self._call_endpoint(args)

    def _call_function(self, args: dict) -> dict:
        declared = {k: v for k, v in args.items() if k in self.arguments}

        try:
            return self.func(**declared)
        except Exception as exc:
            return {"error": f"{self.name} failed: {exc}"}

    def _call_endpoint(self, args: dict) -> dict:
        try:
            response = self._request(args)
            response.raise_for_status()

            return response.json()
        except requests.RequestException as exc:
            return {"error": f"request to {self.url} failed: {exc}"}

    def _request(self, args: dict) -> requests.Response:
        if self.method.upper() == "GET":
            return requests.get(self.url, params=args, timeout=REQUEST_TIMEOUT_SECONDS)

        return requests.request(self.method, self.url, json=args, timeout=REQUEST_TIMEOUT_SECONDS)

    def describe(self, brief: bool = False) -> str:
        if brief:
            return f"- {self.name}: {self._first_sentence()}"

        return f"- {self.name}: {self.description} Arguments: {{{self._argument_list()}}}"

    def _first_sentence(self) -> str:
        return SENTENCE_END.split(self.description.strip(), maxsplit=1)[0]

    def _argument_list(self) -> str:
        return ", ".join(f'"{k}": {v}' for k, v in self.arguments.items())


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
    TOOLS.append(tool)
    TOOLS_BY_NAME[tool.name] = tool


def tools_prompt() -> str:
    lines = "\n".join(t.describe() for t in TOOLS)

    return (
        "You can use these tools:\n"
        f"{lines}\n"
        "To use a tool, reply with ONLY a single-line JSON object and nothing else, "
        'in the form {"tool": "<name>", "arguments": {...}}. '
        "You will then receive the tool's result and must answer the user in plain language. "
        "If no tool is needed, answer normally."
    )


def balanced_json(text: str, start: int) -> str | None:
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
    for match in OPENING_BRACE.finditer(text):
        candidate = balanced_json(text, match.start())

        if candidate is None:
            return None

        call = tool_call_in(candidate)

        if call is not None:
            return call

    return None


def tool_call_in(candidate: str) -> tuple[str, dict] | None:
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict) or "tool" not in data:
        return None

    args = data.get("arguments") or {}

    return str(data["tool"]), (args if isinstance(args, dict) else {})


def run_tool(name: str, args: dict) -> dict:
    tool = TOOLS_BY_NAME.get(name)

    if tool is None:
        return {"error": f"unknown tool '{name}'", "available": list(TOOLS_BY_NAME)}

    return tool.call(args)

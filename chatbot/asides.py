"""Quiet side questions to the model: which tool a message needs, and whether it states a fact.

Both are short, separate generations with nothing printed. Small models answer
a direct question far more reliably than they follow a standing instruction
buried in a long system prompt, which is why these exist.
"""

from __future__ import annotations

import re

from chatbot.conversation import Conversation
from chatbot.model import chat_once
from tools import TOOLS, Tool

ROUTE_PROMPT = (
    "Decide if one of these tools is needed to answer the message.\n{tools}\n"
    "- NONE: everything else (greetings, chat, questions about the user, general knowledge, unclear speech)\n\n"
    "Only pick a tool when the message clearly asks for what that tool does. "
    "Examples: {examples}; 'where do I live' -> NONE; 'hello' -> NONE; 'tell me a joke' -> NONE.\n\n"
    "Message: {message}\nAnswer with one word, the tool name or NONE:"
)


def route_prompt(tools: list[Tool], message: str) -> str:
    """The routing question for just these tools, with an example per tool taken from its ``example``."""
    lines = "\n".join(t.describe(brief=True) for t in tools)
    examples = "; ".join(f"'{t.example[0]}' -> {t.name}" for t in tools if t.example)
    return ROUTE_PROMPT.format(tools=lines, examples=examples, message=message)


_warned: set[str] = set()


def _check_fits(engine, session, prompt: str, what: str) -> None:
    """Warn once if a side prompt has outgrown its cache: it then re-encodes in full on every turn."""
    if session is None or what in _warned:
        return
    _, processor = engine
    n = len(processor.tokenizer.encode(prompt))
    if n + 40 > session.capacity:
        _warned.add(what)
        print(
            f"(the {what} prompt is {n} tokens, more than its cache of {session.capacity}: every turn will "
            "re-encode it in full; shorten the tool descriptions or raise SIDE_CACHE_TOKENS)",
            flush=True,
        )


def warm_up(engine, system: str | None, tools: list[Tool] | None = None) -> None:
    """Encode the constant prefixes now, so the first turn is as quick as the rest.

    The chat session gets the system prompt, the router its question with the
    tool list of the current mode; the first real turn then diverges only at
    the message.
    """
    tools = list(TOOLS) if tools is None else tools
    for session, prompt in (
        (getattr(engine, "chat", None), Conversation(system).prompt()),
        (getattr(engine, "router", None), Conversation().add_user_and_prompt(route_prompt(tools, "")) if tools else ""),
    ):
        if session is None or not prompt:
            continue
        _, processor = engine
        inputs = processor(prompt, None)
        if inputs["input_ids"].shape[-1] + 64 <= session.capacity:
            session.prepare(inputs, [], 64)


def route_tool(engine, message: str, tools: list[Tool] | None = None) -> str | None:
    """Ask the model, in a quiet call, which of ``tools`` (all registered ones if None) the message needs.

    With no tools to choose from there is nothing to ask: None straight away.

    Small models drift out of "reply with JSON" mode after a few turns of
    chat, especially once they have described themselves as a mere program. A
    direct pick-a-name question is answered reliably, and the chosen tool's
    JSON is then pre-filled so only the arguments are left to generate.
    """
    tools = list(TOOLS) if tools is None else tools
    if not tools:
        return None
    probe = Conversation()
    probe.add_user(route_prompt(tools, message), [])
    session = getattr(engine, "router", None)
    _check_fits(engine, session, probe.prompt(), "routing")
    reply = chat_once(engine, probe, max_tokens=12, quiet=True, session=session).strip().lower()
    for tool in sorted(tools, key=lambda t: len(t.name), reverse=True):  # longest name first: "set_tempo" before "set"
        if tool.name in reply:
            return tool.name
    return None


PERSONAL = re.compile(r"\b(i|i'm|i am|i've|my|mine|remember)\b", re.I)
NOT_A_FACT = re.compile(r"\b(ask|asked|asks|request|did not|didn't|wants? (you|me) to)\b", re.I)
EXTRACT_PROMPT = (
    "Does the message below state a lasting fact about the user: their name, where they live, "
    "their family or pets, what they like, their plans, or something they asked you to remember? "
    'If yes, reply with that fact as one short sentence starting with "The user". '
    "Questions, requests, instructions to you, opinions, and general knowledge are not facts about "
    "the user: reply NONE for those. "
    "If no, reply with exactly NONE.\n\nMessage: {message}"
)


def extract_fact(engine, message: str) -> str | None:
    """Ask the model, in a separate quiet call, whether ``message`` holds a fact worth keeping.

    Small models will not reliably call a "remember" tool, but they answer a
    direct yes/no extraction question well. Only messages that look personal
    (first person, or the word "remember") are checked, to keep latency down.
    """
    if message.rstrip().endswith("?") or not PERSONAL.search(message):
        return None
    probe = Conversation()
    probe.add_user(EXTRACT_PROMPT.format(message=message), [])
    reply = (
        chat_once(engine, probe, max_tokens=40, quiet=True, session=getattr(engine, "extractor", None))
        .splitlines()[0]
        .strip()
    )
    if not reply.lower().startswith("the user") or "[" in reply or "none" in reply.lower():
        return None
    if NOT_A_FACT.search(reply):
        return None
    return reply.rstrip(".") + "."

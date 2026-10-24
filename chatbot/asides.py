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
EXTRACT_PROMPT = (
    "Does the message below state a lasting fact about the user: their name, where they live, "
    "their family or pets, what they like, their plans, or something they asked you to remember? "
    'If yes, reply with that fact as one short sentence starting with "The user". '
    "Questions, requests, instructions to you, opinions, and general knowledge are not facts about "
    "the user: reply NONE for those. "
    "If no, reply with exactly NONE.\n\nMessage: {message}"
)
PERSONAL = re.compile(r"\b(i|i'm|i am|i've|my|mine|remember)\b", re.I)
NOT_A_FACT = re.compile(r"\b(ask|asked|asks|request|did not|didn't|wants? (you|me) to)\b", re.I)
ROUTE_MAX_TOKENS = 12
EXTRACT_MAX_TOKENS = 40
WARM_UP_RESERVE_TOKENS = 64
SIDE_PROMPT_HEADROOM_TOKENS = 40

_cache_warnings_given: set[str] = set()


def route_prompt(tools: list[Tool], message: str) -> str:
    lines = "\n".join(t.describe(brief=True) for t in tools)
    examples = "; ".join(f"'{t.example[0]}' -> {t.name}" for t in tools if t.example)

    return ROUTE_PROMPT.format(tools=lines, examples=examples, message=message)


def warn_once_if_prompt_outgrows_cache(engine, session, prompt: str, what: str) -> None:
    if session is None or what in _cache_warnings_given:
        return

    _, processor = engine
    n = len(processor.tokenizer.encode(prompt))

    if n + SIDE_PROMPT_HEADROOM_TOKENS <= session.capacity:
        return

    _cache_warnings_given.add(what)
    print(
        f"(the {what} prompt is {n} tokens, more than its cache of {session.capacity}: every turn will "
        "re-encode it in full; shorten the tool descriptions or raise SIDE_CACHE_TOKENS)",
        flush=True,
    )


def warm_up(engine, system: str | None, tools: list[Tool] | None = None) -> None:
    tools = list(TOOLS) if tools is None else tools
    prime(engine, getattr(engine, "chat", None), Conversation(system).prompt())

    if tools:
        prime(engine, getattr(engine, "router", None), Conversation().add_user_and_prompt(route_prompt(tools, "")))


def prime(engine, session, prompt: str) -> None:
    if session is None or not prompt:
        return

    _, processor = engine
    inputs = processor(prompt, None)

    if inputs["input_ids"].shape[-1] + WARM_UP_RESERVE_TOKENS <= session.capacity:
        session.prepare(inputs, [], WARM_UP_RESERVE_TOKENS)


def route_tool(engine, message: str, tools: list[Tool] | None = None) -> str | None:
    tools = list(TOOLS) if tools is None else tools

    if not tools:
        return None

    probe = Conversation()
    probe.add_user(route_prompt(tools, message), [])
    session = getattr(engine, "router", None)
    warn_once_if_prompt_outgrows_cache(engine, session, probe.prompt(), "routing")
    reply = chat_once(engine, probe, max_tokens=ROUTE_MAX_TOKENS, quiet=True, session=session).strip().lower()

    for tool in longest_name_first(tools):
        if tool.name in reply:
            return tool.name

    return None


def longest_name_first(tools: list[Tool]) -> list[Tool]:
    return sorted(tools, key=lambda t: len(t.name), reverse=True)


def extract_fact(engine, message: str) -> str | None:
    if message.rstrip().endswith("?") or not PERSONAL.search(message):
        return None

    probe = Conversation()
    probe.add_user(EXTRACT_PROMPT.format(message=message), [])
    session = getattr(engine, "extractor", None)
    reply = chat_once(engine, probe, max_tokens=EXTRACT_MAX_TOKENS, quiet=True, session=session).splitlines()[0].strip()

    if not reply.lower().startswith("the user") or "[" in reply or "none" in reply.lower():
        return None

    if NOT_A_FACT.search(reply):
        return None

    return reply.rstrip(".") + "."

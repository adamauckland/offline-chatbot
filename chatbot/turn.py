from __future__ import annotations

import json
import re
from typing import Protocol

from PIL import Image

import events
from chatbot.asides import extract_fact, route_tool
from chatbot.control import Control
from chatbot.conversation import Conversation, load_image
from chatbot.model import chat_once
from chatbot.modes import is_quiet, mode_for_tool
from memory import Memory
from speech import SentenceStream
from tools import TOOLS_BY_NAME, parse_tool_call, run_tool


class Voice(Protocol):
    def say(self, text: str) -> None: ...

    def wait(self) -> None: ...


# The model's own way of saying it did not follow. Spoken, this is tiresome; the "huh" cue says it in
# two falling notes instead, and the rest of such a reply is not read out either.
APOLOGY = re.compile(
    r"^\W*(i'?m |i am )?sorry[,. ]+(but )?(i )?(don'?t|do not|didn'?t|did not|can'?t|cannot|could not|couldn'?t) "
    r"(quite )?(understand|catch|follow|hear|make out)|^\W*(can|could) you (please )?(rephrase|repeat|say that again)",
    re.I,
)


class Mouth:
    """Speaks a reply's sentences, except an apology for not understanding, which becomes the cue."""

    def __init__(self, voice: Voice | None) -> None:
        self.voice = voice
        self.first = True
        self.muted = False

    def say(self, text: str) -> None:
        if self.voice is None or self.muted:
            return
        if self.first and APOLOGY.match(text):
            self.muted = True
            huh(self.voice)
            return
        self.first = False
        self.voice.say(text)


def huh(voice: Voice | None) -> None:
    """Two falling notes: the bot did not understand."""
    audio = getattr(voice, "audio", None)
    if audio is not None:
        from audio.input import cue

        cue(audio, "huh", wait=False)


MAX_TOOL_CALLS = 3
MAX_TOOL_IMAGE_SIDE = 1280
CONTROL_TOOLS = ("exit", "clear_context")
PICTURE_TAKEN = "Picture taken. Examining the picture now. This will take approximately 10 seconds to complete."
NOT_UNDERSTOOD = "Sorry, I did not catch that. Could you say it again?"
GAVE_UP = "Sorry, I could not complete that."
NO_TOOLS_IN_THIS_MODE = (
    "In this mode you have no tools at all: you cannot play notes or drums, take pictures, "
    "control lights or look up the time, and you must not pretend to. "
)
HOW_TO_SWITCH = (
    "If the user asks for something that needs a tool you cannot use now, do not describe the tool and "
    'do not invent a result: say that it needs another mode and that they can say "let\'s do music", '
    '"let\'s do photos", "let\'s do home" or "let\'s chat" to switch.'
)
ANSWER_IN_PLAIN_LANGUAGE = "Now answer the user's request in plain language."


def say(voice: Voice | None, text: str) -> None:
    if voice is not None:
        voice.say(text)


def tell(voice: Voice | None, text: str) -> str:
    print(text, flush=True)
    say(voice, text)

    return text


def mode_line(tools: list, mode: str) -> str:
    names = [t.name for t in tools if t.name not in CONTROL_TOOLS]

    if names:
        usable = f"In this mode you can only use these tools: {', '.join(names)}. "
    else:
        usable = NO_TOOLS_IN_THIS_MODE

    return f"Current mode: {mode or 'default'}. " + usable + HOW_TO_SWITCH


def turn_note(memory: Memory | None, user_text: str, tools: list | None, mode: str) -> str:
    parts = []

    if tools is not None:
        parts.append(mode_line(tools, mode))

    if memory is not None:
        parts += [memory.facts_section(), memory.recall_section(user_text)]

    return "\n\n".join(p for p in parts if p)


def routed_start(chosen: str | None) -> tuple[str, str]:
    if not chosen:
        return "", ""

    arguments = TOOLS_BY_NAME[chosen].arguments

    if not arguments:
        return json.dumps({"tool": chosen, "arguments": {}}), ""

    first = next(iter(arguments))
    prefill = json.dumps({"tool": chosen})[:-1] + ', "arguments": ' + json.dumps({first: ""})[:-2]

    return "", prefill


def something_playing(voice: Voice | None) -> bool:
    """Whether a note or the drums are sounding (the speaker knows its audio stream)."""
    audio = getattr(voice, "audio", None)
    return bool(getattr(audio, "has_layers", False))


def announce_and_run(name: str, args: dict, voice: Voice | None, quiet: bool = False) -> dict:
    """Say what is about to happen, wait to be heard, then run the tool.

    In a quiet mode (music) the announcement is only spoken while nothing is
    playing yet; over the drums it is just printed.
    """
    announcement = f"I will {name.replace('_', ' ')}."
    print(announcement, flush=True)

    if voice is not None and not (quiet and something_playing(voice)):
        voice.say(announcement)
        voice.wait()

    events.state("tool", name=name, args=args)
    result = run_tool(name, args)
    print(f"(result {result})", flush=True)
    events.emit("tool", name=name, args=args, result=result)

    return result


def feed_back(convo: Conversation, name: str, result: dict, voice: Voice | None) -> None:
    image = tool_image(result)

    if image is not None:
        attach_picture(convo, name, result, image, voice)
    else:
        convo.turns.append(("user", f"Result of {name}: {result}\n{ANSWER_IN_PLAIN_LANGUAGE}"))


def remember_if_fact(engine, memory: Memory | None, user_text: str, reply: str) -> None:
    if memory is None:
        return

    memory.add_exchange(user_text, reply)
    fact = extract_fact(engine, user_text)

    if fact and memory.remember(fact).get("ok"):
        print(f"(remembered: {fact})", flush=True)
        events.emit("remembered", fact=fact)


def answer(
    engine,
    convo: Conversation,
    max_tokens: int,
    use_tools: bool,
    memory: Memory | None = None,
    voice: Voice | None = None,
    control: Control | None = None,
    tools: list | None = None,
    mode: str = "",
) -> str:
    control = control or Control()
    quiet = is_quiet(mode)  # music mode: tool results are printed, not spoken over the music
    mouth = Mouth(voice)
    stream = SentenceStream(mouth.say) if voice is not None else None
    user_text = convo.turns[-1][1]
    note = turn_note(memory, user_text, tools, mode)
    allowed = {t.name for t in tools} if tools is not None else None
    chosen = route_tool(engine, user_text) if use_tools else None

    if chosen and allowed is not None and chosen not in allowed:
        return offer_switch(chosen, mode, control, voice)

    reply, prefill = routed_start(chosen)

    for _ in range(MAX_TOOL_CALLS):
        if not reply:
            reply = chat_once(
                engine,
                convo,
                max_tokens,
                use_tools,
                prefill=prefill,
                on_text=stream.feed if stream is not None else None,
                session=getattr(engine, "chat", None),
                note=note,
            )

        call = parse_tool_call(reply) if use_tools else None

        if call is None and prefill:
            call = salvage_tool_call(chosen, reply)

            if call is None:
                print(NOT_UNDERSTOOD, flush=True)
                huh(voice)  # the cue, not the sentence
                reply = NOT_UNDERSTOOD
                break

        prefill = ""

        if call is None:
            break

        name, args = call

        if allowed is not None and name not in allowed:
            return offer_switch(name, mode, control, voice)

        print(f"(calling {name} {args})", flush=True)

        if stream is not None:
            stream.flush()

        result = announce_and_run(name, args, voice, quiet)

        if control.exit_requested or control.reset_requested:
            reply = tell(voice, "Goodbye." if control.exit_requested else "OK, starting fresh.")
            events.emit("reply", text=reply)

            return reply

        events.state("thinking")
        convo.add_assistant(reply)
        fixed = fixed_reply(name, result)

        if fixed is not None:
            convo.turns.append(("user", f"Result of {name}: {result}"))
            reply = tell(None if quiet else voice, fixed)  # quiet mode: shown, not said over the music
            break

        feed_back(convo, name, result, voice)
        reply = ""
    else:
        print("phi> (too many tool calls, giving up)")
        reply = GAVE_UP
        say(voice, reply)

    if stream is not None:
        stream.flush()

    events.emit("reply", text=reply)
    remember_if_fact(engine, memory, user_text, reply)

    return reply


def offer_switch(tool: str, mode: str, control: Control, voice: Voice | None) -> str:
    needed = mode_for_tool(tool)

    if needed is None:
        reply = f"I can't use {tool.replace('_', ' ')} in {mode or 'this'} mode."
    else:
        reply = f"That needs {needed} mode. Shall I switch to it?"
        control.wanted_mode = needed

    tell(voice, reply)
    events.emit("reply", text=reply)

    return reply


def fixed_reply(name: str, result: dict) -> str | None:
    tool = TOOLS_BY_NAME.get(name)

    if tool is None or tool.reply is None or not isinstance(result, dict) or "error" in result:
        return None

    try:
        return tool.reply(result)
    except Exception:
        return None


def attach_picture(convo: Conversation, name: str, result: dict, image: Image.Image, voice: Voice | None) -> None:
    print(f"(looking at {result['image']})", flush=True)
    tell(voice, PICTURE_TAKEN)

    if convo.tool_image is not None:
        convo.drop_image(convo.tool_image)

    number = len(convo.images) + 1
    convo.add_user(
        f"Result of {name}: {result}\nThe new picture is image {number}, attached here. "
        f"Now answer the user's request in plain language, saying what you can see in image {number}.",
        [image],
    )
    convo.tool_image = number


def tool_image(result: dict) -> Image.Image | None:
    source = result.get("image") if isinstance(result, dict) else None

    if not isinstance(source, str) or not result.get("ok", True):
        return None

    try:
        image = load_image(source)
    except Exception as exc:
        print(f"(could not load {source}: {exc})", flush=True)
        return None

    image.thumbnail((MAX_TOOL_IMAGE_SIDE, MAX_TOOL_IMAGE_SIDE))

    return image


def salvage_tool_call(name: str, text: str) -> tuple[str, dict] | None:
    tool = TOOLS_BY_NAME.get(name)

    if tool is None:
        return None

    args = {}

    for key in tool.arguments:
        match = re.search(rf'"{re.escape(key)}"\s*:\s*"([^"]*)"', text)

        if match:
            args[key] = match.group(1)

    return (name, args) if args else None

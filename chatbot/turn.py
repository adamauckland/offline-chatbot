"""One turn: produce the reply to the latest user message, running tools as the model asks."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Protocol

from PIL import Image

import events
from chatbot.asides import extract_fact, route_tool
from chatbot.control import Control
from chatbot.conversation import Conversation, load_image
from chatbot.model import chat_once
from chatbot.modes import mode_for_tool
from memory import Memory
from speech import SentenceStream
from tools import TOOLS_BY_NAME, parse_tool_call, run_tool


class Voice(Protocol):
    """What ``answer`` needs from the speaker."""

    def say(self, text: str) -> None: ...

    def wait(self) -> None: ...


MAX_TOOL_CALLS = 3  # per user message, to stop a confused model looping
MAX_TOOL_IMAGE_SIDE = 1280  # the model resizes to its crop grid anyway; this just keeps the copy in memory small
# Said aloud as soon as the webcam shot is in, since looking at it takes the model a while.
PICTURE_TAKEN = "Picture taken. Examining the picture now. This will take approximately 10 seconds to complete."


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
    """Produce the final assistant reply, running tool calls when the model asks for them.

    Phi-3 has no tool role, so a tool's result is fed back as a user turn. Both
    the request and the result stay in the history so the model remembers what
    it did (for example, that the lights are now on).

    With ``memory``, known facts and relevant past exchanges are added to the
    system prompt for this turn, and the finished exchange is stored.

    ``voice`` is the speaker (anything with ``say`` and ``wait``): the reply is
    spoken sentence by sentence as it is generated, so speech starts after the
    first sentence rather than after the whole reply. Before a tool runs the
    bot says what it is about to do and waits for that to be heard, so the
    announcement never plays over the tool's own sound. Nothing else speaks
    the reply. ``control`` carries the exit and
    clear-context requests raised by tools.
    ``tools`` are the ones the current ``mode`` allows (all registered ones if
    None); the router only chooses among them, and a call the model makes
    to any other is refused with a hint to switch mode.
    """
    control = control or Control()
    stream = SentenceStream(voice.say) if voice is not None else None
    user_text = convo.turns[-1][1]
    # Facts and recalled conversations travel as a trailing note, not in the system prompt, so
    # that a new fact does not invalidate the model's cache of the whole conversation.
    extra_system = ""
    note = ""
    if memory is not None:
        note = "\n\n".join(p for p in (memory.facts_section(), memory.recall_section(user_text)) if p)
    if tools is not None:
        # The system prompt lists every tool (it must stay the same for the cache); the note, which
        # changes freely, says which are usable in this mode, so the model does not reach for the rest.
        names = [t.name for t in tools if t.name not in ("exit", "clear_context")]
        mode_line = (
            f"Current mode: {mode or 'default'}. "
            + (
                f"In this mode you can only use these tools: {', '.join(names)}. "
                if names
                else "In this mode you have no tools at all: you cannot play notes or drums, take pictures, "
                "control lights or look up the time, and you must not pretend to. "
            )
            + "If the user asks for something that needs a tool you cannot use now, do not describe the tool and "
            'do not invent a result: say that it needs another mode and that they can say "let\'s do music", '
            '"let\'s do photos", "let\'s do home" or "let\'s chat" to switch.'
        )
        note = "\n\n".join(p for p in (mode_line, note) if p)
    prefill = ""
    reply = ""
    # The router sees every tool, whatever the mode: it has to recognise a request for a tool the
    # current mode does not open, so the bot can offer to switch instead of pretending in words.
    allowed = {t.name for t in tools} if tools is not None else None
    chosen = route_tool(engine, user_text) if use_tools else None
    if chosen and allowed is not None and chosen not in allowed:
        return offer_switch(chosen, mode, control, voice)
    if chosen and not TOOLS_BY_NAME[chosen].arguments:
        # Nothing for the model to fill in, so do not give it the chance to get it wrong.
        reply = json.dumps({"tool": chosen, "arguments": {}})
    elif chosen:
        # End on the opening brace of the arguments so the model cannot skip it
        # (a trailing space here makes the 4B model drop the "{" and emit junk).
        first = next(iter(TOOLS_BY_NAME[chosen].arguments))
        prefill = json.dumps({"tool": chosen})[:-1] + ', "arguments": ' + json.dumps({first: ""})[:-2]
    for _ in range(MAX_TOOL_CALLS):
        if not reply:
            reply = chat_once(
                engine,
                convo,
                max_tokens,
                use_tools,
                extra_system,
                prefill=prefill,
                on_text=stream.feed if stream is not None else None,
                session=getattr(engine, "chat", None),
                note=note,
            )
        call = parse_tool_call(reply) if use_tools else None
        if call is None and prefill:
            call = salvage_tool_call(chosen, reply)
            if call is None:
                reply = "Sorry, I did not catch that. Could you say it again?"
                print(reply, flush=True)
                if voice is not None:
                    voice.say(reply)
                break
        prefill = ""
        if call is None:
            break
        name, args = call
        if allowed is not None and name not in allowed:
            return offer_switch(name, mode, control, voice)  # the model reached for it on its own
        print(f"(calling {name} {args})", flush=True)
        if stream is not None:
            stream.flush()  # anything the model said before the call is spoken first
        announce = f"I will {name.replace('_', ' ')}."  # "I will set lights."
        print(announce, flush=True)
        if voice is not None:
            voice.say(announce)
            voice.wait()  # heard in full before the tool makes any sound of its own
        events.state("tool", name=name, args=args)
        result = run_tool(name, args)
        print(f"(result {result})", flush=True)
        events.emit("tool", name=name, args=args, result=result)
        if control.exit_requested or control.reset_requested:
            # Neither needs a second model pass: the context is about to be dropped anyway.
            reply = "Goodbye." if control.exit_requested else "OK, starting fresh."
            print(reply, flush=True)
            if voice is not None:
                voice.say(reply)
            events.emit("reply", text=reply)
            return reply
        events.state("thinking")
        convo.add_assistant(reply)
        fixed = fixed_reply(name, result)
        if fixed is not None:
            # The tool words its own reply: exact, and no second model pass. The exchange stays
            # in the history so a follow-up ("play it again") still makes sense to the model.
            convo.turns.append(("user", f"Result of {name}: {result}"))
            reply = fixed
            print(reply, flush=True)
            if voice is not None:
                voice.say(reply)
            break
        image = tool_image(result)
        if image is not None:
            attach_picture(convo, name, result, image, voice)
        else:
            convo.turns.append(
                ("user", f"Result of {name}: {result}\nNow answer the user's request in plain language.")
            )
        reply = ""
    else:
        print("phi> (too many tool calls, giving up)")
        reply = "Sorry, I could not complete that."
        if voice is not None:
            voice.say(reply)
    if stream is not None:
        stream.flush()  # the last sentence; the reply is now being read out
    events.emit("reply", text=reply)
    if memory is not None:
        # Deliberately after the reply is on its way to the speaker: the extraction is
        # another model call, and this way it overlaps the speech instead of delaying it.
        memory.add_exchange(user_text, reply)
        fact = extract_fact(engine, user_text)
        if fact and memory.remember(fact).get("ok"):
            print(f"(remembered: {fact})", flush=True)
            events.emit("remembered", fact=fact)
    return reply


def offer_switch(tool: str, mode: str, control: Control, voice: Voice | None) -> str:
    """The request needs a tool another mode opens: say so and offer the switch, without running anything.

    The loop remembers the request; a "yes" switches mode and carries it out.
    """
    needed = mode_for_tool(tool)
    if needed is None:
        reply = f"I can't use {tool.replace('_', ' ')} in {mode or 'this'} mode."
    else:
        reply = f"That needs {needed} mode. Shall I switch to it?"
        control.wanted_mode = needed
    print(reply, flush=True)
    if voice is not None:
        voice.say(reply)
    events.emit("reply", text=reply)
    return reply


def fixed_reply(name: str, result: dict) -> str | None:
    """The reply a tool supplies for its result, or None to let the model word one."""
    tool = TOOLS_BY_NAME.get(name)
    if tool is None or tool.reply is None or not isinstance(result, dict) or "error" in result:
        return None
    try:
        return tool.reply(result)
    except Exception:  # noqa: BLE001 - a bad template should not end the turn; the model can word it
        return None


def attach_picture(convo: Conversation, name: str, result: dict, image: Image.Image, voice: Voice | None) -> None:
    """Feed a tool's result back together with the picture it took, numbered so the model looks at the right one."""
    print(f"(looking at {result['image']})", flush=True)
    print(PICTURE_TAKEN, flush=True)
    if voice is not None:
        voice.say(PICTURE_TAKEN)  # plays while the model works; the reply is queued behind it
    if convo.tool_image is not None:
        # Only the newest webcam picture is kept: every picture in the history is
        # re-encoded on every turn, and an older one is what the model tended to describe.
        convo.drop_image(convo.tool_image)
    number = len(convo.images) + 1
    convo.add_user(
        f"Result of {name}: {result}\nThe new picture is image {number}, attached here. "
        f"Now answer the user's request in plain language, saying what you can see in image {number}.",
        [image],
    )
    convo.tool_image = number


def tool_image(result: dict) -> Image.Image | None:
    """The image a tool result asks the model to look at (its ``image`` key), scaled down, or None."""
    source = result.get("image") if isinstance(result, dict) else None
    if not isinstance(source, str) or not result.get("ok", True):
        return None
    try:
        image = load_image(source)
    except Exception as exc:  # noqa: BLE001 - a missing file should not end the turn
        print(f"(could not load {source}: {exc})", flush=True)
        return None
    image.thumbnail((MAX_TOOL_IMAGE_SIDE, MAX_TOOL_IMAGE_SIDE))
    return image


def salvage_tool_call(name: str, text: str) -> tuple[str, dict] | None:
    """Pull declared arguments out of a tool call the model mangled.

    After a prefill the 4B model sometimes emits almost-JSON ("state": "on",
    "room": "kitchen"}} with a brace missing or doubled). The values are still
    there, so pick them out by argument name rather than throw the turn away.
    """
    tool = TOOLS_BY_NAME.get(name)
    if tool is None:
        return None
    args = {}
    for key in tool.arguments:
        match = re.search(rf'"{re.escape(key)}"\s*:\s*"([^"]*)"', text)
        if match:
            args[key] = match.group(1)
    return (name, args) if args else None

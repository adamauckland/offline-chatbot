from __future__ import annotations

from dataclasses import dataclass, field

from PIL import Image

import events
from audio.input import AudioIO
from audio.notes import asked_to_stop
from chatbot.conversation import Conversation, load_image
from chatbot.modes import DEFAULT_MODE, MODES, allowed_tools, answer_to_offer, detect_switch
from memory import Memory
from speech import Speaker


@dataclass
class ChatState:
    system: str | None
    memory: Memory | None = None
    speaker: Speaker | None = None
    audio: AudioIO | None = None
    mode: str = DEFAULT_MODE
    offered: tuple[str, str] | None = None
    awake: bool = False
    convo: Conversation = field(init=False)
    pending: list[Image.Image] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.convo = Conversation(self.system)

    def tools(self) -> list:
        return allowed_tools(self.mode)

    def switch_to(self, mode: str) -> None:
        self.mode = mode
        greeting = MODES[mode].greeting
        print(f"(mode: {mode})\n{greeting}\n", flush=True)
        events.emit("mode", mode=mode)

        if self.speaker is not None:
            self.speaker.say(greeting)

    def resolve_offer(self, line: str) -> str | None:
        if self.offered is None:
            return None

        mode, request = self.offered
        self.offered = None
        reply = answer_to_offer(line)

        if reply == "yes":
            self.switch_to(mode)
            return request

        if reply == "no":
            print(f"(staying in {self.mode} mode)\n", flush=True)
            return ""

        return None

    def switch_mode_if_asked(self, line: str) -> bool:
        mode = detect_switch(line)

        if mode is None:
            return False

        self.switch_to(mode)

        return True

    def stop_note_if_asked(self, line: str) -> bool:
        if self.audio is None or not self.audio.has_layers or not asked_to_stop(line):
            return False

        self.audio.stop_layers()
        print("(stopped)\n", flush=True)

        return True

    def reset(self) -> None:
        self.convo = Conversation(self.system)
        self.pending = []

        if self.memory is not None:
            self.memory.new_session()

        print("(conversation cleared)\n")


def reset(state: ChatState, _: str) -> None:
    state.reset()


def show_memory(state: ChatState, _: str) -> None:
    if state.memory is None:
        print("(memory is off)\n")
        return

    facts = state.memory.facts()
    print(f"({state.memory.count()} exchanges stored; {len(facts)} facts)")

    for fact in facts:
        print(f"- {fact}")

    print()


def remember(state: ChatState, text: str) -> None:
    if state.memory is None or not text:
        print("usage: /remember <fact>\n")
        return

    print(f"(remembered: {state.memory.remember(text)['remembered']})\n")


def forget(state: ChatState, text: str) -> None:
    if state.memory is None or not text:
        print("usage: /forget <text>\n")
        return

    print(f"(forgot {state.memory.forget(text)} fact(s))\n")


def toggle_mute(state: ChatState, _: str) -> None:
    speaker = state.speaker

    if speaker is None:
        print("(speech is not loaded; start without --no-speak)\n")
        return

    speaker.muted = not speaker.muted

    if speaker.muted:
        speaker.interrupt()

    print("(speech off)\n" if speaker.muted else "(speech on)\n")


def attach_image(state: ChatState, source: str) -> None:
    if not source:
        print("usage: /image <path-or-url>\n")
        return

    try:
        state.pending.append(load_image(source))
        print(f"(attached {source}; {len(state.pending)} image(s) pending)\n")
    except Exception as exc:
        print(f"could not load image: {exc}\n")


COMMANDS = {
    "/reset": reset,
    "/memory": show_memory,
    "/remember": remember,
    "/forget": forget,
    "/mute": toggle_mute,
    "/image": attach_image,
}


def handle_command(line: str, state: ChatState) -> bool:
    if state.stop_note_if_asked(line) or state.switch_mode_if_asked(line):
        return True

    if not line.startswith("/"):
        return False

    command, _, argument = line.partition(" ")
    handler = COMMANDS.get(command)

    if handler is None:
        print("unknown command\n")
    else:
        handler(state, argument.strip())

    return True

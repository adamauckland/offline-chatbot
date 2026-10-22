"""The state of one chat session and the slash commands that act on it."""

from __future__ import annotations

from dataclasses import dataclass, field

from PIL import Image

import events

from audio.input import AudioIO
from audio.notes import asked_to_stop
from chatbot.modes import DEFAULT_MODE, MODES, allowed_tools, answer_to_offer, detect_switch
from chatbot.conversation import Conversation, load_image
from memory import Memory
from speech import Speaker


@dataclass
class ChatState:
    """What the chat loop keeps between turns."""

    system: str | None
    memory: Memory | None = None
    speaker: Speaker | None = None
    audio: AudioIO | None = None  # for stopping a held note
    mode: str = DEFAULT_MODE  # which tools are open; see chatbot/modes.py
    offered: tuple[str, str] | None = None  # (mode, request) when the bot has asked "shall I switch?"
    convo: Conversation = field(init=False)
    pending: list[Image.Image] = field(default_factory=list)  # images attached with /image, for the next message

    def __post_init__(self) -> None:
        self.convo = Conversation(self.system)

    def tools(self) -> list:
        """The tools the current mode allows."""
        return allowed_tools(self.mode)

    def switch_to(self, mode: str) -> None:
        self.mode = mode
        greeting = MODES[mode].greeting
        print(f"(mode: {mode})\n{greeting}\n", flush=True)
        events.emit("mode", mode=mode)
        if self.speaker is not None:
            self.speaker.say(greeting)

    def resolve_offer(self, line: str) -> str | None:
        """When a switch has been offered, read ``line`` as the answer.

        Returns the original request to carry out now (after switching) on a
        yes; "" on a no; None when ``line`` is something else, in which case the
        offer is dropped and the line is handled as usual.
        """
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
        """When ``line`` asks for a mode ("let's do music"), switch and say so; True if it did.

        No model call: the phrase is recognised here, and the mode's greeting
        is the reply.
        """
        mode = detect_switch(line)
        if mode is None:
            return False
        self.switch_to(mode)
        return True

    def stop_note_if_asked(self, line: str) -> bool:
        """When a note or the drums are playing and ``line`` is just "stop", stop them; True if so.

        Checked before anything else is done with the line: in always-on mode
        "stop" would otherwise send the bot to sleep, and it never needs the model.
        """
        if self.audio is None or not self.audio.has_layers or not asked_to_stop(line):
            return False
        self.audio.stop_layers()
        print("(stopped)\n", flush=True)
        return True

    def reset(self) -> None:
        """Forget the conversation (what /reset and the clear_context tool do); long-term memory stays."""
        self.convo = Conversation(self.system)
        self.pending = []
        if self.memory is not None:
            self.memory.new_session()  # earlier turns become recallable history
        print("(conversation cleared)\n")


def handle_command(line: str, state: ChatState) -> bool:
    """Run ``line`` if it is a slash command; True when it was one (the loop then skips the model).

    ``/quit`` is left to the loop, since it has to end it.
    """
    memory, speaker = state.memory, state.speaker
    if state.stop_note_if_asked(line):
        return True
    if state.switch_mode_if_asked(line):
        return True
    if line == "/reset":
        state.reset()
    elif line == "/memory":
        if memory is None:
            print("(memory is off)\n")
        else:
            facts = memory.facts()
            print(f"({memory.count()} exchanges stored; {len(facts)} facts)")
            for fact in facts:
                print(f"- {fact}")
            print()
    elif line.startswith("/remember"):
        text = line[len("/remember") :].strip()
        if memory is None or not text:
            print("usage: /remember <fact>\n")
        else:
            print(f"(remembered: {memory.remember(text)['remembered']})\n")
    elif line.startswith("/forget"):
        text = line[len("/forget") :].strip()
        if memory is None or not text:
            print("usage: /forget <text>\n")
        else:
            print(f"(forgot {memory.forget(text)} fact(s))\n")
    elif line == "/mute":
        if speaker is None:
            print("(speech is not loaded; start without --no-speak)\n")
        else:
            speaker.muted = not speaker.muted
            if speaker.muted:
                speaker.interrupt()
            print("(speech off)\n" if speaker.muted else "(speech on)\n")
    elif line.startswith("/image"):
        source = line[len("/image") :].strip()
        if not source:
            print("usage: /image <path-or-url>\n")
        else:
            try:
                state.pending.append(load_image(source))
                print(f"(attached {source}; {len(state.pending)} image(s) pending)\n")
            except Exception as exc:  # noqa: BLE001 - show any load failure to the user
                print(f"could not load image: {exc}\n")
    elif line.startswith("/"):
        print("unknown command\n")
    else:
        return False
    return True

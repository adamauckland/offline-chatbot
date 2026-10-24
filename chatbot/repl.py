from __future__ import annotations

from collections.abc import Callable

import queue
import threading

import events
from audio.input import AudioIO, cue
from audio.listen import Listener
from audio.wake import WakeListener
from chatbot.commands import ChatState, handle_command
from chatbot.control import Control
from chatbot.turn import answer
from chatbot.voice import SLEEP_PHRASES, next_spoken_command
from memory import Memory
from speech import Speaker

QUIT_COMMANDS = ("/quit", "/exit")


class Repl:
    def __init__(
        self,
        preload,
        system: str | None,
        max_tokens: int,
        use_tools: bool,
        speaker: Speaker | None,
        listener: Listener | None,
        hands_free: bool,
        memory: Memory | None,
        wake: WakeListener | None,
        ask: Callable[[str], str],
        control: Control | None,
        audio: AudioIO | None,
        mode: str,
    ) -> None:
        self.preload = preload
        self.max_tokens = max_tokens
        self.use_tools = use_tools
        self.speaker = speaker
        self.listener = listener
        self.hands_free = hands_free
        self.memory = memory
        self.wake = wake
        self.ask = ask
        self.control = control or Control()
        self.audio = audio
        self.state = ChatState(system, memory, speaker, audio, mode=mode)
        # In always-on mode a typed line (the TUI's input box) cuts listening short and is taken instead.
        self._typed: queue.Queue[str] = queue.Queue()
        self._typed_ready = threading.Event()

    def run(self) -> None:
        events.emit("mode", mode=self.state.mode)
        events.subscribe(self._on_event)
        self._print_welcome()

        while True:
            try:
                heard = self._next_line()
            except EOFError:
                print()
                break
            except KeyboardInterrupt:
                print()
                self._interrupt_speech()
                return

            if heard is None:
                continue

            line, spoken = heard

            if line in QUIT_COMMANDS:
                break

            line = self._after_offer(line)

            if line is None or handle_command(line, self.state):
                continue

            reply = self._take_turn(line, spoken)

            if reply is None:
                continue

            self._note_offer(line)

            if self.control.exit_requested:
                break

            self._reset_if_requested()
            self.state.awake = True

        events.unsubscribe(self._on_event)
        self._wait_for_speech()

    def _on_event(self, name: str, data: dict) -> None:
        if name == "cancel":
            self._interrupt_speech()
        elif name == "typed" and self.wake is not None:
            self._typed.put(data.get("text", ""))
            self._typed_ready.set()

    def _pending_typed(self) -> str | None:
        try:
            line = self._typed.get_nowait()
        except queue.Empty:
            self._typed_ready.clear()
            return None
        if self._typed.empty():
            self._typed_ready.clear()
        return line.strip()

    def _interrupt_speech(self) -> None:
        """Escape in the TUI, Ctrl-C in the terminal: stop talking, and stop the music too.

        Spoken "stop" cannot be heard over the drums, so the key is the sure
        way to end a backing loop or a held note.
        """
        if self.speaker is not None:
            self.speaker.interrupt()
        if self.audio is not None and self.audio.has_layers:
            self.audio.stop_layers()
            print("(stopped)\n", flush=True)

    def _wait_for_speech(self) -> None:
        if self.speaker is not None:
            self.speaker.wait()

    def _print_welcome(self) -> None:
        if self.wake is not None:
            print(
                f'Always listening. Say "{self.wake.phrase}" and then your question; '
                f'say "go to sleep" to cancel. Ctrl-C to exit.\n'
            )
            return

        print(
            "Phi-3.5-vision chatbot. Type /image <path-or-url> to attach an image, "
            "/mute to toggle speech, /reset to start over, /quit to exit."
            + (" Press Enter on an empty line to talk." if self.listener else "")
            + "\n"
        )

    def _next_line(self) -> tuple[str, bool] | None:
        if self.wake is not None:
            typed = self._pending_typed()
            if typed is not None:
                return (typed, False) if typed else None
            line = self._next_spoken_line()
            if line:
                return (line, True)
            typed = self._pending_typed()  # listening was cut short by a typed line
            return (typed, False) if typed else None

        line = self._next_typed_line()

        if not line and self.listener is not None:
            return self._record_from_microphone()

        return (line, False) if line else None

    def _next_spoken_line(self) -> str | None:
        holding = self.audio is not None and self.audio.has_layers
        awake = self.state.awake or holding
        line, woke = next_spoken_command(
            self.wake, self.listener, self.speaker, awake, keep_awake=holding, abort=self._typed_ready
        )
        self.state.awake = False
        if not line and self._typed_ready.is_set():
            return None  # a typed line is waiting: not a "yes?", not going to sleep

        if not line:
            if woke:
                print("(yes?)", flush=True)
                self.state.awake = True

            return None

        if self.state.stop_note_if_asked(line):
            return None

        if line.lower().strip(" .!") in SLEEP_PHRASES:
            print("(something is playing; say stop to end it)\n" if holding else "(going to sleep)\n")
            return None

        print(f"you (mic)> {line}")

        return line

    def _next_typed_line(self) -> str:
        if self.hands_free:
            self._wait_for_speech()
            return ""

        events.state("idle")

        return self.ask("you> ").strip()

    def _record_from_microphone(self) -> tuple[str, bool] | None:
        line = self.listener.listen()

        if not line:
            return None

        print(f"you (mic)> {line}")

        return line, True

    def _after_offer(self, line: str) -> str | None:
        taken = self.state.resolve_offer(line)

        if taken == "":
            return None

        return line if taken is None else taken

    def _take_turn(self, line: str, spoken: bool) -> str | None:
        convo = self.state.convo
        convo.add_user(line, self.state.pending)
        self.state.pending = []
        events.emit("user", text=line, spoken=spoken)
        self._interrupt_speech()

        if spoken and self.listener is not None:
            cue(self.listener.audio, "think", wait=False)

        print("phi> ", end="", flush=True)
        events.CANCEL.clear()
        events.state("thinking")
        n_turns = len(convo.turns)

        try:
            reply = answer(
                self.preload,
                convo,
                self.max_tokens,
                self.use_tools,
                self.memory,
                voice=self.speaker,
                control=self.control,
                tools=self.state.tools(),
                mode=self.state.mode,
            )
        except KeyboardInterrupt:
            self._abandon_turn(convo, n_turns)
            return None

        convo.add_assistant(reply)

        return reply

    def _abandon_turn(self, convo, n_turns: int) -> None:
        print("\n(generation interrupted)\n")
        events.CANCEL.clear()
        self._interrupt_speech()
        convo.truncate(n_turns - 1)

    def _note_offer(self, line: str) -> None:
        if self.control.wanted_mode is not None:
            self.state.offered = (self.control.wanted_mode, line)
            self.control.wanted_mode = None

    def _reset_if_requested(self) -> None:
        if self.control.reset_requested:
            self.control.reset_requested = False
            self.state.reset()


def run_repl(
    preload,
    system: str | None,
    max_tokens: int,
    use_tools: bool,
    speaker: Speaker | None,
    listener: Listener | None,
    hands_free: bool,
    memory: Memory | None,
    wake: WakeListener | None = None,
    ask: Callable[[str], str] = input,
    control: Control | None = None,
    audio: AudioIO | None = None,
    mode: str = "chat",
) -> None:
    Repl(
        preload, system, max_tokens, use_tools, speaker, listener, hands_free, memory, wake, ask, control, audio, mode
    ).run()

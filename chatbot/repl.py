"""The chat loop: read a line (typed or spoken), answer it, repeat."""

from __future__ import annotations

from collections.abc import Callable

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
    """The chat loop. ``ask(prompt)`` reads one typed line (``input`` in the terminal; the TUI
    supplies its own) and raises EOFError when there is no more input. ``control`` is the
    object the exit and clear_context tools were registered on."""
    control = control or Control()
    state = ChatState(system, memory, speaker, audio, mode=mode)
    events.emit("mode", mode=mode)

    def on_event(name: str, data: dict) -> None:  # noqa: ARG001
        if name == "cancel" and speaker is not None:
            speaker.interrupt()  # a display's "stop" also cuts off speech, like Ctrl-C

    events.subscribe(on_event)
    if wake is not None:
        print(
            f'Always listening. Say "{wake.phrase}" and then your question; '
            f'say "go to sleep" to cancel. Ctrl-C to exit.\n'
        )
    else:
        print(
            "Phi-3.5-vision chatbot. Type /image <path-or-url> to attach an image, "
            "/mute to toggle speech, /reset to start over, /quit to exit."
            + (" Press Enter on an empty line to talk." if listener else "")
            + "\n"
        )
    awake = False
    while True:
        spoken = False  # did this line come from the microphone?
        try:
            if wake is not None:
                # A note or the drums are waiting for "stop": keep listening without the wake phrase.
                holding = audio is not None and audio.has_layers
                line, woke = next_spoken_command(wake, listener, speaker, awake or holding, keep_awake=holding)
                awake = False
                spoken = True
                if not line:
                    if woke:
                        print("(yes?)", flush=True)  # heard the phrase but no command: keep listening
                        awake = True
                    continue
                if state.stop_note_if_asked(line):
                    continue  # "stop" ends a held note; it only means sleep when nothing is playing
                if line.lower().strip(" .!") in SLEEP_PHRASES:
                    print("(something is playing; say stop to end it)\n" if holding else "(going to sleep)\n")
                    continue
                print(f"you (mic)> {line}")
            elif hands_free:
                if speaker is not None:
                    speaker.wait()  # do not record our own voice
                line = ""
            else:
                events.state("idle")
                line = ask("you> ").strip()
            if not line and listener is not None:
                line = listener.listen()
                spoken = True
                if not line:
                    continue
                print(f"you (mic)> {line}")
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print()
            if speaker is not None:
                speaker.interrupt()  # Ctrl-C means stop talking now
            return
        if not line:
            continue
        if line in ("/quit", "/exit"):
            break
        if state.offered is not None:  # "shall I switch to music mode?" was asked last turn
            taken = state.resolve_offer(line)
            if taken == "":
                continue  # declined
            if taken is not None:
                line = taken  # accepted: the mode has changed, now do what was asked
        if handle_command(line, state):
            continue

        convo = state.convo
        convo.add_user(line, state.pending)
        state.pending = []
        events.emit("user", text=line, spoken=spoken)
        if speaker is not None:
            speaker.interrupt()  # a new question cuts off any answer still being read out
        if spoken and listener is not None:
            cue(listener.audio, "think", wait=False)  # transcript ready, handing it to the model
        print("phi> ", end="", flush=True)
        events.CANCEL.clear()
        events.state("thinking")
        n_turns = len(convo.turns)
        try:
            reply = answer(
                preload,
                convo,
                max_tokens,
                use_tools,
                memory,
                voice=speaker,
                control=control,
                tools=state.tools(),
                mode=state.mode,
            )
        except KeyboardInterrupt:
            print("\n(generation interrupted)\n")
            events.CANCEL.clear()
            if speaker is not None:
                speaker.interrupt()  # the sentences already spoken stop too
            convo.truncate(n_turns - 1)  # drop the unanswered user turn, any tool exchange and its picture
            continue
        convo.add_assistant(reply)  # answer() has already spoken it, sentence by sentence
        if control.wanted_mode is not None:
            state.offered = (control.wanted_mode, line)  # the next line answers the offer
            control.wanted_mode = None
        if control.exit_requested:
            break
        if control.reset_requested:
            control.reset_requested = False
            state.reset()
        awake = True  # allow a follow-up question without repeating the wake phrase
    events.unsubscribe(on_event)
    if speaker is not None:
        speaker.wait()  # /quit or Ctrl-D: finish the sentence, then leave

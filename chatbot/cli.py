#!/usr/bin/env python3
"""Example multi-turn terminal chatbot built on phi-3-vision-mlx.

Runs Microsoft's Phi-3.5-vision-instruct locally on Apple silicon via MLX.
You can chat in plain text, or attach one or more images (local paths or URLs)
to a message and ask questions about them. The full conversation, including
every image shared so far, is re-sent to the model on each turn, so follow-up
questions about an earlier image just work.

The model can also call local API endpoints ("tools") to get things done, for
example "turn the kitchen lights on". See tools/registry.py for the registry and
tools/server.py for an example server. Start the server first:
  uv run -m tools.server

Replies are also spoken aloud through the speakers with Kyutai's Pocket TTS
(see speech/speaker.py). Pass --no-speak to keep it quiet, or /mute inside the chat.

You can talk instead of typing: press Enter on an empty line to record from the
microphone until you pause (see audio/listen.py), or start with --hands-free for a
continuous spoken conversation. With --wake-word the microphone stays open and
a tiny keyword spotter (see audio/wake.py) wakes the bot when you say the phrase, so
you never touch the keyboard: "hey computer, turn the kitchen lights on".

It also has long-term memory (see memory/store.py): facts it is told are kept in a
SQLite file and shown to it every turn, and earlier conversations that look
relevant to the current message are pulled in automatically.

With --tui the chat runs inside a full-screen terminal UI (see tui/app.py): a robot
face that reacts to what the bot is doing, a VU meter for the microphone and
the speakers, and the conversation log. Everything below works the same there.

Commands inside the chat:
  <empty line>           record a question from the microphone
  /image <path-or-url>   attach an image to your next message (repeatable)
  /memory                list remembered facts
  /remember <fact>       store a fact yourself
  /forget <text>         delete facts containing <text>
  /mute                  toggle speech on or off
  /reset                 forget the current conversation (long-term memory stays);
                         or just say "start over" (the clear_context tool)
  /quit                  exit (Ctrl-D also works)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from audio.input import AudioIO
from audio.listen import Listener
from audio.stt import CHOICES as STT_CHOICES, describe as describe_stt
from audio.drums import drum_tools
from audio.notes import note_tools
from audio.turnaround import Turnaround
from audio.wake import DEFAULT_WAKE_WORD, WakeListener
from chatbot.asides import warm_up
from chatbot.modes import DEFAULT_MODE, MODES, allowed_tools
from chatbot.control import Control
from chatbot.conversation import Conversation, load_image
from chatbot.model import CACHE_TOKENS, DEFAULT_IMAGE_CROPS, load_model
from chatbot.repl import run_repl
from chatbot.turn import answer
from memory import DEFAULT_DB, Memory
from speech import DEFAULT_MODEL as DEFAULT_TTS_MODEL, DEFAULT_VOICE, Speaker
from tools import BASE_URL, register, tools_prompt

DEFAULT_SYSTEM = (
    "You are Scrudley, a friendly voice assistant in the user's home. You happily chat about "
    "anything, write and explain things, and you can also see images, control the lights and "
    "look up the time with your tools. You remember what the user tells you about themselves "
    "and answer questions about it from the facts you have been given. Keep answers short, "
    "one or two sentences unless asked for more."
)


def ensure_tools_server() -> subprocess.Popen | None:
    """Start tools/server.py if nothing is listening on its address; return the process."""
    import time

    import requests

    try:
        requests.get(f"{BASE_URL}/time", timeout=1)
        return None
    except requests.RequestException:
        pass
    root = Path(__file__).resolve().parent.parent  # the project root, where the packages live
    if not (root / "tools" / "server.py").is_file():
        return None
    proc = subprocess.Popen(
        [sys.executable, "-m", "tools.server"], cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    for _ in range(50):
        time.sleep(0.2)
        try:
            requests.get(f"{BASE_URL}/time", timeout=1)
            print(f"(started tools/server.py on {BASE_URL})", file=sys.stderr)
            return proc
        except requests.RequestException:
            continue
    print("(could not start tools/server.py; tool calls will fail)", file=sys.stderr)
    return proc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chatbot", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="use the full fp16 weights instead of the 4-bit quantized model (slower, ~8 GB RAM more)",
    )
    parser.add_argument("--max-tokens", type=int, default=512, help="maximum tokens per reply (default 512)")
    parser.add_argument(
        "--image-crops",
        type=int,
        default=DEFAULT_IMAGE_CROPS,
        metavar="N",
        help="detail per image: the model cuts each picture into up to N tiles of 336 px, and every "
        "picture in the conversation is re-encoded each turn; 1 is fastest, 16 the model's "
        f"full detail, good for reading text (default {DEFAULT_IMAGE_CROPS})",
    )
    parser.add_argument(
        "--cache-tokens",
        type=int,
        default=CACHE_TOKENS,
        metavar="N",
        help="how much of the conversation the model keeps encoded between turns, in tokens; a turn then "
        "only encodes what is new. 0 re-encodes everything every turn (slower, less memory); the "
        f"maximum is {CACHE_TOKENS} (about 1.6 GB) (default {CACHE_TOKENS})",
    )
    parser.add_argument(
        "--mode",
        default=DEFAULT_MODE,
        choices=list(MODES),
        help="the mode to start in, which decides the tools on offer (default chat: none); switch by saying "
        '"let\'s do music", "let\'s do photos", "let\'s do home" or "let\'s chat"',
    )
    parser.add_argument("--system", default=DEFAULT_SYSTEM, help="system prompt (pass an empty string to disable)")
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        metavar="PATH_OR_URL",
        help="image to attach to the first message (repeatable); use with --prompt",
    )
    parser.add_argument("--prompt", help="run a single prompt instead of starting the interactive chat")
    parser.add_argument(
        "--no-tools", action="store_true", help="do not offer the tools in tools/registry.py to the model"
    )
    parser.add_argument(
        "--no-tools-server",
        action="store_true",
        help="do not start tools/server.py automatically when nothing answers on its port",
    )
    parser.add_argument("--no-memory", action="store_true", help="no long-term memory")
    parser.add_argument("--memory-db", default=DEFAULT_DB, help=f"SQLite file for memory (default {DEFAULT_DB})")
    parser.add_argument("--no-speak", action="store_true", help="do not read replies aloud")
    parser.add_argument("--no-listen", action="store_true", help="do not load the microphone transcriber")
    parser.add_argument(
        "--hands-free", action="store_true", help="keep listening after each spoken reply instead of waiting for Enter"
    )
    parser.add_argument(
        "--stt",
        default="auto",
        choices=STT_CHOICES,
        help="speech-to-text engine: whisper (Whisper on MLX), apple (the macOS Speech framework, macOS 26+), "
        "kyutai (Kyutai STT 1B on MLX, in its own helper process) or sherpa (Moonshine via sherpa-onnx, any "
        "CPU); auto takes whisper where it runs and sherpa elsewhere (default auto). --list-stt shows which "
        "can run here",
    )
    parser.add_argument(
        "--stt-model",
        default=None,
        metavar="MODEL",
        help="the engine's model: a Hugging Face repo for whisper or kyutai, a sherpa-onnx Moonshine release "
        "name for sherpa (default: each engine's own)",
    )
    parser.add_argument("--list-stt", action="store_true", help="show the speech-to-text engines and exit")
    parser.add_argument(
        "--wake-word",
        nargs="?",
        const=DEFAULT_WAKE_WORD,
        metavar="PHRASE",
        help=f"always listen and wake on PHRASE (default '{DEFAULT_WAKE_WORD}')",
    )
    parser.add_argument(
        "--wake-threshold",
        type=float,
        default=0.2,
        metavar="T",
        help="spotter sensitivity, lower fires more easily (default 0.2; try 0.1)",
    )
    parser.add_argument(
        "--mic", metavar="DEVICE", help="input device name or index (list them with: uv run python -m sounddevice)"
    )
    parser.add_argument(
        "--mic-channel", type=int, default=1, metavar="N", help="input channel on that device, 1-based (default 1)"
    )
    parser.add_argument(
        "--mic-gain",
        type=float,
        default=1.0,
        metavar="X",
        help="fixed boost applied to the microphone before auto gain (default 1.0)",
    )
    parser.add_argument(
        "--no-auto-gain", action="store_true", help="do not boost quiet microphone signals automatically"
    )
    parser.add_argument("--speaker", metavar="DEVICE", help="output device name or index")
    parser.add_argument(
        "--no-turnaround",
        action="store_true",
        help="no random notes while the bot works on a spoken answer (see audio/turnaround.py)",
    )
    parser.add_argument(
        "--no-robot",
        action="store_true",
        help="skip the robot-voice effect (octave-up pitch shift and flanger, see audio/dsp.py)",
    )
    parser.add_argument(
        "--voice",
        default=DEFAULT_VOICE,
        help=f"pocket-tts voice name or path to a .wav to clone (default {DEFAULT_VOICE})",
    )
    parser.add_argument(
        "--tts-model",
        default=DEFAULT_TTS_MODEL,
        help=f"pocket-tts model, e.g. english_7m_2026-10 or english (default {DEFAULT_TTS_MODEL})",
    )
    parser.add_argument(
        "--tui", action="store_true", help="full-screen terminal UI with a robot face and VU meter (see tui/app.py)"
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.tui and args.prompt is not None:
        parser.error("--tui is for the interactive chat; drop --prompt")
    if args.list_stt:
        print("speech-to-text engines on this machine:\n" + describe_stt())
        return

    print("Loading model (first run downloads ~8 GB from Hugging Face and converts it)...", file=sys.stderr)
    preload = load_model(quantize=not args.no_quantize, image_crops=args.image_crops, cache_tokens=args.cache_tokens)
    audio = None
    if not (args.no_speak and args.no_listen):
        audio = AudioIO(
            mic=args.mic,
            channel=args.mic_channel,
            speaker=args.speaker,
            mic_gain=args.mic_gain,
            auto_gain=not args.no_auto_gain,
        )
    turnaround = Turnaround(audio) if audio is not None and not args.no_turnaround else None
    speaker = None
    if not args.no_speak:
        print("Loading speech model...", file=sys.stderr)
        speaker = Speaker(audio, model=args.tts_model, voice=args.voice, robot=not args.no_robot)
    listener = None
    if not args.no_listen and args.prompt is None:
        print(f"Loading speech recognition ({args.stt})...", file=sys.stderr)
        listener = Listener(audio, args.stt, args.stt_model)
        print(f"(speech recognition: {listener.transcriber!r})", file=sys.stderr)
    if (args.hands_free or args.wake_word) and listener is None:
        parser.error("--hands-free and --wake-word need the microphone; drop --no-listen")
    wake = None
    if args.wake_word:
        print("Loading wake-word models...", file=sys.stderr)
        wake = WakeListener(audio, args.wake_word, threshold=args.wake_threshold)
    memory = None if args.no_memory else Memory(args.memory_db)
    use_tools = not args.no_tools
    tools_proc = ensure_tools_server() if use_tools and not args.no_tools_server else None
    control = Control()
    if use_tools:
        control.register_tools()
        if audio is not None:
            for tool in note_tools(audio) + drum_tools(audio):  # they need the speakers, so only with audio
                register(tool)
    system = args.system or ""
    if use_tools:
        system = (system + "\n\n" + tools_prompt()).strip()
    system = system or None
    print("Warming up the model...", file=sys.stderr)
    warm_up(
        preload, system, allowed_tools(args.mode)
    )  # the system prompt and the routing prefix, before the first turn

    if args.prompt is not None:
        convo = Conversation(system)
        convo.add_user(args.prompt, [load_image(src) for src in args.image])
        answer(
            preload,
            convo,
            args.max_tokens,
            use_tools,
            memory,
            voice=speaker,
            control=control,
            tools=allowed_tools(args.mode),
            mode=args.mode,
        )
        if speaker is not None:
            speaker.wait()  # answer() spoke the reply as it was generated
        return

    if args.image:
        parser.error("--image requires --prompt; in interactive mode use the /image command")
    try:
        if args.tui:
            from tui import run_tui

            run_tui(
                lambda ask: run_repl(
                    preload,
                    system,
                    args.max_tokens,
                    use_tools,
                    speaker,
                    listener,
                    args.hands_free,
                    memory,
                    wake,
                    ask=ask,
                    control=control,
                    audio=audio,
                    mode=args.mode,
                ),
                typed=wake is None and not args.hands_free,
                wake_phrase=wake.phrase if wake else None,
            )
        else:
            run_repl(
                preload,
                system,
                args.max_tokens,
                use_tools,
                speaker,
                listener,
                args.hands_free,
                memory,
                wake,
                control=control,
                audio=audio,
                mode=args.mode,
            )
    finally:
        if turnaround is not None:
            turnaround.close()
        if wake is not None:
            wake.close()
        if listener is not None:
            listener.close()
        if audio is not None:
            audio.close()
        if tools_proc is not None:
            tools_proc.terminate()
            try:
                tools_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                tools_proc.kill()


if __name__ == "__main__":
    main()

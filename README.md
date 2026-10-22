# scrudley

An example multi-turn terminal chatbot for **Phi-3.5-vision-instruct**, running
locally on Apple silicon through [phi-3-vision-mlx](https://github.com/JosefAlbers/Phi-3-Vision-MLX).
It can look at images, call your local API endpoints as tools, reads its
answers aloud with [Pocket TTS](https://github.com/kyutai-labs/pocket-tts),
listens to you through the microphone with Whisper on MLX (press Enter, or
always-on with a wake word), and remembers facts and past conversations
between runs.

You can chat in plain text, or attach images (local files or URLs) and ask
questions about them. Follow-up questions about an earlier image work because
the whole conversation is re-sent to the model each turn.

## Requirements

- A Mac with Apple silicon (MLX does not run anywhere else)
- Python 3.12 (the library pins dependencies that have no 3.13 wheels)
- [uv](https://docs.astral.sh/uv/) (or plain `pip` in a venv)
- ~10 GB of free disk for the model download and the quantized copy
- Speakers or headphones if you want spoken replies, and a microphone to talk
  (macOS will ask once to allow your terminal to use it)

## Setup

```sh
uv sync
```

Formatting is black (`uv run black .`, 120 columns, settings in `pyproject.toml`).

Or without uv:

```sh
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Run

```sh
uv run -m chatbot        # or: uv run scrudley
```

The first run downloads `microsoft/Phi-3.5-vision-instruct` from Hugging Face
and writes a 4-bit quantized copy into `./microsoftPhi35visioninstruct_q/`
(about 2.5 GB). Later runs start in a few seconds.

Inside the chat:

| Command                 | Effect                                            |
| ----------------------- | ------------------------------------------------- |
| empty line (just Enter) | record a question from the microphone             |
| `/image <path-or-url>`  | attach an image to your next message (repeatable) |
| `/mute`                 | toggle spoken replies on or off                   |
| `/memory`               | list remembered facts                             |
| `/remember <fact>`      | store a fact yourself                             |
| `/forget <text>`        | delete facts containing that text                 |
| `/reset`                | forget the current conversation (memory stays); or say "start over" |
| `/quit` or Ctrl-D       | exit                                              |

Example session:

```
you> /image https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg
(attached ...; 1 image(s) pending)

you> What animal is this and what is it doing?
phi> The image shows a cat ...
you> Suggest a name for it.
phi> ...
```

One-shot mode, handy for scripts:

```sh
uv run -m chatbot --image photo.jpg --prompt "Describe this picture in one sentence."
uv run -m chatbot --prompt "Write a haiku about MLX."
```

Options: `--max-tokens N` (default 512), `--system "..."` to change the system
prompt, `--no-quantize` to run the full fp16 weights, `--no-speak` to stay
silent, `--voice` and `--tts-model` to change the voice, `--no-listen` to skip
the microphone, `--hands-free` for a continuous spoken conversation, and
`--stt-model` to change the speech recogniser, `--no-memory` to run without
long-term memory and `--memory-db` to choose where it is kept (see below).

## Project layout

`chatbot/` is the program itself, started with `uv run -m chatbot`:

| Module            | What is in it                                                                 |
|-------------------|-------------------------------------------------------------------------------|
| `cli.py`          | the command line, model and audio set-up, and wiring                          |
| `repl.py`         | the chat loop: read a typed or spoken line, answer it, repeat                 |
| `commands.py`     | the session state and the slash commands (`/image`, `/memory`, `/reset`...)   |
| `turn.py`         | `answer`: one reply, running tools and feeding their results (and pictures) back |
| `asides.py`       | the quiet side questions to the model: which tool, and is this a fact         |
| `model.py`        | loading Phi-3.5-vision and `chat_once`, the token loop                        |
| `conversation.py` | the history in Phi-3's chat format, with image numbering                      |
| `control.py`      | the built-in `exit` and `clear_context` tools and the flags the loop acts on  |
| `voice.py`        | the always-on turn: wake, record, transcribe, strip the wake phrase           |

Everything else is a package named for what it does:

| Package   | What is in it                                                                                   |
|-----------|-------------------------------------------------------------------------------------------------|
| `audio/`  | the shared microphone/speaker stream (`input.py`, run as a child process by `server.py`), the robot-voice DSP (`dsp.py`), recording (`listen.py`), the speech-to-text engines (`stt/`), the wake word and VAD (`wake.py`), notes and drums (`notes.py`, `drums.py`, `patterns.py`, `turnaround.py`), and the live microphone check (`uv run -m audio.mic_check`) |
| `speech/` | reading replies aloud with Pocket TTS (`speaker.py`)                                            |
| `images/` | taking a picture with the webcam (`webcam.py`, try it with `uv run -m images.webcam`)           |
| `memory/` | long-term memory: facts and past conversations in SQLite (`store.py`)                            |
| `tools/`  | the tool registry the model picks from (`registry.py`) and the example HTTP server (`uv run -m tools.server`) |
| `tui/`    | the full-screen terminal UI with the robot face and VU meter (`app.py`; `uv run -m tui` for a demo) |
| `events/` | the small event bus the chatbot reports its activity on (`bus.py`)                               |

Each package's `__init__.py` re-exports its public names, so code imports
`from memory import Memory` or `from tools import Tool`, not the file inside.

## Always on: wake word

```sh
uv run -m chatbot --wake-word                 # "hey computer"
uv run -m chatbot --wake-word "ok scrudley"   # any phrase, no training
```

The microphone stays open and nothing is typed. Say the phrase, hear a rising
two-note chime, and speak your question; the reply is read out, and for the
next five seconds you can ask a follow-up without repeating the phrase. Saying
"go to sleep", "stop" or "never mind" after the chime cancels. Ctrl-C exits.

Three different sounds mark the stages of a spoken turn (`CUES` in
`audio/input.py`, each a short list of notes you can retune):

| cue     | sound                 | when                                                         |
| ------- | --------------------- | ------------------------------------------------------------ |
| `wake`  | rising two-note chime | the wake phrase was heard; recording starts                  |
| `parse` | quick double tick     | you stopped talking; Whisper is transcribing the clip        |
| `think` | falling three-note    | the transcript is ready and is being handed to the Phi model |

The `parse` and `think` cues also play when you talk after pressing Enter or
in `--hands-free` mode; typed questions stay silent.

The wait after that, the turnaround phase from transcribing your words until
the bot starts to speak, is not silent either: a short random note from a
pentatonic scale plays every 250 ms at a low gain (0.2) so you can hear it is
working, including while it calls a tool or looks at a picture (a typed
question gets them too, from the moment the model starts). The notes
stop the moment speech starts and never sound over a cue or between chunks
of speech. `audio/turnaround.py` holds it, driven by the bot's state events;
`--no-turnaround` turns it off.
`audio/wake.py` holds it, built on [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx):

- Two small CPU models run on every 100 ms frame. **Silero VAD** decides
  whether the frame is speech; in silence nothing else runs. The **keyword
  spotter** (an open-vocabulary zipformer, 3.3M parameters, trained on
  GigaSpeech) only sees frames the VAD flagged. Together they cost well under
  one percent of a core: 2.2 s of audio is processed in about 17 ms.
- The wake phrase is plain text. It is turned into the model's BPE tokens and
  written to a keywords file, so any English phrase works with no training.
  Phrases of two or three real, clearly pronounced words are the most
  reliable. A made-up word like "scrudley" becomes six sub-word tokens, which
  is harder to match; the spotter runs with a wide beam (16 paths) to cope,
  which on synthetic voices took "ok scrudley" from 6 of 9 clips to 9 of 9
  with no false fires. Registering several spellings does not help, it makes
  the spellings compete. The wide beam's one cost: a sentence that contains
  the words in order, like "hey, the computer is broken", can wake it.
- Both models download once into `models/` (about 18 MB).
- Whisper, the vision model and text-to-speech only run after the wake word,
  so the always-on cost is just the two small models.

Details that matter for not clipping words: the spotter fires a few frames
after the phrase ends, so the last half second of audio it consumed is
carried into the recording (measured on synthetic clips: 0.8 s brought the
end of the phrase along and Whisper turned it into "Skritly" or "Certainly",
0.3 s lost the first word, 0.5 s kept every command intact); the VAD keeps
its state across the wake so a sentence that runs straight on from the phrase
is tracked; and a half second of pre-roll is kept before speech is confirmed.
Every clip also gets 0.3 s of silence put in front of it before Whisper sees
it: a clip that starts on speech loses its first consonant otherwise ("set a
timer" came back as "at a timer").

If the phrase itself leaks into the transcript it is stripped: a leading word
that is nearly the phrase ("Computer, lights off", or "Scrudly" for a made-up
name) or a literal fragment of its end, and a lone word of any resemblance
when nothing else was said. Only that much. An earlier version dropped any
leading word half-similar to a phrase word, which with "hey computer" ate
"where", "when", "the", "open", "put" and "hello" from the start of commands.
If you say the phrase and nothing else, the bot prints `(yes?)` and stays
awake for the follow-up window. While the bot is talking the microphone is
ignored, and anything captured while it was busy is discarded before it
listens again, so it never answers itself.

Tuning: `--wake-threshold` trades false wakes against missed ones (default
0.2; try 0.1 if it misses you). The follow-up window is `FOLLOW_UP_SECONDS`
in `chatbot/voice.py`.

### Sensitivity: what was learned the hard way

If it only wakes when you shout, three things in earlier versions were to
blame and are now fixed; the measurements are worth knowing when tuning.

- **The spotter's stream must be recreated, not reset, between wakes.** The
  streaming encoder keeps left context, and with `reset_stream` alone the
  same clip fired or not from run to run. With a fresh stream the clean
  phrase fires at every peak level from -3 dBFS down to -40 dBFS, identically
  five times in a row, and fails below -50 dBFS.
- **The spotter sees every frame.** Gating it behind the voice detector (to
  save CPU) made it miss the phrase even at high volume. It costs about 1.5 %
  of a core, so there is nothing to save.
- **Automatic gain must be steady across a phrase.** The gain is set from the
  loudest frame of the last second, with instant attack, a one-second hold and
  a 3 % per frame release, so it recovers within seconds of a loud sound. Two
  versions failed: a 0.5 % release left normal speech under-amplified for a
  minute after one shout, and a 10 % release pumped through the phrase and
  broke detection. Gain is also capped so the noise floor never rises above
  -40 dBFS, otherwise the voice detector hears amplified hiss as speech and
  recordings never end.

Over the UMC1820's very quiet floor (-70 dBFS RMS) the phrase fires at every
level down to -50 dBFS. Over a noisy microphone (the webcam's floor is
-41 dBFS) detection is limited by signal-to-noise below about -30 dBFS, and no
gain can fix that; a closer or better microphone can. Playing the phrase
through loudspeakers across the room into the webcam is a hard case and
fires about two times in three.

### If it does not wake: check what it hears

```sh
uv run -m audio.mic_check                       # live meter on the default input
uv run -m audio.mic_check --mic Webcam          # or another device
uv run -m audio.mic_check --mic UMC --mic-channel 3
```

Each line is one 100 ms frame: raw level, the automatic gain applied, a bar,
`speech` when the voice detector fires and `WAKE` when the phrase is spotted.
Talk normally: the bar should reach the middle and `speech` should appear. If
the raw level stays near -60 dBFS while you talk, the bot is on an input with
no microphone (a multi-channel interface defaults to channel 1) or the gain
on the interface is down. Pick the right device with `--mic` and the right
input with `--mic-channel`; both flags work for `uv run -m chatbot` too.

Quiet inputs are boosted automatically (up to 60x, see `audio/input.py`), so
a low gain setting is tolerated; a dead input is not. `--mic-gain 4` adds a
fixed boost in front of that if the automatic one is not enough, and
`--no-auto-gain` turns the automatic boost off.

### One audio stream for everything, in its own process

Everything that goes in or out of the sound card goes through one
full-duplex stream (`AudioIO` in `audio/input.py`): the microphone frames
for the wake word, the VAD and the recognisers, and the speech, cues, notes
and drums for the speakers. Opening a second stream on an interface while
another is running made CoreAudio throw "Audio Unit: cannot do in current
context" and the input silently stopped, so there is only ever one. Output
is a queue of chunks (speech, cues) plus *layers* mixed under it (a held
note, the drum loop), turned down by half while the queue plays so the
words stay clear.

The stream runs in a child process, `audio/server.py`, started by `AudioIO`
and talked to over pipes: commands in, and ten times a second a microphone
frame with the engine's state and levels out. The reason is the interpreter
lock. The stream's callback is Python, and in the chatbot process the TUI
redraw, the model's token loop and the TTS all compete with it for the
lock; measured with all three running, the callback was scheduled six times
in five seconds instead of fifty, and the drum loop stuttered. In its own
process nothing competes with it: the same test gives a steady ten frames a
second. If the chatbot ever falls behind reading frames, the engine drops
the oldest rather than wait, so the audio never pauses for it.

## A face for it: the TUI

```sh
uv run -m chatbot --tui                    # typed chat with the face beside the log
uv run -m chatbot --tui --wake-word        # always listening, no keyboard needed
uv run -m tui                              # demo: the face alone, no models loaded
```

`--tui` runs the same chat inside a full-screen terminal UI (see `tui/app.py`,
built on [Textual](https://textual.textualize.io/)). A round robot face with
big eyes shows what the bot is doing, there is a VU meter for the microphone
and the speakers (off unless `SHOW_VU_METER` in `tui/app.py` is set to True),
and everything the chatbot prints goes to a log pane. The
input box at the bottom replaces the `you>` prompt: Enter sends a line, an
empty line records from the microphone, and all the `/commands` work. In
`--wake-word` and `--hands-free` modes the input box is hidden because those
modes never read it.

| The bot is...                   | The face                                                            |
| ------------------------------- | ------------------------------------------------------------------- |
| asleep (waiting for the phrase) | eyes closed, "z z Z" drifting up, grey light                        |
| listening / recording           | eyes wide, pupils big, cyan light; red and blinking while recording |
| transcribing, thinking, calling a tool | eyes rolling round and round while the turnaround notes play; amber, violet and pink light |
| speaking                        | mouth opens and closes with the speaker output, green light         |
| idle                            | smile, blinks, glances around, blue light                           |

The VU meter shows dBFS of the last 100 ms: the microphone after the chatbot's
own gain (so it is what the recognisers hear) and the speaker output, with a
one second peak hold. Green is below -18 dB, amber to -6 dB, red above.

Keys: **Esc** stops the current reply or speech (what Ctrl-C does in the plain
terminal), **Ctrl-Q** quits.

### How the pieces talk: events/bus.py

The chatbot does not know about the face. The chat loop, the microphone code
and the speaker report what they are doing through a tiny event bus
(`events/bus.py`): `events.state("thinking")`, `events.emit("reply", text=...)`,
and the audio callback sends a `level` event ten times a second with the
microphone and speaker loudness. The TUI subscribes and reacts; nothing else
changes when it is not running. Any other display (a web page, LEDs on a
Raspberry Pi) can subscribe the same way. The events and their data are
listed at the top of `events/bus.py`.

## Long-term memory

The chatbot keeps two kinds of memory in one SQLite file, `memory.db`, so they
survive restarts. `memory/store.py` holds all of it. Text only: image turns are
stored by their text.

- **Facts.** Short notes like "The user's name is Adam and they have a dog
  named Biscuit." All of them go at the very top of the system prompt on
  every turn, worded as facts the model knows for certain, and the persona
  says it remembers what it is told. That ordering matters: with the facts
  after the tool list, the model would say "I don't have personal information
  about you" and then stay consistent with its own refusal. They are
  collected automatically: after each message that looks personal (first
  person, or the word "remember"), the model is asked in a separate quiet
  call whether the message states a lasting fact about the user, and if so
  the fact is stored and echoed as `(remembered: ...)`. That call runs while
  the reply is being read out, so it adds nothing to the wait. Questions are never
  checked, nor are requests ("tell me a poem"). The model occasionally keeps
  an opinion it should not; `/memory`
  shows the list and `/forget <text>` prunes it. `/remember <fact>` stores
  something directly.
- **Past conversations.** Every exchange (your message and the reply) is
  stored with an embedding from [model2vec](https://github.com/MinishLab/model2vec)
  (`minishlab/potion-base-8M`, a 30 MB static model that runs on numpy in
  microseconds). Before each turn your message is embedded and the closest
  exchanges from earlier sessions, above a similarity cutoff, are added to the
  prompt as "Earlier conversations that may be relevant". Search is a cosine
  similarity over a numpy array, which is instant at personal-history scale.
  Exchanges from the current session are skipped because they are already in
  the prompt; `/reset` starts a new session so what you just said becomes
  recallable history.

Why not a vector database or a memory framework: at this scale numpy plus
SQLite does the job with no extra services, and frameworks like mem0 or Letta
lean on a large hosted model for fact extraction, which a local 4B model does
not do well enough. If the history ever grows past tens of thousands of
exchanges, `sqlite-vec` or `faiss-cpu` drop in behind the same `Memory` class.

An earlier attempt gave the model a `remember` tool instead of the extraction
call. The 4B model would say "Sure, I'll remember that" or parrot the
instruction without ever producing the tool JSON, so the extraction question
replaced it.

## Talking to it: the microphone

Press Enter on an empty line and the chatbot records from the default input
device. It waits for you to start speaking, stops after about a second of
silence, transcribes the clip and sends it as your message, echoing it as
`you (mic)>`. With `--hands-free` it does this on its own after every spoken
reply, so you can hold a conversation without touching the keyboard; Ctrl-C
ends it. `audio/listen.py` holds all of it:

- Speech recognition is [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper)
  with `mlx-community/whisper-small.en-mlx` (English only, about 500 MB on
  first use, roughly a tenth of a second per utterance after that). Swap it
  with `--stt-model`, for example `mlx-community/whisper-base.en-mlx` to go
  smaller or `mlx-community/whisper-large-v3-turbo` for other languages (then
  also change `language` in `audio/listen.py`).
- `VoiceGate` is a small loudness-based voice-activity detector. The room's
  noise level is the quietest block heard so far (capped, so your own voice
  cannot pass for room noise if you start talking at once), and anything
  clearly louder is speech. If you are in a noisy room and it never stops
  recording, raise `boost` or `min_threshold`; if it misses quiet speech,
  lower them.
- Whisper likes to hear "Thank you." in silence, so clips without detected
  speech are never transcribed, and a few classic hallucinations are dropped.
- In hands-free mode the chatbot waits for its own voice to finish before
  listening again, so it does not transcribe itself.
- Recording uses channel 1 of the default input device. If your microphone is
  elsewhere (an audio interface with many inputs, a USB mic that is not the
  default), list devices with `uv run python -m sounddevice` and pass
  `--mic <name or index>` and, for an interface, `--mic-channel <n>`.
  `uv run -m audio.mic_check` shows live what the bot hears.

## Speech to text: four engines, chosen at start-up

Transcription sits behind one small interface (`audio/stt/base.py`): a
`Transcriber` turns 16 kHz audio into text, and one that can work while you
are still speaking also takes the utterance frame by frame. Each engine is
its own module in `audio/stt/`, with an `available()` that says whether it
can run on this machine; `--stt` picks one and `--list-stt` shows them:

| `--stt`   | what it is                                   | runs on                     | time to text after you stop |
|-----------|----------------------------------------------|-----------------------------|-----------------------------|
| `whisper` | Whisper small on MLX, the original           | Apple silicon               | about 0.25 s                |
| `apple`   | the macOS Speech framework (SpeechAnalyzer)  | macOS 26 or later           | about 0.15 s, no download   |
| `kyutai`  | Kyutai STT 1B on MLX, streaming              | Apple silicon, needs `uv`   | about 0.8 s, having listened all along (3 s if given the clip afterwards) |
| `sherpa`  | Moonshine tiny via sherpa-onnx               | any CPU, e.g. an Intel NUC  | about 0.04 s                |

`auto` (the default) takes `whisper` where it runs and `sherpa` elsewhere, so
the same code runs on an Intel NUC with nothing changed but the engine
underneath. `--stt-model` names a different model for the engine (a Hugging
Face repo for `whisper` or `kyutai`, a sherpa-onnx Moonshine release name for
`sherpa`). All four were word-perfect on the test clips.

Two engines run as helper processes, talked to over a pipe (`audio/stt/helper.py`):
the Speech framework is only reachable from Swift, so `apple_helper.swift` is
compiled with `swiftc` into `models/` on first use (needs the Xcode command
line tools); and Kyutai's `moshi-mlx` pins a huggingface-hub this project
cannot share, so `kyutai_helper.py` runs under `uv run --with moshi-mlx` in an
environment uv builds and caches, and the model (about 2.5 GB) is downloaded
once. Kyutai transcribes at only about twice real time, so it is only worth
it as a streaming engine: the recorders hand it each frame as they keep it,
and the text is ready under a second after you stop; its built-in end-of-turn
detection is not used yet.

Everything else in the bot is already portable: Pocket TTS runs on the CPU,
the wake word and VAD are sherpa-onnx, memory is numpy and SQLite. The one
piece that is not is the language model itself, which runs on MLX; an Intel
NUC would need a different model runner for that.

## Speech: reading replies aloud

Replies are spoken through the default output device using Kyutai's
[Pocket TTS](https://github.com/kyutai-labs/pocket-tts), a CPU text-to-speech
model, so it does not compete with the vision model for the GPU. `speech/speaker.py`
holds all of it:

- The default model is `english_18m_2026-10`, an 18M-parameter English model
  (about 100 MB download on first use, two seconds to load after that). It
  synthesises several times faster than real time; speech starts about a tenth
  of a second after a reply finishes. Smaller `english_7m_2026-10` and
  `english_2m_2026-10` models exist, and `english` is the full 100M model with
  the best quality. Pick one with `--tts-model`.
- The default voice is `alba`. Other built-in names are listed in the Pocket
  TTS README, or pass a path to a short clean `.wav` to clone a voice:
  `--voice ~/me.wav`.
- The reply is spoken sentence by sentence while it is still being generated
  (`SentenceStream` in `speech/stream.py` cuts the streamed words into
  sentences), so speech starts after the first sentence rather than after the
  whole reply. Audio is generated in chunks and written to a `sounddevice`
  output stream on a background thread, so you can type the next message
  while the previous answer is still being read. The first 200 ms of each
  sentence are collected before playback starts and the next sentence is
  synthesised while the current one plays, so the output queue never runs
  dry mid-speech: a dry queue is a hole of a few tens of milliseconds, which
  is heard as a pop. Asking a new question cuts off any speech still
  playing. Markdown and code are stripped before synthesis. Tool-call JSON is
  never spoken, only the final answer.
- Each chunk runs through the robot-voice effect in `audio/dsp.py` before playback:
  a delay-line pitch shifter raises the voice an octave, then a flanger (a
  slowly swept few-millisecond delay with feedback) adds the metallic sweep.
  Both are plain numpy and keep their state between chunks, so streaming
  playback stays gapless. `--no-robot` plays the untreated voice.

`/mute` toggles speech during a chat; `--no-speak` skips loading the speech
model entirely.

Note on dependencies: `phi-3-vision-mlx` pins numpy 1.26 while Pocket TTS
needs numpy 2. The vision code runs fine on numpy 2, so `pyproject.toml` lifts
the pin with a uv dependency override.

## Modes: which tools are on offer

The bot is in one mode at a time (`chatbot/modes.py`), and the mode decides
which tools it can reach:

| mode           | tools                                              | say                                   |
|----------------|----------------------------------------------------|---------------------------------------|
| `chat`         | none: just talk (the default)                      | "let's chat", "chat mode"             |
| `photographer` | the webcam                                         | "let's do photos", "camera mode"      |
| `music`        | drums, tempo, notes, stop                          | "let's do music", "let's do drums"    |
| `home`         | the example server's lights and clock              | "let's do home", "home mode"          |

"Exit" and "start over" work in every mode. Switching is recognised in the
chat loop without a model call, like "stop": the bot says the mode's
greeting ("Music mode. Drums, tempo and notes.") and the TUI's status line
shows the mode. `--mode` picks the starting one.

The routing question ("which tool, if any?") still sees every tool, by name
and first sentence with one example each, so it can recognise a request for
a tool the current mode does not open. When that happens nothing runs and
the model is not asked to word anything: the bot says "That needs music
mode. Shall I switch to it?", and a "yes" switches mode and carries out the
original request; a "no" stays put. (Before this, in chat mode the model
would promise drums in words with no drums.) Modes keep the small model
from reaching for tools nobody wanted, and keep the tool prompts short.
Adding a tool to a mode is one name in its list; the example in the tool's
registry entry is what the router learns from.

## Tools: letting the model call your API

The chatbot can call local HTTP endpoints to get things done ("turn the
kitchen lights on"). Two files are involved:

- `tools/server.py` is an example FastAPI server with `POST /lights`,
  `GET /lights`, `GET /time` and `POST /shutdown`. Replace its handlers with whatever really does
  the work (a smart-home bridge, a database, a shell script). Any server that
  takes JSON or query parameters and returns JSON will do.
- `tools/registry.py` is the registry. Each `Tool` entry gives the model a name, a
  description, the argument names it may send, and either the URL and HTTP
  method the chatbot should call or a Python function (`func`) to run in the
  chatbot itself. A tool may also supply its own `reply`, a function from
  the result to the sentence to say; the chatbot then uses that instead of
  asking the model to word one, which is exact and saves a model pass (the
  note tools do this: a small model asked to summarise "played E3 for 0.6 s"
  said "E4 for one second"). Adding a tool is one entry in the `TOOLS` list:

```python
Tool(
    name="set_thermostat",
    description="Set the target temperature in degrees Celsius.",
    url=f"{BASE_URL}/thermostat",
    method="POST",
    arguments={"celsius": "number between 15 and 25"},
),
```

A Python tool runs inside the chatbot, which suits things that need the
machine the chatbot is on. `take_picture` (in `images/webcam.py`) is one: "take a
picture" or "what can you see" opens the webcam with OpenCV, lets the
exposure settle, and saves a JPEG to `webcam-pictures/` named by date and time
(if OpenCV cannot open the camera and ffmpeg is installed, ffmpeg takes the
shot instead). macOS asks once to let the terminal use the camera. Take a test
shot with `uv run -m images.webcam`. As soon as the shot is in, the bot says "Picture
taken. Examining the picture now. This will take approximately 10 seconds to
complete." while the model looks at it, and the answer follows. The result
says which program took the shot (`via`) and carries a `warning` when the
picture is almost black, which is what a covered camera, a camera another app
is using, or a frame grabbed before the exposure settled all look like; the
model then says so instead of describing "a dark room".

Whenever a tool is called the bot first says what it is doing, "I will set
lights", "I will take picture", from the tool's name, and waits until that
has been heard before the tool runs, so a note or a held tone never starts
under the announcement.

`play_note` (in `audio/notes.py`) is another Python tool, registered only
when the speakers are open: "play a note", "play me a C sharp" or "play B
flat 3 for two seconds" plays that note through the speakers, named the
usual way (letter, sharp or flat, octave; `A4` is 440 Hz and the octave
defaults to 4). `keep_playing_note` holds a note instead ("keep playing a
G", "hold an A until I say stop"): it is mixed into the output inside the
audio callback, so it is continuous and the bot's speech plays over it. Say
"stop" on its own, typed or spoken, and it stops at once without a model
call; in always-on mode that comes before "stop" meaning go to sleep, and
while a note is held the bot keeps listening for it without the wake phrase
instead of going back to sleep after the usual follow-up window.
"Stop the note" or "that's enough of that" go through the `stop_sound` tool.

`play_drums` (in `audio/drums.py`) plays a drum loop the same way: "play some
drums", "give me a dance beat at 100", "drums with kit 3". The sounds come
from a General MIDI drum SoundFont, `assets/drums-soundfont.sf2` (not in git;
any GM percussion bank will do), rendered by TinySoundFont and loaded on
first use. The patterns live in `audio/patterns.py`, written as drum lanes
one character per step (`x` a stroke, `X` an accent, `g` a ghost note), so
adding one is a few lines: grooves in common styles (rock, pop, ballad,
dance, shuffle, jazz, funk, hip-hop, reggae, bossa nova, a waltz in three,
march, metal, punk, train), Latin and world grooves (salsa with its clave and
cascara, samba, afrobeat, the Afro-Cuban 6/8 bembe, cumbia, reggaeton, ska,
bhangra, tango) and the standard rudiments on the snare (single and double
stroke rolls, the paradiddles, five- and seven-stroke rolls, flams, flam
taps and accents, drags), with aliases for how people ask ("swing", "four
on the floor", "boom bap", "mambo", "dhol", "paradiddle"). A request that
names no style ("give me a beat") gets rock, and a misheard one is matched
to the nearest name by spelling ("scar drums" is ska, "sulsa" is salsa),
since that is what speech recognition makes of them. A style that has a
usual tempo (punk 180, metal 160, ballad 70) starts at it unless you ask
for another. All over the bank's twelve kits. The tool's description lists the names, so the model can offer them
when asked; it is built when the chatbot starts, so a running chatbot has
to be restarted to see new ones. The loop is a layer on the audio stream like the
held note, mixed into the output under speech and cues (turned down by half
while they play, so the words stay clear) inside the audio engine's own
process, so it keeps time sample-accurately whatever the chatbot is doing. `set_tempo` changes the speed while it plays, "faster",
"much slower" or "set the tempo to 100", and like a pattern or kit change it
takes effect at the start of the next bar, as a drum machine would. "Stop"
ends it; the bot keeps listening for that without the wake phrase, as for a
held note. (`tinysoundfont` declares `pyaudio`, which has no wheel for Apple
silicon; `pyproject.toml` overrides it away, since the drums are rendered to
buffers and played through the shared stream.)

The picture also goes to the model: a tool result may carry an `image` key
(a local path or URL), and the chatbot attaches that image to the turn that
feeds the result back, named by its number ("the new picture is image 3"), so
the vision model answers while looking at the right one. The picture stays in
the conversation like one added with `/image`, so follow-up questions about
it work, but only the newest webcam picture is kept: when the next one is
taken the previous one is removed from the history, because every picture in
the conversation is re-encoded on every turn (see `--image-crops` below).

```
you> what can you see?
phi> (calling take_picture {})
(result {'ok': True, 'image': 'webcam-pictures/2026-10-22_22-21-07.jpg', 'width': 1920, 'height': 1080})
(looking at webcam-pictures/2026-10-22_22-21-07.jpg)
I can see you sitting at a desk in a dimly lit room, with a bookshelf behind you.
```

The chatbot starts `tools/server.py` itself if nothing answers on its address,
and stops it on exit, so plain `uv run -m chatbot` is enough. Run the server
yourself in another terminal to watch its log, or pass `--no-tools-server` to
never auto-start it (for example when your real server lives elsewhere):

```sh
uv run -m tools.server
uv run -m chatbot
```

```
you> turn the kitchen lights on
phi> OK, turning the lights in the kitchen on.
(calling set_lights {'state': 'on', 'room': 'kitchen'})
I will set lights.
(result {'ok': True, 'rooms': ['kitchen'], 'state': 'on'})
The lights in the kitchen are now on.
```

There is also a built-in `exit` tool: say "exit", "quit", "shut down" or
"goodbye" and the bot says goodbye, asks the tools server to stop (through
`POST /shutdown`, which works whether the chatbot started the server or you
did) and closes. `/quit` still exits the chat by keyboard, but leaves a server
you started yourself running.

How it works under the hood: Phi-3.5-vision has no built-in function calling,
so it is done in two steps. First a short, separate "routing" call shows the
model the tool list and the message and asks which tool applies, answered with
just a name or NONE. The routing prompt spells out that NONE covers greetings,
chat, questions about the user and unclear speech, with a few examples; an
earlier, looser version sent "Where do I live?" to the time tool and a
misheard wake word to the lights (on a 14-message test set the loose prompt
got 12 right, the current one 14). If it names a tool that takes no
arguments (`get_lights`, `get_time`, `exit`) the chatbot calls it straight
away without another model turn. Otherwise the assistant's reply is pre-filled
up to the first argument's value, `{"tool": "set_lights", "arguments":
{"state": "`, so the model only has to finish the JSON. The exact prefill
matters: ending it with a trailing space (`"arguments": `) made the model skip
the opening brace and emit `"state": "on", "room": ...}}` a few turns into a
conversation, which failed to parse and leaked raw JSON into the reply. If the
JSON still fails to parse, the declared argument names are picked out of the
text with a regex, and if even that finds nothing the bot asks you to repeat
yourself rather than print the junk. The chatbot stops generating as soon as
the JSON object is complete, POSTs or GETs the arguments to the tool's URL,
appends the JSON response to the conversation as a user turn (Phi-3 has no
tool role), and generates again to get the plain-language answer. At most
three tool calls are made per message. Pass `--no-tools` to run without any of
this.

Why the routing step: the first version only asked for JSON in the system
prompt. That worked for the first message, but after a greeting the 4B model
had described itself as "just a computer program" and from then on answered
"I don't have access to the time" instead of calling the tool. A direct
pick-a-name question is answered reliably regardless of what came before, and
the pre-fill removes the chance of drifting out of JSON. The routing call adds
roughly half a second per message. The default persona (`DEFAULT_SYSTEM` in
`chatbot/cli.py`) also tells the model it is a home assistant with tools, which
helps; it is written to still chat, write poems and so on.

The model is small, so phrase tool descriptions plainly and keep the argument
list short. The routing question sees only each tool's name and first
sentence, because its prompt has to fit the router's 1024-token cache: when
the drum tool's full description (27 pattern names) was copied into it the
prompt reached 1089 tokens and every turn re-encoded it in full, 3 s a
turn. The bot now prints a warning at the first routing call if that prompt
outgrows the cache again. Put the detail the model needs for arguments
after the first sentence of a description; the first sentence is what the
router reads.

## How it works

The chatbot package is deliberately small:

- `load_model` in `chatbot/model.py` returns the library's `(model, processor)`
  pair, reused for every turn, with the image tile count applied.
- `Conversation` in `chatbot/conversation.py` stores the turns and renders them
  in Phi-3's chat format (`<|user|>`, `<|assistant|>`, `<|end|>`), numbering
  image placeholders `<|image_N|>` across the whole conversation.
- `Speaker` in `speech/speaker.py` queues each final reply and plays it as described above,
  passing each chunk through `RobotVoice` in `audio/dsp.py` unless started with `--no-robot`.
- `Listener` in `audio/listen.py` records one utterance from the microphone and hands it to the
  chosen speech-to-text engine (`audio/stt/`).
- `Memory` in `memory/store.py` stores facts and embedded exchanges and renders the
  memory section of the system prompt; `extract_fact` in `chatbot/asides.py` decides
  what to store.
- `audio/input.py` opens the microphone (device, channel, automatic gain) for
  both `audio/listen.py` and `audio/wake.py`; `audio/mic_check.py` is the live meter.
- `WakeListener` in `audio/wake.py` keeps the microphone open, spots the wake phrase
  and records the command that follows; `next_spoken_command` in `chatbot/voice.py`
  turns that into text.
- `chat_once` in `chatbot/model.py` runs the processor on that prompt plus the
  PIL images, then does greedy decoding with the model's KV cache, printing
  words as they are produced and stopping at `<|end|>`. It is a trimmed copy
  of the library's own `generate` loop, inlined so the end-of-turn token never
  reaches the screen.
- `answer` in `chatbot/turn.py` wraps that for one turn: it asks `route_tool`
  which tool the message needs, runs the tool, feeds the result (and any
  picture) back, and hands the finished exchange to memory. `run_repl` in
  `chatbot/repl.py` is the loop around it.

Every turn re-sends the whole conversation (text and images); the cache
below means only the new part is encoded, but the prompt still grows, and
a very long chat will not fit the cache. When that happens, `/reset`
or just saying "start over" (the `clear_context` tool) drops the conversation
and the next turn is quick again; long-term memory keeps what you told it.

### Not re-encoding what the model has already read

The model keeps its key/value cache between turns (`Session` in
`chatbot/model.py`). Each turn the new prompt is tokenised, the longest
prefix it shares with what the cache holds is kept, and only the rest is
fed to the model, typically the previous reply's end and your new message.
An interrupted reply or a dropped picture simply means encoding again from
the point of difference. The "which tool?" question and the fact extraction
have their own sessions, since each of those prompts is a constant prefix
plus the message. Measured on a seven-turn chat:

| turn                         | re-encoding everything | with the cache |
|------------------------------|-----------------------:|---------------:|
| "my name is Adam and I like tea" | 3.1 s              | 0.7 s          |
| "turn the kitchen lights on" (a tool call) | 5.2 s     | 1.3 s          |
| "what is my name?"           | 3.5 s                  | 0.8 s          |
| the warm "which tool?" call  | 1.5 s                  | 0.2 s          |

For this to work the start of the prompt has to be the same from turn to
turn. So the facts about the user and the recalled earlier conversations,
which change as the chat goes on, are not in the system prompt: they are
rendered as a user turn of their own just before your latest message, for
that call only. A new fact then costs its own few tokens instead of a full
re-encode (which grows with the conversation: 1 s at 400 tokens, 11 s at
3000). Placing that note *after* your message was tried first and the
model answered the note instead of the question; before it, the model
answers "Your name is Adam and you live in Bristol" as it should.

The cache is 1.6 GB at its full 4096 tokens; `--cache-tokens` makes it
smaller, and 0 turns it off. When a conversation would no longer fit, the
oldest exchanges are dropped in one go, down to about 60 % of the cache,
and the bot prints `(forgetting the oldest N exchange(s) ...)`: one
re-encode of the remaining turns every so often rather than a full one on
every turn once the cap is reached. Pictures go with the exchanges that
showed them, and everything dropped is still in long-term memory, so it
can come back as a recalled conversation. The cap is not more than 4096
because beyond the model's original context the library switches rotary
scaling, which would change every reply.

Decoding is greedy, so a cached turn should produce exactly the tokens an
uncached one would, and in testing it does, apart from the odd near-tie:
encoding a prompt in one pass and in pieces differs by floating-point
noise, which once in a while flips the choice between ending a reply and
adding one more sentence.

What a picture costs is set by `--image-crops`, not by its resolution: the
library resizes every image to a grid of up to that many 336-pixel tiles
whatever size it came in. At the model's full 16 tiles a picture is about
2350 prompt tokens; the default of 4 is about 750 and 1 is about 310. On an
M-series Mac a turn with two pictures in the history took 44 s at 16 tiles,
24 s at 4 and 11 s at 1. Use 16 when the model needs to read text in images,
1 when it only has to say what is in front of the webcam.

The library also ships its own Gradio web UI; after `uv sync` you can try it
with `uv run phi3v` (note that it downloads the text-only Phi-3.5-mini model
as well).


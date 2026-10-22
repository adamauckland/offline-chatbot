"""One shared, full-duplex audio stream for the whole chatbot, run in its own process.

Why one stream: opening a second stream on an interface while another is
running (the beep, or a spoken reply, while the microphone stays open) makes
CoreAudio throw "Audio Unit: cannot do in current context" and the input
silently stops delivering frames. So the microphone and the speakers share a
single ``sounddevice.Stream`` that is opened once at start-up and never
closed until exit.

Why its own process: the stream's callback is Python and needs the
interpreter lock, which the TUI redraw, the model's token loop and the TTS
all compete for; under all three the callback was starved and the output
stuttered. ``audio/server.py`` runs the stream and the mixer in a child
process, and ``AudioIO`` here is the proxy the rest of the code talks to.

The stream runs at 48 kHz, a rate every interface supports natively. Input is
decimated 3:1 to the 16 kHz the recognisers want; output written at 24 kHz
(Pocket TTS's rate) is upsampled 2:1. Both filters keep their state between
blocks, so block edges are seamless.

Quiet inputs are boosted: a fixed ``mic_gain`` first, then ``AutoGain``,
which tracks the recent peak and scales speech to a normal level.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

import numpy as np

import events

STREAM_RATE = 48000
MIC_RATE = 16000  # what Whisper, the VAD and the spotter expect
TTS_RATE = 24000  # what Pocket TTS produces
BLOCK = STREAM_RATE // 10  # 100 ms
HELD_NOTE_VOLUME = 0.2  # a note held with hold_note(); speech and cues play over it
DUCK = 0.5  # layers are turned down to this while speech or a cue plays, so the words stay clear


class Layer:
    """Something mixed into the output for as long as it is attached: a held note, a drum loop.

    ``render(frames)`` returns that many samples at ``STREAM_RATE``, either mono
    ``(frames,)`` or stereo ``(frames, 2)`` float32, and is called from the
    audio callback, so it must be quick and must not block. ``gain`` scales it.
    The stream ramps a layer in when it is added and out when it is removed,
    so neither clicks.
    """

    gain = 1.0

    def render(self, frames: int) -> np.ndarray:
        raise NotImplementedError


class HeldNote(Layer):
    """A sine at ``hz``, phase-continuous across blocks; the pitch can change while it plays."""

    gain = HELD_NOTE_VOLUME

    def __init__(self, hz: float) -> None:
        self.hz = hz
        self._phase = 0.0

    def render(self, frames: int) -> np.ndarray:
        step = 2 * np.pi * self.hz / STREAM_RATE
        wave = np.sin(self._phase + step * np.arange(frames, dtype=np.float64))
        self._phase = (self._phase + step * frames) % (2 * np.pi)
        return wave.astype(np.float32)


class AutoGain:
    """Bring quiet microphones up to a normal level without mangling speech.

    Gain is set from the loudest frame of the last second (instant attack, a
    one-second hold, then a 3 % per frame release), so it is steady across a
    phrase instead of pumping syllable by syllable, yet recovers within a few
    seconds of a loud sound. A fast 10 %-per-frame release was tried and broke
    wake-word detection: the spotter reads the loudness envelope, and a gain
    that climbs through the phrase distorts it.

    The gain is also capped so the room's noise floor is never lifted above
    ``noise_ceiling``; otherwise silence between words comes out at full scale
    and the spotter sees speech buried in amplified hiss.
    """

    def __init__(
        self,
        target: float = 0.6,
        max_gain: float = 60.0,
        release: float = 0.97,
        hold_frames: int = 10,
        noise_ceiling: float = 0.01,
    ) -> None:
        self.target = target
        self.max_gain = max_gain
        self.release = release
        self.hold_frames = hold_frames
        self.noise_ceiling = noise_ceiling
        self.peak = 1e-3
        self.hold = 0
        self.noise = None  # running estimate of the quietest recent frame RMS

    def process(self, frame: np.ndarray) -> np.ndarray:
        peak = float(np.abs(frame).max())
        rms = float(np.sqrt(np.mean(frame * frame)))
        self.noise = rms if self.noise is None else min(self.noise * 1.02, rms)  # tracks the floor, drifts up slowly
        if peak >= self.peak:
            self.peak, self.hold = peak, self.hold_frames
        elif self.hold > 0:
            self.hold -= 1
        else:
            self.peak = max(self.peak * self.release, 1e-4)
        gain = min(self.max_gain, self.target / self.peak, self.noise_ceiling / max(self.noise, 1e-5))
        return np.clip(frame * max(gain, 1.0), -1.0, 1.0).astype(np.float32)


class Resampler:
    """Streaming polyphase resampler by an integer ratio, exact across block boundaries.

    Zero-stuff by ``up``, low-pass with a FIR whose filter state carries over
    between calls, then keep every ``down``-th sample, tracking the phase. The
    FIR introduces a fixed delay of about ``taps / 2`` input samples (a few
    milliseconds), which is irrelevant here.
    """

    def __init__(self, up: int, down: int, taps: int | None = None) -> None:
        from scipy.signal import firwin

        self.up, self.down = up, down
        n = max(up, down)
        taps = taps or 20 * n + 1
        self.b = (firwin(taps, 1.0 / n) * up).astype(np.float32)
        self.zi = np.zeros(taps - 1, dtype=np.float32)
        self.phase = 0

    def process(self, block: np.ndarray) -> np.ndarray:
        from scipy.signal import lfilter

        x = np.asarray(block, dtype=np.float32).reshape(-1)
        if self.up > 1:
            stuffed = np.zeros(len(x) * self.up, dtype=np.float32)
            stuffed[:: self.up] = x
            x = stuffed
        y, self.zi = lfilter(self.b, 1.0, x, zi=self.zi)
        out = y[self.phase :: self.down]
        taken = len(out)
        self.phase = self.phase + taken * self.down - len(x)
        return out.astype(np.float32)


def resolve_device(name: str | None, kind: str = "input") -> int | None:
    """Device index from a name fragment or index string; None keeps the default."""
    import sounddevice as sd

    if name is None:
        return None
    if name.isdigit():
        return int(name)
    key = f"max_{kind}_channels"
    for i, d in enumerate(sd.query_devices()):
        if name.lower() in d["name"].lower() and d[key] > 0:
            return i
    raise ValueError(f"no {kind} device matching '{name}'; list them with: uv run python -m sounddevice")


class AudioIO:
    """The shared stream, served by ``audio/server.py`` in its own process.

    Subscribe callbacks for 16 kHz microphone frames; queue 24 kHz audio to
    play; hold a note or run the drums as layers under it. The engine runs in
    a child process so the chatbot's own threads (the TUI, the model, the TTS)
    can never starve the audio callback; this class is the proxy, and keeps
    a mirror of the engine's state from the status that arrives with every
    frame, ten times a second.
    """

    def __init__(
        self,
        mic: str | None = None,
        channel: int = 1,
        speaker: str | None = None,
        mic_gain: float = 1.0,
        auto_gain: bool = True,
    ) -> None:
        import pickle
        import subprocess
        import sys
        from pathlib import Path

        self._pickle = pickle
        root = Path(__file__).resolve().parent.parent
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "audio.server"], cwd=root, stdin=subprocess.PIPE, stdout=subprocess.PIPE
        )
        self._lock = threading.Lock()
        self._changed = threading.Condition(self._lock)
        self._listeners: list[Callable[[np.ndarray], None]] = []
        self._seq = 0  # last chunk sent
        self._played = 0  # last chunk the engine reports played
        self._queue_empty = True
        self._layers = 0
        self._holding: float | None = None
        self._drums: dict | None = None
        self._replies: dict[int, dict] = {}
        self._request = 0
        self._closed = False
        self._send({"mic": mic, "channel": channel, "speaker": speaker, "mic_gain": mic_gain, "auto_gain": auto_gain})
        first = pickle.load(self._proc.stdout)
        if first[0] != "ready":
            raise RuntimeError(first[1] if len(first) > 1 else "audio engine did not start")
        self.channel = channel
        print(f"(microphone: {first[1][0]}, channel {channel}; speaker: {first[1][1]})", flush=True)
        threading.Thread(target=self._reader, daemon=True, name="audio-reader").start()

    # -- the pipe ---------------------------------------------------------------------
    def _send(self, message) -> None:
        try:
            self._pickle.dump(message, self._proc.stdin, protocol=self._pickle.HIGHEST_PROTOCOL)
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            if not self._closed:
                raise RuntimeError(f"the audio engine is gone: {exc}") from exc

    def _reader(self) -> None:
        while True:
            try:
                message = self._pickle.load(self._proc.stdout)
            except (EOFError, OSError, ValueError):
                return
            kind = message[0]
            if kind == "frame":
                _, frame, info = message
                with self._changed:
                    self._played = info["played"]
                    self._queue_empty = info["queue_empty"]
                    self._layers = info["layers"]
                    self._holding = info["holding"]
                    self._drums = info["drums"]
                    listeners = list(self._listeners)
                    self._changed.notify_all()
                if frame is not None:
                    for cb in listeners:
                        try:
                            cb(frame)
                        except Exception:  # noqa: BLE001 - one listener must not stop the others
                            pass
                events.emit("level", mic=info["mic"], out=info["out"])
            elif kind == "reply":
                with self._changed:
                    self._replies[message[1]] = message[2]
                    self._changed.notify_all()

    def _call(self, kind: str, *args, timeout: float = 30.0) -> dict:
        """Send a command that gets a reply, and wait for it."""
        with self._changed:
            self._request += 1
            request_id = self._request
        self._send((kind, request_id, *args))
        with self._changed:
            if not self._changed.wait_for(lambda: request_id in self._replies, timeout=timeout):
                raise RuntimeError(f"the audio engine did not answer {kind}")
            return self._replies.pop(request_id)

    # -- microphone -----------------------------------------------------------
    def subscribe(self, callback: Callable[[np.ndarray], None]) -> None:
        """``callback(frame)`` gets each 100 ms frame (1600 float32 samples at 16 kHz)."""
        with self._lock:
            self._listeners.append(callback)

    def unsubscribe(self, callback: Callable[[np.ndarray], None]) -> None:
        with self._lock:
            if callback in self._listeners:
                self._listeners.remove(callback)

    # -- speaker ----------------------------------------------------------------
    def play(self, audio_24k: np.ndarray) -> None:
        """Queue mono 24 kHz audio; returns at once. Use ``wait`` to block until it has played."""
        with self._changed:
            self._seq += 1
            seq = self._seq
            self._queue_empty = False
        self._send(("play", seq, np.asarray(audio_24k, dtype=np.float32).reshape(-1)))

    def wait(self) -> None:
        """Block until everything queued so far has played."""
        with self._changed:
            self._changed.wait_for(lambda: self._closed or (self._played >= self._seq and self._queue_empty))

    def is_idle(self) -> bool:
        """True when nothing of ours is sounding: nothing queued, and no layer playing."""
        return self.playback_idle() and not self.has_layers

    def playback_idle(self) -> bool:
        """True when the queue of audio to play is empty (a held note does not count)."""
        with self._lock:
            return self._queue_empty and self._played >= self._seq

    def stop_playback(self) -> None:
        with self._changed:
            self._played = self._seq
            self._queue_empty = True
            self._changed.notify_all()
        self._send(("stop_playback",))

    # -- layers: a held note, the drums -----------------------------------------------
    def stop_layers(self) -> None:
        """Fade out everything that is playing on (a note, the drums)."""
        with self._lock:
            self._layers = 0
            self._holding = None
        self._send(("stop_layers",))

    @property
    def has_layers(self) -> bool:
        with self._lock:
            return self._layers > 0

    def hold_note(self, frequency: float | None) -> None:
        """Start holding a note at ``frequency`` Hz (changing it if one is held), or stop with None."""
        with self._lock:
            if frequency is None:
                self._layers = max(0, self._layers - (1 if self._holding is not None else 0))
            elif self._holding is None:
                self._layers += 1
            self._holding = frequency
        self._send(("hold_note", frequency))

    @property
    def holding(self) -> float | None:
        """The frequency being held, or None."""
        with self._lock:
            return self._holding

    def drums(self, action: str, **params) -> dict:
        """Start, change or stop the drum loop; returns the engine's state (``error`` if it refused)."""
        state = self._call("drums", action, params)
        with self._lock:
            self._drums = state
            if "error" not in state:
                self._layers = max(self._layers, 1) if state.get("playing") else self._layers
        return state

    @property
    def drum_state(self) -> dict | None:
        with self._lock:
            return self._drums

    def close(self) -> None:
        with self._changed:
            self._closed = True
            self._changed.notify_all()
        try:
            self._send(("close",))
            self._proc.stdin.close()
        except (RuntimeError, OSError):
            pass
        try:
            self._proc.wait(timeout=3)
        except Exception:  # noqa: BLE001
            self._proc.kill()


# Short sounds that mark the stages of a spoken turn, as (frequency Hz, seconds)
# notes played back to back; 0 Hz is a rest. Each has a different shape so you
# can tell them apart without looking at the screen.
CUES: dict[str, list[tuple[float, float]]] = {
    "wake": [(660.0, 0.08), (990.0, 0.14)],  # rising pair: the wake word was heard
    "parse": [(1320.0, 0.05), (0.0, 0.04), (1320.0, 0.05)],  # quick double tick: transcribing
    "think": [(990.0, 0.07), (770.0, 0.07), (550.0, 0.14)],  # falling trio: handed to the model
}


def tone(frequency: float, seconds: float, volume: float = 0.2) -> np.ndarray:
    """A sine note at 24 kHz with a Hann envelope so it does not click."""
    t = np.arange(int(TTS_RATE * seconds)) / TTS_RATE
    if frequency <= 0:
        return np.zeros(len(t), dtype=np.float32)
    return (volume * np.sin(2 * np.pi * frequency * t) * np.hanning(len(t))).astype(np.float32)


def cue(audio: AudioIO, name: str, wait: bool = True) -> None:
    """Play the named sound from ``CUES``; with ``wait`` block until it has finished."""
    audio.play(np.concatenate([tone(f, sec) for f, sec in CUES[name]]))
    if wait:
        audio.wait()


def beep(audio: AudioIO, frequency: float = 880.0, seconds: float = 0.12) -> None:
    audio.play(tone(frequency, seconds))
    audio.wait()

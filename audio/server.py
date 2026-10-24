"""The audio engine, in a process of its own: the stream, the mixer, the held note and the drums.

Why a process: the audio callback is Python, so it needs the interpreter lock
every 100 ms. In the chatbot process the TUI redraw, the model's token loop
and the TTS all compete for that lock, and under all three together the
callback was scheduled six times in five seconds. The drum loop stuttered and
speech could too. Here nothing competes with it.

``audio.input.AudioIO`` starts this with ``python -m audio.server`` and talks
to it over its pipes with pickled tuples: commands in on stdin, and on stdout
one ``("frame", mic_frame, status)`` per block plus replies to commands.
The engine never blocks on the pipe: frames go out through a bounded queue
and are dropped if the chatbot is slow to read them, which only loses
microphone audio the chatbot could not have kept up with anyway.
"""

from __future__ import annotations

import collections
import pickle
import queue
import sys
import threading
from pathlib import Path

import numpy as np

from audio.input import (
    BLOCK,
    DUCK,
    STREAM_RATE,
    AutoGain,
    HeldNote,
    Layer,
    Resampler,
    resolve_device,
)


class AudioEngine:
    """Owns the full-duplex stream; mixes queued audio and layers; reports frames and levels."""

    def __init__(self, config: dict, send) -> None:
        import sounddevice as sd

        self.send = send
        in_dev = resolve_device(config.get("mic"), "input")
        out_dev = resolve_device(config.get("speaker"), "output")
        in_info = sd.query_devices(in_dev if in_dev is not None else sd.default.device[0])
        out_info = sd.query_devices(out_dev if out_dev is not None else sd.default.device[1])
        self.channel = max(1, min(int(config.get("channel", 1)), in_info["max_input_channels"]))
        out_channels = min(2, out_info["max_output_channels"])
        self.mic_gain = float(config.get("mic_gain", 1.0))
        self.auto = AutoGain() if config.get("auto_gain", True) else None
        self._down = Resampler(1, 3)
        self._up = Resampler(2, 1)
        self._lock = threading.Lock()
        self._out: collections.deque[tuple[int, np.ndarray]] = collections.deque()  # (seq, chunk)
        self._out_pos = 0
        self._played = 0  # seq of the last chunk played to the end
        self._layers: list[list] = []  # [layer, level, target]
        self._duck = 1.0
        self._held: Layer | None = None  # the held note: a SynthNote, or a HeldNote sine as fallback
        self._band = None
        self.devices = (in_info["name"], out_info["name"])
        self.stream = sd.Stream(
            samplerate=STREAM_RATE,
            blocksize=BLOCK,
            dtype="float32",
            device=(in_dev, out_dev),
            channels=(self.channel, out_channels),
            callback=self._callback,
        )
        self.stream.start()

    # -- commands from the chatbot ---------------------------------------------------
    def play(self, seq: int, audio_24k: np.ndarray) -> None:
        chunk = self._up.process(np.asarray(audio_24k, dtype=np.float32).reshape(-1))
        with self._lock:
            self._out.append((seq, chunk))

    def stop_playback(self) -> None:
        with self._lock:
            if self._out:
                self._played = max(self._played, self._out[-1][0])
            self._out.clear()
            self._out_pos = 0

    def add_layer(self, layer: Layer) -> None:
        with self._lock:
            for entry in self._layers:
                if entry[0] is layer:
                    entry[2] = 1.0
                    return
            self._layers.append([layer, 0.0, 1.0])

    def remove_layer(self, layer: Layer) -> None:
        with self._lock:
            for entry in self._layers:
                if entry[0] is layer:
                    entry[2] = 0.0

    def stop_layers(self) -> None:
        with self._lock:
            for entry in self._layers:
                entry[2] = 0.0
        self._held = None
        if self._band is not None:
            self._band.silence_all()  # so a later start begins from silence, on a fresh bar

    def playing_layers(self) -> list[Layer]:
        with self._lock:
            return [layer for layer, _, target in self._layers if target > 0]

    def hold_note(self, hz: float | None) -> None:
        """The sine fallback for a held note (used when the GM SoundFont is not available)."""
        if hz is None:
            if self._held is not None:
                self.remove_layer(self._held)
                self._held = None
        elif self._held is None:
            self._held = HeldNote(hz)
            self.add_layer(self._held)
        else:
            self._held.hz = hz

    def note(self, action: str, params: dict) -> dict:
        """A piano note from the GM SoundFont: ``play`` one, ``hold`` one (re-struck until stopped)."""
        from audio.synth import SynthNote

        try:
            if action == "play":
                self.add_layer(SynthNote(params["midi"], seconds=float(params.get("seconds", 0.6))))
            elif action == "hold":
                held = self._held
                if isinstance(held, SynthNote) and held in self.playing_layers():
                    held.retune(params["midi"])
                else:
                    self.hold_note(None)
                    self._held = SynthNote(params["midi"], repeat=float(params.get("repeat", 1.6)))
                    self.add_layer(self._held)
            else:
                return {"error": f"unknown note action {action!r}"}
        except Exception as exc:  # noqa: BLE001 - the tool falls back to a plain tone
            return {"error": str(exc)}
        return {"ok": True, "holding": self._held is not None and self._held in self.playing_layers()}

    def band(self, action: str, params: dict) -> dict:
        """Drums, chords and bass on one clock (see audio/band.py); attached while any track plays."""
        from audio.band import Band
        from audio.drums import DrumMachine

        if self._band is None:
            self._band = Band(Path(params.get("soundfont") or DrumMachine.soundfont_default()))
        band = self._band
        try:
            state = band.request(action, params)
        except Exception as exc:  # noqa: BLE001 - reported to the tool, which tells the model
            return {"error": str(exc), **band.state()}
        if band.active and band not in self.playing_layers():
            self.add_layer(band)
        elif not band.active and band in self.playing_layers():
            self.remove_layer(band)
        return state

    def drums(self, action: str, params: dict) -> dict:
        """The drum tool's requests, routed to the band's drum track."""
        return self.band({"start": "drums_start", "set": "drums_set", "stop": "drums_stop"}[action], params)

    def status(self) -> dict:
        with self._lock:
            queue_empty = not self._out
            played = self._played
        return {
            "played": played,
            "queue_empty": queue_empty,
            "layers": len(self.playing_layers()),
            "holding": getattr(self._held, "hz", getattr(self._held, "midi", None)) if self._held is not None else None,
            "band": self._band.state() if self._band is not None else None,
        }

    def close(self) -> None:
        self.stream.stop()
        self.stream.close()

    # -- the PortAudio callback ---------------------------------------------------
    def _mix_layers(self, outdata: np.ndarray, playing_queue: bool) -> None:
        with self._lock:
            entries = list(self._layers)
        if not entries:
            return
        n, channels = outdata.shape
        duck_to = DUCK if playing_queue else 1.0
        duck = np.linspace(self._duck, duck_to, n, endpoint=False, dtype=np.float32)
        self._duck = duck_to
        mix = np.zeros((n, channels), dtype=np.float32)
        for entry in entries:
            layer, level, target = entry
            try:
                samples = np.asarray(layer.render(n), dtype=np.float32)
                if getattr(layer, "finished", False):
                    entry[2] = 0.0  # a one-shot note has died away: let it go
            except Exception:  # noqa: BLE001 - a failing layer is dropped, never the stream
                samples, entry[2] = np.zeros(n, dtype=np.float32), 0.0
            if samples.ndim == 1:
                samples = samples[:, None]
            elif samples.shape[1] != channels:
                samples = samples.mean(axis=1, keepdims=True)
            envelope = np.linspace(level, target, n, endpoint=False, dtype=np.float32) * duck * layer.gain
            mix += samples * envelope[:, None]
            entry[1] = target
        with self._lock:
            self._layers = [e for e in self._layers if not (e[1] == 0.0 and e[2] == 0.0)]
        outdata[:, :] = np.clip(outdata + mix, -1.0, 1.0)

    def _callback(self, indata, outdata, frames, time_info, status) -> None:  # noqa: ARG002
        frame = None
        mic_level = 0.0
        try:
            frame = self._down.process(indata[:, self.channel - 1] * self.mic_gain)
            if self.auto is not None:
                frame = self.auto.process(frame)
            mic_level = float(np.sqrt(np.mean(frame * frame)))
        except Exception:  # noqa: BLE001 - a bad frame must not stop the stream
            pass
        outdata.fill(0)
        filled = 0
        with self._lock:
            while filled < frames and self._out:
                seq, head = self._out[0]
                take = min(frames - filled, len(head) - self._out_pos)
                outdata[filled : filled + take, :] = head[self._out_pos : self._out_pos + take, None]
                filled += take
                self._out_pos += take
                if self._out_pos >= len(head):
                    self._out.popleft()
                    self._out_pos = 0
                    self._played = seq
        self._mix_layers(outdata, playing_queue=filled > 0)
        out_level = float(np.sqrt(np.mean(outdata[:, 0] ** 2))) if filled or self._layers else 0.0
        info = self.status()
        info.update(mic=mic_level, out=out_level)
        self.send(("frame", frame, info))


def main() -> None:
    """Serve the chatbot over stdin/stdout until it closes the pipe."""
    out = sys.stdout.buffer
    sys.stdout = sys.stderr  # anything printed must not land in the data stream
    pending: queue.Queue = queue.Queue(maxsize=50)

    def send(message) -> None:
        try:
            pending.put_nowait(message)
        except queue.Full:  # the chatbot is not keeping up: drop the oldest frame, never block audio
            try:
                pending.get_nowait()
            except queue.Empty:
                pass
            pending.put_nowait(message)

    def sender() -> None:
        while True:
            message = pending.get()
            if message is None:
                return
            try:
                pickle.dump(message, out, protocol=pickle.HIGHEST_PROTOCOL)
                out.flush()
            except (BrokenPipeError, OSError):
                return

    threading.Thread(target=sender, daemon=True).start()
    config = pickle.load(sys.stdin.buffer)
    try:
        engine = AudioEngine(config, send)
    except Exception as exc:  # noqa: BLE001
        send(("error", f"could not open the audio stream: {exc}"))
        pending.put(None)
        return
    send(("ready", engine.devices))
    try:
        while True:
            try:
                message = pickle.load(sys.stdin.buffer)
            except (EOFError, OSError):
                break  # the chatbot has gone
            kind, *rest = message
            if kind == "play":
                engine.play(rest[0], rest[1])
            elif kind == "stop_playback":
                engine.stop_playback()
            elif kind == "hold_note":
                engine.hold_note(rest[0])
            elif kind == "stop_layers":
                engine.stop_layers()
            elif kind == "drums":
                request_id, action, params = rest
                send(("reply", request_id, engine.drums(action, params)))
            elif kind == "band":
                request_id, action, params = rest
                send(("reply", request_id, engine.band(action, params)))
            elif kind == "note":
                request_id, action, params = rest
                send(("reply", request_id, engine.note(action, params)))
            elif kind == "close":
                break
    finally:
        engine.close()
        pending.put(None)


if __name__ == "__main__":
    main()

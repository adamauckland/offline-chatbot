from __future__ import annotations

import collections
import threading
import queue
import sys
import tarfile
import time
import urllib.request
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np

import events
from audio.input import AudioIO, cue

SAMPLE_RATE = 16000
FRAME_SECONDS = 0.1
FRAME = int(SAMPLE_RATE * FRAME_SECONDS)
MODELS_DIR = Path("models")
KWS_NAME = "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01"
KWS_URL = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/{KWS_NAME}.tar.bz2"
KWS_CHECKPOINT = "epoch-12-avg-2-chunk-16-left-64.int8.onnx"
VAD_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"
DEFAULT_WAKE_WORD = "hey computer"
FRAMES_KEPT_AFTER_PHRASE = 5
PRE_ROLL_FRAMES = 5
FLUSH_FRAMES = 5


def download(url: str, dest: Path) -> None:
    print(f"Downloading {url.rsplit('/', 1)[-1]}...", file=sys.stderr)
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)


def download_and_unpack(url: str, archive: Path, into: Path) -> None:
    download(url, archive)

    with tarfile.open(archive) as tar:
        tar.extractall(into, filter="data")

    archive.unlink()


def ensure_models(models_dir: Path = MODELS_DIR) -> tuple[Path, Path]:
    kws_dir = models_dir / KWS_NAME
    vad = models_dir / "silero_vad.onnx"

    if not kws_dir.is_dir():
        download_and_unpack(KWS_URL, models_dir / f"{KWS_NAME}.tar.bz2", models_dir)

    if not vad.is_file():
        download(VAD_URL, vad)

    return kws_dir, vad


def phrase_as_bpe_tokens(kws_dir: Path, phrase: str) -> list[str]:
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=str(kws_dir / "bpe.model"))

    return sp.encode(phrase.upper(), out_type=str)


def write_keywords_file(kws_dir: Path, models_dir: Path, phrase: str, boost: float, threshold: float) -> Path:
    tokens = phrase_as_bpe_tokens(kws_dir, phrase)
    display = phrase.replace(" ", "_")
    keywords_file = models_dir / f"keywords-{display}.txt"
    keywords_file.write_text(f"{' '.join(tokens)} :{boost} #{threshold} @{display}\n")

    return keywords_file


class WakeWordDetector:
    def __init__(
        self,
        phrase: str = DEFAULT_WAKE_WORD,
        models_dir: Path = MODELS_DIR,
        threshold: float = 0.2,
        boost: float = 1.5,
        max_active_paths: int = 16,
    ) -> None:
        import sherpa_onnx as so

        kws_dir, _ = ensure_models(models_dir)
        self.phrase = phrase.strip().lower()
        keywords_file = write_keywords_file(kws_dir, models_dir, self.phrase, boost, threshold)
        self.kws = so.KeywordSpotter(
            tokens=str(kws_dir / "tokens.txt"),
            encoder=str(kws_dir / f"encoder-{KWS_CHECKPOINT}"),
            decoder=str(kws_dir / f"decoder-{KWS_CHECKPOINT}"),
            joiner=str(kws_dir / f"joiner-{KWS_CHECKPOINT}"),
            keywords_file=str(keywords_file),
            num_threads=1,
            keywords_threshold=threshold,
            max_active_paths=max_active_paths,
        )
        self.stream = self._stream_without_left_context()

    def feed(self, samples: np.ndarray) -> bool:
        self.stream.accept_waveform(SAMPLE_RATE, samples)
        heard = False

        while self.kws.is_ready(self.stream):
            self.kws.decode_stream(self.stream)

            if self.kws.get_result(self.stream):
                heard = True
                self.kws.reset_stream(self.stream)

        return heard

    def reset(self) -> None:
        self.stream = self._stream_without_left_context()

    def _stream_without_left_context(self):
        return self.kws.create_stream()


class SpeechDetector:
    def __init__(self, models_dir: Path = MODELS_DIR, threshold: float = 0.5) -> None:
        import sherpa_onnx as so

        _, vad_path = ensure_models(models_dir)
        cfg = so.VadModelConfig()
        cfg.silero_vad.model = str(vad_path)
        cfg.silero_vad.threshold = threshold
        cfg.silero_vad.min_silence_duration = 0.3
        cfg.silero_vad.min_speech_duration = 0.1
        cfg.sample_rate = SAMPLE_RATE
        self.vad = so.VoiceActivityDetector(cfg, buffer_size_in_seconds=10)

    def feed(self, samples: np.ndarray) -> bool:
        self.vad.accept_waveform(samples)
        self._discard_finished_segments()

        return self.vad.is_speech_detected()

    def _discard_finished_segments(self) -> None:
        while not self.vad.empty():
            self.vad.pop()

    def reset(self) -> None:
        self.vad.reset()


def detect_in_clip(detector: WakeWordDetector, audio: np.ndarray) -> float | None:
    detector.reset()

    for start in range(0, len(audio), FRAME):
        if detector.feed(audio[start : start + FRAME]):
            return (start + FRAME) / SAMPLE_RATE

    if detector.feed(np.zeros(FRAME * FLUSH_FRAMES, dtype=np.float32)):
        return len(audio) / SAMPLE_RATE

    return None


class Utterance:
    def __init__(self, on_frame: Callable[[np.ndarray], None] | None) -> None:
        self.on_frame = on_frame
        self.frames: list[np.ndarray] = []

    def add(self, frames: Iterable[np.ndarray]) -> None:
        for frame in frames:
            self.frames.append(frame)

            if self.on_frame is not None:
                self.on_frame(frame)

    @property
    def seconds(self) -> float:
        return len(self.frames) * FRAME_SECONDS

    def audio(self) -> np.ndarray:
        return np.concatenate(self.frames)


class WakeListener:
    def __init__(
        self, audio: AudioIO, phrase: str = DEFAULT_WAKE_WORD, models_dir: Path = MODELS_DIR, threshold: float = 0.2
    ) -> None:
        self.audio = audio
        self.phrase = phrase
        self.wake = WakeWordDetector(phrase, models_dir, threshold=threshold)
        self.vad = SpeechDetector(models_dir)
        self._frames: queue.Queue[np.ndarray] = queue.Queue()
        self.recent_frames: collections.deque[np.ndarray] = collections.deque(maxlen=FRAMES_KEPT_AFTER_PHRASE)

        audio.subscribe(self._frames.put)

    def _drain(self) -> None:
        while True:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                return

    def _next_frame(self, timeout: float = 0.5) -> np.ndarray | None:
        try:
            return self._frames.get(timeout=timeout)
        except queue.Empty:
            return None

    def wait_for_wake(self, timeout: float | None = None, abort: threading.Event | None = None) -> bool:
        """Block until the phrase is heard (True), ``timeout`` passes, or ``abort`` is set (False)."""
        self._drain()
        self.wake.reset()
        self.vad.reset()
        self.recent_frames.clear()
        deadline = None if timeout is None else time.monotonic() + timeout

        while deadline is None or time.monotonic() < deadline:
            if abort is not None and abort.is_set():
                return False
            frame = self._next_frame()

            if frame is None:
                continue

            self.recent_frames.append(frame)
            self.vad.feed(frame)

            if self.wake.feed(frame):
                return True

        return False

    def record_utterance(
        self,
        wait_seconds: float = 6.0,
        silence_seconds: float = 0.8,
        max_seconds: float = 30.0,
        carry: list[np.ndarray] | None = None,
        on_frame: Callable[[np.ndarray], None] | None = None,
        abort: threading.Event | None = None,
    ) -> np.ndarray | None:
        """Record until silence; None if nobody spoke in time, or ``abort`` was set (a typed line came in)."""
        utterance = Utterance(on_frame)
        utterance.add(carry or [])
        started = bool(utterance.frames)
        pre_roll: collections.deque[np.ndarray] = collections.deque(maxlen=PRE_ROLL_FRAMES)
        quiet = 0.0
        waited = 0.0

        if started:
            self._announce_recording()
        else:
            events.state("listening")

        while True:
            if abort is not None and abort.is_set():
                return None
            frame = self._next_frame()
            if frame is None:
                continue
            speech = self.vad.feed(frame)

            if not started and not speech:
                pre_roll.append(frame)
                waited += FRAME_SECONDS

                if waited >= wait_seconds:
                    return None

                continue

            if not started:
                started = True
                utterance.add(pre_roll)
                self._announce_recording()

            utterance.add([frame])
            quiet = 0.0 if speech else quiet + FRAME_SECONDS

            if quiet >= silence_seconds or utterance.seconds >= max_seconds:
                return utterance.audio()

    def _announce_recording(self) -> None:
        print("(recording)", flush=True)
        events.state("recording")

    def listen(
        self,
        awake: bool = False,
        wait_seconds: float = 6.0,
        on_frame: Callable[[np.ndarray], None] | None = None,
        abort: threading.Event | None = None,
    ) -> np.ndarray | None:
        """Wait for the phrase (unless ``awake``), then record; None when nothing came, or on ``abort``."""
        if awake:
            self._drain()
            self.vad.reset()
            carry = None
        else:
            carry = self._sleep_until_woken(abort)
            if carry is None:
                return None

        return self.record_utterance(wait_seconds=wait_seconds, carry=carry, on_frame=on_frame, abort=abort)

    def _sleep_until_woken(self, abort: threading.Event | None = None) -> list[np.ndarray] | None:
        print(f'(asleep; say "{self.phrase}")', flush=True)
        events.state("asleep", phrase=self.phrase)
        if not self.wait_for_wake(abort=abort):
            return None  # aborted: a typed line is waiting
        cue(self.audio, "wake")
        print("(awake)", flush=True)

        return list(self.recent_frames)

    def close(self) -> None:
        self.audio.unsubscribe(self._frames.put)

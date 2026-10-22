"""Always-on listening with a wake word, on sherpa-onnx.

Two small CPU models run on every 100 ms frame from an input stream that is
never closed:

* **Silero VAD** (speech / not speech). In silence nothing else runs.
* **Keyword spotter**: sherpa-onnx's open-vocabulary zipformer (3.3M params,
  English, GigaSpeech). The wake phrase is plain text, turned into BPE tokens
  with ``text2token``; no training, any phrase. It only sees frames the VAD
  flagged as speech.

When the phrase is heard the listener plays the "wake" cue (see ``CUES`` in
audio/input.py), keeps recording until the VAD says you stopped talking, and
hands the clip (minus the wake phrase) to whoever transcribes it. Both models are downloaded once into ``models/``.
"""

from __future__ import annotations

import collections
from collections.abc import Callable
import queue
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np

import events
from audio.input import AudioIO, cue

SAMPLE_RATE = 16000
FRAME = 1600  # 100 ms at 16 kHz
MODELS_DIR = Path("models")
KWS_NAME = "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01"
KWS_URL = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/{KWS_NAME}.tar.bz2"
VAD_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"
DEFAULT_WAKE_WORD = "hey computer"


def _download(url: str, dest: Path) -> None:
    print(f"Downloading {url.rsplit('/', 1)[-1]}...", file=sys.stderr)
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)


def ensure_models(models_dir: Path = MODELS_DIR) -> tuple[Path, Path]:
    kws_dir = models_dir / KWS_NAME
    if not kws_dir.is_dir():
        archive = models_dir / f"{KWS_NAME}.tar.bz2"
        _download(KWS_URL, archive)
        with tarfile.open(archive) as tar:
            tar.extractall(models_dir, filter="data")
        archive.unlink()
    vad = models_dir / "silero_vad.onnx"
    if not vad.is_file():
        _download(VAD_URL, vad)
    return kws_dir, vad


class WakeWordDetector:
    """Feed 16 kHz float32 audio; ``feed`` returns True on the frame the phrase completes."""

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
        tokens = kws_dir / "tokens.txt"
        # The model's BPE vocabulary is upper case. (sherpa's own text2token helper
        # would do this, but it imports a Chinese pinyin package we do not need.)
        import sentencepiece as spm

        sp = spm.SentencePieceProcessor(model_file=str(kws_dir / "bpe.model"))
        encoded = sp.encode(self.phrase.upper(), out_type=str)
        keywords_file = models_dir / f"keywords-{self.phrase.replace(' ', '_')}.txt"
        # Format: tokens, then optional :boost #threshold, then @display-text (no spaces)
        keywords_file.write_text(f"{' '.join(encoded)} :{boost} #{threshold} @{self.phrase.replace(' ', '_')}\n")
        self.kws = so.KeywordSpotter(
            tokens=str(tokens),
            encoder=str(kws_dir / "encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx"),
            decoder=str(kws_dir / "decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx"),
            joiner=str(kws_dir / "joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx"),
            keywords_file=str(keywords_file),
            num_threads=1,
            keywords_threshold=threshold,
            # A wider beam matters for made-up words that become many BPE tokens
            # ("scrudley" is six): 4 paths caught 6 of 9 test clips, 16 caught all.
            max_active_paths=max_active_paths,
        )
        self.stream = self.kws.create_stream()

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
        # A fresh stream, not reset_stream(): the streaming encoder keeps left
        # context, and resetting alone left detection depending on what the
        # stream had heard before (the same clip fired or not run to run).
        self.stream = self.kws.create_stream()


class SpeechDetector:
    """Silero VAD wrapper: ``feed`` returns True while the current frame is speech."""

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
        while not self.vad.empty():  # discard finished segments; we only use the live flag
            self.vad.pop()
        return self.vad.is_speech_detected()

    def reset(self) -> None:
        self.vad.reset()


def detect_in_clip(detector: WakeWordDetector, audio: np.ndarray) -> float | None:
    """Offline helper for tests: seconds into ``audio`` at which the phrase fired, or None."""
    detector.reset()
    for start in range(0, len(audio), FRAME):
        if detector.feed(audio[start : start + FRAME]):
            return (start + FRAME) / SAMPLE_RATE
    # Flush with a little silence so a phrase at the very end can complete.
    if detector.feed(np.zeros(FRAME * 5, dtype=np.float32)):
        return len(audio) / SAMPLE_RATE
    return None


class WakeListener:
    """Keep the microphone open; return one utterance each time the wake phrase is heard."""

    def __init__(
        self, audio: AudioIO, phrase: str = DEFAULT_WAKE_WORD, models_dir: Path = MODELS_DIR, threshold: float = 0.2
    ) -> None:
        self.audio = audio
        self.phrase = phrase
        self.wake = WakeWordDetector(phrase, models_dir, threshold=threshold)
        self.vad = SpeechDetector(models_dir)
        self._frames: queue.Queue[np.ndarray] = queue.Queue()
        # The spotter fires a few frames after the phrase ends; whatever arrived
        # meanwhile is the start of the command, so keep it to hand to the recorder.
        # Half a second is the sweet spot: measured on synthetic clips, 0.8 s
        # brought the end of the phrase along ("scridly turn the lights off"),
        # 0.3 s lost the first word; 0.5 s kept every command intact.
        self.recent: collections.deque[np.ndarray] = collections.deque(maxlen=5)
        audio.subscribe(self._frames.put)

    def _drain(self) -> None:
        while True:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                return

    def wait_for_wake(self, timeout: float | None = None) -> bool:
        """Block until the phrase is heard (True) or ``timeout`` seconds pass (False)."""
        self._drain()  # ignore whatever happened while we were busy (e.g. our own voice)
        self.wake.reset()
        self.vad.reset()
        deadline = None if timeout is None else time.monotonic() + timeout
        # Every frame goes to the spotter. An earlier version only fed it frames the
        # VAD flagged as speech (with a short pre-roll) to save CPU, and that made it
        # miss: in an acoustic test the gated spotter missed the phrase even at high
        # volume while the ungated one fired. The spotter costs ~1.5 % of a core, so
        # gating buys nothing worth that. The VAD still tracks state for the recorder.
        self.recent.clear()
        while deadline is None or time.monotonic() < deadline:
            try:
                frame = self._frames.get(timeout=0.5)
            except queue.Empty:
                continue
            self.recent.append(frame)
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
    ) -> np.ndarray | None:
        """Record until the VAD hears ``silence_seconds`` of quiet; None if nobody spoke.

        ``on_frame`` gets every frame that is kept, as it is kept, so a
        streaming recogniser can work on the utterance while it is spoken.

        ``carry`` is audio that already belongs to the utterance (the frames the
        wake detector consumed after the phrase ended). The VAD is not reset:
        straight after the wake phrase the user is mid-sentence and it must keep
        tracking. The last few frames before speech is confirmed are kept as
        pre-roll so soft starts and the VAD's ~0.2 s reaction do not clip words.
        """
        kept: list[np.ndarray] = list(carry or [])
        if on_frame is not None:
            for frame in kept:
                on_frame(frame)
        pre_roll: collections.deque[np.ndarray] = collections.deque(maxlen=5)
        quiet = 0.0
        waited = 0.0
        started = bool(kept)
        if started:
            print("(recording)", flush=True)
            events.state("recording")
        else:
            events.state("listening")
        while True:
            frame = self._frames.get()
            speech = self.vad.feed(frame)
            if not started:
                if not speech:
                    pre_roll.append(frame)
                    waited += FRAME / SAMPLE_RATE
                    if waited >= wait_seconds:
                        return None
                    continue
                started = True
                kept.extend(pre_roll)
                if on_frame is not None:
                    for early in pre_roll:
                        on_frame(early)
                print("(recording)", flush=True)
                events.state("recording")
            kept.append(frame)
            if on_frame is not None:
                on_frame(frame)
            quiet = 0.0 if speech else quiet + FRAME / SAMPLE_RATE
            if quiet >= silence_seconds or len(kept) * FRAME / SAMPLE_RATE >= max_seconds:
                return np.concatenate(kept)

    def listen(
        self, awake: bool = False, wait_seconds: float = 6.0, on_frame: Callable[[np.ndarray], None] | None = None
    ) -> np.ndarray | None:
        """Wait for the wake phrase (unless already ``awake``), beep, and record what follows."""
        carry = None
        if not awake:
            print(f'(asleep; say "{self.phrase}")', flush=True)
            events.state("asleep", phrase=self.phrase)
            self.wait_for_wake()
            cue(self.audio, "wake")
            print("(awake)", flush=True)
            carry = list(self.recent)
        else:
            self._drain()
            self.vad.reset()
        return self.record_utterance(wait_seconds=wait_seconds, carry=carry, on_frame=on_frame)

    def close(self) -> None:
        self.audio.unsubscribe(self._frames.put)

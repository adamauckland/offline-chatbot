from __future__ import annotations

import logging
import queue
import re
import threading
import time

import numpy as np

import events
from audio.dsp import RobotVoice
from audio.input import AudioIO

DEFAULT_MODEL = "english_18m_2026-10"
DEFAULT_VOICE = "alba"
PREBUFFER_SECONDS = 0.2
PLAYBACK_POLL_SECONDS = 0.02

FENCED_CODE = re.compile(r"```.*?```", re.S)
INLINE_CODE = re.compile(r"`([^`]*)`")
EMPHASIS_HEADINGS_QUOTES = re.compile(r"[*_#>]+")
LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
BULLET = re.compile(r"^\s*[-•]\s+", re.M)
WHITESPACE = re.compile(r"\s+")

logging.getLogger("pocket_tts").setLevel(logging.WARNING)


def clean_for_speech(text: str) -> str:
    text = FENCED_CODE.sub(" ", text)
    text = INLINE_CODE.sub(r"\1", text)
    text = EMPHASIS_HEADINGS_QUOTES.sub("", text)
    text = LINK.sub(r"\1", text)
    text = BULLET.sub("", text)

    return WHITESPACE.sub(" ", text).strip()


class Prebuffer:
    def __init__(self, target_samples: float) -> None:
        self.target_samples = target_samples
        self.chunks: list[np.ndarray] | None = []
        self.count = 0

    @property
    def holding(self) -> bool:
        return self.chunks is not None

    @property
    def full(self) -> bool:
        return self.count >= self.target_samples

    def add(self, samples: np.ndarray) -> None:
        self.chunks.append(samples)
        self.count += len(samples)

    def release(self) -> np.ndarray:
        audio = np.concatenate(self.chunks)
        self.chunks = None

        return audio


class Speaker:
    def __init__(
        self, audio: AudioIO, model: str = DEFAULT_MODEL, voice: str = DEFAULT_VOICE, robot: bool = True
    ) -> None:
        from pocket_tts import TTSModel

        self.audio = audio
        self.model = TTSModel.load_model(language=model)
        self.voice = self.model.get_state_for_audio_prompt(voice)
        self.sample_rate = self.model.sample_rate
        self.effect = RobotVoice(self.sample_rate) if robot else None
        self.muted = False
        self._state_before_speaking = "idle"
        self._queue: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._speaking = False

        events.subscribe(self._track_state)
        threading.Thread(target=self._worker, daemon=True).start()

    def say(self, text: str) -> None:
        text = clean_for_speech(text)

        if text and not self.muted:
            self._queue.put(text)

    def wait(self) -> None:
        self._queue.join()
        self.audio.wait()

    def interrupt(self) -> None:
        self._discard_queued()
        self._stop.set()
        self.audio.stop_playback()

    def _discard_queued(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                return

    def _worker(self) -> None:
        while True:
            text = self._queue.get()

            if not self._speaking:
                self._begin_speaking()

            try:
                self._speak(text)
            except Exception as exc:
                print(f"(speech failed: {exc})", flush=True)
            finally:
                self._queue.task_done()

            self._wait_for_playback_or_next_sentence()

            if self._queue.empty():
                self._finish_speaking()

    def _begin_speaking(self) -> None:
        self._speaking = True

        if self.effect is not None:
            self.effect.reset()

        events.state("speaking")

    def _wait_for_playback_or_next_sentence(self) -> None:
        while self._queue.empty() and not self.audio.playback_idle():
            time.sleep(PLAYBACK_POLL_SECONDS)

    def _finish_speaking(self) -> None:
        self._speaking = False
        events.state(self._state_before_speaking)

    def _track_state(self, name: str, data: dict) -> None:
        if name == "state" and data.get("state") != "speaking":
            self._state_before_speaking = data["state"]

    def _speak(self, text: str) -> None:
        self._stop.clear()
        prebuffer = Prebuffer(PREBUFFER_SECONDS * self.sample_rate)

        for chunk in self.model.generate_audio_stream(self.voice, text, stop=self._stop):
            if self._stop.is_set():
                break

            samples = self._samples_of(chunk)

            if prebuffer.holding:
                prebuffer.add(samples)

                if prebuffer.full or not self.audio.playback_idle():
                    self.audio.play(prebuffer.release())

                continue

            self.audio.play(samples)

        if prebuffer.holding and prebuffer.count:
            self.audio.play(prebuffer.release())

    def _samples_of(self, chunk) -> np.ndarray:
        samples = np.asarray(chunk.numpy(), dtype=np.float32).reshape(-1)

        if self.effect is not None:
            samples = self.effect.process(samples)

        return samples

"""Speak chatbot replies through the speakers with Kyutai's Pocket TTS.

Pocket TTS runs on the CPU, so it does not compete with the MLX vision model
for the GPU. The default here is the 18M-parameter English model, which is a
few seconds to load and faster than real time on Apple silicon. Other choices:
``english_7m_2026-10``, ``english_2m_2026-10`` (smaller, rougher) or
``english`` (the full 100M model, best quality).

Audio is generated in chunks and written straight to a sounddevice output
stream, so speech starts after the first chunk rather than after the whole
reply has been synthesised. Synthesis runs on a background thread so you can
type the next message while the previous answer is still being spoken.

By default each chunk goes through the robot-voice effect in audio/dsp.py (an
octave-up pitch shift into a flanger) before it reaches the speakers; pass
``robot=False`` to hear the plain voice.
"""

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
DEFAULT_VOICE = "alba"  # see the pocket-tts README for the voice catalogue, or pass a .wav path
# Audio collected before a sentence starts to play. Synthesis runs many times faster than real
# time, so this costs about ten milliseconds, and it stops the output queue running dry between
# the first chunks, which made audible holes (clicks) at the start of sentences.
PREBUFFER_SECONDS = 0.2

logging.getLogger("pocket_tts").setLevel(logging.WARNING)


def clean_for_speech(text: str) -> str:
    """Drop markdown and code so the voice reads prose, not asterisks."""
    text = re.sub(r"```.*?```", " ", text, flags=re.S)  # fenced code blocks
    text = re.sub(r"`([^`]*)`", r"\1", text)  # inline code
    text = re.sub(r"[*_#>]+", "", text)  # emphasis, headings, quotes
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # links -> link text
    text = re.sub(r"^\s*[-•]\s+", "", text, flags=re.M)  # bullet markers
    return re.sub(r"\s+", " ", text).strip()


class Speaker:
    """Queue text and speak it through the default output device."""

    def __init__(
        self, audio: AudioIO, model: str = DEFAULT_MODEL, voice: str = DEFAULT_VOICE, robot: bool = True
    ) -> None:
        from pocket_tts import TTSModel  # slow import (torch), keep it local

        self.audio = audio
        self.model = TTSModel.load_model(language=model)
        self.voice = self.model.get_state_for_audio_prompt(voice)
        self.sample_rate = self.model.sample_rate
        self.effect = RobotVoice(self.sample_rate) if robot else None
        self.muted = False
        self._resume = "idle"  # the bot's state before it started speaking, restored afterwards
        events.subscribe(self._track_state)
        self._queue: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._speaking = False  # True from the first queued sentence until the queue has drained
        threading.Thread(target=self._worker, daemon=True).start()

    def say(self, text: str) -> None:
        """Queue ``text`` to be spoken; returns immediately."""
        text = clean_for_speech(text)
        if text and not self.muted:
            self._queue.put(text)

    def wait(self) -> None:
        """Block until everything queued has been synthesised and played."""
        self._queue.join()
        self.audio.wait()

    def interrupt(self) -> None:
        """Stop the current utterance and forget anything still queued."""
        while True:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break
        self._stop.set()
        self.audio.stop_playback()

    def _worker(self) -> None:
        # A reply arrives one sentence at a time (see speech/stream.py). The sentences
        # are synthesised back to back and count as one "speaking" span: the state only
        # goes back when the queue has drained and the last sample has played.
        while True:
            text = self._queue.get()
            if not self._speaking:
                self._speaking = True
                if self.effect is not None:
                    self.effect.reset()
                events.state("speaking")
            try:
                self._speak(text)
            except Exception as exc:  # noqa: BLE001 - never let speech kill the chat
                print(f"(speech failed: {exc})", flush=True)
            finally:
                self._queue.task_done()
            # Hold the "speaking" state while the audio is still playing, but take the next
            # sentence the moment it is queued, so it is synthesised while this one plays and
            # the two join without a hole.
            while self._queue.empty() and not self.audio.playback_idle():
                time.sleep(0.02)
            if self._queue.empty():
                self._speaking = False
                events.state(self._resume)

    def _track_state(self, name: str, data: dict) -> None:
        # Speech can play while the model is still busy (a "picture taken" notice), so when
        # it ends the state goes back to what it was, not to idle.
        if name == "state" and data.get("state") != "speaking":
            self._resume = data["state"]

    def _speak(self, text: str) -> None:
        """Synthesise one sentence into the output queue; playback runs on from the previous one."""
        self._stop.clear()
        held: list[np.ndarray] | None = []  # the first chunks, released together (see PREBUFFER_SECONDS)
        held_samples = 0
        for chunk in self.model.generate_audio_stream(self.voice, text, stop=self._stop):
            if self._stop.is_set():
                break
            samples = np.asarray(chunk.numpy(), dtype=np.float32).reshape(-1)
            if self.effect is not None:
                samples = self.effect.process(samples)
            if held is not None:
                held.append(samples)
                held_samples += len(samples)
                # Release once there is enough in hand, or at once if the previous sentence is
                # still playing, since then the queue already has material ahead of us.
                if held_samples >= PREBUFFER_SECONDS * self.sample_rate or not self.audio.playback_idle():
                    self.audio.play(np.concatenate(held))
                    held = None
                continue
            self.audio.play(samples)
        if held:  # a sentence too short to reach the threshold
            self.audio.play(np.concatenate(held))

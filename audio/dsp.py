"""Robot-voice effect for the streamed speech output.

The chain is an octave-up pitch shifter followed by a flanger. Both are
written to process audio one chunk at a time: each keeps a short history
buffer of what it has already heard, so feeding it the chunks that
``Speaker`` streams to the speakers gives the same result as processing the
whole utterance in one go. Pure numpy, no extra dependencies.

* ``PitchShifter`` is a delay-line ("Doppler") shifter. Two read taps sweep a
  sawtooth delay across the buffer so they replay the audio faster than it
  arrives; crossfading the taps hides the jump when each one wraps.
* ``Flanger`` mixes the signal with a copy delayed by a few milliseconds,
  sweeping that delay with a slow sine and feeding the output back in, which
  gives the familiar jet-engine comb-filter sweep.
"""

from __future__ import annotations

import numpy as np


def _read_interpolated(buf: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Linear-interpolated reads from ``buf`` at fractional positions ``pos``."""
    i0 = np.floor(pos).astype(np.intp)
    frac = (pos - i0).astype(np.float32)
    return buf[i0] * (1.0 - frac) + buf[i0 + 1] * frac


class PitchShifter:
    """Shift pitch by ``semitones`` (12 = one octave up) without changing speed."""

    def __init__(self, sample_rate: int, semitones: float = 12.0, window_ms: float = 40.0) -> None:
        self.ratio = 2.0 ** (semitones / 12.0)
        self.window = max(2, int(sample_rate * window_ms / 1000))
        self._hist_len = self.window + 2
        self.reset()

    def reset(self) -> None:
        self._hist = np.zeros(self._hist_len, dtype=np.float32)
        self._phase = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        if n == 0 or self.ratio == 1.0:
            return x
        # One trailing zero so the i0 + 1 read never falls off the end.
        buf = np.concatenate([self._hist, x.astype(np.float32, copy=False), np.zeros(1, np.float32)])
        base = self._hist_len + np.arange(n)
        # Each tap's phase runs 0 -> 1 as its delay sweeps across the window.
        inc = abs(self.ratio - 1.0) / self.window
        phase = (self._phase + inc * np.arange(n)) % 1.0
        self._phase = (self._phase + inc * n) % 1.0
        out = np.zeros(n, dtype=np.float32)
        for offset in (0.0, 0.5):
            q = (phase + offset) % 1.0
            # Shifting up means the read head catches up with the write head, so
            # the delay shrinks; shifting down it grows.
            delay = self.window * (1.0 - q) if self.ratio > 1.0 else self.window * q
            gain = np.sin(np.pi * q).astype(np.float32)  # silent where the tap wraps
            out += gain * _read_interpolated(buf, base - delay)
        self._hist = buf[-(self._hist_len + 1) : -1]
        return out


class Flanger:
    """Sine-swept short delay with feedback, mixed with the dry signal."""

    def __init__(
        self,
        sample_rate: int,
        rate_hz: float = 0.4,
        min_ms: float = 1.0,
        max_ms: float = 7.0,
        feedback: float = 0.6,
        mix: float = 0.5,
    ) -> None:
        self.sample_rate = sample_rate
        self.rate_hz = rate_hz
        self.min_delay = max(2.0, sample_rate * min_ms / 1000)
        self.max_delay = max(self.min_delay + 1.0, sample_rate * max_ms / 1000)
        self.feedback = feedback
        self.mix = mix
        # Feedback means each sample depends on earlier output, but never on
        # anything newer than ``min_delay`` samples ago, so blocks shorter than
        # that can be computed vectorised.
        self._block = max(1, int(self.min_delay) - 1)
        self._hist_len = int(self.max_delay) + 2
        self.reset()

    def reset(self) -> None:
        self._hist = np.zeros(self._hist_len, dtype=np.float32)
        self._phase = 0.0  # LFO phase in radians

    def process(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        if n == 0:
            return x
        x = x.astype(np.float32, copy=False)
        h = self._hist_len
        buf = np.concatenate([self._hist, np.zeros(n + 1, np.float32)])
        lfo = self._phase + 2.0 * np.pi * self.rate_hz * np.arange(n) / self.sample_rate
        self._phase = (self._phase + 2.0 * np.pi * self.rate_hz * n / self.sample_rate) % (2.0 * np.pi)
        delay = self.min_delay + (self.max_delay - self.min_delay) * (1.0 + np.sin(lfo)) / 2.0
        wet = np.empty(n, dtype=np.float32)
        for start in range(0, n, self._block):
            end = min(n, start + self._block)
            idx = np.arange(start, end)
            d = _read_interpolated(buf, h + idx - delay[start:end])
            wet[start:end] = d
            buf[h + start : h + end] = x[start:end] + self.feedback * d
        self._hist = buf[-(h + 1) : -1]
        return (1.0 - self.mix) * x + self.mix * wet


class RobotVoice:
    """Octave-up pitch shift into a flanger: the chain ``Speaker`` applies to each chunk."""

    def __init__(self, sample_rate: int, semitones: float = 12.0) -> None:
        self.pitch = PitchShifter(sample_rate, semitones=semitones)
        self.flanger = Flanger(sample_rate)

    def reset(self) -> None:
        """Forget the previous utterance so the next one does not start with its tail."""
        self.pitch.reset()
        self.flanger.reset()
        self._primed = False

    def process(self, x: np.ndarray) -> np.ndarray:
        if not getattr(self, "_primed", True):
            # Straight after a reset the delay lines are empty and the first 10 ms come out
            # as digital silence followed by a hard start, which pops. Run the first chunk
            # through once unheard so the lines hold real audio before the output begins.
            self.flanger.process(self.pitch.process(x))
            self._primed = True
        y = self.flanger.process(self.pitch.process(x))
        return np.clip(y, -1.0, 1.0)

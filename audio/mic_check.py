#!/usr/bin/env python3
"""See what the bot hears: a live level meter with the voice detector and wake-word spotter.

    uv run -m audio.mic_check                      # default device, channel 1, "hey computer"
    uv run -m audio.mic_check --mic UMC --mic-channel 3 --wake-word "ok scrudley"

Each line is one 100 ms frame: raw level in dBFS (after --mic-gain), the auto gain applied, a bar,
whether Silero VAD thinks it is speech, and WAKE when the phrase is spotted.
Speak normally; the bar should hit the middle and "speech" should appear.
If the raw level stays below about -40 dBFS while you talk, the microphone
is on another channel or its gain is turned down.
"""

from __future__ import annotations

import argparse
import queue
import time

import numpy as np

from audio.input import AudioIO
from audio.wake import DEFAULT_WAKE_WORD, SpeechDetector, WakeWordDetector


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mic")
    ap.add_argument("--mic-channel", type=int, default=1)
    ap.add_argument("--wake-word", default=DEFAULT_WAKE_WORD)
    ap.add_argument("--wake-threshold", type=float, default=0.2)
    ap.add_argument("--mic-gain", type=float, default=1.0)
    ap.add_argument("--no-auto-gain", action="store_true")
    ap.add_argument("--seconds", type=float, default=30.0)
    args = ap.parse_args()

    vad, wake = SpeechDetector(), WakeWordDetector(args.wake_word, threshold=args.wake_threshold)
    audio = AudioIO(mic=args.mic, channel=args.mic_channel, mic_gain=args.mic_gain, auto_gain=not args.no_auto_gain)
    raw_peaks: queue.Queue[float] = queue.Queue()
    frames: queue.Queue[np.ndarray] = queue.Queue()
    # Tap the raw level before gain by temporarily wrapping the auto-gain stage.
    auto = audio.auto

    class Tap:
        def process(self, frame):
            raw_peaks.put(float(np.abs(frame).max()))
            return auto.process(frame) if auto else frame

    audio.auto = Tap()
    audio.subscribe(frames.put)
    print(f'Say "{args.wake_word}". Ctrl-C to stop.\n')
    end = time.monotonic() + args.seconds
    try:
        while time.monotonic() < end:
            frame = frames.get()
            peak = raw_peaks.get()
            db = 20 * np.log10(max(peak, 1e-6))
            g = float(np.abs(frame).max()) / max(peak, 1e-6)
            speech = vad.feed(frame)
            fired = wake.feed(frame)
            bar = "#" * int(min(1.0, float(np.abs(frame).max())) * 40)
            print(
                f"{db:6.1f} dBFS  x{g:5.1f}  {bar:<40} {'speech' if speech else '      '} {'WAKE' if fired else ''}",
                flush=True,
            )
    except KeyboardInterrupt:
        pass
    finally:
        audio.close()


if __name__ == "__main__":
    main()

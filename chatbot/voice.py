from __future__ import annotations

import difflib

import threading

import events
from audio.input import cue
from audio.listen import Listener
from audio.wake import WakeListener
from speech import Speaker

SLEEP_PHRASES = {"go to sleep", "stop listening", "never mind", "nevermind", "stop", "cancel"}
WAKE_WAIT_SECONDS = 6.0
FOLLOW_UP_SECONDS = 20.0
HOLD_SECONDS = 3600.0
PUNCTUATION = " ,.!?"
LONE_WORD_SIMILARITY = 0.4
LONE_TAIL_SIMILARITY = 0.5
PHRASE_WORD_SIMILARITY = 0.75
MIN_FRAGMENT_LETTERS = 3


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def bare(word: str) -> str:
    return word.lower().strip(PUNCTUATION)


def sounds_like_the_phrase(word: str, targets: list[str]) -> bool:
    if not word:
        return False

    tail = "".join(targets)[-len(word) :]

    return (
        max(similarity(word, t) for t in targets) >= LONE_WORD_SIMILARITY
        or similarity(word, tail) >= LONE_TAIL_SIMILARITY
    )


def is_phrase_word(word: str, targets: list[str]) -> bool:
    return any(similarity(word, t) >= PHRASE_WORD_SIMILARITY for t in targets)


def is_phrase_fragment(word: str, targets: list[str]) -> bool:
    return MIN_FRAGMENT_LETTERS <= len(word) < len(targets[-1]) and "".join(targets).endswith(word)


def count_leading_phrase_words(words: list[str], targets: list[str]) -> int:
    dropped = 0

    for word in words[: len(targets)]:
        w = bare(word)

        if not is_phrase_word(w, targets) and not is_phrase_fragment(w, targets):
            break

        dropped += 1

    return dropped


def strip_wake_phrase(text: str, phrase: str) -> str:
    words = text.split()

    if not words:
        return text

    targets = phrase.lower().split()

    if len(words) == 1:
        return "" if sounds_like_the_phrase(bare(words[0]), targets) else text

    dropped = count_leading_phrase_words(words, targets)

    return " ".join(words[dropped:]).lstrip(PUNCTUATION)


def listening_window(awake: bool, keep_awake: bool) -> tuple[bool, float]:
    if keep_awake:
        return True, HOLD_SECONDS

    return awake, FOLLOW_UP_SECONDS if awake else WAKE_WAIT_SECONDS


def next_spoken_command(
    wake: WakeListener,
    listener: Listener,
    speaker: Speaker | None,
    awake: bool,
    keep_awake: bool = False,
    abort: threading.Event | None = None,
) -> tuple[str, bool]:
    if speaker is not None:
        speaker.wait()

    listener.start()
    awake, wait = listening_window(awake, keep_awake)
    audio = wake.listen(awake=awake, wait_seconds=wait, on_frame=listener.feed, abort=abort)

    if audio is None:
        listener.finish()
        return "", not awake

    cue(listener.audio, "parse", wait=False)
    print("(transcribing)", flush=True)
    events.state("transcribing")
    text = strip_wake_phrase(listener.finish(), wake.phrase)

    return text, not awake

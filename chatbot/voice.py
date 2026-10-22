"""The spoken side of a turn in always-on mode: wake, record, transcribe, tidy the transcript."""

from __future__ import annotations

import events
from audio.input import cue
from audio.listen import Listener
from audio.wake import WakeListener
from speech import Speaker

SLEEP_PHRASES = {"go to sleep", "stop listening", "never mind", "nevermind", "stop", "cancel"}
FOLLOW_UP_SECONDS = 20.0  # after a reply, how long we stay awake for a follow-up without the wake word
HOLD_SECONDS = 3600.0  # while a note is held for "stop", how long we keep listening without the wake word


def strip_wake_phrase(text: str, phrase: str) -> str:
    """Drop the wake phrase, or the tail of it, when it leaks into the transcript.

    The clip handed to Whisper starts half a second before the spotter fired,
    so it can hold the end of the phrase. Measured on synthetic clips, that
    tail hardly ever comes back as a word when a command follows it; what
    does come back is the phrase itself when the user says the name again
    ("Computer, lights off"), a literal fragment of it ("ter", "ley"), or,
    when nothing follows, a lone junk word ("here", "Skritly").

    So mid-sentence only a near-exact phrase word or a literal suffix of the
    phrase is dropped. An earlier looser rule (any leading word half-similar
    to a phrase word) ate ordinary first words: "where", "open", "put", "the".
    A lone word is judged loosely, since there is no command to lose.
    """
    import difflib

    words = text.split()
    if not words:
        return text
    targets = phrase.lower().split()
    joined = "".join(targets)

    def similarity(a: str, b: str) -> float:
        return difflib.SequenceMatcher(None, a, b).ratio()

    def bare(w: str) -> str:
        return w.lower().strip(" ,.!?")

    if len(words) == 1:
        w = bare(words[0])
        tail = joined[-len(w) :] if w else ""
        if w and (max(similarity(w, t) for t in targets) >= 0.4 or similarity(w, tail) >= 0.5):
            return ""
        return text
    dropped = 0
    for word in words[: len(targets)]:
        w = bare(word)
        phrase_word = any(similarity(w, t) >= 0.75 for t in targets)
        fragment = len(w) >= 3 and len(w) < len(targets[-1]) and joined.endswith(w)
        if phrase_word or fragment:
            dropped += 1
        else:
            break
    return " ".join(words[dropped:]).lstrip(" ,.!?")


def next_spoken_command(
    wake: WakeListener, listener: Listener, speaker: Speaker | None, awake: bool, keep_awake: bool = False
) -> tuple[str, bool]:
    """One always-on turn: wait for the wake phrase (unless ``awake``), record, transcribe.

    Returns (text, woke): ``woke`` is True when the wake phrase was heard this
    turn, so the caller can stay awake for a follow-up even if nothing usable
    was said after the phrase. With ``keep_awake`` (a note is being held until
    the user says "stop") the bot listens on without the wake phrase and does
    not go to sleep when nothing is said.
    """
    if speaker is not None:
        speaker.wait()  # never listen to our own voice
    listener.start()
    if keep_awake:
        awake, wait = True, HOLD_SECONDS
    else:
        wait = FOLLOW_UP_SECONDS if awake else 6.0
    audio = wake.listen(awake=awake, wait_seconds=wait, on_frame=listener.feed)
    if audio is None:
        listener.finish()  # nothing was fed; closes the utterance
        return "", not awake  # nothing said in the follow-up window: back to sleep
    cue(listener.audio, "parse", wait=False)
    print("(transcribing)", flush=True)
    events.state("transcribing")
    text = listener.finish()  # a streaming backend has been working on it all along
    # People say the name again even when the bot is already awake ("Scrudley, lights off"),
    # and the clip right after a wake can start with the phrase's tail: drop it either way.
    text = strip_wake_phrase(text, wake.phrase)
    return text, not awake

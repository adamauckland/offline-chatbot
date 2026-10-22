"""Text-to-speech: reading replies aloud with Pocket TTS and the robot-voice effect."""

from speech.speaker import DEFAULT_MODEL, DEFAULT_VOICE, Speaker, clean_for_speech
from speech.stream import SentenceStream

__all__ = ["DEFAULT_MODEL", "DEFAULT_VOICE", "SentenceStream", "Speaker", "clean_for_speech"]

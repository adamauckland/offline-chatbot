"""The full-screen terminal UI: robot face, VU meter and chat log (``--tui``).
``uv run -m tui`` shows it on its own with pretend events."""

from tui.app import FaceApp, demo, run_tui

__all__ = ["FaceApp", "demo", "run_tui"]

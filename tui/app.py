"""Full-screen terminal UI for the chatbot: a robot face, a VU meter and the conversation.

Start the chatbot with ``--tui`` to use it. The face reacts to the events in
events/bus.py: eyes closed while asleep, wide open while listening, looking up
while thinking, down while calling a tool, and the mouth moves with the
speaker output while a reply is read aloud. The VU meter shows the microphone
(after the chatbot's own gain) and the speaker output, in dBFS with a one
second peak hold. Everything the chatbot prints lands in the log pane, and
the input box at the bottom replaces the ``you>`` prompt (an empty line still
records from the microphone).

Keys: Enter sends a line, Esc stops the current reply or speech (what Ctrl-C
does in the terminal), Ctrl-Q quits.

The chat loop runs on a worker thread; this module only reads the event bus,
so it never touches the model or the audio stream. The face is drawn with
half-block characters, which gives two "pixels" per terminal row and makes
each pixel roughly square, so circles come out round.

See the face without loading any models (it cycles through every state with
pretend audio):
  uv run -m tui
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import math
import queue
import random
import sys
import threading
import time
import traceback
from collections.abc import Callable

from rich.style import Style
from rich.text import Text
from textual import events as tevents
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widget import Widget
from textual.widgets import Input, RichLog, Static

import events

FPS = 15
SHOW_VU_METER = False  # the MIC/OUT level bars under the face; off for now
ROLL_STATES = {"transcribing", "thinking", "tool"}  # the turnaround phase: the eyes roll round and round
ROLL_SECONDS = 1.2  # one full circle
NARROW = 76  # below this many columns the face goes above the log instead of beside it

STATE_COLOURS = {
    "asleep": "#6b7a8a",
    "idle": "#60a5fa",
    "listening": "#22d3ee",
    "recording": "#f87171",
    "transcribing": "#fbbf24",
    "thinking": "#a78bfa",
    "tool": "#f472b6",
    "speaking": "#4ade80",
}
HEAD, RIM, EYE, PUPIL = "#3b4a5a", "#64798f", "#eef3f8", "#0f172a"
PANEL = "#161a21"  # the face panel's background; empty pixels are painted with it too


class BotState:
    """What the event handlers record; the widgets read it on every frame."""

    def __init__(self, wake_phrase: str | None = None) -> None:
        self.state = "idle"
        self.detail: dict = {}
        self.since = time.monotonic()
        self.wake_phrase = wake_phrase
        self.mode = "chat"
        self.mic = 0.0
        self.out = 0.0

    def handle(self, name: str, data: dict) -> None:
        if name == "level":  # ten times a second from the audio thread: store and leave
            self.mic = data.get("mic", 0.0)
            self.out = data.get("out", 0.0)
        elif name == "mode":
            self.mode = data.get("mode", self.mode)
        elif name == "state":
            if data["state"] != self.state:
                self.since = time.monotonic()
            self.state = data["state"]
            self.detail = data
            if "phrase" in data:
                self.wake_phrase = data["phrase"]

    def label(self, typed: bool) -> str:
        s = self.state
        if s == "asleep":
            return f'asleep, say "{self.wake_phrase}"' if self.wake_phrase else "asleep"
        if s == "idle":
            return "ready, type or press Enter to talk" if typed else "ready"
        if s == "tool":
            return f"calling {self.detail.get('name', 'a tool')}"
        return {
            "listening": "listening, speak now",
            "recording": "recording",
            "transcribing": "transcribing",
            "thinking": "thinking",
            "speaking": "speaking",
        }.get(s, s)


def to_db(level: float) -> float:
    return max(-60.0, min(0.0, 20 * math.log10(max(level, 1e-6))))


# -- the face ------------------------------------------------------------------


class Face(Widget):
    """A round robot head with big eyes, drawn from a pixel buffer two pixels per row."""

    DEFAULT_CSS = "Face { width: 100%; height: 1fr; }"

    def __init__(self, bot: BotState, **kwargs) -> None:
        super().__init__(**kwargs)
        self.bot = bot
        self.gaze = [0.0, 0.0]
        self.target = (0.0, 0.0)
        self.wander_at = 0.0
        self.blink_at = time.monotonic() + 2.0
        self.blink_until = 0.0
        self.openness = 1.0
        self.mouth = 0.0
        self._styles: dict[tuple[str, str], Style] = {}

    # Animation state, advanced once per frame.
    def animate(self, t: float) -> None:
        s = self.bot.state
        if s in ROLL_STATES:
            # Working: the pupils circle the eyes, in step with the turnaround notes.
            phase = 2 * math.pi * (t - self.bot.since) / ROLL_SECONDS
            self.target = (0.85 * math.cos(phase), 0.8 * math.sin(phase))
            self.gaze = list(self.target)  # no smoothing, or the circle shrinks to a wobble
        elif s in ("listening", "recording"):
            self.target = (0.0, 0.0)
        elif s == "asleep":
            self.target = (0.0, 0.3)
        elif t >= self.wander_at:  # idle or speaking: glance around now and then
            self.target = (random.uniform(-0.6, 0.6), random.uniform(-0.4, 0.3))
            self.wander_at = t + random.uniform(1.5, 4.0)
        self.gaze[0] += (self.target[0] - self.gaze[0]) * 0.3
        self.gaze[1] += (self.target[1] - self.gaze[1]) * 0.3

        if s == "asleep":
            want = 0.06
        elif t < self.blink_until:
            want = 0.06
        else:
            if t >= self.blink_at:
                self.blink_until = t + 0.14
                self.blink_at = t + random.uniform(2.5, 6.0)
            want = {"listening": 1.15, "recording": 1.2}.get(s, 1.0)
        self.openness += (want - self.openness) * (0.7 if want < self.openness else 0.5)

        level = max(0.0, min(1.0, (to_db(self.bot.out) + 40) / 30))
        self.mouth += (level - self.mouth) * (0.7 if level > self.mouth else 0.35)

    def render(self) -> Text:
        w, h = self.size.width, self.size.height
        if w < 12 or h < 6:
            return Text("(too small)", justify="center")
        px: list[list[str | None]] = [[None] * w for _ in range(2 * h)]
        self._draw(px, w, 2 * h)
        return self._to_text(px, w, h)

    def _draw(self, px: list[list[str | None]], w: int, h: int) -> None:
        bot = self.bot
        colour = STATE_COLOURS.get(bot.state, STATE_COLOURS["idle"])
        t = time.monotonic()
        antenna = 4
        r = min((w - 2) / 2, (h - antenna - 1) / 2)
        if r < 4:
            return
        cx, cy = w / 2, antenna + 1 + r
        # Head and rim
        disk(px, cx, cy, r, r, HEAD)
        ring(px, cx, cy, r, 1.3, RIM)
        # Antenna: a stalk and a light that shows the state
        top = int(cy - r)
        for y in range(top - 2, top):
            put(px, int(cx), y, RIM)
        glow = colour
        if bot.state == "recording" and int(t * 4) % 2:
            glow = HEAD  # fast blink while recording
        elif bot.state in ("listening", "thinking") and math.sin(t * 5) < -0.3:
            glow = RIM  # slow pulse
        disk(px, cx, top - 3, 1.5, 1.5, glow)
        # Eyes
        ex, ey = r * 0.42, r * 0.2
        rx, ry = r * 0.3, r * 0.36 * self.openness
        pr = r * (0.15 if bot.state in ("listening", "recording") else 0.12)
        gx, gy = self.gaze
        for side in (-1, 1):
            x0, y0 = cx + side * ex, cy - ey
            if self.openness <= 0.12:  # closed: a flat line
                for x in range(int(x0 - rx), int(x0 + rx) + 1):
                    put(px, x, int(y0), PUPIL)
                continue
            disk(px, x0, y0, rx, max(ry, 1.0), EYE)
            ix, iy = x0 + gx * rx * 0.45, y0 + gy * ry * 0.4
            disk(px, ix, iy, pr, pr, colour)  # iris in the state colour
            disk(px, ix, iy, pr * 0.55, pr * 0.55, PUPIL)
            if pr >= 2.5:
                put(px, int(ix - pr * 0.45), int(iy - pr * 0.45), EYE)  # glint
        # Mouth
        mx, my, mw = cx, cy + r * 0.52, r * 0.55
        thick = 1 if r < 10 else 2
        if bot.state == "speaking" and self.mouth > 0.12:
            disk(px, mx, my, mw * (0.6 + 0.4 * self.mouth), 1 + self.mouth * r * 0.3, PUPIL)
        elif bot.state in ("listening", "recording"):
            disk(px, mx, my, r * 0.12, r * 0.12 * (1.3 if bot.state == "recording" else 1.0), PUPIL)
        elif bot.state in ("idle", "speaking"):
            curve = r * 0.12
            for x in range(int(mx - mw), int(mx + mw) + 1):
                y = my + curve * (1 - ((x + 0.5 - mx) / mw) ** 2)
                for k in range(thick):
                    put(px, x, int(y) + k, PUPIL)
        else:
            shift = r * 0.15 if bot.state == "thinking" else 0.0
            width = mw * (0.6 if bot.state == "asleep" else 0.8)
            for x in range(int(mx + shift - width), int(mx + shift + width) + 1):
                for k in range(thick):
                    put(px, x, int(my) + k, PUPIL)

    def _to_text(self, px: list[list[str | None]], w: int, h: int) -> Text:
        text = Text(no_wrap=True, end="")
        bot = self.bot
        zees: dict[tuple[int, int], str] = {}
        if bot.state == "asleep":
            antenna = 4
            r = min((w - 2) / 2, (2 * h - antenna - 1) / 2)
            col, row = int(w / 2 + r * 0.8), int((antenna + 1 + r - r * 0.95) / 2)
            for k in range(int(time.monotonic() * 1.5) % 3 + 1):
                zees[(row - k, col + k)] = "z" if k < 2 else "Z"
        for row in range(h):
            run: list[str] = []
            run_style: Style | None = None
            for col in range(w):
                top, bottom = px[2 * row][col], px[2 * row + 1][col]
                if (row, col) in zees:
                    char, style = zees[(row, col)], self._style(STATE_COLOURS["asleep"], PANEL)
                elif top is None and bottom is None:
                    char, style = " ", None
                else:
                    # Always a half block with both colours set: a fully painted
                    # cell, however the terminal's font shapes its block glyphs.
                    char, style = "▀", self._style(top or PANEL, bottom or PANEL)
                if style is not run_style and run:
                    text.append("".join(run), run_style)
                    run = []
                run.append(char)
                run_style = style
            if run:
                text.append("".join(run), run_style)
            if row < h - 1:
                text.append("\n")
        return text

    def _style(self, fg: str, bg: str) -> Style:
        key = (fg, bg)
        if key not in self._styles:
            self._styles[key] = Style(color=fg, bgcolor=bg)
        return self._styles[key]


def put(px: list[list[str | None]], x: int, y: int, colour: str) -> None:
    if 0 <= y < len(px) and 0 <= x < len(px[0]):
        px[y][x] = colour


def disk(px: list[list[str | None]], cx: float, cy: float, rx: float, ry: float, colour: str) -> None:
    """Fill an ellipse; a pixel is inside when its centre is."""
    for y in range(max(0, int(cy - ry) - 1), min(len(px), int(cy + ry) + 2)):
        for x in range(max(0, int(cx - rx) - 1), min(len(px[0]), int(cx + rx) + 2)):
            if ((x + 0.5 - cx) / rx) ** 2 + ((y + 0.5 - cy) / ry) ** 2 <= 1.0:
                px[y][x] = colour


def ring(px: list[list[str | None]], cx: float, cy: float, r: float, width: float, colour: str) -> None:
    for y in range(max(0, int(cy - r) - 1), min(len(px), int(cy + r) + 2)):
        for x in range(max(0, int(cx - r) - 1), min(len(px[0]), int(cx + r) + 2)):
            d = math.hypot(x + 0.5 - cx, y + 0.5 - cy)
            if r - width <= d <= r:
                px[y][x] = colour


# -- the VU meter --------------------------------------------------------------


class Channel:
    """One meter: fast attack, slow release, and a peak that holds for a second."""

    def __init__(self) -> None:
        self.db = -60.0
        self.peak = -60.0
        self.peak_at = 0.0

    def update(self, level: float, t: float) -> None:
        db = to_db(level)
        self.db = db if db > self.db else max(db, self.db - 1.2)
        if db >= self.peak:
            self.peak, self.peak_at = db, t
        elif t - self.peak_at > 1.0:
            self.peak = max(db, self.peak - 2.0)


class VUMeter(Widget):
    """Two bars, MIC and OUT, with a dB scale underneath."""

    DEFAULT_CSS = "VUMeter { width: 100%; height: 3; }"

    def __init__(self, bot: BotState, **kwargs) -> None:
        super().__init__(**kwargs)
        self.bot = bot
        self.mic = Channel()
        self.out = Channel()

    def animate(self, t: float) -> None:
        self.mic.update(self.bot.mic, t)
        self.out.update(self.bot.out, t)

    def render(self) -> Text:
        w = self.size.width
        n = w - 12  # "MIC " + bar + " -60dB"
        if n < 8:
            return Text("(too narrow)")
        text = Text(no_wrap=True, end="")
        live = self.bot.state
        self._bar(text, "MIC", self.mic, n, live in ("listening", "recording", "asleep"))
        text.append("\n")
        self._bar(text, "OUT", self.out, n, live == "speaking")
        text.append("\n")
        scale = [" "] * n
        for db in (-60, -40, -18, -6, 0):
            label = str(db)
            at = min(n - len(label), max(0, round((db + 60) / 60 * (n - 1)) - len(label) // 2))
            scale[at : at + len(label)] = list(label)
        text.append("    " + "".join(scale), Style(dim=True))
        return text

    @staticmethod
    def _bar(text: Text, name: str, ch: Channel, n: int, live: bool) -> None:
        text.append(f"{name} ", Style(bold=live, color="#e2e8f0" if live else "#64748b"))
        filled = round((ch.db + 60) / 60 * n)
        peak = min(n - 1, round((ch.peak + 60) / 60 * (n - 1)))
        for i in range(n):
            db = -60 + 60 * (i + 0.5) / n
            colour = "#4ade80" if db < -18 else "#fbbf24" if db < -6 else "#f87171"
            if i == peak and ch.peak > -59:
                text.append("▌", Style(color="#f8fafc", bgcolor=colour if i < filled else None))
            elif i < filled:
                text.append("█", Style(color=colour))
            else:
                text.append("▒", Style(color="#334155"))
        text.append(f"{ch.db:5.0f}dB", Style(color="#e2e8f0" if live else "#64748b"))


# -- the app -------------------------------------------------------------------


class FaceApp(App[str | None]):
    """Runs ``chat(ask)`` on a worker thread and shows what it does."""

    TITLE = "scrudley"
    CSS = f"""
    Screen {{ layout: horizontal; }}
    Screen.narrow {{ layout: vertical; }}
    #left {{ width: 46; height: 100%; border: round #475569; border-title-color: #94a3b8; padding: 0 1;
             background: {PANEL}; }}
    Screen.narrow #left {{ width: 100%; height: 55%; }}
    #right {{ width: 1fr; height: 100%; }}
    Screen.narrow #right {{ height: 1fr; }}
    #status {{ height: 1; }}
    #log {{ height: 1fr; border: round #475569; border-title-color: #94a3b8; border-subtitle-color: #64748b;
            padding: 0 1; }}
    #live {{ height: auto; max-height: 4; padding: 0 2; }}
    #input {{ dock: bottom; }}
    """
    BINDINGS = [
        Binding("escape", "cancel", "Stop", priority=True),
        Binding("ctrl+q", "quit", "Quit", priority=True),
    ]

    def __init__(
        self, chat: Callable[[Callable[[str], str]], None], typed: bool = True, wake_phrase: str | None = None
    ) -> None:
        super().__init__()
        self.chat = chat
        self.typed = typed
        self.bot = BotState(wake_phrase)
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._partial = ""
        self._error: str | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="left") as left:
            left.border_title = "scrudley"
            yield Face(self.bot, id="face")
            if SHOW_VU_METER:
                yield VUMeter(self.bot, id="meter")
            yield Static(id="status")
        with Vertical(id="right"):
            log = RichLog(id="log", wrap=True, min_width=20, markup=False, highlight=False)
            log.border_title = "conversation"
            log.border_subtitle = "Esc stops · Ctrl-Q quits"
            yield log
            yield Static(id="live")
            if self.typed:
                yield Input(id="input", placeholder="type a message, or press Enter alone to talk")

    def on_mount(self) -> None:
        events.subscribe(self.bot.handle)
        self.begin_capture_print(self)
        self.set_interval(1 / FPS, self._tick)
        self._on_resize_check(self.size.width)
        if self.typed:
            self.query_one("#input", Input).focus()
        threading.Thread(target=self._run_chat, daemon=True, name="chat").start()

    def on_unmount(self) -> None:
        events.unsubscribe(self.bot.handle)
        events.CANCEL.set()
        self._lines.put(None)  # unblock ask(): the chat loop sees EOF and ends

    def on_resize(self, event: tevents.Resize) -> None:
        self._on_resize_check(event.size.width)

    def _on_resize_check(self, width: int) -> None:
        self.screen.set_class(width < NARROW, "narrow")

    # -- the chat loop's side ---------------------------------------------------
    def _run_chat(self) -> None:
        try:
            self.chat(self._ask)
        except Exception:  # noqa: BLE001 - show it after the screen is restored
            self._error = traceback.format_exc()
        try:
            self.call_from_thread(self.exit, self._error)
        except (RuntimeError, asyncio.CancelledError, concurrent.futures.CancelledError):
            pass  # the app is already closing (loop gone, or our call was cancelled with it)

    def _ask(self, prompt: str) -> str:  # noqa: ARG002 - the input box is the prompt
        line = self._lines.get()
        if line is None:
            raise EOFError
        return line

    # -- the screen's side -------------------------------------------------------
    def _tick(self) -> None:
        t = time.monotonic()
        face = self.query_one("#face", Face)
        face.animate(t)
        face.refresh()
        for meter in self.query(VUMeter):  # none when SHOW_VU_METER is off
            meter.animate(t)
            meter.refresh()
        colour = STATE_COLOURS.get(self.bot.state, STATE_COLOURS["idle"])
        status = Text(no_wrap=True, overflow="ellipsis")
        status.append("● ", Style(color=colour))
        status.append(self.bot.label(self.typed), Style(color="#e2e8f0"))
        status.append(f"  · {self.bot.mode} mode", Style(dim=True))
        self.query_one("#status", Static).update(status)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        line = event.value
        event.input.value = ""
        if line.strip():
            self._write(f"you> {line.strip()}")
        self._lines.put(line)

    def on_print(self, event: tevents.Print) -> None:
        """The chatbot's prints, one write at a time: complete lines go to the log,
        the unfinished one (a reply being streamed word by word) is shown underneath."""
        self._partial += event.text
        *done, self._partial = self._partial.split("\n")
        for line in done:
            self._write(line.rstrip("\r"), dim=event.stderr)
        self.query_one("#live", Static).update(self._styled(self._partial))

    def _write(self, line: str, dim: bool = False) -> None:
        if line.strip() == "":
            return
        self.query_one("#log", RichLog).write(self._styled(line, dim))

    @staticmethod
    def _styled(line: str, dim: bool = False) -> Text:
        text = Text(line)
        if dim:
            text.stylize("dim")
        elif line.startswith("you"):
            text.stylize("bold #22d3ee", 0, line.find(">") + 1)
        elif line.startswith("phi>"):
            text.stylize("bold #4ade80", 0, 4)
        elif line.startswith("("):
            text.stylize("dim")
        return text

    def action_cancel(self) -> None:
        events.cancel()
        self._write("(stopping)")


def run_tui(chat: Callable[[Callable[[str], str]], None], typed: bool = True, wake_phrase: str | None = None) -> None:
    """Show the UI while ``chat(ask)`` runs on a worker thread; returns when either ends.

    ``ask(prompt)`` reads a line from the input box and raises EOFError when
    the UI has closed. ``typed`` hides the input box for the always-listening
    modes, which never read it.
    """
    error = FaceApp(chat, typed=typed, wake_phrase=wake_phrase).run()
    if error:
        print(error, file=sys.stderr)


# -- demo ----------------------------------------------------------------------


def demo(ask: Callable[[str], str]) -> None:
    """Cycle through the states with made-up audio, so the face can be seen without the models."""
    phrase = "hey computer"
    script = [
        ("asleep", 3.0, {"phrase": phrase}, f'(asleep; say "{phrase}")'),
        ("listening", 1.5, {}, "(awake)"),
        ("recording", 2.0, {}, "(recording)"),
        ("transcribing", 1.0, {}, "(transcribing)"),
        ("thinking", 2.0, {}, "you (mic)> turn the kitchen lights on\nphi> "),
        (
            "tool",
            1.2,
            {"name": "set_lights", "args": {"state": "on", "room": "kitchen"}},
            "(calling set_lights {'state': 'on', 'room': 'kitchen'})",
        ),
        ("thinking", 1.5, {}, "(result {'ok': True})"),
        ("speaking", 3.5, {}, "The kitchen lights are now on.\n"),
        ("idle", 2.5, {}, ""),
    ]

    def echo() -> None:
        while True:
            try:
                line = ask("you> ")
            except EOFError:
                return
            print(f"(demo: you typed {line!r}; nothing listens in the demo)" if line else "(demo: Enter)")

    threading.Thread(target=echo, daemon=True).start()
    rng = random.Random(1)
    while True:
        for state, seconds, data, line in script:
            if line:
                print(line, end="" if line.endswith("> ") else "\n", flush=True)
            events.state(state, **data)
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                mic = out = rng.uniform(0.002, 0.006)
                if state == "recording" and rng.random() < 0.7:
                    mic = rng.uniform(0.05, 0.4)
                if state == "speaking" and rng.random() < 0.75:
                    out = rng.uniform(0.03, 0.3)
                events.emit("level", mic=mic, out=out)
                time.sleep(0.1)


if __name__ == "__main__":
    run_tui(demo, typed=True, wake_phrase="hey computer")

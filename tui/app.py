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
SHOW_VU_METER = False
ROLL_STATES = {"transcribing", "thinking", "tool"}
ROLL_SECONDS = 1.2
NARROW_COLUMNS = 76
ANTENNA_ROWS = 4
CLOSED_EYE_OPENNESS = 0.12
BLINK_SECONDS = 0.14
METER_LABEL_COLUMNS = 12

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
PANEL = "#161a21"

Pixels = list[list[str | None]]


class BotState:
    def __init__(self, wake_phrase: str | None = None) -> None:
        self.state = "idle"
        self.detail: dict = {}
        self.since = time.monotonic()
        self.wake_phrase = wake_phrase
        self.mode = "chat"
        self.mic = 0.0
        self.out = 0.0

    def handle(self, name: str, data: dict) -> None:
        if name == "level":
            self._on_level(data)
        elif name == "mode":
            self.mode = data.get("mode", self.mode)
        elif name == "state":
            self._on_state(data)

    def _on_level(self, data: dict) -> None:
        self.mic = data.get("mic", 0.0)
        self.out = data.get("out", 0.0)

    def _on_state(self, data: dict) -> None:
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


def put(px: Pixels, x: int, y: int, colour: str) -> None:
    if 0 <= y < len(px) and 0 <= x < len(px[0]):
        px[y][x] = colour


def disk(px: Pixels, cx: float, cy: float, rx: float, ry: float, colour: str) -> None:
    for y in range(max(0, int(cy - ry) - 1), min(len(px), int(cy + ry) + 2)):
        for x in range(max(0, int(cx - rx) - 1), min(len(px[0]), int(cx + rx) + 2)):
            if ((x + 0.5 - cx) / rx) ** 2 + ((y + 0.5 - cy) / ry) ** 2 <= 1.0:
                px[y][x] = colour


def ring(px: Pixels, cx: float, cy: float, r: float, width: float, colour: str) -> None:
    for y in range(max(0, int(cy - r) - 1), min(len(px), int(cy + r) + 2)):
        for x in range(max(0, int(cx - r) - 1), min(len(px[0]), int(cx + r) + 2)):
            d = math.hypot(x + 0.5 - cx, y + 0.5 - cy)

            if r - width <= d <= r:
                px[y][x] = colour


def head_radius(w: int, h: int) -> float:
    return min((w - 2) / 2, (h - ANTENNA_ROWS - 1) / 2)


class Face(Widget):
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

    def animate(self, t: float) -> None:
        self._aim_eyes(t)
        self._blink(t)
        self._move_mouth()

    def _aim_eyes(self, t: float) -> None:
        s = self.bot.state

        if s in ROLL_STATES:
            phase = 2 * math.pi * (t - self.bot.since) / ROLL_SECONDS
            self.target = (0.85 * math.cos(phase), 0.8 * math.sin(phase))
            self.gaze = list(self.target)
        elif s in ("listening", "recording"):
            self.target = (0.0, 0.0)
        elif s == "asleep":
            self.target = (0.0, 0.3)
        elif t >= self.wander_at:
            self.target = (random.uniform(-0.6, 0.6), random.uniform(-0.4, 0.3))
            self.wander_at = t + random.uniform(1.5, 4.0)

        self.gaze[0] += (self.target[0] - self.gaze[0]) * 0.3
        self.gaze[1] += (self.target[1] - self.gaze[1]) * 0.3

    def _blink(self, t: float) -> None:
        want = self._wanted_openness(t)
        self.openness += (want - self.openness) * (0.7 if want < self.openness else 0.5)

    def _wanted_openness(self, t: float) -> float:
        s = self.bot.state

        if s == "asleep" or t < self.blink_until:
            return 0.06

        if t >= self.blink_at:
            self.blink_until = t + BLINK_SECONDS
            self.blink_at = t + random.uniform(2.5, 6.0)

        return {"listening": 1.15, "recording": 1.2}.get(s, 1.0)

    def _move_mouth(self) -> None:
        level = max(0.0, min(1.0, (to_db(self.bot.out) + 40) / 30))
        self.mouth += (level - self.mouth) * (0.7 if level > self.mouth else 0.35)

    def render(self) -> Text:
        w, h = self.size.width, self.size.height

        if w < 12 or h < 6:
            return Text("(too small)", justify="center")

        px: Pixels = [[None] * w for _ in range(2 * h)]
        self._draw(px, w, 2 * h)

        return self._to_text(px, w, h)

    def _draw(self, px: Pixels, w: int, h: int) -> None:
        colour = STATE_COLOURS.get(self.bot.state, STATE_COLOURS["idle"])
        r = head_radius(w, h)

        if r < 4:
            return

        cx, cy = w / 2, ANTENNA_ROWS + 1 + r
        disk(px, cx, cy, r, r, HEAD)
        ring(px, cx, cy, r, 1.3, RIM)
        self._draw_antenna(px, cx, cy, r, colour)
        self._draw_eyes(px, cx, cy, r, colour)
        self._draw_mouth(px, cx, cy, r)

    def _draw_antenna(self, px: Pixels, cx: float, cy: float, r: float, colour: str) -> None:
        top = int(cy - r)

        for y in range(top - 2, top):
            put(px, int(cx), y, RIM)

        disk(px, cx, top - 3, 1.5, 1.5, self._antenna_glow(colour))

    def _antenna_glow(self, colour: str) -> str:
        t = time.monotonic()
        state = self.bot.state

        if state == "recording" and int(t * 4) % 2:
            return HEAD

        if state in ("listening", "thinking") and math.sin(t * 5) < -0.3:
            return RIM

        return colour

    def _draw_eyes(self, px: Pixels, cx: float, cy: float, r: float, colour: str) -> None:
        ex, ey = r * 0.42, r * 0.2
        rx, ry = r * 0.3, r * 0.36 * self.openness
        pupil = r * (0.15 if self.bot.state in ("listening", "recording") else 0.12)

        for side in (-1, 1):
            self._draw_eye(px, cx + side * ex, cy - ey, rx, ry, pupil, colour)

    def _draw_eye(self, px: Pixels, x0: float, y0: float, rx: float, ry: float, pupil: float, colour: str) -> None:
        if self.openness <= CLOSED_EYE_OPENNESS:
            self._draw_closed_eye(px, x0, y0, rx)
            return

        gx, gy = self.gaze
        ix, iy = x0 + gx * rx * 0.45, y0 + gy * ry * 0.4
        disk(px, x0, y0, rx, max(ry, 1.0), EYE)
        disk(px, ix, iy, pupil, pupil, colour)
        disk(px, ix, iy, pupil * 0.55, pupil * 0.55, PUPIL)

        if pupil >= 2.5:
            put(px, int(ix - pupil * 0.45), int(iy - pupil * 0.45), EYE)

    def _draw_closed_eye(self, px: Pixels, x0: float, y0: float, rx: float) -> None:
        for x in range(int(x0 - rx), int(x0 + rx) + 1):
            put(px, x, int(y0), PUPIL)

    def _draw_mouth(self, px: Pixels, cx: float, cy: float, r: float) -> None:
        state = self.bot.state
        x, y, half_width = cx, cy + r * 0.52, r * 0.55
        thick = 1 if r < 10 else 2

        if state == "speaking" and self.mouth > 0.12:
            disk(px, x, y, half_width * (0.6 + 0.4 * self.mouth), 1 + self.mouth * r * 0.3, PUPIL)
        elif state in ("listening", "recording"):
            disk(px, x, y, r * 0.12, r * 0.12 * (1.3 if state == "recording" else 1.0), PUPIL)
        elif state in ("idle", "speaking"):
            self._draw_smile(px, x, y, half_width, r * 0.12, thick)
        else:
            shift = r * 0.15 if state == "thinking" else 0.0
            width = half_width * (0.6 if state == "asleep" else 0.8)
            self._draw_flat_mouth(px, x + shift, y, width, thick)

    def _draw_smile(self, px: Pixels, x: float, y: float, half_width: float, curve: float, thick: int) -> None:
        for col in range(int(x - half_width), int(x + half_width) + 1):
            row = y + curve * (1 - ((col + 0.5 - x) / half_width) ** 2)

            for k in range(thick):
                put(px, col, int(row) + k, PUPIL)

    def _draw_flat_mouth(self, px: Pixels, x: float, y: float, half_width: float, thick: int) -> None:
        for col in range(int(x - half_width), int(x + half_width) + 1):
            for k in range(thick):
                put(px, col, int(y) + k, PUPIL)

    def _to_text(self, px: Pixels, w: int, h: int) -> Text:
        text = Text(no_wrap=True, end="")
        zees = self._sleep_zees(w, h) if self.bot.state == "asleep" else {}

        for row in range(h):
            self._append_row(text, px, row, w, zees)

            if row < h - 1:
                text.append("\n")

        return text

    def _sleep_zees(self, w: int, h: int) -> dict[tuple[int, int], str]:
        r = head_radius(w, 2 * h)
        col, row = int(w / 2 + r * 0.8), int((ANTENNA_ROWS + 1 + r - r * 0.95) / 2)
        count = int(time.monotonic() * 1.5) % 3 + 1

        return {(row - k, col + k): "z" if k < 2 else "Z" for k in range(count)}

    def _append_row(self, text: Text, px: Pixels, row: int, w: int, zees: dict[tuple[int, int], str]) -> None:
        run: list[str] = []
        run_style: Style | None = None

        for col in range(w):
            char, style = self._cell(px, row, col, zees)

            if style is not run_style and run:
                text.append("".join(run), run_style)
                run = []

            run.append(char)
            run_style = style

        if run:
            text.append("".join(run), run_style)

    def _cell(self, px: Pixels, row: int, col: int, zees: dict[tuple[int, int], str]) -> tuple[str, Style | None]:
        if (row, col) in zees:
            return zees[(row, col)], self._style(STATE_COLOURS["asleep"], PANEL)

        top, bottom = px[2 * row][col], px[2 * row + 1][col]

        if top is None and bottom is None:
            return " ", None

        return "▀", self._style(top or PANEL, bottom or PANEL)

    def _style(self, fg: str, bg: str) -> Style:
        key = (fg, bg)

        if key not in self._styles:
            self._styles[key] = Style(color=fg, bgcolor=bg)

        return self._styles[key]


class Channel:
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
        n = self.size.width - METER_LABEL_COLUMNS

        if n < 8:
            return Text("(too narrow)")

        text = Text(no_wrap=True, end="")
        live = self.bot.state
        self._bar(text, "MIC", self.mic, n, live in ("listening", "recording", "asleep"))
        text.append("\n")
        self._bar(text, "OUT", self.out, n, live == "speaking")
        text.append("\n")
        text.append("    " + self._scale(n), Style(dim=True))

        return text

    @staticmethod
    def _scale(n: int) -> str:
        scale = [" "] * n

        for db in (-60, -40, -18, -6, 0):
            label = str(db)
            at = min(n - len(label), max(0, round((db + 60) / 60 * (n - 1)) - len(label) // 2))
            scale[at : at + len(label)] = list(label)

        return "".join(scale)

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


class FaceApp(App[str | None]):
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

            yield Input(
                id="input",
                placeholder=(
                    "type a message, or press Enter alone to talk"
                    if self.typed
                    else "type a message (the microphone stays on); Esc stops"
                ),
            )

    def on_mount(self) -> None:
        events.subscribe(self.bot.handle)
        self.begin_capture_print(self)
        self.set_interval(1 / FPS, self._tick)
        self._fit_layout_to(self.size.width)

        self.query_one("#input", Input).focus()

        threading.Thread(target=self._run_chat, daemon=True, name="chat").start()

    def on_unmount(self) -> None:
        events.unsubscribe(self.bot.handle)
        events.CANCEL.set()
        self._lines.put(None)

    def on_resize(self, event: tevents.Resize) -> None:
        self._fit_layout_to(event.size.width)

    def _fit_layout_to(self, width: int) -> None:
        self.screen.set_class(width < NARROW_COLUMNS, "narrow")

    def _run_chat(self) -> None:
        try:
            self.chat(self._ask)
        except Exception:
            self._error = traceback.format_exc()

        try:
            self.call_from_thread(self.exit, self._error)
        except (RuntimeError, asyncio.CancelledError, concurrent.futures.CancelledError):
            pass

    def _ask(self, prompt: str) -> str:
        line = self._lines.get()

        if line is None:
            raise EOFError

        return line

    def _tick(self) -> None:
        t = time.monotonic()
        face = self.query_one("#face", Face)
        face.animate(t)
        face.refresh()

        for meter in self.query(VUMeter):
            meter.animate(t)
            meter.refresh()

        self.query_one("#status", Static).update(self._status_line())

    def _status_line(self) -> Text:
        colour = STATE_COLOURS.get(self.bot.state, STATE_COLOURS["idle"])
        status = Text(no_wrap=True, overflow="ellipsis")
        status.append("● ", Style(color=colour))
        status.append(self.bot.label(self.typed), Style(color="#e2e8f0"))
        status.append(f"  · {self.bot.mode} mode", Style(dim=True))

        return status

    def on_input_submitted(self, event: Input.Submitted) -> None:
        line = event.value
        event.input.value = ""

        if line.strip():
            self._write(f"you> {line.strip()}")

        if self.typed:
            self._lines.put(line)  # the loop reads typed lines through ask()
        elif line.strip():
            events.emit("typed", text=line.strip())  # always-on mode: the loop takes it instead of the microphone

    def on_print(self, event: tevents.Print) -> None:
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
    error = FaceApp(chat, typed=typed, wake_phrase=wake_phrase).run()

    if error:
        print(error, file=sys.stderr)


DEMO_PHRASE = "hey computer"
DEMO_SCRIPT = [
    ("asleep", 3.0, {"phrase": DEMO_PHRASE}, f'(asleep; say "{DEMO_PHRASE}")'),
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


def echo_typed_lines(ask: Callable[[str], str]) -> None:
    while True:
        try:
            line = ask("you> ")
        except EOFError:
            return

        print(f"(demo: you typed {line!r}; nothing listens in the demo)" if line else "(demo: Enter)")


def pretend_levels(state: str, rng: random.Random) -> tuple[float, float]:
    mic = out = rng.uniform(0.002, 0.006)

    if state == "recording" and rng.random() < 0.7:
        mic = rng.uniform(0.05, 0.4)

    if state == "speaking" and rng.random() < 0.75:
        out = rng.uniform(0.03, 0.3)

    return mic, out


def play_demo_step(state: str, seconds: float, data: dict, line: str, rng: random.Random) -> None:
    if line:
        print(line, end="" if line.endswith("> ") else "\n", flush=True)

    events.state(state, **data)
    end = time.monotonic() + seconds

    while time.monotonic() < end:
        mic, out = pretend_levels(state, rng)
        events.emit("level", mic=mic, out=out)
        time.sleep(0.1)


def demo(ask: Callable[[str], str]) -> None:
    threading.Thread(target=echo_typed_lines, args=(ask,), daemon=True).start()
    rng = random.Random(1)

    while True:
        for state, seconds, data, line in DEMO_SCRIPT:
            play_demo_step(state, seconds, data, line, rng)


if __name__ == "__main__":
    run_tui(demo, typed=True, wake_phrase=DEMO_PHRASE)

"""Take a picture with the webcam: the ``take_picture`` tool.

OpenCV (the headless build, no GUI) opens the default camera, lets the
auto-exposure settle over a few frames, grabs one, and saves it as a JPEG in
``webcam-pictures/`` named by date and time. The camera is released straight
after, so other apps can use it again. If OpenCV cannot open the camera and
ffmpeg is installed (``brew install ffmpeg``), ffmpeg takes the picture instead.

macOS asks once to allow the terminal (or editor) running the chatbot to use
the camera; if that was declined, allow it under System Settings > Privacy &
Security > Camera. ``device`` is the camera index in the order macOS lists
them; 0 is the built-in camera. Phones that offer Continuity Camera take the
next numbers.

Try it on its own:
  uv run -m images.webcam           # saves one picture and prints its path
  uv run -m images.webcam --device 1
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

PICTURES_DIR = Path("webcam-pictures")
WARMUP_FRAMES = 10  # auto-exposure and white balance need a few frames to settle
FFMPEG_WARMUP_FRAMES = 24  # the same, for the ffmpeg path (a frame count: a duration limit never stops a device)
DARK_THRESHOLD = 12.0  # mean brightness (0..255) below which a picture is reported as black


def take_picture(folder: str | Path = PICTURES_DIR, device: int = 0, warmup: int = WARMUP_FRAMES) -> dict:
    """Grab one frame from the webcam and save it; returns the path (or an error) for the model.

    The path is returned under the key ``image``, which tells the chatbot to
    show the picture to the model as well (see ``tools/registry.py``), so it can say
    what it sees.

    OpenCV does the capture. If it cannot open the camera and ffmpeg is
    installed, ffmpeg's AVFoundation input is tried instead: it is a separate
    program, so it can succeed where this process's camera access is blocked.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{datetime.now():%Y-%m-%d_%H-%M-%S}.jpg"
    via = "opencv"
    error = _capture_opencv(path, device, warmup)
    if error and shutil.which("ffmpeg"):
        via = "ffmpeg"
        error = _capture_ffmpeg(path, device) or ""
    if error:
        return {"ok": False, "error": error}
    width, height, brightness = _inspect(path)
    result = {"ok": True, "image": str(path), "width": width, "height": height, "via": via}
    if brightness < DARK_THRESHOLD:
        # A black frame is what you get from a covered camera, or from a camera that had no
        # time to set its exposure; say so, or the model will describe "a dark room".
        result["warning"] = "the picture is almost completely black; the camera may be covered or still waking up"
    return result


def _capture_opencv(path: Path, device: int, warmup: int) -> str:
    """Save one frame to ``path``; returns "" on success or the reason it failed."""
    import cv2  # slow import, keep it local so tools/registry.py stays light

    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        return (
            "could not open the webcam; another app may be using it, or camera access is denied for "
            "the terminal running the chatbot (allow it under System Settings > Privacy & Security > Camera)"
        )
    try:
        frame = None
        for _ in range(warmup + 1):
            ok, frame = cap.read()
            if not ok or frame is None:
                return "the webcam opened but delivered no picture"
    finally:
        cap.release()
    if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        return f"could not write {path}"
    return ""


def _capture_ffmpeg(path: Path, device: int) -> str:
    """The same with ffmpeg; ``nv12`` is a pixel format every Apple camera offers.

    ``FFMPEG_WARMUP_FRAMES`` frames are captured and the file is rewritten with
    each one, so the frame kept is the last: the very first frame comes out
    almost black before the auto-exposure has settled.
    """
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "avfoundation",
        "-framerate",
        "30",
        "-pixel_format",
        "nv12",
        "-i",
        str(device),
        "-frames:v",
        str(FFMPEG_WARMUP_FRAMES),
        "-update",
        "1",
        "-q:v",
        "2",
        str(path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    except subprocess.TimeoutExpired:
        return "ffmpeg timed out waiting for the webcam"
    if proc.returncode != 0 or not path.is_file():
        return f"ffmpeg could not take the picture: {proc.stderr.strip() or 'no output'}"
    return ""


def _inspect(path: Path) -> tuple[int, int, float]:
    """Width, height and mean brightness (0..255) of the saved picture."""
    from PIL import Image, ImageStat

    with Image.open(path) as img:
        width, height = img.size
        brightness = ImageStat.Stat(img.convert("L")).mean[0]
    return width, height, brightness


def main() -> None:
    parser = argparse.ArgumentParser(description="Take one picture with the webcam.")
    parser.add_argument("--device", type=int, default=0, help="camera index (default 0, the built-in camera)")
    parser.add_argument("--folder", default=str(PICTURES_DIR), help=f"where to save it (default {PICTURES_DIR})")
    args = parser.parse_args()
    result = take_picture(args.folder, args.device)
    print(result)
    raise SystemExit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()

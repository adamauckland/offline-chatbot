from __future__ import annotations

import argparse
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

PICTURES_DIR = Path("webcam-pictures")
WARMUP_FRAMES = 10
FFMPEG_WARMUP_FRAMES = 24
DARK_THRESHOLD = 12.0
JPEG_QUALITY = 92
FFMPEG_TIMEOUT_SECONDS = 20
BLACK_PICTURE_WARNING = "the picture is almost completely black; the camera may be covered or still waking up"
CAMERA_UNAVAILABLE = (
    "could not open the webcam; another app may be using it, or camera access is denied for "
    "the terminal running the chatbot (allow it under System Settings > Privacy & Security > Camera)"
)


def take_picture(folder: str | Path = PICTURES_DIR, device: int = 0, warmup: int = WARMUP_FRAMES) -> dict:
    path = new_picture_path(Path(folder))
    via, error = capture(path, device, warmup)

    if error:
        return {"ok": False, "error": error}

    width, height, brightness = picture_stats(path)
    result = {"ok": True, "image": str(path), "width": width, "height": height, "via": via}

    if brightness < DARK_THRESHOLD:
        result["warning"] = BLACK_PICTURE_WARNING

    return result


def new_picture_path(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)

    return folder / f"{datetime.now():%Y-%m-%d_%H-%M-%S}.jpg"


def capture(path: Path, device: int, warmup: int) -> tuple[str, str]:
    error = capture_with_opencv(path, device, warmup)

    if error and shutil.which("ffmpeg"):
        return "ffmpeg", capture_with_ffmpeg(path, device)

    return "opencv", error


def capture_with_opencv(path: Path, device: int, warmup: int) -> str:
    import cv2

    cap = cv2.VideoCapture(device)

    if not cap.isOpened():
        return CAMERA_UNAVAILABLE

    try:
        frame = last_frame_of(cap, warmup + 1)
    finally:
        cap.release()

    if frame is None:
        return "the webcam opened but delivered no picture"

    if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
        return f"could not write {path}"

    return ""


def last_frame_of(cap, count: int):
    frame = None

    for _ in range(count):
        ok, frame = cap.read()

        if not ok or frame is None:
            return None

    return frame


def ffmpeg_command(path: Path, device: int) -> list[str]:
    quiet = ["-hide_banner", "-loglevel", "error", "-y"]
    camera = ["-f", "avfoundation", "-framerate", "30", "-pixel_format", "nv12", "-i", str(device)]
    keep_last_frame = ["-frames:v", str(FFMPEG_WARMUP_FRAMES), "-update", "1", "-q:v", "2"]

    return ["ffmpeg", *quiet, *camera, *keep_last_frame, str(path)]


def capture_with_ffmpeg(path: Path, device: int) -> str:
    try:
        proc = subprocess.run(
            ffmpeg_command(path, device), capture_output=True, text=True, timeout=FFMPEG_TIMEOUT_SECONDS
        )
    except subprocess.TimeoutExpired:
        return "ffmpeg timed out waiting for the webcam"

    if proc.returncode != 0 or not path.is_file():
        return f"ffmpeg could not take the picture: {proc.stderr.strip() or 'no output'}"

    return ""


def picture_stats(path: Path) -> tuple[int, int, float]:
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

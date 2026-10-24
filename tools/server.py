#!/usr/bin/env python3
import os
import threading
from datetime import datetime

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="chatbot tools")

LIGHTS: dict[str, bool] = {"living room": False, "kitchen": False, "bedroom": False}
EXIT_DELAY_SECONDS = 0.3


class LightsRequest(BaseModel):
    state: str
    room: str = "living room"


@app.post("/lights")
def set_lights(req: LightsRequest) -> dict:
    room = req.room.lower().strip()
    rooms = list(LIGHTS) if room == "all" else [room]

    if any(r not in LIGHTS for r in rooms):
        return {"ok": False, "error": f"unknown room '{room}'", "rooms": list(LIGHTS) + ["all"]}

    on = req.state.lower() == "on"

    for r in rooms:
        LIGHTS[r] = on
        print(f"[tools_server] lights in {r} -> {'on' if on else 'off'}")

    return {"ok": True, "rooms": rooms, "state": "on" if on else "off"}


@app.get("/lights")
def get_lights() -> dict:
    return {room: ("on" if on else "off") for room, on in LIGHTS.items()}


@app.get("/time")
def get_time() -> dict:
    now = datetime.now()

    return {"time": now.strftime("%H:%M"), "date": now.strftime("%A %d %B %Y")}


@app.post("/shutdown")
def shutdown() -> dict:
    print("[tools_server] shutdown requested", flush=True)
    exit_after_reply_is_sent()

    return {"ok": True, "stopping": True}


def exit_after_reply_is_sent() -> None:
    threading.Timer(EXIT_DELAY_SECONDS, lambda: os._exit(0)).start()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)

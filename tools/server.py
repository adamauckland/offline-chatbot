#!/usr/bin/env python3
"""Example local API the chatbot can call. Run with:  uv run -m tools.server

Replace these endpoints with whatever actually does the work in your setup
(a smart-home bridge, a database, a shell script...). The chatbot only needs
each endpoint to accept JSON or query parameters and return JSON.
"""

import os
import threading
from datetime import datetime

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="chatbot tools")

# In-memory state standing in for real devices.
LIGHTS: dict[str, bool] = {"living room": False, "kitchen": False, "bedroom": False}


class LightsRequest(BaseModel):
    state: str  # "on" or "off"
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
    """Stop this server (the chatbot's "exit" tool calls it). Replies first, then exits."""
    print("[tools_server] shutdown requested", flush=True)
    # Hard exit after the reply has gone out: uvicorn's graceful shutdown can
    # wait on idle keep-alive connections, and this server holds no state worth saving.
    threading.Timer(0.3, lambda: os._exit(0)).start()
    return {"ok": True, "stopping": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)

"""Sprint 05/09 check — the web app without a browser.

Starts the real server on a free port, connects a real WebSocket client, pulls N
frames and reports what came down the wire, including whether Live ML (Sprint
09) said anything:

    uv run python scripts/check_web.py --frames 30
    uv run python scripts/check_web.py --model backend/models/gesture_classifier.joblib

Counterpart of `check_motion.py`: same pipeline, but it answers "does the stream
work, and how big is it" instead of "what is the arm doing". A browser can hide
a broken payload behind a pretty canvas; this cannot.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

import uvicorn

from backend.camera import CameraConfig
from backend.ml import DEFAULT_MODEL_PATH
from backend.vision import DEFAULT_FACE_MODEL_PATH
from backend.web import WebConfig, create_app


def free_port() -> int:
    """Ask the OS for an unused port instead of guessing one."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


async def pull(url: str, frames: int, timeout: float) -> dict[str, Any]:
    """Collect `frames` messages, or stop early on a pipeline error."""
    import websockets

    sizes: list[int] = []
    jpeg_bytes: list[int] = []
    fps_seen: list[float] = []
    detected = 0
    ml_signals = 0
    face_available = 0
    face_detected = 0

    async with websockets.connect(url, max_size=None) as connection:
        started = time.perf_counter()
        hello = json.loads(await asyncio.wait_for(connection.recv(), timeout))
        if hello.get("type") != "hello":
            return {"error": f"first message was {hello.get('type')!r}, expected 'hello'"}

        first_frame: Optional[float] = None
        for _ in range(frames):
            raw = await asyncio.wait_for(connection.recv(), timeout)
            message = json.loads(raw)
            if message.get("type") == "error":
                return {"error": message.get("message", "unknown pipeline error")}
            if first_frame is None:
                first_frame = time.perf_counter() - started
            sizes.append(len(raw))
            jpeg_bytes.append(message["image"]["bytes"])
            fps_seen.append(message.get("fps") or 0.0)
            detected += 1 if message.get("detected") else 0
            ml_signals += 1 if message.get("ml") else 0
            face = message.get("face") or {}
            face_available += 1 if face.get("available") else 0
            face_detected += 1 if face.get("detected") else 0

    return {
        "hello": hello,
        "frames": len(sizes),
        "startup_s": first_frame or 0.0,
        "detection_rate": detected / len(sizes) if sizes else 0.0,
        "ml_rate": ml_signals / len(sizes) if sizes else 0.0,
        "face_available_rate": face_available / len(sizes) if sizes else 0.0,
        "face_detection_rate": face_detected / face_available if face_available else 0.0,
        "fps": max(fps_seen) if fps_seen else 0.0,
        "message_kb": sum(sizes) / len(sizes) / 1024 if sizes else 0.0,
        "jpeg_kb": sum(jpeg_bytes) / len(jpeg_bytes) / 1024 if jpeg_bytes else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Headless check of the Sprint 05 web app.")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--preview-width", type=int, default=640)
    parser.add_argument("--quality", type=int, default=70)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--face", action="store_true", help="Sprint 11 facial signals — off by default"
    )
    parser.add_argument("--face-model", type=Path, default=DEFAULT_FACE_MODEL_PATH)
    parser.add_argument("--timeout", type=float, default=15.0, help="seconds to wait per message")
    args = parser.parse_args()

    port = free_port()
    app = create_app(
        WebConfig(
            camera=CameraConfig(args.camera, args.width, args.height),
            preview_width=args.preview_width,
            jpeg_quality=args.quality,
            model_path=args.model,
            face_enabled=args.face,
            face_model_path=args.face_model,
        )
    )
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, name="check-web-server", daemon=True)
    thread.start()

    deadline = time.monotonic() + 10.0
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.05)
    if not server.started:
        print("FAIL: the server did not start.", file=sys.stderr)
        return 1

    try:
        result = asyncio.run(pull(f"ws://127.0.0.1:{port}/ws", args.frames, args.timeout))
    except (asyncio.TimeoutError, TimeoutError):
        print(
            f"FAIL: no frame within {args.timeout:.0f}s. The camera is probably blocked — "
            "check System Settings > Privacy & Security > Camera.",
            file=sys.stderr,
        )
        return 1
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)

    if "error" in result:
        print(f"FAIL: {result['error']}", file=sys.stderr)
        return 1

    hello = result["hello"]
    print(f"handshake     {len(hello['chains'])} bones, {len(hello['feature_names'])} features")
    print(f"first frame   {result['startup_s']:.2f} s after connecting")
    print(f"frames        {result['frames']}  at {result['fps']:.1f} fps")
    print(f"detection     {result['detection_rate'] * 100:.0f}% of frames had a body")
    ml_status = f"{result['ml_rate'] * 100:.0f}% of frames" if hello["ml_classes"] else "no model"
    print(f"live ml       {ml_status}  (classes: {', '.join(hello['ml_classes']) or '-'})")
    if args.face:
        print(
            f"face          {result['face_available_rate'] * 100:.0f}% of frames available, "
            f"{result['face_detection_rate'] * 100:.0f}% of those had a face"
        )
    else:
        print("face          off (pass --face to enable)")
    print(f"payload       {result['message_kb']:.1f} KB/msg  (jpeg {result['jpeg_kb']:.1f} KB)")
    print(f"bandwidth     ~{result['message_kb'] * result['fps'] / 1024:.2f} MB/s at that rate")
    print("OK: the browser would have received a drawable stream.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

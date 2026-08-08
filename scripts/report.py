"""Sprint 10 — Motion Lab: metrics + explainability, in one report.

Acceptance criterion: "Eu abro um arquivo e vejo por que o modelo decide o que
decide, e se o sinal ao vivo bate com a regra que já tínhamos."

    uv run python scripts/train_model.py --simulate   # (or a real recording)
    uv run python scripts/report.py                   # writes the HTML report

Two independent halves. The training-time half (confusion matrix, per-class
precision/recall, feature importance) always renders — it is already sitting
in the model bundle from Sprint 08, no camera required. The live half (FPS,
inference time, detection rate, rule-vs-ML agreement) needs an actual camera
session; if the camera is busy or missing, this script says so and still
writes the report with just the training half — a report you can open is
better than a stack trace because your webcam was in use by another window.
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
from backend.gestures import GestureConfig
from backend.ml import (
    DEFAULT_MODEL_PATH,
    ModelError,
    compute_agreement,
    load_model,
    render_report,
)
from backend.web import WebConfig, create_app


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


async def pull_session(url: str, duration_s: float, timeout: float) -> dict[str, Any]:
    """Collect every `frame` message for `duration_s` seconds of wall clock."""
    import websockets

    frames: list[dict[str, Any]] = []
    async with websockets.connect(url, max_size=None) as connection:
        hello = json.loads(await asyncio.wait_for(connection.recv(), timeout))
        if hello.get("type") != "hello":
            return {"error": f"first message was {hello.get('type')!r}, expected 'hello'"}

        deadline = time.perf_counter() + duration_s
        while time.perf_counter() < deadline:
            raw = await asyncio.wait_for(connection.recv(), timeout)
            message = json.loads(raw)
            if message.get("type") == "error":
                return {"error": message.get("message", "unknown pipeline error")}
            if message.get("type") == "frame":
                frames.append(message)
    return {"frames": frames}


def summarize_live(
    frames: list[dict[str, Any]], open_threshold: float
) -> Optional[dict[str, Any]]:
    if not frames:
        return None
    fps_values = [f["fps"] for f in frames if f.get("fps") is not None]
    inference_values = sorted(
        f["inference_ms"] for f in frames if f.get("inference_ms") is not None
    )
    p95_index = max(0, round(0.95 * len(inference_values)) - 1) if inference_values else 0
    duration = frames[-1]["timestamp"] - frames[0]["timestamp"] if len(frames) > 1 else 0.0

    return {
        "frames": len(frames),
        "duration_s": duration,
        "fps_mean": sum(fps_values) / len(fps_values) if fps_values else 0.0,
        "inference_ms_mean": (
            sum(inference_values) / len(inference_values) if inference_values else 0.0
        ),
        "inference_ms_p95": inference_values[p95_index] if inference_values else 0.0,
        "detection_rate": sum(1 for f in frames if f.get("detected")) / len(frames),
        "ml_rate": sum(1 for f in frames if f.get("ml")) / len(frames),
        "agreement": compute_agreement(frames, open_threshold),
    }


def run_live_session(
    model_path: Path, camera: int, width: int, height: int, seconds: float
) -> dict[str, Any]:
    """Self-host the real web app on a free port and sample it like a client would."""
    port = free_port()
    app = create_app(WebConfig(camera=CameraConfig(camera, width, height), model_path=model_path))
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, name="report-server", daemon=True)
    thread.start()
    # A little longer than the sampling window: startup (camera open, first
    # inference) eats into it otherwise and the sample runs short.
    timeout = seconds + 15.0
    try:
        deadline = time.monotonic() + 10.0
        while not server.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.05)
        if not server.started:
            return {"error": "the server did not start"}
        return asyncio.run(pull_session(f"ws://127.0.0.1:{port}/ws", seconds, timeout))
    except (asyncio.TimeoutError, TimeoutError):
        return {"error": f"no frame within {timeout:.0f}s — camera busy or blocked"}
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render the Motion Lab metrics + explainability report."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--out", type=Path, default=None, help="default: <model>.report.html")
    parser.add_argument("--seconds", type=float, default=8.0, help="live sampling window")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--no-live", action="store_true", help="skip the camera session entirely")
    args = parser.parse_args()

    try:
        bundle = load_model(args.model)
    except ModelError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    training = {
        "trained_at": bundle["trained_at"],
        "n_samples": bundle["n_samples"],
        "counts": bundle["counts"],
        "accuracy": bundle["accuracy"],
        "feature_importances": bundle["feature_importances"],
        "class_labels": bundle["classes"],
        "confusion_matrix": bundle["confusion_matrix"],
        "per_class": bundle["per_class"],
    }

    live = None
    if args.no_live:
        print("Skipping the live session (--no-live).")
    else:
        print(f"Sampling a live session for {args.seconds:.0f}s...")
        result = run_live_session(args.model, args.camera, args.width, args.height, args.seconds)
        if "error" in result:
            print(f"WARN: live session unavailable ({result['error']}) — training-only report.")
        else:
            live = summarize_live(result["frames"], GestureConfig().open_threshold)
            print(f"Sampled {live['frames']} frames." if live else "No frames arrived in time.")

    out_path = args.out or args.model.with_name(args.model.stem + ".report.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_report(training, live))

    print(f"\naccuracy   {training['accuracy'] * 100:.1f}%  ({training['n_samples']} samples)")
    if live and live["agreement"]["rate"] is not None:
        print(f"agreement  {live['agreement']['rate'] * 100:.0f}% (rule vs ML)")
    print(f"report     {out_path}")
    print("OK: open it in a browser. No image was used to generate it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
from backend.dataset.stats import dataset_stats
from backend.dataset.writer import DEFAULT_DATASET_PATH
from backend.gestures import GestureConfig
from backend.ml import (
    DEFAULT_MODEL_PATH,
    DEFAULT_V2_MODEL_PATH,
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


def _training_summary(bundle: dict[str, Any]) -> dict[str, Any]:
    """Select report fields from a validated v1 or v2 model bundle."""
    return {
        "trained_at": bundle["trained_at"],
        "n_samples": bundle["n_samples"],
        "counts": bundle["counts"],
        "accuracy": bundle["accuracy"],
        "macro_f1": bundle.get("macro_f1", 0.0),
        "balanced_accuracy": bundle.get("balanced_accuracy", 0.0),
        "feature_importances": bundle["feature_importances"],
        "class_labels": bundle["classes"],
        "confusion_matrix": bundle["confusion_matrix"],
        "per_class": bundle["per_class"],
        "cv": bundle.get("cv"),
    }


def _comparison_summary(bundle: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": bundle.get("schema_version", 1),
        "n_samples": bundle.get("n_samples", 0),
        "classes": bundle.get("classes", []),
        "cv": bundle.get("cv"),
        "window_seconds": bundle.get("window_seconds"),
        "max_gap": bundle.get("max_gap"),
        "min_samples": bundle.get("min_samples"),
        "deadband": bundle.get("deadband"),
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
    parser.add_argument("--model", type=Path, default=None, help="v1 model (default: standard v1 bundle)")
    parser.add_argument(
        "--model-v2",
        type=Path,
        default=None,
        help="v2 model (default: standard v2 bundle)",
    )
    parser.add_argument("--out", type=Path, default=None, help="default: <model>.report.html")
    parser.add_argument("--seconds", type=float, default=8.0, help="live sampling window")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--no-live", action="store_true", help="skip the camera session entirely")
    args = parser.parse_args()

    v1_path = args.model or DEFAULT_MODEL_PATH
    v2_path = args.model_v2 or DEFAULT_V2_MODEL_PATH
    bundles: dict[str, dict[str, Any]] = {}
    paths = {"v1": (v1_path, 1), "v2": (v2_path, 2)}
    try:
        for name, (path, schema) in paths.items():
            if path.exists():
                bundles[name] = load_model(path, schema=schema)
    except ModelError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    if not bundles:
        print(
            f"FAIL: no model bundle found at {v1_path} or {v2_path}. Train schema v1 or v2 first.",
            file=sys.stderr,
        )
        return 1

    # The existing --model path remains the v1 report/live target. If only a
    # v2 bundle exists, use it for the training report and keep live sampling
    # disabled because PipelineRunner has no temporal-window inference yet.
    primary_name = "v1" if "v1" in bundles else "v2"
    bundle = bundles[primary_name]
    model_path = paths[primary_name][0]
    training = _training_summary(bundle)
    schema_comparison = {
        name: _comparison_summary(bundles[name]) if name in bundles else None
        for name in ("v1", "v2")
    }

    dataset_path = Path(bundle.get("dataset_path", DEFAULT_DATASET_PATH))
    dataset = dataset_stats(dataset_path)

    live = None
    if primary_name == "v2":
        print("Skipping live inference: schema v2 still needs temporal-window support in PipelineRunner.")
    elif args.no_live:
        print("Skipping the live session (--no-live).")
    else:
        print(f"Sampling a live session for {args.seconds:.0f}s...")
        result = run_live_session(model_path, args.camera, args.width, args.height, args.seconds)
        if "error" in result:
            print(f"WARN: live session unavailable ({result['error']}) — training-only report.")
        else:
            live = summarize_live(result["frames"], GestureConfig().open_threshold)
            print(f"Sampled {live['frames']} frames." if live else "No frames arrived in time.")

    out_path = args.out or model_path.with_name(model_path.stem + ".report.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        render_report(training, live, dataset, schema_comparison=schema_comparison)
    )

    for name in ("v1", "v2"):
        if name not in bundles:
            print(f"{name}: bundle indisponível")
            continue
        summary = schema_comparison[name]
        assert summary is not None
        cv_summary = summary.get("cv") or {}
        mean, std = cv_summary.get("macro_f1_mean"), cv_summary.get("macro_f1_std")
        score = "indisponível" if mean is None or std is None else f"{mean:.3f} ± {std:.3f}"
        print(f"{name}: macro-F1 CV {score} ({summary['n_samples']} amostras/janelas)")

    cv = training.get("cv")
    if cv and cv.get("macro_f1_oof") is not None:
        print(f"\nmacro-F1 LOSO  {cv['macro_f1_oof']:.3f}  (pooled out-of-fold predictions)")
    if cv and cv.get("macro_f1_mean") is not None and cv.get("macro_f1_std") is not None:
        print(
            f"\nmacro-F1 CV  {cv['macro_f1_mean']:.3f} ± {cv['macro_f1_std']:.3f}  "
            f"({cv['folds_used']}/{cv['n_splits']} folds; {cv['folds_skipped']} skipped)"
        )
    else:
        print("\nmacro-F1 CV  unavailable (model predates grouped cross-validation)")
    print(f"split acc    {training['accuracy'] * 100:.1f}%  ({training['n_samples']} samples/janelas)")
    if live and live["agreement"]["rate"] is not None:
        print(f"agreement  {live['agreement']['rate'] * 100:.0f}% (rule vs ML)")
    print(f"report     {out_path}")
    print("OK: open it in a browser. No image was used to generate it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

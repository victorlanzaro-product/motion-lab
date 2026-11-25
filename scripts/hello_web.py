"""Sprint 05 — Web App.

Acceptance criterion: "Vejo a câmera, o esqueleto e os números do pipeline no
navegador, ao vivo."

    uv run python scripts/hello_web.py            # http://127.0.0.1:8000
    uv run python scripts/hello_web.py --open     # e abre o navegador

Same pipeline as `hello_motion.py`, different front end: OpenCV drew the overlay
into the frame, here the browser draws it from the landmark payload. The camera
only turns on once a page is actually connected.
"""

from __future__ import annotations

import argparse
import ipaddress
import sys
import webbrowser

import uvicorn

from pathlib import Path

from backend.camera import CameraConfig
from backend.dataset.writer import DEFAULT_DATASET_PATH, DEFAULT_V2_DATASET_PATH
from backend.vision import (
    DEFAULT_FACE_MODEL_PATH,
    FaceModelError,
    PoseModelError,
    ensure_face_model,
    ensure_model,
)
from backend.web import WebConfig, create_app


def main() -> int:
    parser = argparse.ArgumentParser(description="Motion Lab live view in the browser.")
    # Loopback by default, and deliberately so: this endpoint streams a webcam
    # with no authentication. Binding 0.0.0.0 puts your living room on the LAN.
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--allow-lan",
        action="store_true",
        help="allow binding beyond loopback; requires at least one explicit --allowed-origin",
    )
    parser.add_argument(
        "--allowed-origin",
        action="append",
        default=[],
        help="exact browser Origin allowed to connect; repeat for multiple Origins",
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--preview-width",
        type=int,
        default=640,
        help="width of the JPEG sent to the browser; inference still uses the full frame",
    )
    parser.add_argument("--quality", type=int, default=70, help="JPEG quality, 1-100")
    parser.add_argument(
        "--model-schema",
        type=int,
        choices=(1, 2),
        default=1,
        help="model/data schema (1: per-frame, 2: temporal windows)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="override the schema-specific model bundle; missing is fine, Live ML stays off",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help=(
            "override the schema-specific recording dataset "
            f"(v1: {DEFAULT_DATASET_PATH}; v2: {DEFAULT_V2_DATASET_PATH})"
        ),
    )
    parser.add_argument(
        "--face",
        action="store_true",
        help="Sprint 11 facial signals (smile, blink, brow, head pose) — off by default",
    )
    parser.add_argument("--face-model", type=Path, default=DEFAULT_FACE_MODEL_PATH)
    parser.add_argument("--open", action="store_true", help="open the browser on startup")
    args = parser.parse_args()

    try:
        host_is_loopback = ipaddress.ip_address(args.host).is_loopback
    except ValueError:
        host_is_loopback = args.host.lower() == "localhost"
    if not host_is_loopback and not args.allow_lan:
        parser.error("non-loopback --host requires --allow-lan")
    if not host_is_loopback and not args.allowed_origin:
        parser.error("--allow-lan requires at least one explicit --allowed-origin")
    origin_host = f"[{args.host}]" if ":" in args.host else args.host
    bound_origin = f"http://{origin_host}:{args.port}"
    allowed_origins = tuple(args.allowed_origin) or (
        bound_origin,
        f"http://127.0.0.1:{args.port}",
        f"http://localhost:{args.port}",
        f"http://[::1]:{args.port}",
    )

    config = WebConfig(
        camera=CameraConfig(args.camera, args.width, args.height),
        preview_width=args.preview_width,
        jpeg_quality=args.quality,
        model_path=args.model,
        model_schema=args.model_schema,
        dataset_path=args.dataset,
        face_enabled=args.face,
        face_model_path=args.face_model,
        allowed_origins=allowed_origins,
    )

    try:
        # Fetch the model now rather than on the first connection: a 5.5 MB
        # download would otherwise look like a page that just never loads.
        ensure_model(config.pose.model_path)
    except PoseModelError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    if args.face:
        try:
            ensure_face_model(config.face_model_path)
        except FaceModelError as exc:
            # Face is optional, never a dependency of the rest of the
            # pipeline (ARQUITETURA FACIAL: falha degradada) — warn and keep
            # going with pose/gestures/ML/web fully working.
            print(f"WARN: face model unavailable, Sprint 11 signals stay off: {exc}")

    display_host = f"[{args.host}]" if ":" in args.host else args.host
    url = f"http://{display_host}:{args.port}"
    print(f"Motion Lab web app on {url} — the camera starts when a page connects.")
    print("Ctrl+C to stop. No video is written to disk.")
    if args.open:
        webbrowser.open(url)

    try:
        uvicorn.run(create_app(config), host=args.host, port=args.port, log_level="warning")
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

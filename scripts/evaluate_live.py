#!/usr/bin/env python3
"""Sprint 16 — local live evaluation (JSON offline or interactive capture).

Offline:

    uv run python scripts/evaluate_live.py \\
        --model-v1 models/gesture_classifier.joblib \\
        --model-v2 models/gesture_classifier_v2.joblib \\
        --frames data/eval/session.json \\
        --segments data/eval/segments.json

Interactive default (validates bundles before camera/pose):

    uv run python scripts/evaluate_live.py \\
        --model-v1 models/gesture_classifier.joblib \\
        --model-v2 models/gesture_classifier_v2.joblib \\

Pass --frames and --segments to use offline JSON mode. Validates both bundles
first (Portuguese error on stderr). Prints one JSON summary to stdout — counts
and rates only, no images or facial data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from backend.camera import CameraConfig, CameraEngine
from backend.gestures import GestureConfig
from backend.ml.live_capture import run_interactive_evaluation
from backend.ml.live_eval import evaluate_from_json, validate_bundles
from backend.ml.train import ModelError
from backend.vision import PoseConfig, PoseDetector


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Avaliação local v1/v2/regra (JSON gravado ou captura interativa)."
    )
    parser.add_argument("--model-v1", type=Path, required=True, help="Bundle schema 1")
    parser.add_argument("--model-v2", type=Path, required=True, help="Bundle schema 2")
    parser.add_argument(
        "--frames",
        type=Path,
        help="JSON com lista de frames (features) — modo offline",
    )
    parser.add_argument(
        "--segments",
        type=Path,
        help="JSON com trechos e rótulo esperado — modo offline",
    )
    parser.add_argument(
        "--capture",
        action="store_true",
        help="Captura interativa na mesma sessão de câmera (sem persistir pixels)",
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--open-threshold", type=float, default=GestureConfig().open_threshold)
    parser.add_argument("--default-warmup", type=int, default=0)
    parser.add_argument("--default-transition", type=int, default=0)
    args = parser.parse_args(argv)

    try:
        v1_bundle, v2_bundle = validate_bundles(args.model_v1, args.model_v2)
    except ModelError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    gesture = GestureConfig(open_threshold=args.open_threshold)

    if args.capture or (not args.frames and not args.segments):
        if args.frames or args.segments:
            print(
                "Use --capture sozinho ou --frames com --segments; não combine os modos.",
                file=sys.stderr,
            )
            return 1
        from backend.ml.live_capture import StdinCapturePrompter

        prompter = StdinCapturePrompter(
            default_warmup=args.default_warmup,
            default_transition=args.default_transition,
        )
        camera = CameraEngine(config=CameraConfig(index=args.camera_index))
        detector = PoseDetector(config=PoseConfig())
        try:
            report = run_interactive_evaluation(
                v1_bundle,
                v2_bundle,
                camera,
                detector,
                prompter,
                gesture_config=gesture,
            )
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1
    else:
        if not args.frames or not args.segments:
            print(
                "Informe --frames e --segments para modo offline; sem esses arquivos, a captura interativa é iniciada.",
                file=sys.stderr,
            )
            return 1
        if not args.frames.is_file():
            print(f"Arquivo de frames não encontrado: {args.frames}", file=sys.stderr)
            return 1
        if not args.segments.is_file():
            print(f"Arquivo de segmentos não encontrado: {args.segments}", file=sys.stderr)
            return 1

        frames_json = json.loads(args.frames.read_text(encoding="utf-8"))
        segments_json = json.loads(args.segments.read_text(encoding="utf-8"))
        if not isinstance(frames_json, list) or not isinstance(segments_json, list):
            print("frames e segments devem ser listas JSON.", file=sys.stderr)
            return 1

        report = evaluate_from_json(
            v1_bundle,
            v2_bundle,
            frames_json,
            segments_json,
            gesture_config=gesture,
        )

    json.dump(report, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

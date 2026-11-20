#!/usr/bin/env python3
"""Sprint 16 — local live evaluation without opening a camera.

    uv run python scripts/evaluate_live.py \\
        --model-v1 models/gesture_classifier.joblib \\
        --model-v2 models/gesture_classifier_v2.joblib \\
        --frames data/eval/session.json \\
        --segments data/eval/segments.json

Validates both bundles first (Portuguese error on stderr). Prints one JSON
summary to stdout — counts and rates only, no images or facial data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from backend.gestures import GestureConfig
from backend.ml.live_eval import evaluate_from_json, validate_bundles
from backend.ml.train import ModelError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Avaliação local v1/v2/regra sobre frames gravados.")
    parser.add_argument("--model-v1", type=Path, required=True, help="Bundle schema 1")
    parser.add_argument("--model-v2", type=Path, required=True, help="Bundle schema 2")
    parser.add_argument("--frames", type=Path, required=True, help="JSON com lista de frames (features)")
    parser.add_argument("--segments", type=Path, required=True, help="JSON com trechos e rótulo esperado")
    parser.add_argument("--open-threshold", type=float, default=GestureConfig().open_threshold)
    args = parser.parse_args(argv)

    try:
        v1_bundle, v2_bundle = validate_bundles(args.model_v1, args.model_v2)
    except ModelError as exc:
        print(str(exc), file=sys.stderr)
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

    gesture = GestureConfig(open_threshold=args.open_threshold)
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

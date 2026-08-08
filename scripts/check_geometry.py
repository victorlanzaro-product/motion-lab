"""Headless feature check — Sprint 03 acceptance evidence.

Prints the feature row the ML layer will eventually receive.

    uv run python scripts/check_geometry.py --image some_photo.jpg   # deterministic
    uv run python scripts/check_geometry.py --frames 30              # live webcam
"""

from __future__ import annotations

import argparse
import sys

import cv2

from backend.camera import CameraConfig, CameraEngine, CameraError
from backend.camera.capture import Frame
from backend.features import FEATURE_NAMES, extract
from backend.vision import PoseConfig, PoseDetector, PoseModelError, ensure_model


def show(features) -> None:
    print(f"  frame {features.frame_index}  complete={features.complete}")
    for name in FEATURE_NAMES:
        value = getattr(features, name)
        if value is None:
            rendered = "-        (joint not visible)"
        elif isinstance(value, bool):
            rendered = "yes" if value else "no"
        else:
            rendered = f"{value:8.2f}"
        print(f"    {name:<28} {rendered}")


def from_image(path: str, detector: PoseDetector) -> int:
    image = cv2.imread(path)
    if image is None:
        print(f"FAIL: could not read image {path}", file=sys.stderr)
        return 1
    snapshot = detector.detect(Frame(image=image, index=0, timestamp=1.0))
    if not snapshot.detected:
        print(f"FAIL: no body found in {path}", file=sys.stderr)
        return 1
    print(f"Feature check (static image {path}, {image.shape[1]}x{image.shape[0]}) ------")
    show(extract(snapshot))
    print("OK: features computed.")
    return 0


def from_camera(args, detector: PoseDetector) -> int:
    camera = CameraEngine(config=CameraConfig(args.camera, args.width, args.height))
    with camera:
        last = None
        complete = 0
        for frame in camera.frames(limit=args.frames):
            features = extract(detector.detect(frame))
            complete += int(features.complete)
            last = features
        if last is None:
            print("FAIL: no frames captured.", file=sys.stderr)
            return 1
        print(f"Feature check (webcam, {camera.frame_count} frames) -----------")
        print(f"  complete rows    : {complete}/{camera.frame_count}")
        print("  last frame:")
        show(last)
        if complete == 0:
            print("FAIL: never saw all six arm joints at once.", file=sys.stderr)
            return 1
        print("OK: features computed.")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Print the per-frame feature row.")
    parser.add_argument("--image", help="run on a still image instead of the webcam")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    try:
        with PoseDetector(config=PoseConfig(model_path=ensure_model())) as detector:
            if args.image:
                return from_image(args.image, detector)
            return from_camera(args, detector)
    except (CameraError, PoseModelError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

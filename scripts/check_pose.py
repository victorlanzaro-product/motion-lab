"""Headless pose check — Sprint 02 acceptance evidence, no GUI window.

Runs the real detector over N webcam frames and reports how often a body was
found and how confident each arm joint was.

    uv run python scripts/check_pose.py --frames 60
"""

from __future__ import annotations

import argparse
import statistics
import sys

from backend.camera import CameraConfig, CameraEngine, CameraError
from backend.vision import (
    ARM_LANDMARKS,
    PoseConfig,
    PoseDetector,
    PoseModelError,
    ensure_model,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check pose detection without opening a window.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    try:
        model_path = ensure_model()
        camera = CameraEngine(config=CameraConfig(args.camera, args.width, args.height))
        with camera, PoseDetector(config=PoseConfig(model_path=model_path)) as detector:
            detected = 0
            inference_ms: list[float] = []
            visibility: dict[int, list[float]] = {int(lm): [] for lm in ARM_LANDMARKS}

            for frame in camera.frames(limit=args.frames):
                snapshot = detector.detect(frame)
                inference_ms.append(snapshot.inference_ms)
                if not snapshot.detected:
                    continue
                detected += 1
                for landmark in ARM_LANDMARKS:
                    point = snapshot.point(landmark)
                    if point is not None:
                        visibility[int(landmark)].append(point.visibility)

            total = camera.frame_count
            print("Pose check --------------------------------------")
            print(f"  model            : {model_path}")
            print(f"  frames processed : {total}")
            print(f"  body detected    : {detected}/{total} ({detected / max(total, 1):.0%})")
            print(
                f"  inference        : {statistics.mean(inference_ms):.1f} ms avg, "
                f"{max(inference_ms):.1f} ms worst"
            )
            print(f"  pipeline FPS     : {camera.fps:.1f}")
            print("  arm visibility (mean over detected frames):")
            for landmark in ARM_LANDMARKS:
                values = visibility[int(landmark)]
                mean = statistics.mean(values) if values else 0.0
                bar = "#" * int(mean * 20)
                print(f"    {landmark.name:<16} {mean:.2f}  {bar}")
            print("  video stored     : no")

            if detected == 0:
                print(
                    "FAIL: no body detected. Stand in front of the camera, "
                    "framed from the hips up, with the room reasonably lit.",
                    file=sys.stderr,
                )
                return 1
            print("OK: pose landmarks are being tracked.")
            return 0
    except (CameraError, PoseModelError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Headless camera check — Sprint 01 acceptance evidence, no GUI window.

Grabs N frames and reports what the driver actually delivered. Useful on macOS
where a missing camera permission looks identical to "no camera" unless you
inspect the result.

    uv run python scripts/check_camera.py --frames 60
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from backend.camera import CameraConfig, CameraEngine, CameraError


def main() -> int:
    parser = argparse.ArgumentParser(description="Check webcam access without opening a window.")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default: 0)")
    parser.add_argument("--frames", type=int, default=60, help="frames to grab (default: 60)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    config = CameraConfig(index=args.camera, width=args.width, height=args.height)

    try:
        with CameraEngine(config=config) as camera:
            granted_w, granted_h = camera.resolution
            brightness: list[float] = []
            last = None
            for frame in camera.frames(limit=args.frames):
                brightness.append(float(np.mean(frame.image)))
                last = frame

            if last is None:
                print("FAIL: camera opened but delivered no frames.", file=sys.stderr)
                return 1

            print("Camera check ------------------------------------")
            print(f"  index            : {config.index}")
            print(f"  requested        : {config.width}x{config.height} @ {config.target_fps}fps")
            print(f"  granted          : {granted_w}x{granted_h}")
            print(f"  frame shape      : {last.width}x{last.height} (BGR, {last.image.dtype})")
            print(f"  frames captured  : {camera.frame_count}")
            print(f"  measured FPS     : {camera.fps:.1f}")
            print(f"  mean brightness  : {np.mean(brightness):.1f} / 255")
            print("  video stored     : no (frames stay in memory)")
            print("OK: webcam is reachable.")
            return 0
    except CameraError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

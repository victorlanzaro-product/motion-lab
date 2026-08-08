"""Headless motion check — Sprint 04 acceptance evidence.

Criterion: "O sistema consegue dizer se meu braço esta parado, subindo ou
descendo."

    uv run python scripts/check_motion.py --simulate   # deterministic, no camera
    uv run python scripts/check_motion.py --frames 90  # live webcam

`--simulate` drives the tracker with a scripted arm: still, then up, then still
at the top, then down. It proves the verdict logic end to end without needing a
body in front of the lens, and it always produces the same output.
"""

from __future__ import annotations

import argparse
import sys

from backend.camera import CameraConfig, CameraEngine, CameraError
from backend.features import Direction, FrameFeatures, MotionTracker, extract
from backend.vision import PoseConfig, PoseDetector, PoseModelError, ensure_model

FPS = 30.0

#: (label, seconds, height at the start, height at the end)
SCRIPT = [
    ("arm down, held", 0.6, 0.00, 0.00),
    ("raising", 0.8, 0.00, 1.20),
    ("held up", 0.6, 1.20, 1.20),
    ("lowering", 0.8, 1.20, 0.00),
]


def simulate() -> int:
    """Replay a scripted gesture and print what the tracker made of each phase."""
    tracker = MotionTracker()
    print("Motion check (simulated, no camera) --------------------")
    print(f"  window {tracker.config.window_seconds}s   "
          f"dead zone {tracker.config.still_threshold} widths/s   {FPS:.0f} fps")

    expected = {
        "arm down, held": Direction.STILL,
        "raising": Direction.UP,
        "held up": Direction.STILL,
        "lowering": Direction.DOWN,
    }
    index = 0
    clock = 0.0
    failures = 0

    for label, seconds, start, end in SCRIPT:
        steps = int(seconds * FPS)
        verdicts: list[Direction] = []
        velocity = None
        for step in range(steps):
            progress = step / steps
            height = start + (end - start) * progress
            state = tracker.update(
                FrameFeatures(
                    frame_index=index,
                    timestamp=clock,
                    left_wrist_height=height,
                    left_wrist_offset_x=0.0,
                    left_elbow_angle=90.0,
                )
            )
            verdicts.append(state.left.direction)
            velocity = state.left.velocity_y
            index += 1
            clock += 1 / FPS

        # The first frames of a phase still carry the previous phase inside the
        # window, so the settled verdict is the one from the second half.
        settled = verdicts[len(verdicts) // 2 :]
        winner = max(set(settled), key=settled.count)
        ok = winner is expected[label]
        failures += int(not ok)
        speed = "   -  " if velocity is None else f"{velocity:+6.2f}"
        print(f"  {label:<16} -> {winner.value:<7} ({speed} widths/s)  {'ok' if ok else 'MISMATCH'}")

    if failures:
        print(f"FAIL: {failures} phase(s) misread.", file=sys.stderr)
        return 1
    print("OK: still / up / down all reported correctly.")
    return 0


def live(args) -> int:
    tracker = MotionTracker()
    camera = CameraEngine(config=CameraConfig(args.camera, args.width, args.height))
    try:
        with camera, PoseDetector(config=PoseConfig(model_path=ensure_model())) as detector:
            print(f"Motion check (webcam, {args.frames} frames) ------------")
            print("  move one arm up and down while this runs.")
            counts: dict[str, int] = {}
            peak = 0.0
            for frame in camera.frames(limit=args.frames):
                state = tracker.update(extract(detector.detect(frame)))
                for arm in (state.left, state.right):
                    counts[arm.direction.value] = counts.get(arm.direction.value, 0) + 1
                    if arm.speed is not None:
                        peak = max(peak, arm.speed)
            total = sum(counts.values()) or 1
            for name in ("up", "down", "still", "unknown"):
                share = counts.get(name, 0) / total
                print(f"  {name:<8} {counts.get(name, 0):4d}  {'#' * int(share * 40)}")
            print(f"  peak speed     : {peak:.2f} shoulder-widths/s")
            if counts.get("unknown", 0) == total:
                print("FAIL: never had enough history — was anyone in frame?", file=sys.stderr)
                return 1
            print("OK: motion tracked. No video written to disk.")
            return 0
    except (CameraError, PoseModelError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Report arm direction over time.")
    parser.add_argument("--simulate", action="store_true", help="scripted gesture, no camera")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    return simulate() if args.simulate else live(args)


if __name__ == "__main__":
    raise SystemExit(main())

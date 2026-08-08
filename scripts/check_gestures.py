"""Headless gesture check — Sprint 06 acceptance evidence.

Criterion: "O sistema reconhece braço levantado, aceno e postura de braços
cruzados/abertos como eventos distintos, um por vez, não um enxame."

    uv run python scripts/check_gestures.py --simulate   # deterministic, no camera
    uv run python scripts/check_gestures.py --frames 300 # live webcam

`--simulate` drives the engine with scripted postures end to end: it proves
edge-triggering, the hold debounce and the cooldown without needing a body in
front of the lens, and it always produces the same output.
"""

from __future__ import annotations

import argparse
import sys

from backend.camera import CameraConfig, CameraEngine, CameraError
from backend.features import ArmMotion, Direction, FrameFeatures, MotionState, MotionTracker, extract
from backend.gestures import GestureConfig, GestureEngine
from backend.vision import PoseConfig, PoseDetector, PoseModelError, ensure_model

FPS = 30.0

#: (label, seconds, feature overrides held for the whole phase, expected gesture or None)
SCRIPT = [
    ("arms down", 0.3, {}, None),
    ("arm raised, held", 1.0, {"left_wrist_above_shoulder": True}, "arm_raised"),
    ("still raised (cooldown)", 1.0, {"left_wrist_above_shoulder": True}, None),
    ("arm lowered", 0.3, {}, None),
    ("arms crossed", 1.0, {"wrists_crossed": True}, "arms_crossed"),
    ("arms open", 1.0, {"wrist_distance": 2.6}, "arms_open"),
]


def _still_motion(index: int, timestamp: float) -> MotionState:
    """No arm swinging in this script; the wave rule stays silent throughout."""
    return MotionState(
        frame_index=index,
        timestamp=timestamp,
        left=ArmMotion(direction=Direction.STILL),
        right=ArmMotion(direction=Direction.STILL),
    )


def simulate() -> int:
    """Replay scripted postures and print what the engine made of each phase."""
    config = GestureConfig(hold_seconds=0.4, cooldown_seconds=2.0)
    engine = GestureEngine(config)
    print("Gesture check (simulated, no camera) --------------------")
    print(f"  hold {config.hold_seconds}s   cooldown {config.cooldown_seconds}s   {FPS:.0f} fps")

    failures = 0
    clock = 0.0
    index = 0

    for label, seconds, overrides, expected in SCRIPT:
        fired: list[str] = []
        for _ in range(int(seconds * FPS)):
            events = engine.update(
                FrameFeatures(frame_index=index, timestamp=clock, **overrides),
                _still_motion(index, clock),
            )
            fired += [event.name for event in events]
            index += 1
            clock += 1 / FPS

        ok = fired == ([expected] if expected else [])
        failures += int(not ok)
        shown = ", ".join(fired) if fired else "-"
        print(f"  {label:<24} -> {shown:<14} {'ok' if ok else 'MISMATCH'}")

    if failures:
        print(f"FAIL: {failures} phase(s) misread.", file=sys.stderr)
        return 1
    print("OK: each posture produced exactly the gesture it should, once.")
    return 0


def live(args) -> int:
    tracker = MotionTracker()
    engine = GestureEngine()
    camera = CameraEngine(config=CameraConfig(args.camera, args.width, args.height))
    try:
        with camera, PoseDetector(config=PoseConfig(model_path=ensure_model())) as detector:
            print(f"Gesture check (webcam, {args.frames} frames) ------------")
            print("  raise an arm, wave, cross or open your arms while this runs.")
            counts: dict[str, int] = {}
            for frame in camera.frames(limit=args.frames):
                features = extract(detector.detect(frame))
                motion = tracker.update(features)
                for event in engine.update(features, motion):
                    key = f"{event.name}:{event.side or 'both'}"
                    counts[key] = counts.get(key, 0) + 1
                    print(f"  frame {event.frame_index:5d}  t={event.timestamp:6.2f}s  {key}")

            if not counts:
                print("FAIL: no gesture fired — was anyone in frame long enough?", file=sys.stderr)
                return 1
            print("  totals:", ", ".join(f"{k}={v}" for k, v in counts.items()))
            print("OK: gestures detected. No video written to disk.")
            return 0
    except (CameraError, PoseModelError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Report gesture events over time.")
    parser.add_argument("--simulate", action="store_true", help="scripted postures, no camera")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    return simulate() if args.simulate else live(args)


if __name__ == "__main__":
    raise SystemExit(main())

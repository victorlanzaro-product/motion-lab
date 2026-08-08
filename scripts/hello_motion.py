"""Sprint 04 — Motion.

Acceptance criterion: "O sistema consegue dizer se meu braço esta parado,
subindo ou descendo."

    uv run python scripts/hello_motion.py     # q/ESC quits, t toggles the trail

The wrist trail is drawn in image space so you can see the trajectory, but the
verdict on the HUD comes from body-space velocity — walk towards the camera and
the trail moves while the verdict stays "still", which is the point.
"""

from __future__ import annotations

import argparse
import sys

import cv2

from backend.camera import CameraConfig, CameraEngine, CameraError, mirror
from backend.features import Direction, MotionTracker, Trail, extract
from backend.vision import (
    PoseConfig,
    PoseDetector,
    PoseLandmark,
    PoseModelError,
    Side,
    draw_arms,
    draw_hud,
    draw_joint_values,
    draw_trail,
    ensure_model,
)

_ARROW = {
    Direction.UP: "^ subindo ",
    Direction.DOWN: "v descendo",
    Direction.STILL: "= parado  ",
    Direction.UNKNOWN: "? --      ",
}


def fmt(value, spec: str = "6.2f", missing: str = "     -") -> str:
    return missing if value is None else format(value, spec)


def main() -> int:
    parser = argparse.ArgumentParser(description="Live arm velocity and direction.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--no-mirror", action="store_true")
    args = parser.parse_args()

    window = "Motion Lab - Sprint 04: Motion"
    tracker = MotionTracker()
    trails = {side: Trail(tracker.config.trail_length) for side in (Side.LEFT, Side.RIGHT)}
    show_trail = True

    try:
        model_path = ensure_model()
        camera = CameraEngine(config=CameraConfig(args.camera, args.width, args.height))
        with camera, PoseDetector(config=PoseConfig(model_path=model_path)) as detector:
            print("Motion view ready. Press q or ESC to quit, t to toggle the trail.")

            for frame in camera.frames():
                snapshot = detector.detect(frame)
                features = extract(snapshot)
                motion = tracker.update(features)

                canvas = frame.image
                if show_trail:
                    for side, trail in trails.items():
                        trail.add_from(snapshot, side)
                        draw_trail(canvas, trail.segments())
                draw_arms(canvas, snapshot)
                draw_joint_values(
                    canvas,
                    snapshot,
                    {
                        int(PoseLandmark.LEFT_WRIST): motion.left.velocity_y,
                        int(PoseLandmark.RIGHT_WRIST): motion.right.velocity_y,
                    },
                )
                view = canvas if args.no_mirror else mirror(canvas)

                draw_hud(
                    view,
                    [
                        "MOTION LAB / motion engine",
                        f"FPS {camera.fps:5.1f}   inference {snapshot.inference_ms:5.1f} ms",
                        f"LEFT   {_ARROW[motion.left.direction]}"
                        f"  vy {fmt(motion.left.velocity_y)}  speed {fmt(motion.left.speed)}",
                        f"RIGHT  {_ARROW[motion.right.direction]}"
                        f"  vy {fmt(motion.right.velocity_y)}  speed {fmt(motion.right.speed)}",
                        f"elbow deg/s  L {fmt(motion.left.elbow_velocity, '7.1f')}"
                        f"  R {fmt(motion.right.elbow_velocity, '7.1f')}",
                        f"height       L {fmt(features.left_wrist_height)}"
                        f"  R {fmt(features.right_wrist_height)}",
                        "units: shoulder-widths per second",
                    ],
                    width_px=560,
                )

                cv2.imshow(window, view)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("t"):
                    show_trail = not show_trail
                    for trail in trails.values():
                        trail.clear()
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break

            print(f"Processed {camera.frame_count} frames. No video written to disk.")
        return 0
    except (CameraError, PoseModelError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())

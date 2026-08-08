"""Sprint 02 — Hello Pose.

Acceptance criterion: "Eu movo meu braço e vejo ombro, cotovelo e punho
acompanhando."

    uv run python scripts/hello_pose.py        # q/ESC quits, L toggles labels
"""

from __future__ import annotations

import argparse
import sys

import cv2

from backend.camera import CameraConfig, CameraEngine, CameraError, mirror
from backend.vision import (
    PoseConfig,
    PoseDetector,
    PoseModelError,
    Side,
    draw_arms,
    draw_hud,
    ensure_model,
)


def arm_line(snapshot, side: Side) -> str:
    """One readable line per arm: which of the three joints are trusted."""
    marks = []
    for landmark in (side.shoulder, side.elbow, side.wrist):
        point = snapshot.point(landmark)
        marks.append("o" if point is not None and point.is_reliable() else ".")
    label = "L" if side is Side.LEFT else "R"
    return f"{label} arm    shoulder {marks[0]}  elbow {marks[1]}  wrist {marks[2]}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Live webcam with arm landmarks drawn on top.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--labels", action="store_true", help="start with landmark names shown")
    parser.add_argument("--no-mirror", action="store_true")
    args = parser.parse_args()

    window = "Motion Lab - Sprint 02: Hello Pose"
    labels = args.labels

    try:
        model_path = ensure_model()
        camera = CameraEngine(config=CameraConfig(args.camera, args.width, args.height))
        with camera, PoseDetector(config=PoseConfig(model_path=model_path)) as detector:
            print("Pose engine ready. Press q or ESC to quit, L to toggle labels.")

            for frame in camera.frames():
                snapshot = detector.detect(frame)

                # Skeleton is drawn on the RAW frame (landmark coords match it),
                # then the whole composite is mirrored for the selfie view. The
                # HUD is drawn after the flip so its text stays readable.
                canvas = frame.image
                draw_arms(canvas, snapshot, labels=labels)
                view = canvas if args.no_mirror else mirror(canvas)

                if snapshot.detected:
                    status = f"POSE  conf {snapshot.confidence() * 100:4.1f}%"
                else:
                    status = "POSE  no body in frame"

                draw_hud(
                    view,
                    [
                        "MOTION LAB / pose engine",
                        f"FPS       {camera.fps:5.1f}    inference {snapshot.inference_ms:5.1f} ms",
                        f"Frames    {camera.frame_count}",
                        status,
                        arm_line(snapshot, Side.LEFT),
                        arm_line(snapshot, Side.RIGHT),
                    ],
                    width_px=420,
                )

                cv2.imshow(window, view)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("l"):
                    labels = not labels
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

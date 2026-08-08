"""Sprint 03 — Arm Geometry.

Acceptance criterion: "A tela mostra o ângulo do meu cotovelo mudando em tempo
real."

    uv run python scripts/hello_geometry.py     # q/ESC quits
"""

from __future__ import annotations

import argparse
import sys

import cv2

from backend.camera import CameraConfig, CameraEngine, CameraError, mirror
from backend.features import extract
from backend.vision import (
    PoseConfig,
    PoseDetector,
    PoseLandmark,
    PoseModelError,
    draw_arms,
    draw_hud,
    draw_joint_values,
    ensure_model,
)


def fmt(value, spec: str = "6.1f", missing: str = "     -") -> str:
    """Missing features print as a dash, never as 0."""
    return missing if value is None else format(value, spec)


def flag(value) -> str:
    return "-" if value is None else ("yes" if value else "no ")


def main() -> int:
    parser = argparse.ArgumentParser(description="Live arm angles, distances and positions.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--no-mirror", action="store_true")
    args = parser.parse_args()

    window = "Motion Lab - Sprint 03: Arm Geometry"

    try:
        model_path = ensure_model()
        camera = CameraEngine(config=CameraConfig(args.camera, args.width, args.height))
        with camera, PoseDetector(config=PoseConfig(model_path=model_path)) as detector:
            print("Geometry view ready. Press q or ESC to quit.")

            for frame in camera.frames():
                snapshot = detector.detect(frame)
                features = extract(snapshot)

                canvas = frame.image
                draw_arms(canvas, snapshot)
                draw_joint_values(
                    canvas,
                    snapshot,
                    {
                        int(PoseLandmark.LEFT_ELBOW): features.left_elbow_angle,
                        int(PoseLandmark.RIGHT_ELBOW): features.right_elbow_angle,
                    },
                    suffix="deg",
                )
                view = canvas if args.no_mirror else mirror(canvas)

                draw_hud(
                    view,
                    [
                        "MOTION LAB / feature engine",
                        f"FPS {camera.fps:5.1f}   inference {snapshot.inference_ms:5.1f} ms",
                        f"elbow      L {fmt(features.left_elbow_angle)}   "
                        f"R {fmt(features.right_elbow_angle)}",
                        f"shoulder   L {fmt(features.left_shoulder_angle)}   "
                        f"R {fmt(features.right_shoulder_angle)}",
                        f"wrist dist   {fmt(features.wrist_distance, '5.2f')} shoulder-widths",
                        f"wrist height L {fmt(features.left_wrist_height, '5.2f')}  "
                        f"R {fmt(features.right_wrist_height, '5.2f')}",
                        f"above shoulder L {flag(features.left_wrist_above_shoulder)}  "
                        f"R {flag(features.right_wrist_above_shoulder)}",
                        f"wrists crossed {flag(features.wrists_crossed)}    "
                        f"complete {flag(features.complete)}",
                    ],
                    width_px=470,
                )

                cv2.imshow(window, view)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
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

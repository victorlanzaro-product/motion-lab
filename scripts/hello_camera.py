"""Sprint 01 — Hello Camera.

Acceptance criterion: "Eu consigo abrir o programa e visualizar minha webcam."

    uv run python scripts/hello_camera.py     # press q or ESC to quit
"""

from __future__ import annotations

import argparse
import sys

import cv2

from backend.camera import CameraConfig, CameraEngine, CameraError, mirror

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_GREEN = (120, 255, 120)
_DIM = (200, 200, 200)


def draw_hud(image, lines: list[str]) -> None:
    """Semi-transparent panel with the live capture stats."""
    panel_h = 26 * len(lines) + 16
    overlay = image.copy()
    cv2.rectangle(overlay, (12, 12), (330, 12 + panel_h), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.45, image, 0.55, 0, dst=image)

    y = 38
    for i, line in enumerate(lines):
        cv2.putText(image, line, (24, y), _FONT, 0.6, _GREEN if i == 0 else _DIM, 1, cv2.LINE_AA)
        y += 26


def main() -> int:
    parser = argparse.ArgumentParser(description="Show the webcam feed with capture stats.")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default: 0)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="show the raw sensor image instead of the selfie view",
    )
    args = parser.parse_args()

    config = CameraConfig(index=args.camera, width=args.width, height=args.height)
    window = "Motion Lab - Sprint 01: Hello Camera"

    try:
        with CameraEngine(config=config) as camera:
            granted_w, granted_h = camera.resolution
            print(f"Camera {config.index} open at {granted_w}x{granted_h}. Press q or ESC to quit.")

            for frame in camera.frames():
                # Mirroring happens here, on the display copy only: the pose
                # stage (Sprint 02) must receive the unflipped image so that
                # LEFT_SHOULDER really is the left shoulder.
                view = frame.image if args.no_mirror else mirror(frame.image)

                draw_hud(
                    view,
                    [
                        "MOTION LAB / camera engine",
                        f"FPS       {camera.fps:5.1f}",
                        f"Frames    {camera.frame_count}",
                        f"Size      {frame.width}x{frame.height}",
                        "Recording  no",
                    ],
                )

                cv2.imshow(window, view)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # q or ESC
                    break
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break  # user closed the window

            print(f"Captured {camera.frame_count} frames. Nothing was written to disk.")
        return 0
    except CameraError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())

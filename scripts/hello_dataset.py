"""Sprint 07 — Dataset Builder.

Acceptance criterion: "Eu seguro uma tecla de rótulo e o sistema grava
amostras rotuladas de features (nunca pixels) em data/training/."

    uv run python scripts/hello_dataset.py     # q/ESC quits

Controls
  1 arm_raised   2 arms_crossed   3 arms_open   4 idle
  hold the digit key to record that label continuously; release to stop.

  "wave" is not one of the keys: it is a class of motion, not posture, and
  this recorder writes one static feature snapshot per frame -- no window of
  reversals for the Forest to actually learn a wave from. GestureEngine's
  rule-based detector recognizes a wave live; this dataset does not train
  the ML side to (`backend/dataset/writer.py:REMOVED_LABELS`).

Recording is edge-free on purpose — this is raw material for Sprint 08's
Random Forest, not a gesture verdict, so there is no hold debounce or
cooldown here the way `GestureEngine` has. Every frame the key is down is
one row, mistakes and all; a bad sample is just a row to filter out later,
not a live decision to get right.

Holding a key relies on the OS's own keyboard auto-repeat: `cv2.waitKey`
has no key-up event, so "still held" is inferred from repeat events arriving
faster than `RELEASE_TIMEOUT`. If your keyboard's repeat rate is unusually
slow, lower `RELEASE_TIMEOUT` or just tap the key faster than once every
quarter second — SPACE always stops recording immediately as a fallback.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

import cv2

from backend.camera import CameraConfig, CameraEngine, CameraError, mirror
from backend.dataset import DEFAULT_LABELS, DatasetWriter
from backend.features import MotionTracker, extract
from backend.vision import PoseConfig, PoseDetector, PoseModelError, draw_arms, draw_hud, ensure_model

#: How long without a repeat event before a held key counts as released.
RELEASE_TIMEOUT = 0.25

#: 1..5 map onto DEFAULT_LABELS in order, so adding a label there updates
#: this automatically instead of drifting out of sync.
LABEL_KEYS = {ord(str(i + 1)): label for i, label in enumerate(DEFAULT_LABELS)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Record labeled feature rows for training.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--no-mirror", action="store_true")
    args = parser.parse_args()

    window = "Motion Lab - Sprint 07: Dataset Builder"
    session_id = time.strftime("%Y%m%dT%H%M%S")
    tracker = MotionTracker()
    active_label: Optional[str] = None
    last_key_time = 0.0

    try:
        model_path = ensure_model()
        camera = CameraEngine(config=CameraConfig(args.camera, args.width, args.height))
        with (
            camera,
            PoseDetector(config=PoseConfig(model_path=model_path)) as detector,
            DatasetWriter(session_id=session_id) as writer,
        ):
            print(f"Recording to {writer.path}  (session {session_id})")
            print("Hold 1-5 to record a label, SPACE to stop, q/ESC to quit.")

            for frame in camera.frames():
                snapshot = detector.detect(frame)
                features = extract(snapshot)
                motion = tracker.update(features)

                now = time.perf_counter()
                if active_label is not None and now - last_key_time > RELEASE_TIMEOUT:
                    active_label = None  # no repeat arrived in time -> key released

                recorded = False
                if active_label is not None:
                    recorded = writer.write(active_label, features, motion)

                canvas = frame.image
                draw_arms(canvas, snapshot)
                view = canvas if args.no_mirror else mirror(canvas)

                status = f"RECORDING {active_label}" if active_label else "not recording"
                if active_label and not recorded:
                    status += "  (skipped: joint occluded)"
                counts_line = "  ".join(
                    f"{name} {writer.counts.get(name, 0)}" for name in DEFAULT_LABELS
                )

                draw_hud(
                    view,
                    [
                        "MOTION LAB / dataset builder",
                        f"FPS {camera.fps:5.1f}   session {session_id}",
                        status,
                        counts_line,
                        f"total {writer.total}   skipped {writer.skipped}",
                        "hold 1-5 to record, SPACE stop, q/ESC quit",
                    ],
                    width_px=560,
                )

                cv2.imshow(window, view)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    active_label = None
                elif key in LABEL_KEYS:
                    active_label = LABEL_KEYS[key]
                    last_key_time = now
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break

            print(
                f"Recorded {writer.total} samples ({writer.skipped} skipped, occluded). "
                "No video written to disk."
            )
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

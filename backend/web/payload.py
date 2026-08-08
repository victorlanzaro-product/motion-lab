"""Wire format — one frame of pipeline state, as JSON the browser can draw.

Three rules shape what travels:

  * **Only arm landmarks go on the wire.** V1 tracks shoulder -> elbow -> wrist.
    Shipping all 33 BlazePose points would inflate every message with joints
    nothing draws.
  * **Coordinates stay raw.** Normalized, unmirrored, exactly as the estimator
    produced them. Mirroring is a display concern (Sprint 01): the browser flips
    at draw time, so the payload keeps left/right meaning what they mean.
  * **The preview is downscaled, the inference is not.** Pose runs on the full
    sensor frame; only the JPEG the browser displays is shrunk. Picture size and
    detection quality stay independent knobs.

The topology (`ARM_CHAINS`) is sent to the client instead of being hardcoded in
JavaScript: the bone table is domain knowledge and belongs in one place.
"""

from __future__ import annotations

import base64
from typing import Any, Optional

import cv2
import numpy as np

from backend.features.features import FEATURE_NAMES, FrameFeatures
from backend.features.velocity import MotionState, Trail
from backend.gestures.engine import GestureEvent
from backend.vision.landmarks import (
    ARM_CHAINS,
    ARM_LANDMARKS,
    DEFAULT_VISIBILITY_THRESHOLD,
    PoseSnapshot,
)

#: Four decimals of a normalized coordinate is 1/10000 of the frame — far under
#: one pixel at any display size, and it keeps the JSON about a third smaller
#: than a full float repr.
COORD_DIGITS = 4
VALUE_DIGITS = 4


class PreviewError(RuntimeError):
    """Raised when a frame cannot be encoded for the browser."""


def _round(value: Any, digits: int = VALUE_DIGITS) -> Any:
    """Round numbers; pass everything else through untouched.

    `bool` is excluded first because it is a subclass of int — `round(True)` is
    1, and a boolean feature has to stay a boolean in JSON. Strings pass through
    for the same reason: `direction` is a verdict ("up"), not a measurement.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    return round(float(value), digits)


def encode_preview(image: np.ndarray, width: int = 640, quality: int = 70) -> dict[str, Any]:
    """BGR frame -> base64 JPEG, downscaled to `width`.

    Never upscales: a 320-wide camera stays 320 wide instead of being blown up
    into bytes that carry no extra detail.
    """
    source_h, source_w = image.shape[:2]
    if width > 0 and source_w > width:
        height = max(1, int(round(source_h * width / source_w)))
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise PreviewError("OpenCV could not encode the frame as JPEG.")

    return {
        "format": "jpeg",
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "bytes": int(buffer.size),
        "base64": base64.b64encode(buffer.tobytes()).decode("ascii"),
    }


def landmarks_payload(snapshot: PoseSnapshot) -> dict[str, dict[str, float]]:
    """Arm joints, keyed by landmark index as a string (JSON has no int keys).

    Low-visibility joints travel too, carrying their score, so the browser can
    dim them the way `draw_arms` does. Dropping them here would leave the client
    unable to tell "occluded" from "no body in frame".
    """
    payload: dict[str, dict[str, float]] = {}
    for landmark in ARM_LANDMARKS:
        point = snapshot.point(landmark)
        if point is None:
            continue
        payload[str(int(landmark))] = {
            "x": _round(point.x, COORD_DIGITS),
            "y": _round(point.y, COORD_DIGITS),
            "v": _round(point.visibility, 3),
        }
    return payload


def trail_payload(trail: Trail) -> list[list[list[float]]]:
    """Wrist trajectory as one polyline per continuous run.

    The gap handling — a lost joint breaks the line instead of teleporting
    across it — already lives in `Trail`. Sending finished segments keeps that
    rule in one tested place instead of re-deriving it in JavaScript.
    """
    return [
        [[_round(p.x, COORD_DIGITS), _round(p.y, COORD_DIGITS)] for p in run]
        for run in trail.segments()
    ]


def features_payload(features: FrameFeatures) -> dict[str, Any]:
    """The ML columns plus the scale reference, rounded.

    `frame_index`/`timestamp` are dropped: they already sit on the envelope.
    """
    payload: dict[str, Any] = {name: _round(getattr(features, name)) for name in FEATURE_NAMES}
    payload["shoulder_width"] = _round(features.shoulder_width)
    payload["complete"] = features.complete
    return payload


def motion_payload(motion: MotionState) -> dict[str, Any]:
    """`MotionState.to_dict` with the envelope fields stripped and floats rounded."""
    return {
        key: _round(value)
        for key, value in motion.to_dict().items()
        if key not in ("frame_index", "timestamp")
    }


def gesture_payload(events: list[GestureEvent]) -> list[dict[str, Any]]:
    """Gestures that fired on THIS frame — usually empty.

    `GestureEngine` is already edge-triggered, so this is never the "is it
    currently held" state, only the instant it crossed into that state. A
    client that wants a toast or a log line reacts to entries appearing here,
    not to polling a boolean every frame.
    """
    return [
        {"name": event.name, "side": event.side, "frame_index": event.frame_index}
        for event in events
    ]


def frame_message(
    *,
    snapshot: PoseSnapshot,
    features: FrameFeatures,
    motion: MotionState,
    preview: dict[str, Any],
    fps: float,
    trails: dict[str, Trail],
    gestures: Optional[list[GestureEvent]] = None,
) -> dict[str, Any]:
    """One `type: "frame"` message: pixels, joints and numbers from the same frame.

    They travel together on purpose. A separate MJPEG stream would be cheaper in
    bytes, but it would let the skeleton drift a frame or two away from the
    picture it is supposed to be drawn on.
    """
    return {
        "type": "frame",
        "frame_index": snapshot.frame_index,
        "timestamp": _round(snapshot.timestamp, 3),
        "fps": _round(fps, 1),
        "inference_ms": _round(snapshot.inference_ms, 1),
        "detected": snapshot.detected,
        "confidence": _round(snapshot.confidence(), 3),
        "image": preview,
        "landmarks": landmarks_payload(snapshot),
        "features": features_payload(features),
        "motion": motion_payload(motion),
        "trail": {name: trail_payload(trail) for name, trail in trails.items()},
        "gestures": gesture_payload(gestures or []),
    }


#: Gesture names the engine can emit — declared here so the frontend's
#: translation table (`DIRECTION_LABEL`-style) has a fixed list to work from
#: instead of inferring it from whatever happens to show up first.
GESTURE_NAMES: tuple[str, ...] = ("arm_raised", "wave", "arms_crossed", "arms_open")


def hello_message(
    *,
    visibility_threshold: float = DEFAULT_VISIBILITY_THRESHOLD,
    still_threshold: float = 0.35,
    preview_width: int = 640,
    jpeg_quality: int = 70,
) -> dict[str, Any]:
    """First message on every connection: everything the client needs to draw.

    Sent up front so the frontend keeps no copy of the landmark table, the bone
    topology or the thresholds — change them in Python and the browser follows.
    """
    return {
        "type": "hello",
        "arm_landmarks": [int(lm) for lm in ARM_LANDMARKS],
        "landmark_names": {str(int(lm)): lm.name for lm in ARM_LANDMARKS},
        "chains": [[int(a), int(b)] for a, b in ARM_CHAINS],
        "feature_names": list(FEATURE_NAMES),
        "gesture_names": list(GESTURE_NAMES),
        "visibility_threshold": visibility_threshold,
        "still_threshold": still_threshold,
        "preview": {"width": preview_width, "quality": jpeg_quality},
    }


def error_message(message: str, *, fatal: bool = True) -> dict[str, Any]:
    """A failure the user has to see — camera denied, model missing, device gone.

    The browser shows this instead of holding a frozen last frame: a live view
    that silently stops looks exactly like a person standing very still.
    """
    return {"type": "error", "message": message, "fatal": fatal}

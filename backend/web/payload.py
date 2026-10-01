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
from typing import Any, Mapping, Optional

import cv2
import numpy as np

from backend.features.face_features import FaceSignals
from backend.features.face_motion import FaceMotionState
from backend.features.features import FEATURE_NAMES, FrameFeatures
from backend.features.velocity import MotionState, Trail
from backend.gestures.engine import GestureEvent
from backend.vision.face_landmarks import (
    FACE_CHAIN_GROUPS,
    FACE_MESH_CONTRACT_VERSION,
    FACE_OVERLAY_LANDMARKS,
)
from backend.vision.landmarks import (
    ARM_CHAINS,
    ARM_LANDMARKS,
    DEFAULT_VISIBILITY_THRESHOLD,
    Point,
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


def face_landmarks_payload(mesh: Mapping[int, Point]) -> dict[str, dict[str, float]]:
    """Face overlay points, keyed by landmark index as a string — same shape
    `landmarks_payload` uses for arm joints.

    Only `x`/`y` travel: `z` is not drawn client-side (2D overlay only), and
    no `v`/visibility field is added the way arm joints get one, because the
    Face Landmarker gives no equivalent per-point confidence to report
    (`face_landmarks.py`'s `FaceSnapshot` docstring) — inventing one would
    look exactly as trustworthy as the real thing while measuring nothing.
    Empty when `mesh` is empty (no face this frame): nothing to draw, not a
    guess at where the face might be.
    """
    return {
        str(index): {"x": _round(point.x, COORD_DIGITS), "y": _round(point.y, COORD_DIGITS)}
        for index, point in mesh.items()
    }


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


def ml_payload(prediction: Optional[tuple[str, dict[str, float]]]) -> Optional[dict[str, Any]]:
    """Raw per-frame classifier signal, or `None` when there is nothing to say.

    `None` covers two cases the client treats the same way — no model has
    been trained yet, or this particular frame had an occluded joint and was
    never handed to the model (`PipelineRunner` skips it rather than feeding
    the model a fabricated value; see Sprint 07's `DatasetWriter` for the
    same rule at training time). Deliberately unsmoothed: this is the raw
    "Sinal X em tempo real" the roadmap asks for, not a debounced verdict —
    `GestureEngine` already owns that job.
    """
    if prediction is None:
        return None
    label, probabilities = prediction
    return {
        "label": label,
        "confidence": _round(probabilities[label], 3),
        "probabilities": {name: _round(value, 3) for name, value in probabilities.items()},
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


def face_payload(
    *,
    enabled: bool,
    error: Optional[str],
    detected: bool = False,
    signals: Optional[FaceSignals] = None,
    motion: Optional[FaceMotionState] = None,
    mesh: Optional[Mapping[int, Point]] = None,
) -> dict[str, Any]:
    """Facial signal for the wire — always present, never an emotion claim.

    `mesh` is `FaceSnapshot.mesh` (the overlay-subset points), or `None`/empty
    when there is nothing to draw — a missing/disabled/errored face never
    reaches `landmarks_payload` at all here, same as `signals`/`motion`.

    Three states the client has to tell apart, so `reason`/`available`/
    `detected` are three separate fields rather than one:

    * `available: false, reason: "disabled"` — the operator turned face
      tracking off for this deployment (`WebConfig.face_enabled`).
    * `available: false, reason: <error>` — face tracking is on but the
      model never loaded this session (missing file, failed download, or a
      runtime error mid-session); Sprint 09's rule for the ML signal, applied
      here too: optional layers degrade, they never take pose/web down.
    * `available: true, detected: false` — the model is fine, nobody's face
      is in frame this instant.

    Every field is a named, observable signal (`smile`, `mouth_open`, ...) or
    a rate of change of one — never a mood, an emotion, or a mental state.
    """
    if not enabled:
        return {"available": False, "detected": False, "reason": "disabled"}
    if signals is None:
        return {"available": False, "detected": False, "reason": error or "no_model"}
    return {
        "available": True,
        "detected": detected,
        "smile": _round(signals.smile, 3),
        "mouth_open": _round(signals.mouth_open, 3),
        "eye_blink_left": _round(signals.eye_blink_left, 3),
        "eye_blink_right": _round(signals.eye_blink_right, 3),
        "brow_raise": _round(signals.brow_raise, 3),
        "head_yaw": _round(signals.head_yaw, 1),
        "head_pitch": _round(signals.head_pitch, 1),
        "head_roll": _round(signals.head_roll, 1),
        "smile_velocity": _round(motion.smile_velocity, 3) if motion else None,
        "mouth_open_velocity": _round(motion.mouth_open_velocity, 3) if motion else None,
        "brow_raise_velocity": _round(motion.brow_raise_velocity, 3) if motion else None,
        "blink_rate_left": _round(motion.blink_rate_left, 2) if motion else None,
        "blink_rate_right": _round(motion.blink_rate_right, 2) if motion else None,
        # Overlay points for THIS frame -- `{}` (never omitted) when nothing
        # was detected, so the client's "no face -> draw nothing" rule
        # (app.js:drawFaceOverlay) never has to distinguish "key missing"
        # from "key present but empty".
        "landmarks": face_landmarks_payload(mesh) if mesh else {},
    }


def frame_message(
    *,
    snapshot: PoseSnapshot,
    features: FrameFeatures,
    motion: MotionState,
    preview: dict[str, Any],
    fps: float,
    trails: dict[str, Trail],
    gestures: Optional[list[GestureEvent]] = None,
    ml: Optional[tuple[str, dict[str, float]]] = None,
    face: Optional[dict[str, Any]] = None,
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
        "ml": ml_payload(ml),
        # Falls back to "disabled" rather than `None`: an omitted `face` key
        # would look, to a client written before Sprint 11, exactly like a
        # face signal that simply is not there yet -- explicit beats absent.
        "face": face if face is not None else face_payload(enabled=False, error=None),
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
    ml_classes: Optional[list[str]] = None,
    face_enabled: bool = False,
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
        # Empty when no model is trained yet -- the client's cue to show
        # "sem modelo treinado" instead of waiting for an "ml" field that
        # will never arrive non-null.
        "ml_classes": list(ml_classes) if ml_classes else [],
        "visibility_threshold": visibility_threshold,
        "still_threshold": still_threshold,
        "preview": {"width": preview_width, "quality": jpeg_quality},
        # Static config, known before the first frame -- lets the client show
        # the face toggle (or not) without waiting on a `frame.face` reason.
        "face_enabled": face_enabled,
        # Face overlay topology (Sprint 11) -- static domain knowledge, sent
        # unconditionally on `face_enabled` for the same reason `chains`/
        # `arm_landmarks` are sent unconditionally on the pose model existing:
        # it costs nothing when unused and the client never has to hardcode a
        # copy that could drift from `face_landmarks.py`.
        "face_landmarks": list(FACE_OVERLAY_LANDMARKS),
        "face_chains": {
            name: [[a, b] for a, b in pairs] for name, pairs in FACE_CHAIN_GROUPS.items()
        },
        # Bumped only if the topology above ever changes shape -- lets a
        # client tell "this server's face overlay contract is different from
        # what I was built against" apart from "no topology sent at all".
        "face_mesh_version": FACE_MESH_CONTRACT_VERSION,
    }


def error_message(message: str, *, fatal: bool = True) -> dict[str, Any]:
    """A failure the user has to see — camera denied, model missing, device gone.

    The browser shows this instead of holding a frozen last frame: a live view
    that silently stops looks exactly like a person standing very still.
    """
    return {"type": "error", "message": message, "fatal": fatal}


def status_message(state: str, message: str = "") -> dict[str, Any]:
    """A non-fatal lifecycle notice — the socket stays open, frames may resume.

    Distinct from `error_message` on purpose: an `error` in this wire format
    always means "the browser should show a stopped state" (see `app.js`).
    A transient camera drop that `PipelineRunner` is actively retrying is not
    that — the client should say "reconectando…", not "parado", and keep
    listening on the same connection for the frames that resume it.
    """
    return {"type": "status", "state": state, "message": message}

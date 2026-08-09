"""Face Feature Engine — one face snapshot in, one row of observable signals out.

Same boundary `features.py` draws for the body: from here on nothing knows
about pixels or MediaPipe, only numbers. And the same boundary this sprint's
product rule draws for the face, stricter than the pose one: these are
**observable facial signals**, never an emotion or mental-state claim. A
"sorriso aparente" (apparent smile) score is a fact about facial-muscle
activation MediaPipe already computed — it is not, and must never be
presented as, "the person is happy".

Every field is `Optional`, same rule as `FrameFeatures`: `None` means "no face
in frame" or "this blendshape wasn't available", never a fabricated 0.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

from backend.vision.face_landmarks import FaceSnapshot

#: Column order for anything that persists or trains on these signals later.
#: Append only — same discipline as `FEATURE_NAMES`.
FACE_SIGNAL_NAMES: tuple[str, ...] = (
    "smile",
    "mouth_open",
    "eye_blink_left",
    "eye_blink_right",
    "brow_raise",
    "head_yaw",
    "head_pitch",
    "head_roll",
)

#: MediaPipe blendshape category names this layer reads. Picking named,
#: already-computed activation scores instead of re-deriving geometry from
#: the 478 raw mesh points — see `face_landmarks.py`'s module docstring.
_SMILE_BLENDSHAPES = ("mouthSmileLeft", "mouthSmileRight")
_BROW_RAISE_BLENDSHAPES = ("browInnerUp", "browOuterUpLeft", "browOuterUpRight")


@dataclass(frozen=True)
class FaceSignals:
    """One frame, described as observable facial signals — never a diagnosis."""

    frame_index: int
    timestamp: float

    #: 0..1, how much the mouth-corner-pull blendshapes are active.
    smile: Optional[float] = None
    #: 0..1, jaw-open blendshape — how open the mouth is.
    mouth_open: Optional[float] = None
    #: 0..1 each, independent per eye (a wink is not a blink).
    eye_blink_left: Optional[float] = None
    eye_blink_right: Optional[float] = None
    #: 0..1, how much the brow-raise blendshapes are active.
    brow_raise: Optional[float] = None
    #: Degrees. Head orientation, not gaze — see `euler_from_matrix`'s
    #: documented limitation on axis convention.
    head_yaw: Optional[float] = None
    head_pitch: Optional[float] = None
    head_roll: Optional[float] = None

    @property
    def complete(self) -> bool:
        """True when every signal column is filled — mirrors `FrameFeatures.complete`."""
        return all(getattr(self, name) is not None for name in FACE_SIGNAL_NAMES)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mean(*values: Optional[float]) -> Optional[float]:
    """Average of whichever of `values` are present; `None` if none are.

    Used when a signal is defined as more than one blendshape (smile is both
    mouth corners, brow raise is three brow segments) — a single occluded
    sub-score should not zero out the whole signal.
    """
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def extract_face(snapshot: FaceSnapshot) -> FaceSignals:
    """Snapshot in, signal row out. Never raises: no face means every field is `None`."""
    base = FaceSignals(frame_index=snapshot.frame_index, timestamp=snapshot.timestamp)
    if not snapshot.detected:
        return base

    return FaceSignals(
        frame_index=snapshot.frame_index,
        timestamp=snapshot.timestamp,
        smile=_mean(*(snapshot.blendshape(name) for name in _SMILE_BLENDSHAPES)),
        mouth_open=snapshot.blendshape("jawOpen"),
        eye_blink_left=snapshot.blendshape("eyeBlinkLeft"),
        eye_blink_right=snapshot.blendshape("eyeBlinkRight"),
        brow_raise=_mean(*(snapshot.blendshape(name) for name in _BROW_RAISE_BLENDSHAPES)),
        head_yaw=snapshot.head_yaw,
        head_pitch=snapshot.head_pitch,
        head_roll=snapshot.head_roll,
    )

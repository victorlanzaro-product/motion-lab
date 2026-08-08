"""Gesture predicates — one frame's features, turned into a yes/no per posture.

Pure and stateless, like the functions in `angles.py` / `distances.py` /
`positions.py`: every rule here answers "is the pose like this RIGHT NOW",
never "for how long". Time — how long a posture must hold, how often it can
refire — lives one layer up, in `GestureEngine` (`engine.py`). Keeping that
split means a rule reads exactly like the feature functions it is built from,
and is just as easy to unit test.

`None` propagates rather than being guessed: an occluded wrist means "we don't
know if the arm is raised", not "the arm is not raised". `GestureEngine`
decides how to treat that uncertainty — conservatively, as "condition not
met" — so a gesture never fires off a joint MediaPipe could not see.
"""

from __future__ import annotations

from typing import Optional

from backend.features.features import FrameFeatures
from backend.vision.landmarks import Side


def arm_raised(features: FrameFeatures, side: Side) -> Optional[bool]:
    """Wrist above its own shoulder — the position half of "raise your arm"."""
    return getattr(features, f"{side.name.lower()}_wrist_above_shoulder")


def arms_crossed(features: FrameFeatures) -> Optional[bool]:
    """Both wrists past the body midline — the geometric core of an X shape."""
    return features.wrists_crossed


def arms_open(features: FrameFeatures, threshold: float) -> Optional[bool]:
    """Wrists spread wider than `threshold` shoulder widths — a T-pose-ish reach.

    `wrist_distance` is already shoulder-width scaled (`distances.py`), so one
    threshold means the same pose whether you stand close to the camera or
    across the room.
    """
    if features.wrist_distance is None:
        return None
    return features.wrist_distance > threshold

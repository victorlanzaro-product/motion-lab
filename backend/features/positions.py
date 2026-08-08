"""Relative positions: where a wrist sits with respect to the body.

Image coordinates put y=0 at the TOP, so "higher up" means a SMALLER y. Every
signed height here is flipped so that positive = above, which is what a human
reading the feature dump expects.
"""

from __future__ import annotations

from typing import Optional

from backend.features.distances import shoulder_width
from backend.vision.landmarks import (
    DEFAULT_VISIBILITY_THRESHOLD,
    PoseLandmark,
    PoseSnapshot,
    Side,
)


def wrist_height(
    snapshot: PoseSnapshot,
    side: Side,
    threshold: float = DEFAULT_VISIBILITY_THRESHOLD,
) -> Optional[float]:
    """Wrist height relative to its own shoulder, in shoulder widths.

    Positive = wrist above the shoulder. Scaled so the number means the same
    whether you are close to the camera or far from it.
    """
    wrist = snapshot.point(side.wrist)
    shoulder = snapshot.point(side.shoulder)
    if wrist is None or shoulder is None:
        return None
    if not (wrist.is_reliable(threshold) and shoulder.is_reliable(threshold)):
        return None
    reference = shoulder_width(snapshot, threshold)
    if reference is None:
        return None
    return (shoulder.y - wrist.y) / reference  # flipped: image y grows downwards


def wrist_above_shoulder(
    snapshot: PoseSnapshot,
    side: Side,
    threshold: float = DEFAULT_VISIBILITY_THRESHOLD,
) -> Optional[bool]:
    height = wrist_height(snapshot, side, threshold)
    return None if height is None else height > 0.0


def wrist_offset_x(
    snapshot: PoseSnapshot,
    side: Side,
    threshold: float = DEFAULT_VISIBILITY_THRESHOLD,
) -> Optional[float]:
    """Horizontal wrist offset from the body midline, in shoulder widths.

    Positive = towards the person's own left in the raw (unmirrored) image,
    which is where LEFT_SHOULDER lives. Negative = the other way.
    """
    wrist = snapshot.point(side.wrist)
    left = snapshot.point(PoseLandmark.LEFT_SHOULDER)
    right = snapshot.point(PoseLandmark.RIGHT_SHOULDER)
    if wrist is None or left is None or right is None:
        return None
    if not all(p.is_reliable(threshold) for p in (wrist, left, right)):
        return None
    reference = shoulder_width(snapshot, threshold)
    if reference is None:
        return None
    midline = (left.x + right.x) / 2
    return (wrist.x - midline) * snapshot.aspect / reference


def wrists_crossed(
    snapshot: PoseSnapshot,
    threshold: float = DEFAULT_VISIBILITY_THRESHOLD,
) -> Optional[bool]:
    """True when each wrist has moved past the body midline to the other side.

    This is the geometric core of an X shape, and a good example of a hand-made
    feature the classifier can lean on instead of raw coordinates.
    """
    left = wrist_offset_x(snapshot, Side.LEFT, threshold)
    right = wrist_offset_x(snapshot, Side.RIGHT, threshold)
    if left is None or right is None:
        return None
    # In raw image space the left shoulder sits at a larger x than the right one,
    # so an uncrossed pose has left offset > 0 and right offset < 0.
    return left < 0.0 and right > 0.0

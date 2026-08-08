"""Distances between joints, in body units.

The central idea of this file: a raw distance is useless to a classifier.
"Wrists 0.2 apart" means a wide stance up close and a narrow one across the
room. Everything user-facing is therefore divided by shoulder width, which
makes the feature invariant to how far you stand from the camera.
"""

from __future__ import annotations

import math
from typing import Optional

from backend.vision.landmarks import (
    DEFAULT_VISIBILITY_THRESHOLD,
    Point,
    PoseLandmark,
    PoseSnapshot,
    Side,
)


def distance(a: Point, b: Point, aspect: float = 1.0) -> float:
    """Straight-line distance in the un-squashed normalized space."""
    return math.hypot((b.x - a.x) * aspect, b.y - a.y)


def _pair(
    snapshot: PoseSnapshot,
    first: PoseLandmark,
    second: PoseLandmark,
    threshold: float,
) -> Optional[tuple[Point, Point]]:
    a, b = snapshot.point(first), snapshot.point(second)
    if a is None or b is None:
        return None
    if not (a.is_reliable(threshold) and b.is_reliable(threshold)):
        return None
    return a, b


def shoulder_width(
    snapshot: PoseSnapshot, threshold: float = DEFAULT_VISIBILITY_THRESHOLD
) -> Optional[float]:
    """The scale reference for every other distance."""
    pair = _pair(snapshot, PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER, threshold)
    if pair is None:
        return None
    width = distance(*pair, snapshot.aspect)
    # Shoulders on top of each other means a bad detection, not a tiny person.
    return width if width > 1e-6 else None


def wrist_distance(
    snapshot: PoseSnapshot,
    threshold: float = DEFAULT_VISIBILITY_THRESHOLD,
    scaled: bool = True,
) -> Optional[float]:
    """Distance between the two wrists, in shoulder widths when `scaled`."""
    pair = _pair(snapshot, PoseLandmark.LEFT_WRIST, PoseLandmark.RIGHT_WRIST, threshold)
    if pair is None:
        return None
    raw = distance(*pair, snapshot.aspect)
    if not scaled:
        return raw
    reference = shoulder_width(snapshot, threshold)
    return None if reference is None else raw / reference


def forearm_length(
    snapshot: PoseSnapshot,
    side: Side,
    threshold: float = DEFAULT_VISIBILITY_THRESHOLD,
    scaled: bool = True,
) -> Optional[float]:
    """Elbow to wrist. Shrinks as the forearm points at the camera — a cheap
    depth cue in a 2D projection."""
    pair = _pair(snapshot, side.elbow, side.wrist, threshold)
    if pair is None:
        return None
    raw = distance(*pair, snapshot.aspect)
    if not scaled:
        return raw
    reference = shoulder_width(snapshot, threshold)
    return None if reference is None else raw / reference

"""Angles between joints.

Everything here returns `None` when the input is missing or unreliable. A joint
we cannot see must not become a number — a fabricated 0 degrees would look like
a fully bent elbow to the classifier.
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


def _vector(origin: Point, target: Point, aspect: float) -> tuple[float, float]:
    """2D vector in a square space.

    x is multiplied by the aspect ratio to undo the normalization squash;
    without this a 90-degree elbow reads as ~101 degrees on a 16:9 frame.
    """
    return ((target.x - origin.x) * aspect, target.y - origin.y)


def angle_between(a: Point, b: Point, c: Point, aspect: float = 1.0) -> Optional[float]:
    """Angle at vertex `b`, in degrees, 0..180. None if a side has zero length."""
    ba = _vector(b, a, aspect)
    bc = _vector(b, c, aspect)

    len_ba = math.hypot(*ba)
    len_bc = math.hypot(*bc)
    if len_ba == 0.0 or len_bc == 0.0:
        return None

    cosine = (ba[0] * bc[0] + ba[1] * bc[1]) / (len_ba * len_bc)
    # Floating point can push this a hair past +-1, which would blow up acos.
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _reliable(
    snapshot: PoseSnapshot, landmarks: tuple[PoseLandmark, ...], threshold: float
) -> Optional[tuple[Point, ...]]:
    points = []
    for landmark in landmarks:
        point = snapshot.point(landmark)
        if point is None or not point.is_reliable(threshold):
            return None
        points.append(point)
    return tuple(points)


def elbow_angle(
    snapshot: PoseSnapshot,
    side: Side,
    threshold: float = DEFAULT_VISIBILITY_THRESHOLD,
) -> Optional[float]:
    """Shoulder-elbow-wrist. 180 = arm straight, 90 = right angle, ~30 = fully flexed."""
    points = _reliable(snapshot, (side.shoulder, side.elbow, side.wrist), threshold)
    if points is None:
        return None
    shoulder, elbow, wrist = points
    return angle_between(shoulder, elbow, wrist, snapshot.aspect)


def shoulder_angle(
    snapshot: PoseSnapshot,
    side: Side,
    threshold: float = DEFAULT_VISIBILITY_THRESHOLD,
) -> Optional[float]:
    """Elbow-shoulder-hip: how far the arm is lifted away from the torso.

    ~0 = arm hanging down beside the body, ~90 = arm out horizontally,
    ~180 = arm straight up overhead.
    """
    points = _reliable(snapshot, (side.elbow, side.shoulder, side.hip), threshold)
    if points is None:
        return None
    elbow, shoulder, hip = points
    return angle_between(elbow, shoulder, hip, snapshot.aspect)

"""Landmark model — what MediaPipe returns, expressed in Motion Lab's own types.

Why not pass MediaPipe objects around: the feature layer (Sprint 03+) should not
import MediaPipe. If we ever swap the pose estimator, only this file and
`pose_detector.py` change.

MediaPipe 1.0 dropped the legacy `mp.solutions` module, so the BlazePose index
table lives here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable, Mapping, Optional


class PoseLandmark(IntEnum):
    """The 33 BlazePose points, by index."""

    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


class Side(IntEnum):
    LEFT = 0
    RIGHT = 1

    @property
    def shoulder(self) -> PoseLandmark:
        return PoseLandmark.LEFT_SHOULDER if self is Side.LEFT else PoseLandmark.RIGHT_SHOULDER

    @property
    def elbow(self) -> PoseLandmark:
        return PoseLandmark.LEFT_ELBOW if self is Side.LEFT else PoseLandmark.RIGHT_ELBOW

    @property
    def wrist(self) -> PoseLandmark:
        return PoseLandmark.LEFT_WRIST if self is Side.LEFT else PoseLandmark.RIGHT_WRIST

    @property
    def hip(self) -> PoseLandmark:
        return PoseLandmark.LEFT_HIP if self is Side.LEFT else PoseLandmark.RIGHT_HIP


#: The V1 scope: arms only. shoulder -> elbow -> wrist, per side.
ARM_LANDMARKS: tuple[PoseLandmark, ...] = (
    PoseLandmark.LEFT_SHOULDER,
    PoseLandmark.LEFT_ELBOW,
    PoseLandmark.LEFT_WRIST,
    PoseLandmark.RIGHT_SHOULDER,
    PoseLandmark.RIGHT_ELBOW,
    PoseLandmark.RIGHT_WRIST,
)

#: Bones to draw, as (from, to) pairs. Includes the shoulder line for reference.
ARM_CHAINS: tuple[tuple[PoseLandmark, PoseLandmark], ...] = (
    (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW),
    (PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST),
    (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW),
    (PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST),
    (PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER),
)

#: Below this, MediaPipe is guessing an occluded joint. Treat it as missing.
DEFAULT_VISIBILITY_THRESHOLD = 0.5


@dataclass(frozen=True)
class Point:
    """A landmark.

    Normalized points: x, y in [0, 1] relative to the image; z roughly in the
    same scale, measured from the hips, negative towards the camera.
    World points: metres, origin at the hip centre — scale-invariant, which is
    what the feature layer will want.
    """

    x: float
    y: float
    z: float
    visibility: float = 0.0
    presence: float = 0.0

    def is_reliable(self, threshold: float = DEFAULT_VISIBILITY_THRESHOLD) -> bool:
        return self.visibility >= threshold


@dataclass(frozen=True)
class PoseSnapshot:
    """One frame's worth of pose, or the absence of one.

    `normalized` is keyed by landmark index so a partial pose stays representable;
    `world` may be empty if the estimator did not provide world coordinates.
    """

    frame_index: int
    timestamp: float
    normalized: Mapping[int, Point]
    world: Mapping[int, Point]
    inference_ms: float = 0.0
    #: image width / height. Carried here because normalized coords are squashed:
    #: x is divided by the width and y by the height, so on a 16:9 frame an angle
    #: measured straight off x/y is wrong. The feature layer un-squashes with this.
    aspect: float = 1.0

    @property
    def detected(self) -> bool:
        return bool(self.normalized)

    def point(self, landmark: PoseLandmark) -> Optional[Point]:
        return self.normalized.get(int(landmark))

    def world_point(self, landmark: PoseLandmark) -> Optional[Point]:
        return self.world.get(int(landmark))

    def arm(self, side: Side) -> tuple[Optional[Point], Optional[Point], Optional[Point]]:
        """(shoulder, elbow, wrist) for one side; entries are None when missing."""
        return (self.point(side.shoulder), self.point(side.elbow), self.point(side.wrist))

    def confidence(
        self,
        landmarks: Iterable[PoseLandmark] = ARM_LANDMARKS,
        threshold: float = DEFAULT_VISIBILITY_THRESHOLD,
    ) -> float:
        """Mean visibility over the landmarks we actually care about.

        Full-body mean would be misleading: legs off-frame would drag the number
        down even with both arms perfectly tracked.
        """
        del threshold  # kept for call-site symmetry with is_reliable
        points = [self.point(lm) for lm in landmarks]
        found = [p.visibility for p in points if p is not None]
        return sum(found) / len(found) if found else 0.0

    @classmethod
    def empty(
        cls,
        frame_index: int,
        timestamp: float,
        inference_ms: float = 0.0,
        aspect: float = 1.0,
    ) -> "PoseSnapshot":
        return cls(
            frame_index=frame_index,
            timestamp=timestamp,
            normalized={},
            world={},
            inference_ms=inference_ms,
            aspect=aspect,
        )


def to_pixel(point: Point, width: int, height: int) -> tuple[int, int]:
    """Normalized coords -> pixel coords, clamped to the image.

    MediaPipe can return values slightly outside [0, 1] for joints just off
    frame; drawing those unclamped throws OpenCV.
    """
    x = min(max(point.x, 0.0), 1.0)
    y = min(max(point.y, 0.0), 1.0)
    return (int(round(x * (width - 1))), int(round(y * (height - 1))))

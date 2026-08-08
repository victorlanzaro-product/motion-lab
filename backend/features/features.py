"""Feature Engine — turns one pose snapshot into one row of tabular data.

This is the file where computer vision stops and machine learning starts. From
here on nothing knows about pixels: the classifier will only ever see numbers
like "elbow at 142 degrees, wrists 0.21 shoulder-widths apart".

Every field is Optional. `None` means "we could not see that joint", which is
information, not an error — Sprint 07/08 decides how to treat it when building
the dataset. Filling it with 0 here would be a lie that the model would learn.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

from backend.features.angles import elbow_angle, shoulder_angle
from backend.features.distances import forearm_length, shoulder_width, wrist_distance
from backend.features.positions import (
    wrist_above_shoulder,
    wrist_height,
    wrist_offset_x,
    wrists_crossed,
)
from backend.vision.landmarks import DEFAULT_VISIBILITY_THRESHOLD, PoseSnapshot, Side

#: Column order for the ML dataset. Append only — reordering silently
#: invalidates every model trained before the change.
FEATURE_NAMES: tuple[str, ...] = (
    "left_elbow_angle",
    "right_elbow_angle",
    "left_shoulder_angle",
    "right_shoulder_angle",
    "wrist_distance",
    "left_wrist_height",
    "right_wrist_height",
    "left_wrist_offset_x",
    "right_wrist_offset_x",
    "left_forearm_length",
    "right_forearm_length",
    "left_wrist_above_shoulder",
    "right_wrist_above_shoulder",
    "wrists_crossed",
)


@dataclass(frozen=True)
class FrameFeatures:
    """One frame, described in body-relative terms."""

    frame_index: int
    timestamp: float

    left_elbow_angle: Optional[float] = None
    right_elbow_angle: Optional[float] = None
    left_shoulder_angle: Optional[float] = None
    right_shoulder_angle: Optional[float] = None

    wrist_distance: Optional[float] = None
    left_forearm_length: Optional[float] = None
    right_forearm_length: Optional[float] = None

    left_wrist_height: Optional[float] = None
    right_wrist_height: Optional[float] = None
    left_wrist_offset_x: Optional[float] = None
    right_wrist_offset_x: Optional[float] = None

    left_wrist_above_shoulder: Optional[bool] = None
    right_wrist_above_shoulder: Optional[bool] = None
    wrists_crossed: Optional[bool] = None

    #: Raw scale reference, kept out of FEATURE_NAMES on purpose: it depends on
    #: distance to the camera, so a model that learned from it would only work
    #: at the distance it was trained at.
    shoulder_width: Optional[float] = None

    @property
    def complete(self) -> bool:
        """True when every ML column is filled — the frames worth training on."""
        return all(getattr(self, name) is not None for name in FEATURE_NAMES)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_vector(self) -> list[Optional[float]]:
        """Feature vector in FEATURE_NAMES order; booleans become 1.0/0.0."""
        vector: list[Optional[float]] = []
        for name in FEATURE_NAMES:
            value = getattr(self, name)
            if value is None:
                vector.append(None)
            elif isinstance(value, bool):
                vector.append(1.0 if value else 0.0)
            else:
                vector.append(float(value))
        return vector


def extract(
    snapshot: PoseSnapshot,
    threshold: float = DEFAULT_VISIBILITY_THRESHOLD,
) -> FrameFeatures:
    """Snapshot in, feature row out. Never raises: missing joints become None."""
    base = FrameFeatures(frame_index=snapshot.frame_index, timestamp=snapshot.timestamp)
    if not snapshot.detected:
        return base

    return FrameFeatures(
        frame_index=snapshot.frame_index,
        timestamp=snapshot.timestamp,
        left_elbow_angle=elbow_angle(snapshot, Side.LEFT, threshold),
        right_elbow_angle=elbow_angle(snapshot, Side.RIGHT, threshold),
        left_shoulder_angle=shoulder_angle(snapshot, Side.LEFT, threshold),
        right_shoulder_angle=shoulder_angle(snapshot, Side.RIGHT, threshold),
        wrist_distance=wrist_distance(snapshot, threshold),
        left_forearm_length=forearm_length(snapshot, Side.LEFT, threshold),
        right_forearm_length=forearm_length(snapshot, Side.RIGHT, threshold),
        left_wrist_height=wrist_height(snapshot, Side.LEFT, threshold),
        right_wrist_height=wrist_height(snapshot, Side.RIGHT, threshold),
        left_wrist_offset_x=wrist_offset_x(snapshot, Side.LEFT, threshold),
        right_wrist_offset_x=wrist_offset_x(snapshot, Side.RIGHT, threshold),
        left_wrist_above_shoulder=wrist_above_shoulder(snapshot, Side.LEFT, threshold),
        right_wrist_above_shoulder=wrist_above_shoulder(snapshot, Side.RIGHT, threshold),
        wrists_crossed=wrists_crossed(snapshot, threshold),
        shoulder_width=shoulder_width(snapshot, threshold),
    )

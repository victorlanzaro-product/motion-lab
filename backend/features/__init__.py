"""Feature Engine: landmarks in, body-relative numbers out."""

from backend.features.angles import angle_between, elbow_angle, shoulder_angle
from backend.features.distances import distance, forearm_length, shoulder_width, wrist_distance
from backend.features.features import FEATURE_NAMES, FrameFeatures, extract
from backend.features.positions import (
    wrist_above_shoulder,
    wrist_height,
    wrist_offset_x,
    wrists_crossed,
)
from backend.features.velocity import (
    ArmMotion,
    Direction,
    MotionConfig,
    MotionState,
    MotionTracker,
    SignalTrack,
    Trail,
    classify,
)

__all__ = [
    "FEATURE_NAMES",
    "ArmMotion",
    "Direction",
    "FrameFeatures",
    "MotionConfig",
    "MotionState",
    "MotionTracker",
    "SignalTrack",
    "Trail",
    "angle_between",
    "classify",
    "distance",
    "elbow_angle",
    "extract",
    "forearm_length",
    "shoulder_angle",
    "shoulder_width",
    "wrist_above_shoulder",
    "wrist_distance",
    "wrist_height",
    "wrist_offset_x",
    "wrists_crossed",
]

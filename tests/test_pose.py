"""Pose Engine tests. No webcam and no model file: both are faked."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from backend.camera.capture import Frame
from backend.vision import (
    ARM_LANDMARKS,
    Point,
    PoseDetector,
    PoseLandmark,
    PoseSnapshot,
    Side,
    draw_arms,
    to_pixel,
)


def point(x=0.5, y=0.5, z=0.0, visibility=0.9) -> Point:
    return Point(x=x, y=y, z=z, visibility=visibility, presence=visibility)


def snapshot(overrides: dict[int, Point] | None = None) -> PoseSnapshot:
    normalized = {int(lm): point() for lm in ARM_LANDMARKS}
    normalized.update(overrides or {})
    return PoseSnapshot(frame_index=0, timestamp=0.0, normalized=normalized, world={})


# --- Side / landmark table --------------------------------------------


def test_side_maps_to_the_right_landmark_indices():
    assert Side.LEFT.shoulder is PoseLandmark.LEFT_SHOULDER
    assert Side.RIGHT.wrist is PoseLandmark.RIGHT_WRIST
    assert int(PoseLandmark.LEFT_SHOULDER) == 11  # BlazePose index table
    assert int(PoseLandmark.RIGHT_WRIST) == 16


def test_arm_landmarks_are_the_six_joints_v1_cares_about():
    assert len(ARM_LANDMARKS) == 6
    assert PoseLandmark.NOSE not in ARM_LANDMARKS


# --- PoseSnapshot ------------------------------------------------------


def test_empty_snapshot_is_not_detected():
    empty = PoseSnapshot.empty(frame_index=3, timestamp=1.5)
    assert empty.detected is False
    assert empty.point(PoseLandmark.LEFT_WRIST) is None
    assert empty.confidence() == 0.0


def test_arm_returns_shoulder_elbow_wrist_in_order():
    snap = snapshot(
        {
            int(PoseLandmark.LEFT_SHOULDER): point(x=0.1),
            int(PoseLandmark.LEFT_ELBOW): point(x=0.2),
            int(PoseLandmark.LEFT_WRIST): point(x=0.3),
        }
    )
    shoulder, elbow, wrist = snap.arm(Side.LEFT)
    assert (shoulder.x, elbow.x, wrist.x) == (0.1, 0.2, 0.3)


def test_missing_joint_comes_back_as_none_not_zero():
    partial = PoseSnapshot(
        frame_index=0,
        timestamp=0.0,
        normalized={int(PoseLandmark.LEFT_SHOULDER): point()},
        world={},
    )
    shoulder, elbow, wrist = partial.arm(Side.LEFT)
    assert shoulder is not None
    assert elbow is None and wrist is None


def test_confidence_averages_only_the_arm_landmarks():
    # A perfectly tracked pair of arms plus a garbage nose must stay at 1.0.
    normalized = {int(lm): point(visibility=1.0) for lm in ARM_LANDMARKS}
    normalized[int(PoseLandmark.NOSE)] = point(visibility=0.0)
    snap = PoseSnapshot(frame_index=0, timestamp=0.0, normalized=normalized, world={})
    assert snap.confidence() == pytest.approx(1.0)


def test_is_reliable_uses_the_visibility_threshold():
    assert point(visibility=0.6).is_reliable() is True
    assert point(visibility=0.4).is_reliable() is False


# --- to_pixel ----------------------------------------------------------


def test_to_pixel_maps_normalized_coords_to_the_image():
    assert to_pixel(point(x=0.0, y=0.0), 640, 480) == (0, 0)
    assert to_pixel(point(x=1.0, y=1.0), 640, 480) == (639, 479)
    assert to_pixel(point(x=0.5, y=0.5), 641, 481) == (320, 240)


def test_to_pixel_clamps_landmarks_that_fall_off_frame():
    # MediaPipe extrapolates joints just outside the image; unclamped values
    # would index outside the array and crash OpenCV drawing.
    assert to_pixel(point(x=-0.4, y=1.9), 640, 480) == (0, 479)


# --- PoseDetector ------------------------------------------------------


class FakeLandmarker:
    """Stand-in for vision.PoseLandmarker."""

    def __init__(self, landmarks: list | None = None):
        self.landmarks = landmarks
        self.timestamps: list[int] = []
        self.closed = False

    def detect_for_video(self, _image, timestamp_ms: int):
        self.timestamps.append(timestamp_ms)
        if not self.landmarks:
            return SimpleNamespace(pose_landmarks=[], pose_world_landmarks=[])
        return SimpleNamespace(
            pose_landmarks=[self.landmarks], pose_world_landmarks=[self.landmarks]
        )

    def close(self):
        self.closed = True


def fake_landmark(x=0.5, y=0.5, z=0.0, visibility=0.9):
    return SimpleNamespace(x=x, y=y, z=z, visibility=visibility, presence=visibility)


def frame(index: int, timestamp: float) -> Frame:
    return Frame(image=np.zeros((48, 64, 3), dtype=np.uint8), index=index, timestamp=timestamp)


def detector_with(landmarker: FakeLandmarker) -> PoseDetector:
    return PoseDetector(landmarker_factory=lambda _cfg: landmarker)


def test_detect_converts_landmarks_into_snapshot_points():
    landmarker = FakeLandmarker([fake_landmark(x=0.25, y=0.75, visibility=0.8)] * 33)
    with detector_with(landmarker) as detector:
        snap = detector.detect(frame(0, 1.0))
    assert snap.detected is True
    wrist = snap.point(PoseLandmark.LEFT_WRIST)
    assert (wrist.x, wrist.y, wrist.visibility) == (0.25, 0.75, 0.8)
    assert snap.world_point(PoseLandmark.LEFT_WRIST) is not None
    assert snap.inference_ms >= 0.0


def test_detect_without_a_body_returns_an_empty_snapshot():
    with detector_with(FakeLandmarker(None)) as detector:
        snap = detector.detect(frame(7, 2.0))
    assert snap.detected is False
    assert snap.frame_index == 7  # still tells us which frame it was


def test_timestamps_are_strictly_increasing():
    # MediaPipe VIDEO mode raises if a timestamp repeats. Two frames captured
    # inside the same millisecond must not collide.
    landmarker = FakeLandmarker(None)
    with detector_with(landmarker) as detector:
        detector.detect(frame(0, 1.0000))
        detector.detect(frame(1, 1.0001))  # same millisecond after rounding
        detector.detect(frame(2, 1.0002))
    assert landmarker.timestamps == sorted(set(landmarker.timestamps))
    assert len(landmarker.timestamps) == 3


def test_closing_the_detector_closes_the_landmarker():
    landmarker = FakeLandmarker(None)
    with detector_with(landmarker):
        pass
    assert landmarker.closed is True


# --- drawing -----------------------------------------------------------


def test_draw_arms_is_a_noop_without_a_pose():
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    draw_arms(image, PoseSnapshot.empty(0, 0.0))
    assert image.sum() == 0


def test_draw_arms_paints_when_landmarks_are_reliable():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    draw_arms(image, snapshot())
    assert image.sum() > 0


def test_low_visibility_joints_do_not_draw_bones():
    hidden = {int(lm): point(visibility=0.1) for lm in ARM_LANDMARKS}
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    draw_arms(image, PoseSnapshot(frame_index=0, timestamp=0.0, normalized=hidden, world={}))
    # Points are still marked (in the "weak" colour) but no bone line is drawn:
    # a guessed joint must not look like a tracked limb.
    painted = np.count_nonzero(image.any(axis=2))
    assert 0 < painted < 2000

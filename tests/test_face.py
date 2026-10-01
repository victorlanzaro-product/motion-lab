"""Face Engine tests. No webcam and no model file: both are faked, same
shape as `test_pose.py`.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from backend.camera.capture import Frame
from backend.features.face_features import (
    FACE_SIGNAL_NAMES,
    FaceSignals,
    extract_face,
)
from backend.features.face_motion import FaceMotionTracker
from backend.vision.face_detector import FaceDetector
from backend.vision.face_landmarks import (
    FACE_CHAIN_GROUPS,
    FACE_CHAINS,
    FACE_OVERLAY_LANDMARKS,
    FaceSnapshot,
    euler_from_matrix,
)

#: A face landmark list long enough that every index in `FACE_OVERLAY_LANDMARKS`
#: resolves to a real point instead of `object()` (see `FakeFaceLandmarker`
#: below) -- MediaPipe's real Face Landmarker always returns all 478.
FULL_MESH_SIZE = 478


def _fake_mesh_landmarks(offset: float = 0.0) -> list[SimpleNamespace]:
    """478 points spread evenly across [0, 1] -- distinctive but deterministic,
    so a test can pick any index in `FACE_OVERLAY_LANDMARKS` and know its
    coordinate without needing the real geometry.
    """
    return [
        SimpleNamespace(x=(i % 100) / 100 + offset, y=(i % 100) / 100 + offset, z=0.0)
        for i in range(FULL_MESH_SIZE)
    ]


# --- euler_from_matrix ---------------------------------------------------


def _rx(deg: float) -> list[list[float]]:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return [[1, 0, 0], [0, c, -s], [0, s, c]]


def _ry(deg: float) -> list[list[float]]:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return [[c, 0, s], [0, 1, 0], [-s, 0, c]]


def _rz(deg: float) -> list[list[float]]:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return [[c, -s, 0], [s, c, 0], [0, 0, 1]]


def test_identity_matrix_is_zero_orientation():
    assert euler_from_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) == pytest.approx(
        (0.0, 0.0, 0.0), abs=1e-6
    )


def test_yaw_only_rotation_is_isolated_from_pitch_and_roll():
    yaw, pitch, roll = euler_from_matrix(_ry(30))
    assert yaw == pytest.approx(30.0, abs=1e-6)
    assert pitch == pytest.approx(0.0, abs=1e-6)
    assert roll == pytest.approx(0.0, abs=1e-6)


def test_pitch_only_rotation_is_isolated_from_yaw_and_roll():
    yaw, pitch, roll = euler_from_matrix(_rx(-60))
    assert pitch == pytest.approx(-60.0, abs=1e-6)
    assert yaw == pytest.approx(0.0, abs=1e-6)
    assert roll == pytest.approx(0.0, abs=1e-6)


def test_roll_only_rotation_is_isolated_from_yaw_and_pitch():
    yaw, pitch, roll = euler_from_matrix(_rz(-70))
    assert roll == pytest.approx(-70.0, abs=1e-6)
    assert yaw == pytest.approx(0.0, abs=1e-6)
    assert pitch == pytest.approx(0.0, abs=1e-6)


# --- FaceSnapshot --------------------------------------------------------


def test_empty_snapshot_is_not_detected():
    empty = FaceSnapshot.empty(frame_index=3, timestamp=1.5)
    assert empty.detected is False
    assert empty.blendshapes == {}
    assert empty.blendshape("jawOpen") is None


# --- FaceDetector --------------------------------------------------------


class FakeCategory:
    def __init__(self, category_name: str, score: float):
        self.category_name = category_name
        self.score = score


class FakeFaceLandmarker:
    """Stand-in for vision.FaceLandmarker."""

    def __init__(
        self,
        blendshapes: dict[str, float] | None = None,
        matrix: list[list[float]] | None = None,
        has_face: bool = True,
    ):
        self.blendshapes = blendshapes or {}
        self.matrix = matrix
        self.has_face = has_face
        self.timestamps: list[int] = []
        self.closed = False

    def detect_for_video(self, _image, timestamp_ms: int):
        self.timestamps.append(timestamp_ms)
        if not self.has_face:
            return SimpleNamespace(
                face_landmarks=[], face_blendshapes=[], facial_transformation_matrixes=[]
            )
        categories = [FakeCategory(name, score) for name, score in self.blendshapes.items()]
        matrixes = [self.matrix] if self.matrix is not None else []
        return SimpleNamespace(
            face_landmarks=[_fake_mesh_landmarks()],
            face_blendshapes=[categories],
            facial_transformation_matrixes=matrixes,
        )

    def close(self):
        self.closed = True


def frame(index: int, timestamp: float) -> Frame:
    return Frame(image=np.zeros((48, 64, 3), dtype=np.uint8), index=index, timestamp=timestamp)


def detector_with(landmarker: FakeFaceLandmarker) -> FaceDetector:
    return FaceDetector(landmarker_factory=lambda _cfg: landmarker)


def test_detect_without_a_face_returns_an_empty_snapshot():
    with detector_with(FakeFaceLandmarker(has_face=False)) as detector:
        snap = detector.detect(frame(7, 2.0))
    assert snap.detected is False
    assert snap.frame_index == 7  # still tells us which frame it was


def test_detect_maps_blendshapes_and_head_pose():
    landmarker = FakeFaceLandmarker(
        blendshapes={"mouthSmileLeft": 0.8, "jawOpen": 0.4},
        matrix=_ry(15),
    )
    with detector_with(landmarker) as detector:
        snap = detector.detect(frame(0, 1.0))
    assert snap.detected is True
    assert snap.blendshape("mouthSmileLeft") == pytest.approx(0.8)
    assert snap.blendshape("jawOpen") == pytest.approx(0.4)
    assert snap.blendshape("eyeBlinkLeft") is None  # not present this frame, not a fabricated 0
    assert snap.head_yaw == pytest.approx(15.0, abs=1e-6)
    assert snap.inference_ms >= 0.0


def test_detect_without_a_transformation_matrix_leaves_head_pose_none():
    landmarker = FakeFaceLandmarker(blendshapes={"jawOpen": 0.1}, matrix=None)
    with detector_with(landmarker) as detector:
        snap = detector.detect(frame(0, 1.0))
    assert snap.detected is True
    assert snap.head_yaw is None and snap.head_pitch is None and snap.head_roll is None


def test_timestamps_are_strictly_increasing():
    landmarker = FakeFaceLandmarker(has_face=False)
    with detector_with(landmarker) as detector:
        detector.detect(frame(0, 1.0000))
        detector.detect(frame(1, 1.0001))  # same millisecond after rounding
        detector.detect(frame(2, 1.0002))
    assert landmarker.timestamps == sorted(set(landmarker.timestamps))
    assert len(landmarker.timestamps) == 3


def test_closing_the_detector_closes_the_landmarker():
    landmarker = FakeFaceLandmarker(has_face=False)
    with detector_with(landmarker):
        pass
    assert landmarker.closed is True


# --- overlay mesh (Sprint 11) ----------------------------------------------


def test_detect_populates_the_overlay_mesh_from_the_same_result():
    """No second inference: the mesh comes out of the exact `result` object
    `detect_for_video` already returned for blendshapes/head pose above."""
    landmarker = FakeFaceLandmarker(blendshapes={"jawOpen": 0.2})
    with detector_with(landmarker) as detector:
        snap = detector.detect(frame(0, 1.0))
    assert set(snap.mesh) == set(FACE_OVERLAY_LANDMARKS)
    sample_index = FACE_OVERLAY_LANDMARKS[0]
    expected = (sample_index % 100) / 100
    assert snap.mesh[sample_index].x == pytest.approx(expected)
    assert snap.mesh[sample_index].y == pytest.approx(expected)


def test_detect_without_a_face_leaves_the_mesh_empty():
    with detector_with(FakeFaceLandmarker(has_face=False)) as detector:
        snap = detector.detect(frame(0, 1.0))
    assert snap.mesh == {}


def test_overlay_mesh_is_a_strict_subset_of_the_full_478_point_mesh():
    """The whole point of `FACE_OVERLAY_LANDMARKS`: fewer than the 478 raw
    mesh points MediaPipe returns, not all of them."""
    assert 0 < len(FACE_OVERLAY_LANDMARKS) < FULL_MESH_SIZE
    assert all(0 <= index < FULL_MESH_SIZE for index in FACE_OVERLAY_LANDMARKS)


def test_every_chain_endpoint_is_covered_by_the_overlay_landmark_list():
    """`FACE_OVERLAY_LANDMARKS` has to be exactly the set of points the chains
    reference -- too few and a chain would look up a missing landmark on the
    wire; too many and points travel that nothing ever draws."""
    endpoints = {index for pair in FACE_CHAINS for index in pair}
    assert endpoints == set(FACE_OVERLAY_LANDMARKS)


def test_chain_groups_cover_the_seven_named_face_regions():
    assert set(FACE_CHAIN_GROUPS) == {
        "face_oval",
        "left_eyebrow",
        "right_eyebrow",
        "left_eye",
        "right_eye",
        "nose",
        "lips",
    }
    # Every group is itself non-empty and only ever references real points.
    for pairs in FACE_CHAIN_GROUPS.values():
        assert pairs
        assert all(0 <= a < FULL_MESH_SIZE and 0 <= b < FULL_MESH_SIZE for a, b in pairs)


# --- extract_face ---------------------------------------------------------


def test_no_face_yields_every_signal_none():
    signals = extract_face(FaceSnapshot.empty(0, 0.0))
    assert signals.complete is False
    for name in FACE_SIGNAL_NAMES:
        assert getattr(signals, name) is None


def test_smile_averages_both_mouth_corners():
    snap = FaceSnapshot(
        frame_index=0,
        timestamp=0.0,
        detected=True,
        blendshapes={"mouthSmileLeft": 0.8, "mouthSmileRight": 0.6},
    )
    assert extract_face(snap).smile == pytest.approx(0.7)


def test_smile_falls_back_to_whichever_corner_is_present():
    # One side occluded must not zero out the whole signal.
    snap = FaceSnapshot(
        frame_index=0, timestamp=0.0, detected=True, blendshapes={"mouthSmileLeft": 0.9}
    )
    assert extract_face(snap).smile == pytest.approx(0.9)


def test_blink_is_independent_per_eye():
    snap = FaceSnapshot(
        frame_index=0,
        timestamp=0.0,
        detected=True,
        blendshapes={"eyeBlinkLeft": 0.9, "eyeBlinkRight": 0.05},
    )
    signals = extract_face(snap)
    assert signals.eye_blink_left == pytest.approx(0.9)
    assert signals.eye_blink_right == pytest.approx(0.05)


def test_brow_raise_averages_available_segments():
    snap = FaceSnapshot(
        frame_index=0,
        timestamp=0.0,
        detected=True,
        blendshapes={"browInnerUp": 0.6, "browOuterUpLeft": 0.2, "browOuterUpRight": 0.4},
    )
    assert extract_face(snap).brow_raise == pytest.approx(0.4)


def test_head_pose_passes_through_from_the_snapshot():
    snap = FaceSnapshot(
        frame_index=0,
        timestamp=0.0,
        detected=True,
        blendshapes={},
        head_yaw=12.0,
        head_pitch=-3.0,
        head_roll=1.5,
    )
    signals = extract_face(snap)
    assert (signals.head_yaw, signals.head_pitch, signals.head_roll) == (12.0, -3.0, 1.5)


# --- FaceMotionTracker ------------------------------------------------------


def face_row(index: int, timestamp: float, **values) -> FaceSignals:
    return FaceSignals(frame_index=index, timestamp=timestamp, **values)


def test_blink_rate_is_none_until_the_signal_is_available():
    tracker = FaceMotionTracker()
    state = tracker.update(face_row(0, 0.0))  # no face this frame
    assert state.blink_rate_left is None
    assert state.blink_rate_right is None


def test_blink_rate_counts_rising_edges_not_frames_held_closed():
    tracker = FaceMotionTracker(blink_window_seconds=60.0, blink_threshold=0.5)
    # One blink: closed for three frames, then open again.
    sequence = [0.1, 0.9, 0.9, 0.9, 0.1, 0.1]
    state = None
    for i, value in enumerate(sequence):
        state = tracker.update(face_row(i, i / 30.0, eye_blink_left=value))
    # 1 blink counted in a 60s window -> rate = 1 * (60/60) = 1.0/minute.
    assert state.blink_rate_left == pytest.approx(1.0)


def test_two_separate_blinks_count_twice():
    tracker = FaceMotionTracker(blink_window_seconds=60.0, blink_threshold=0.5)
    sequence = [0.1, 0.9, 0.1, 0.1, 0.9, 0.1]
    state = None
    for i, value in enumerate(sequence):
        state = tracker.update(face_row(i, i / 30.0, eye_blink_left=value))
    assert state.blink_rate_left == pytest.approx(2.0)


def test_smile_velocity_reads_a_constant_rise():
    tracker = FaceMotionTracker()
    state = None
    fps = 30.0
    for i in range(20):
        t = i / fps
        state = tracker.update(face_row(i, t, smile=0.5 * t))
    assert state.smile_velocity == pytest.approx(0.5, abs=0.05)


def test_reset_clears_blink_history_and_trend():
    tracker = FaceMotionTracker(blink_window_seconds=60.0, blink_threshold=0.5)
    tracker.update(face_row(0, 0.0, eye_blink_left=0.9, smile=0.5))
    tracker.update(face_row(1, 1.0, eye_blink_left=0.1, smile=0.6))
    tracker.reset()
    state = tracker.update(face_row(2, 2.0, eye_blink_left=0.9, smile=0.5))
    # Fresh after reset: this frame's own rising edge is the only blink seen,
    # and there is no trend history left over from before the reset.
    assert state.blink_rate_left == pytest.approx(1.0)
    assert state.smile_velocity is None

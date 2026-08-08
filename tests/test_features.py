"""Feature Engine tests, built on synthetic poses with known geometry."""

from __future__ import annotations

import pytest

from backend.features import (
    FEATURE_NAMES,
    angle_between,
    elbow_angle,
    extract,
    shoulder_angle,
    shoulder_width,
    wrist_distance,
    wrist_height,
    wrists_crossed,
)
from backend.vision.landmarks import Point, PoseLandmark, PoseSnapshot, Side


def p(x: float, y: float, visibility: float = 0.9) -> Point:
    return Point(x=x, y=y, z=0.0, visibility=visibility, presence=visibility)


def pose(points: dict, aspect: float = 1.0) -> PoseSnapshot:
    return PoseSnapshot(
        frame_index=0,
        timestamp=0.0,
        normalized={int(k): v for k, v in points.items()},
        world={},
        aspect=aspect,
    )


def standing(overrides: dict | None = None) -> PoseSnapshot:
    """Arms hanging down, seen head-on. Image x grows to the right, so the
    person's LEFT shoulder sits at the larger x."""
    points = {
        PoseLandmark.LEFT_SHOULDER: p(0.60, 0.40),
        PoseLandmark.RIGHT_SHOULDER: p(0.40, 0.40),
        PoseLandmark.LEFT_ELBOW: p(0.62, 0.55),
        PoseLandmark.RIGHT_ELBOW: p(0.38, 0.55),
        PoseLandmark.LEFT_WRIST: p(0.64, 0.70),
        PoseLandmark.RIGHT_WRIST: p(0.36, 0.70),
        PoseLandmark.LEFT_HIP: p(0.57, 0.75),
        PoseLandmark.RIGHT_HIP: p(0.43, 0.75),
    }
    points.update(overrides or {})
    return pose(points)


# --- angle_between -----------------------------------------------------


def test_right_angle_is_ninety_degrees():
    assert angle_between(p(0.0, 1.0), p(0.0, 0.0), p(1.0, 0.0)) == pytest.approx(90.0)


def test_straight_line_is_one_hundred_eighty_degrees():
    assert angle_between(p(0.0, 0.0), p(0.5, 0.0), p(1.0, 0.0)) == pytest.approx(180.0)


def test_folded_back_on_itself_is_zero_degrees():
    assert angle_between(p(1.0, 0.0), p(0.0, 0.0), p(1.0, 0.0)) == pytest.approx(0.0)


def test_zero_length_side_has_no_angle():
    assert angle_between(p(0.5, 0.5), p(0.5, 0.5), p(1.0, 0.0)) is None


def test_aspect_ratio_correction_recovers_the_true_angle():
    # A physical 45-degree arm filmed on a 16:9 frame: 200px right, 200px up.
    width, height = 1280, 720
    vertex = p(0.5, 0.5)
    horizontal = p(0.5 + 200 / width, 0.5)
    diagonal = p(0.5 + 200 / width, 0.5 - 200 / height)

    assert angle_between(horizontal, vertex, diagonal, aspect=width / height) == pytest.approx(45.0)
    # Ignoring the squash skews it by ~15 degrees — enough to poison a classifier.
    assert angle_between(horizontal, vertex, diagonal, aspect=1.0) != pytest.approx(45.0, abs=1.0)


# --- joint angles ------------------------------------------------------


def test_straight_arm_reads_about_one_eighty():
    snapshot = standing(
        {
            PoseLandmark.LEFT_SHOULDER: p(0.6, 0.3),
            PoseLandmark.LEFT_ELBOW: p(0.6, 0.5),
            PoseLandmark.LEFT_WRIST: p(0.6, 0.7),
        }
    )
    assert elbow_angle(snapshot, Side.LEFT) == pytest.approx(180.0)


def test_bent_arm_reads_ninety():
    snapshot = standing(
        {
            PoseLandmark.LEFT_SHOULDER: p(0.6, 0.3),
            PoseLandmark.LEFT_ELBOW: p(0.6, 0.5),
            PoseLandmark.LEFT_WRIST: p(0.8, 0.5),
        }
    )
    assert elbow_angle(snapshot, Side.LEFT) == pytest.approx(90.0)


def test_arm_raised_sideways_reads_ninety_at_the_shoulder():
    # elbow straight out to the side, hip straight below the shoulder
    snapshot = standing(
        {
            PoseLandmark.LEFT_SHOULDER: p(0.5, 0.4),
            PoseLandmark.LEFT_ELBOW: p(0.7, 0.4),
            PoseLandmark.LEFT_HIP: p(0.5, 0.7),
        }
    )
    assert shoulder_angle(snapshot, Side.LEFT) == pytest.approx(90.0)


def test_hidden_wrist_yields_no_angle_instead_of_a_wrong_one():
    snapshot = standing({PoseLandmark.LEFT_WRIST: p(0.64, 0.70, visibility=0.2)})
    assert elbow_angle(snapshot, Side.LEFT) is None
    assert elbow_angle(snapshot, Side.RIGHT) is not None  # the other arm still works


# --- distances ---------------------------------------------------------


def test_shoulder_width_is_the_gap_between_shoulders():
    assert shoulder_width(standing()) == pytest.approx(0.20)


def test_collapsed_shoulders_are_rejected_as_a_bad_detection():
    snapshot = standing({PoseLandmark.RIGHT_SHOULDER: p(0.60, 0.40)})
    assert shoulder_width(snapshot) is None
    assert wrist_distance(snapshot) is None


def test_wrist_distance_is_measured_in_shoulder_widths():
    # wrists 0.28 apart, shoulders 0.20 apart -> 1.4
    assert wrist_distance(standing()) == pytest.approx(1.4)


def test_wrist_distance_survives_stepping_away_from_the_camera():
    """The whole point of scaling: same pose, half the size, same feature."""
    near = standing()
    far = pose(
        {
            index: p(0.5 + (point.x - 0.5) / 2, 0.5 + (point.y - 0.5) / 2)
            for index, point in near.normalized.items()
        }
    )
    assert wrist_distance(far) == pytest.approx(wrist_distance(near))
    # the unscaled version does NOT survive it, which is why we scale
    assert wrist_distance(far, scaled=False) == pytest.approx(
        wrist_distance(near, scaled=False) / 2
    )


# --- positions ---------------------------------------------------------


def test_wrist_above_shoulder_is_positive_despite_y_growing_downwards():
    raised = standing({PoseLandmark.LEFT_WRIST: p(0.60, 0.20)})
    assert wrist_height(raised, Side.LEFT) == pytest.approx(1.0)  # 0.20 above / 0.20 width
    assert wrist_height(raised, Side.RIGHT) < 0  # the other arm still hangs down


def test_wrists_crossed_detects_each_wrist_past_the_midline():
    assert wrists_crossed(standing()) is False
    crossed = standing(
        {
            PoseLandmark.LEFT_WRIST: p(0.40, 0.35),
            PoseLandmark.RIGHT_WRIST: p(0.60, 0.35),
        }
    )
    assert wrists_crossed(crossed) is True


# --- extract -----------------------------------------------------------


def test_extract_on_a_frame_without_a_body_is_all_none():
    features = extract(PoseSnapshot.empty(frame_index=4, timestamp=2.0))
    assert features.frame_index == 4
    assert features.complete is False
    assert all(getattr(features, name) is None for name in FEATURE_NAMES)


def test_extract_fills_every_column_for_a_clean_pose():
    features = extract(standing())
    assert features.complete is True
    assert features.left_elbow_angle is not None
    assert features.shoulder_width == pytest.approx(0.20)


def test_a_partial_pose_stays_partial_instead_of_being_faked():
    features = extract(standing({PoseLandmark.RIGHT_WRIST: p(0.36, 0.70, visibility=0.1)}))
    assert features.complete is False
    assert features.right_elbow_angle is None
    assert features.wrist_distance is None
    assert features.left_elbow_angle is not None  # the visible half is still reported


def test_vector_matches_the_declared_column_order():
    vector = extract(standing()).to_vector()
    assert len(vector) == len(FEATURE_NAMES)
    crossed_index = FEATURE_NAMES.index("wrists_crossed")
    assert vector[crossed_index] == 0.0  # booleans arrive as floats, not True/False


def test_vector_keeps_missing_values_as_none():
    vector = extract(PoseSnapshot.empty(0, 0.0)).to_vector()
    assert vector == [None] * len(FEATURE_NAMES)

"""Dataset Writer tests — a temp CSV, synthetic rows, no camera involved."""

from __future__ import annotations

import csv

import pytest

from backend.dataset import DATASET_COLUMNS, DatasetWriter
from backend.features import FEATURE_NAMES, ArmMotion, Direction, FrameFeatures, MotionState


def complete_features(**overrides) -> FrameFeatures:
    """Every FEATURE_NAMES column filled — the only kind of row worth training on."""
    values = {name: 1.0 for name in FEATURE_NAMES}
    for name in ("wrists_crossed", "left_wrist_above_shoulder", "right_wrist_above_shoulder"):
        values[name] = False
    values.update(overrides)
    return FrameFeatures(frame_index=0, timestamp=0.0, **values)


def motion(direction: Direction = Direction.STILL, **overrides) -> MotionState:
    left = ArmMotion(direction=direction, **overrides)
    return MotionState(frame_index=0, timestamp=0.0, left=left, right=ArmMotion())


def read_rows(path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


# --- header / file lifecycle --------------------------------------------


def test_writing_creates_the_file_with_a_header(tmp_path):
    path = tmp_path / "dataset.csv"
    with DatasetWriter(path, session_id="s1") as writer:
        writer.write("idle", complete_features(), motion())

    with path.open(newline="") as handle:
        header = next(csv.reader(handle))
    assert tuple(header) == DATASET_COLUMNS


def test_multiple_sessions_append_without_duplicating_the_header(tmp_path):
    path = tmp_path / "dataset.csv"
    with DatasetWriter(path, session_id="s1") as writer:
        writer.write("idle", complete_features(), motion())
    with DatasetWriter(path, session_id="s2") as writer:
        writer.write("wave", complete_features(), motion())

    rows = read_rows(path)
    assert len(rows) == 2
    assert [row["session_id"] for row in rows] == ["s1", "s2"]


# --- incomplete frames ----------------------------------------------------


def test_incomplete_frames_are_skipped_and_counted(tmp_path):
    path = tmp_path / "dataset.csv"
    incomplete = FrameFeatures(frame_index=0, timestamp=0.0)  # every column None

    with DatasetWriter(path, session_id="s1") as writer:
        ok = writer.write("idle", incomplete, motion())
        assert ok is False
        assert writer.skipped == 1
        assert writer.total == 0

    assert read_rows(path) == []  # header only, no data row


# --- counts -----------------------------------------------------------------


def test_counts_track_samples_per_label(tmp_path):
    path = tmp_path / "dataset.csv"
    with DatasetWriter(path, session_id="s1") as writer:
        writer.write("idle", complete_features(), motion())
        writer.write("idle", complete_features(), motion())
        writer.write("wave", complete_features(), motion())

    assert writer.counts == {"idle": 2, "wave": 1}
    assert writer.total == 3


# --- row content --------------------------------------------------------


def test_a_feature_value_round_trips_through_the_csv(tmp_path):
    path = tmp_path / "dataset.csv"
    with DatasetWriter(path, session_id="s1") as writer:
        writer.write("arm_raised", complete_features(left_elbow_angle=123.0), motion())

    row = read_rows(path)[0]
    assert float(row["left_elbow_angle"]) == pytest.approx(123.0)
    assert row["label"] == "arm_raised"


def test_motion_columns_carry_velocity_for_the_wave_label(tmp_path):
    """The whole reason motion rides along: a static feature snapshot alone
    cannot tell a wave from any other momentary arm position."""
    path = tmp_path / "dataset.csv"
    swinging = motion(velocity_y=2.5, direction=Direction.UP)
    with DatasetWriter(path, session_id="s1") as writer:
        writer.write("wave", complete_features(), swinging)

    row = read_rows(path)[0]
    assert float(row["left_velocity_y"]) == pytest.approx(2.5)
    assert row["left_direction"] == "up"

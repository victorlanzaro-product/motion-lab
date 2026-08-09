"""Dataset Writer tests — a temp CSV, synthetic rows, no camera involved."""

from __future__ import annotations

import csv

import pytest

from backend.dataset import DATASET_COLUMNS, DEFAULT_LABELS, DatasetWriter, DatasetWriterError
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
        writer.write("arm_raised", complete_features(), motion())

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
        writer.write("arm_raised", complete_features(), motion())

    assert writer.counts == {"idle": 2, "arm_raised": 1}
    assert writer.total == 3


# --- row content --------------------------------------------------------


def test_a_feature_value_round_trips_through_the_csv(tmp_path):
    path = tmp_path / "dataset.csv"
    with DatasetWriter(path, session_id="s1") as writer:
        writer.write("arm_raised", complete_features(left_elbow_angle=123.0), motion())

    row = read_rows(path)[0]
    assert float(row["left_elbow_angle"]) == pytest.approx(123.0)
    assert row["label"] == "arm_raised"


def test_motion_columns_carry_velocity_regardless_of_label(tmp_path):
    """`MOTION_COLUMNS` rides along for every label, not just a hypothetical
    motion-based one — this is what would let a future label separate on arm
    speed, even though "wave" itself never became a trainable label (see
    `test_write_rejects_the_removed_wave_label` below)."""
    path = tmp_path / "dataset.csv"
    swinging = motion(velocity_y=2.5, direction=Direction.UP)
    with DatasetWriter(path, session_id="s1") as writer:
        writer.write("arm_raised", complete_features(), swinging)

    row = read_rows(path)[0]
    assert float(row["left_velocity_y"]) == pytest.approx(2.5)
    assert row["left_direction"] == "up"


# --- "wave" removed from the trainable vocabulary (Sprint 11, product decision) --


def test_default_labels_no_longer_include_wave():
    """Product decision: `wave` stays a GestureEngine rule, never an ML class,
    until the per-frame vector gets a real temporal window."""
    assert "wave" not in DEFAULT_LABELS
    assert DEFAULT_LABELS == ("arm_raised", "arms_crossed", "arms_open", "idle")


def test_write_rejects_the_removed_wave_label(tmp_path):
    path = tmp_path / "dataset.csv"
    with DatasetWriter(path, session_id="s1") as writer:
        with pytest.raises(DatasetWriterError, match="trainable ML vocabulary"):
            writer.write("wave", complete_features(), motion())
    assert read_rows(path) == []  # header only, the rejected row never landed


# --- only DEFAULT_LABELS is accepted (post-Sprint-11 hardening) -----------


def test_write_accepts_every_default_label(tmp_path):
    path = tmp_path / "dataset.csv"
    with DatasetWriter(path, session_id="s1") as writer:
        for label in DEFAULT_LABELS:
            assert writer.write(label, complete_features(), motion()) is True
    assert writer.total == len(DEFAULT_LABELS)


def test_write_rejects_an_unknown_label(tmp_path):
    path = tmp_path / "dataset.csv"
    with DatasetWriter(path, session_id="s1") as writer:
        with pytest.raises(DatasetWriterError, match="not a recognized label"):
            writer.write("sitting", complete_features(), motion())
    assert read_rows(path) == []


def test_write_rejects_a_label_with_the_wrong_capitalization(tmp_path):
    """A typo'd/mis-cased label must fail loudly, not silently start a new,
    slightly different class next to the real `idle`."""
    path = tmp_path / "dataset.csv"
    with DatasetWriter(path, session_id="s1") as writer:
        with pytest.raises(DatasetWriterError, match="Did you mean 'idle'"):
            writer.write("Idle", complete_features(), motion())
    assert read_rows(path) == []


# --- hardened invariants (post-Sprint-11 review) ---------------------------


def test_write_rejects_an_empty_session_id(tmp_path):
    path = tmp_path / "dataset.csv"
    with DatasetWriter(path) as writer:  # no session_id ever set
        with pytest.raises(DatasetWriterError, match="session_id"):
            writer.write("idle", complete_features(), motion())


def test_write_rejects_an_empty_label(tmp_path):
    path = tmp_path / "dataset.csv"
    with DatasetWriter(path, session_id="s1") as writer:
        with pytest.raises(DatasetWriterError, match="label"):
            writer.write("   ", complete_features(), motion())


def test_write_rejects_features_and_motion_from_different_frames(tmp_path):
    """A caller bug (pairing this frame's posture with a stale/future motion
    reading) must fail loudly, not silently write a mismatched row."""
    path = tmp_path / "dataset.csv"
    mismatched = MotionState(frame_index=7, timestamp=7 / 30.0, left=ArmMotion(), right=ArmMotion())
    with DatasetWriter(path, session_id="s1") as writer:
        with pytest.raises(DatasetWriterError, match="not the same frame"):
            writer.write("idle", complete_features(), mismatched)


def test_write_rejects_a_non_finite_value(tmp_path):
    path = tmp_path / "dataset.csv"
    with DatasetWriter(path, session_id="s1") as writer:
        with pytest.raises(DatasetWriterError, match="non-finite"):
            writer.write("idle", complete_features(wrist_distance=float("nan")), motion())


def test_opening_an_existing_file_with_a_different_header_is_rejected(tmp_path):
    path = tmp_path / "dataset.csv"
    path.write_text("session_id,frame_index,timestamp,label\ns1,0,0.0,idle\n")
    with pytest.raises(DatasetWriterError, match="different header"):
        DatasetWriter(path, session_id="s1").open()


def test_reopening_a_file_with_the_current_header_appends_fine(tmp_path):
    path = tmp_path / "dataset.csv"
    with DatasetWriter(path, session_id="s1") as writer:
        writer.write("idle", complete_features(), motion())
    with DatasetWriter(path, session_id="s2") as writer:
        writer.write("idle", complete_features(), motion())
    assert len(read_rows(path)) == 2

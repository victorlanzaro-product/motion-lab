"""Recorder unit tests: the CSV boundary without a camera or WebSocket."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pytest

from backend.dataset.writer import DATASET_COLUMNS, REMOVED_LABELS, DatasetWriterError
from backend.features.features import FEATURE_NAMES, FrameFeatures
from backend.features.velocity import ArmMotion, MotionState
import backend.web.recorder as recorder_module
from backend.web.recorder import Recorder


def frame(index: int, *, complete: bool = True) -> tuple[FrameFeatures, MotionState]:
    timestamp = index / 30.0
    if complete:
        values = {name: 0.5 for name in FEATURE_NAMES}
        for name in (
            "left_wrist_above_shoulder",
            "right_wrist_above_shoulder",
            "wrists_crossed",
        ):
            values[name] = False
        features = FrameFeatures(
            frame_index=index, timestamp=timestamp, shoulder_width=0.3, **values
        )
    else:
        features = FrameFeatures(frame_index=index, timestamp=timestamp)
    still = ArmMotion(
        velocity_y=0.0,
        velocity_x=0.0,
        speed=0.0,
        elbow_velocity=0.0,
        moving=False,
    )
    motion = MotionState(
        frame_index=index, timestamp=timestamp, left=still, right=still
    )
    return features, motion


def test_start_write_stop_uses_dataset_schema_and_counts_skips(tmp_path: Path):
    path = tmp_path / "nested" / "dataset.csv"
    recorder = Recorder(path)
    owner = object()

    started = recorder.request_start("arm_raised", owner)
    assert started["error"] is None
    assert started["recording"] is True
    assert not path.exists(), "the event-loop request must not open the CSV"
    session_id = started["session_id"]
    assert re.fullmatch(r"web-\d{8}-\d{6}-[0-9a-f]{6}", session_id)

    for index in range(4):
        features, motion = frame(index, complete=index != 1)
        assert recorder.record_frame(features, motion) is None
    assert recorder.status() == {
        "recording": True,
        "label": "arm_raised",
        "session_id": session_id,
        "written": 3,
        "skipped": 1,
        "error": None,
    }

    assert recorder.request_stop(owner) is None
    features, motion = frame(4)
    stopped = recorder.record_frame(features, motion)
    assert stopped == {
        "recording": False,
        "label": "arm_raised",
        "session_id": session_id,
        "written": 3,
        "skipped": 1,
        "error": None,
    }
    assert recorder.status()["label"] is None
    assert recorder.status()["session_id"] is None

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert tuple(reader.fieldnames or ()) == DATASET_COLUMNS
    assert len(rows) == 3
    assert {row["session_id"] for row in rows} == {session_id}
    assert {row["label"] for row in rows} == {"arm_raised"}


def test_two_sessions_get_distinct_session_ids(tmp_path: Path):
    recorder = Recorder(tmp_path / "dataset.csv")
    owner = object()
    session_ids = []

    for index, label in enumerate(("idle", "arms_open")):
        started = recorder.request_start(label, owner)
        assert started["error"] is None
        features, motion = frame(index)
        recorder.record_frame(features, motion)
        session_ids.append(started["session_id"])
        assert recorder.request_stop(owner) is None
        stopped = recorder.record_frame(features, motion)
        assert stopped["recording"] is False
        assert stopped["session_id"] == session_ids[-1]

    assert session_ids[0] != session_ids[1]
    with (tmp_path / "dataset.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["session_id"] for row in rows] == session_ids


def test_session_id_suffix_collision_is_retried(tmp_path: Path, monkeypatch):
    suffixes = iter(("abc123", "abc123", "def456"))
    monkeypatch.setattr(
        recorder_module.secrets, "token_hex", lambda _bytes: next(suffixes)
    )
    recorder = Recorder(tmp_path / "dataset.csv")
    owner = object()

    first = recorder.request_start("idle", owner)
    recorder.finish()
    second = recorder.request_start("idle", owner)

    assert first["session_id"].endswith("-abc123")
    assert second["session_id"].endswith("-def456")
    assert first["session_id"] != second["session_id"]


def test_repeated_session_id_suffix_collision_returns_error_without_hanging(
    tmp_path: Path, monkeypatch
):
    path = tmp_path / "dataset.csv"
    recorder = Recorder(path)
    owner = object()
    first = recorder.request_start("idle", owner)
    recorder.finish()
    used_suffix = first["session_id"][-6:]
    attempts = 0

    def repeat_used_suffix(_bytes):
        nonlocal attempts
        attempts += 1
        return used_suffix

    monkeypatch.setattr(
        recorder_module.secrets, "token_hex", repeat_used_suffix
    )
    rejected = recorder.request_start("arms_open", owner)

    assert attempts == Recorder.SESSION_SUFFIX_ATTEMPTS
    assert rejected == {
        "recording": False,
        "label": None,
        "session_id": None,
        "written": 0,
        "skipped": 0,
        "error": (
            "Could not generate a unique Recorder session ID suffix after "
            f"{Recorder.SESSION_SUFFIX_ATTEMPTS} attempts."
        ),
    }
    assert not path.exists()


def test_session_id_suffix_space_exhaustion_returns_error(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(Recorder, "SESSION_SUFFIX_SPACE", 0)
    recorder = Recorder(tmp_path / "dataset.csv")

    rejected = recorder.request_start("idle", object())

    assert rejected["recording"] is False
    assert rejected["label"] is None
    assert rejected["session_id"] is None
    assert rejected["written"] == rejected["skipped"] == 0
    assert rejected["error"] == "Recorder session ID suffix space is exhausted."
    assert recorder.status()["recording"] is False
    assert not (tmp_path / "dataset.csv").exists()


def test_session_id_generator_exception_becomes_a_rejection(
    tmp_path: Path, monkeypatch
):
    def fail(_bytes):
        raise OSError("secure randomness unavailable")

    monkeypatch.setattr(recorder_module.secrets, "token_hex", fail)
    recorder = Recorder(tmp_path / "dataset.csv")

    rejected = recorder.request_start("idle", object())

    assert rejected["recording"] is False
    assert rejected["label"] is None
    assert rejected["session_id"] is None
    assert rejected["error"] == "secure randomness unavailable"
    assert not (tmp_path / "dataset.csv").exists()


@pytest.mark.parametrize(
    ("label", "message"),
    [
        ("wave", REMOVED_LABELS["wave"]),
        ("unknown_pose", "is not a recognized label"),
    ],
)
def test_invalid_labels_are_rejected_before_file_open(
    tmp_path: Path, label: str, message: str
):
    path = tmp_path / "dataset.csv"
    recorder = Recorder(path)

    rejected = recorder.request_start(label, object())

    assert rejected["recording"] is False
    assert message in rejected["error"]
    assert not path.exists()
    assert recorder.status()["recording"] is False


def test_only_one_start_and_only_owner_can_stop(tmp_path: Path):
    recorder = Recorder(tmp_path / "dataset.csv")
    owner = object()
    other = object()

    started = recorder.request_start("idle", owner)
    assert started["error"] is None
    conflict = recorder.request_start("arms_open", other)
    assert "already active" in conflict["error"]
    denied = recorder.request_stop(other)
    assert denied is not None
    assert "Only the tab" in denied["error"]

    features, motion = frame(0)
    recorder.record_frame(features, motion)
    assert recorder.status()["recording"] is True
    assert recorder.request_stop(owner) is None
    assert recorder.record_frame(features, motion)["recording"] is False


def test_owner_disconnect_queues_a_final_stop(tmp_path: Path):
    recorder = Recorder(tmp_path / "dataset.csv")
    owner = object()
    other = object()
    recorder.request_start("idle", owner)
    features, motion = frame(0)
    recorder.record_frame(features, motion)

    assert recorder.owner_disconnected(other) is False
    assert recorder.status()["recording"] is True
    assert recorder.owner_disconnected(owner) is True
    stopped = recorder.record_frame(features, motion)
    assert stopped["recording"] is False
    assert stopped["label"] == "idle"


def test_writer_failure_closes_recording_and_becomes_status(tmp_path: Path):
    closed = []

    class BrokenWriter:
        def __init__(self, path: Path, session_id: str):
            self.path = path
            self.session_id = session_id
            self.total = 0
            self.skipped = 0

        def open(self):
            return self

        def write(self, _label, _features, _motion):
            raise DatasetWriterError("forced write failure")

        def close(self):
            closed.append(True)

    recorder = Recorder(tmp_path / "dataset.csv", writer_factory=BrokenWriter)
    owner = object()
    assert recorder.request_start("idle", owner)["error"] is None

    features, motion = frame(0)
    failed = recorder.record_frame(features, motion)

    assert failed["recording"] is False
    assert failed["error"] == "forced write failure"
    assert recorder.status()["recording"] is False
    assert closed == [True]


def test_stream_close_cancels_start_before_opening_file(tmp_path: Path):
    path = tmp_path / "dataset.csv"
    recorder = Recorder(path)
    owner = object()
    accepted = recorder.request_start("idle", owner)

    stopped = recorder.finish("camera stream ended")

    assert stopped["recording"] is False
    assert stopped["label"] == "idle"
    assert stopped["session_id"] == accepted["session_id"]
    assert stopped["error"] == "camera stream ended"
    assert not path.exists()
    assert recorder.status()["recording"] is False

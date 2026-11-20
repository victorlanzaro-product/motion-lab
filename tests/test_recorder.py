"""Recorder unit tests: the CSV boundary without a camera or WebSocket."""

from __future__ import annotations

import asyncio
import csv
import re
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from backend.camera import CameraReadError, Frame
from backend.dataset.writer import (
    DATASET_COLUMNS,
    REMOVED_LABELS,
    DatasetWriter,
    DatasetWriterError,
)
from backend.features.features import FEATURE_NAMES, FrameFeatures
from backend.features.velocity import ArmMotion, MotionState
from backend.vision.landmarks import PoseSnapshot
import backend.web.recorder as recorder_module
from backend.web.pipeline import PipelineRunner, Subscriber, WebConfig
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


class _PipelineCamera:
    """Small pipeline fake that exposes camera release and frame timing."""

    def __init__(self, frame_generator, released: threading.Event):
        self._frame_generator = frame_generator
        self._released = released
        self.fps = 30.0

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self._released.set()

    def frames(self):
        yield from self._frame_generator()


class _PipelineDetector:
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        pass

    def detect(self, frame: Frame) -> PoseSnapshot:
        return PoseSnapshot.empty(frame.index, frame.timestamp)


def _pipeline_frame(index: int) -> Frame:
    return Frame(
        image=np.zeros((8, 8, 3), dtype=np.uint8),
        index=index,
        timestamp=index / 30.0,
    )


class _TrackedWriter(DatasetWriter):
    def __init__(self, path: Path, session_id: str, opened, closed):
        super().__init__(path, session_id=session_id)
        self._opened_event = opened
        self._closed_event = closed

    def open(self):
        result = super().open()
        self._opened_event.set()
        return result

    def close(self):
        try:
            super().close()
        finally:
            self._closed_event.set()


def _make_runner(
    tmp_path: Path,
    frame_generator,
    camera_released: threading.Event,
    writer_opened: threading.Event,
    writer_closed: threading.Event,
    **config_overrides,
) -> PipelineRunner:
    config = WebConfig(
        model_path=tmp_path / "missing-model.joblib",
        dataset_path=tmp_path / "pipeline-dataset.csv",
        idle_seconds=0.03,
        retry_backoff_base=0.001,
        retry_backoff_cap=0.001,
        **config_overrides,
    )
    runner = PipelineRunner(
        config,
        camera_factory=lambda _config: _PipelineCamera(frame_generator, camera_released),
        detector_factory=lambda _config: _PipelineDetector(),
    )
    runner.recorder = Recorder(
        config.dataset_path,
        writer_factory=lambda path, session_id: _TrackedWriter(
            path, session_id, writer_opened, writer_closed
        ),
    )
    return runner


def test_stop_start_disconnect_race_keeps_new_session_disconnect_signal(tmp_path: Path):
    """An old stop completion must not erase a newer owner's disconnect."""
    async def scenario():
        runner = PipelineRunner(WebConfig(dataset_path=tmp_path / "race.csv"))
        runner.state = "live"
        runner._accept_record_starts = True
        old_owner = Subscriber(asyncio.get_running_loop())
        new_owner = Subscriber(asyncio.get_running_loop())
        runner._subscribers.extend((old_owner, new_owner))

        first = runner.record_start("idle", old_owner)
        assert first["error"] is None
        features, motion = frame(0)
        assert runner.recorder.record_frame(features, motion) is None
        assert runner.record_stop(old_owner) is None
        old_stop = runner.recorder.record_frame(features, motion)
        assert old_stop is not None and old_stop["session_id"] == first["session_id"]

        second = runner.record_start("arms_open", new_owner)
        assert second["error"] is None
        runner.unsubscribe(new_owner)
        assert runner._owner_disconnected_session == second["session_id"]

        # This is the pipeline's post-record_frame cleanup for the older stop.
        with runner._lock:
            runner._clear_owner_disconnect_for_locked(old_stop["session_id"])
        assert runner._owner_disconnected_session == second["session_id"]

        # The zero-subscriber stream path must now close the newer reservation.
        runner._process_owner_disconnect()
        assert runner.recorder.status()["recording"] is False
        assert runner._owner_disconnected_session is None

    asyncio.run(scenario())


def test_stop_start_disconnect_publishes_sessions_in_linear_order(tmp_path: Path):
    """A new reservation cannot overtake the prior session's final status."""

    async def scenario():
        runner = PipelineRunner(WebConfig(dataset_path=tmp_path / "ordered.csv"))
        runner.state = "live"
        runner._accept_record_starts = True
        owner_a = Subscriber(asyncio.get_running_loop())
        owner_b = Subscriber(asyncio.get_running_loop())
        observer = Subscriber(asyncio.get_running_loop())
        runner._subscribers.extend((owner_a, owner_b, observer))

        start_a = runner.record_start("idle", owner_a)
        assert start_a["error"] is None
        observed_start_a = await asyncio.wait_for(observer.get(), timeout=1.0)
        assert observed_start_a["session_id"] == start_a["session_id"]
        assert observed_start_a["recording"] is True
        owner_b_start_a = await asyncio.wait_for(owner_b.get(), timeout=1.0)
        assert owner_b_start_a["session_id"] == start_a["session_id"]

        assert runner.record_stop(owner_a) is None
        frame_ready = threading.Event()
        allow_frame = threading.Event()
        final_a_ready = threading.Event()
        allow_final_a_publish = threading.Event()
        original_record_frame = runner.recorder.record_frame
        original_status = runner.recorder.status

        def pause_after_final_a(features, motion):
            frame_ready.set()
            if not allow_frame.wait(timeout=3.0):
                raise TimeoutError("test did not release frame processing")
            event = original_record_frame(features, motion)
            if event is not None and event["session_id"] == start_a["session_id"]:
                final_a_ready.set()
                if not allow_final_a_publish.wait(timeout=3.0):
                    raise TimeoutError("test did not release final A publication")
            return event

        runner.recorder.record_frame = pause_after_final_a
        features, motion = frame(1)
        frame_task = asyncio.create_task(
            asyncio.to_thread(runner._record_and_publish_frame, features, motion)
        )
        try:
            assert await asyncio.to_thread(frame_ready.wait, 3.0)

            def active_snapshot_then_finish():
                snapshot = original_status()
                allow_frame.set()
                if not final_a_ready.wait(timeout=3.0):
                    raise TimeoutError("Recorder did not finalize A")
                return snapshot

            # The writer can finish while B validates an active snapshot.
            # Even in that case, the closed gate must reject the reservation.
            runner.recorder.status = active_snapshot_then_finish
            try:
                racing_b = runner.record_start("arms_open", owner_b)
            finally:
                runner.recorder.status = original_status
            assert racing_b["recording"] is True
            assert racing_b["session_id"] == start_a["session_id"]
            assert racing_b["error"] == "A recording is already active."
            assert await asyncio.wait_for(owner_b.get(), timeout=1.0) == racing_b | {
                "type": "record_status",
                "owner": False,
            }
            assert await asyncio.to_thread(final_a_ready.wait, 3.0)

            # Recorder state has closed A, but the worker is paused before it
            # can publish A-final. B must not be accepted into this gap.
            early_b = runner.record_start("arms_open", owner_b)
            assert early_b["recording"] is False
            assert early_b["error"] is not None
            assert runner._accept_record_starts is False
            early_b_message = await asyncio.wait_for(owner_b.get(), timeout=1.0)
            assert early_b_message["error"] is not None

            allow_final_a_publish.set()
            await frame_task
            observed_final_a = await asyncio.wait_for(observer.get(), timeout=1.0)
            assert observed_final_a["recording"] is False
            assert observed_final_a["session_id"] == start_a["session_id"]

            start_b = runner.record_start("arms_open", owner_b)
            assert start_b["error"] is None
            observed_start_b = await asyncio.wait_for(observer.get(), timeout=1.0)
            assert observed_start_b["recording"] is True
            assert observed_start_b["session_id"] == start_b["session_id"]

            await asyncio.to_thread(
                runner._record_and_publish_frame, features, motion
            )  # open B's writer before the owner disconnects
            runner.unsubscribe(owner_b)
            assert runner._owner_disconnected_session == start_b["session_id"]
            runner._process_owner_disconnect()
            observed_final_b = await asyncio.wait_for(observer.get(), timeout=1.0)
            assert observed_final_b["recording"] is False
            assert observed_final_b["session_id"] == start_b["session_id"]

            lifecycle = [
                observed_start_a,
                observed_final_a,
                observed_start_b,
                observed_final_b,
            ]
            assert [message["session_id"] for message in lifecycle] == [
                start_a["session_id"],
                start_a["session_id"],
                start_b["session_id"],
                start_b["session_id"],
            ]
            assert [message["recording"] for message in lifecycle] == [
                True,
                False,
                True,
                False,
            ]
        finally:
            allow_frame.set()
            allow_final_a_publish.set()
            await asyncio.gather(frame_task, return_exceptions=True)
            runner.stop()

    asyncio.run(scenario())


def test_idle_frame_defers_start_stop_commands_until_the_next_frame(tmp_path: Path):
    """An idle check cannot consume a new session before its gate is closed."""

    async def scenario():
        runner = PipelineRunner(WebConfig(dataset_path=tmp_path / "idle-order.csv"))
        runner.state = "live"
        runner._accept_record_starts = True
        owner_b = Subscriber(asyncio.get_running_loop())
        owner_c = Subscriber(asyncio.get_running_loop())
        observer = Subscriber(asyncio.get_running_loop())
        runner._subscribers.extend((owner_b, owner_c, observer))

        original_record_frame = runner.recorder.record_frame
        record_frame_calls = []

        def count_record_frames(features, motion):
            record_frame_calls.append((features.frame_index, motion.frame_index))
            return original_record_frame(features, motion)

        runner.recorder.record_frame = count_record_frames
        features, motion = frame(2)
        try:
            assert runner._record_and_publish_frame(features, motion) is None
            assert record_frame_calls == []

            start_b = runner.record_start("arms_open", owner_b)
            assert start_b["error"] is None
            observed_start_b = await asyncio.wait_for(observer.get(), timeout=1.0)
            assert observed_start_b["recording"] is True
            assert observed_start_b["session_id"] == start_b["session_id"]

            assert runner.record_stop(owner_b) is None
            early_c = runner.record_start("arms_crossed", owner_c)
            assert early_c["error"] is not None
            assert early_c["recording"] is True
            assert early_c["session_id"] == start_b["session_id"]

            await asyncio.to_thread(runner._record_and_publish_frame, features, motion)
            assert len(record_frame_calls) == 1
            observed_final_b = await asyncio.wait_for(observer.get(), timeout=1.0)
            assert observed_final_b["recording"] is False
            assert observed_final_b["session_id"] == start_b["session_id"]

            start_c = runner.record_start("arms_crossed", owner_c)
            assert start_c["error"] is None
            observed_start_c = await asyncio.wait_for(observer.get(), timeout=1.0)
            assert observed_start_c["recording"] is True
            assert observed_start_c["session_id"] == start_c["session_id"]

            lifecycle = [observed_start_b, observed_final_b, observed_start_c]
            assert [message["session_id"] for message in lifecycle] == [
                start_b["session_id"],
                start_b["session_id"],
                start_c["session_id"],
            ]
            assert [message["recording"] for message in lifecycle] == [
                True,
                False,
                True,
            ]

            runner.unsubscribe(owner_c)
            runner._process_owner_disconnect()
        finally:
            runner.stop()

    asyncio.run(scenario())


def test_sole_owner_disconnect_closes_writer_and_releases_camera(tmp_path: Path):
    async def scenario():
        released = threading.Event()
        opened = threading.Event()
        closed = threading.Event()

        def frames():
            index = 0
            while True:
                time.sleep(0.002)
                yield _pipeline_frame(index)
                index += 1

        runner = _make_runner(tmp_path, frames, released, opened, closed)
        subscriber = runner.subscribe(asyncio.get_running_loop())
        try:
            assert await asyncio.to_thread(
                lambda: _wait_for(lambda: runner.state == "live")
            )
            started = runner.record_start("idle", subscriber)
            assert started["error"] is None
            assert await asyncio.to_thread(opened.wait, 3.0)

            runner.unsubscribe(subscriber)
            assert await asyncio.to_thread(closed.wait, 3.0)
            assert await asyncio.to_thread(released.wait, 3.0)
            assert runner.recorder.status()["recording"] is False
        finally:
            runner.stop()

        assert closed.is_set(), "owner disconnect must close the active writer"
        assert released.is_set(), "last client disconnect must release the camera"

    asyncio.run(scenario())


def test_camera_read_error_during_recording_closes_writer_and_stream_resumes(
    tmp_path: Path,
):
    async def scenario():
        first_frame_ready = threading.Event()
        allow_second_frame = threading.Event()
        allow_camera_error = threading.Event()
        recovered_frame = threading.Event()
        released = threading.Event()
        opened = threading.Event()
        closed = threading.Event()
        attempts = 0

        def frames_for_attempt():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                yield _pipeline_frame(0)
                first_frame_ready.set()
                if not allow_second_frame.wait(3.0):
                    raise AssertionError("test did not release first camera frame")
                yield _pipeline_frame(1)
                if not allow_camera_error.wait(3.0):
                    raise AssertionError("test did not trigger camera read failure")
                raise CameraReadError("camera disconnected during active recording")

            index = 0
            while not runner._stop.is_set():
                recovered_frame.set()
                yield _pipeline_frame(index)
                index += 1
                time.sleep(0.002)

        runner = _make_runner(
            tmp_path,
            frames_for_attempt,
            released,
            opened,
            closed,
            max_camera_retries=2,
        )
        subscriber = runner.subscribe(asyncio.get_running_loop())
        try:
            assert await asyncio.to_thread(first_frame_ready.wait, 3.0)
            started = runner.record_start("idle", subscriber)
            assert started["error"] is None
            allow_second_frame.set()
            assert await asyncio.to_thread(opened.wait, 3.0)
            allow_camera_error.set()

            final = None
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                message = await asyncio.wait_for(subscriber.get(), timeout=1.0)
                if (
                    message.get("type") == "record_status"
                    and not message.get("recording")
                    and message.get("session_id") == started["session_id"]
                ):
                    final = message
                    break
            assert final is not None
            assert final["error"] == "camera disconnected during active recording"
            assert await asyncio.to_thread(closed.wait, 3.0)
            assert await asyncio.to_thread(recovered_frame.wait, 3.0)
            assert await asyncio.to_thread(
                lambda: _wait_for(lambda: runner.frames_served >= 3)
            )
            assert runner.state == "live"
        finally:
            allow_second_frame.set()
            allow_camera_error.set()
            runner.stop()

        assert closed.is_set(), "camera read failure must close the writer"
        assert attempts >= 2, "camera retry should create a new stream"
        assert runner.frames_served >= 3, "frames must resume after camera recovery"

    asyncio.run(scenario())


def _wait_for(predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return bool(predicate())

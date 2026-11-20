"""Temporal feature tests built from synthetic, time-stamped wrist signals."""

from __future__ import annotations

import math

import pytest

from backend.features import FrameFeatures, WindowFeatures, rebuild_windows


def frame(index: int, timestamp: float, left_height, right_height=0.25,
          left_offset=0.0, right_offset=0.0) -> FrameFeatures:
    return FrameFeatures(
        frame_index=index,
        timestamp=timestamp,
        left_wrist_height=left_height,
        right_wrist_height=right_height,
        left_wrist_offset_x=left_offset,
        right_wrist_offset_x=right_offset,
    )


def feed(tracker: WindowFeatures, heights, *, fps: float, offsets=None):
    state = None
    for index, height in enumerate(heights):
        timestamp = index / fps
        offset = 0.0 if offsets is None else offsets[index]
        state = tracker.push(timestamp, frame(index, timestamp, height, left_offset=offset))
    return state


def test_wave_window_separates_repeated_waving_from_one_arm_raise():
    """The endpoint frame is identical; temporal reversals and spread differ."""
    fps = 40.0
    times = [i / fps for i in range(int(fps) + 1)]
    wave = [0.5 + 0.45 * math.sin(2 * math.pi * 3 * timestamp) for timestamp in times]
    raised_once = [min(2.0 * timestamp, 0.5) for timestamp in times]
    wave[-1] = raised_once[-1]  # make the shared endpoint exactly identical

    # A frame classifier sees byte-for-byte equivalent feature vectors at the
    # endpoint; only their preceding histories distinguish the two motions.
    wave_endpoint = frame(len(times) - 1, times[-1], wave[-1])
    raised_endpoint = frame(len(times) - 1, times[-1], raised_once[-1])
    assert wave_endpoint.to_vector() == raised_endpoint.to_vector()

    waving = feed(WindowFeatures(), wave, fps=fps)
    raised = feed(WindowFeatures(), raised_once, fps=fps)

    assert waving is not None
    assert raised is not None
    assert waving.left.direction_reversals >= 4
    assert raised.left.direction_reversals == 0
    assert waving.left.std_wrist_height > raised.left.std_wrist_height
    assert waving.left.range_wrist_height > raised.left.range_wrist_height


def test_a_window_is_none_until_it_spans_the_requested_time_and_sample_count():
    tracker = WindowFeatures(window_seconds=1.0, max_gap=0.5, min_samples=4)
    assert tracker.push(0.0, frame(0, 0.0, 0.4)) is None
    assert tracker.push(0.4, frame(1, 0.4, 0.4)) is None
    assert tracker.push(0.8, frame(2, 0.8, 0.4)) is None
    assert tracker.push(1.0, frame(3, 1.0, 0.4)) is not None


def test_min_samples_must_be_an_integer_count():
    with pytest.raises(ValueError, match="integer"):
        WindowFeatures(min_samples=2.5)


def test_a_gap_larger_than_max_gap_invalidates_then_requires_a_fresh_window():
    tracker = WindowFeatures(window_seconds=1.0, max_gap=0.3, min_samples=4)
    state = None
    for index in range(11):
        timestamp = index / 10
        state = tracker.push(timestamp, frame(index, timestamp, 0.5))
    assert state is not None

    assert tracker.push(1.5, frame(11, 1.5, 0.5)) is None
    for index in range(16, 25):
        timestamp = index / 10
        assert tracker.push(timestamp, frame(index, timestamp, 0.5)) is None
    assert tracker.push(2.5, frame(25, 2.5, 0.5)) is not None


def test_missing_wrist_is_none_not_zero_and_recovers_from_fresh_samples():
    tracker = WindowFeatures(window_seconds=0.5, max_gap=0.3, min_samples=4)
    for index in range(6):
        timestamp = index / 10
        tracker.push(timestamp, frame(index, timestamp, 0.7))

    assert tracker.push(0.6, frame(6, 0.6, None)) is None
    state = None
    for index in range(7, 13):
        timestamp = index / 10
        state = tracker.push(timestamp, frame(index, timestamp, 0.7))

    assert state is not None
    assert state.left.mean_wrist_height == pytest.approx(0.7)
    assert state.left.range_wrist_height == pytest.approx(0.0)


def test_max_speed_is_two_dimensional_and_per_second_at_different_frame_rates():
    outputs = []
    for fps in (10.0, 50.0):
        tracker = WindowFeatures(window_seconds=1.0, min_samples=4)
        state = None
        for index in range(int(fps) + 1):
            timestamp = index / fps
            # 0.9 shoulder-width/s vertically and 1.2 horizontally => 1.5 total.
            state = tracker.push(
                timestamp,
                frame(
                    index,
                    timestamp,
                    left_height=0.3 + 0.9 * timestamp,
                    left_offset=0.2 + 1.2 * timestamp,
                ),
            )
        assert state is not None
        outputs.append(state.left.max_speed)

    assert outputs[0] == pytest.approx(1.5, abs=1e-6)
    assert outputs[1] == pytest.approx(outputs[0], abs=1e-6)


def test_moving_fraction_weights_elapsed_time_instead_of_frame_count():
    tracker = WindowFeatures(window_seconds=1.0, max_gap=0.5, min_samples=4)
    # Only the first 0.1 s moves. The remaining 0.9 s is still, despite having
    # the same number of recorded samples as the moving segment.
    samples = [
        (0.0, 0.0),
        (0.1, 0.1),
        (0.5, 0.1),
        (1.0, 0.1),
    ]
    state = None
    for index, (timestamp, height) in enumerate(samples):
        state = tracker.push(timestamp, frame(index, timestamp, height))

    assert state is not None
    assert state.left.moving_fraction == pytest.approx(0.1)


def test_batch_rebuild_sorts_and_never_mixes_session_windows():
    rows = []
    # Session B follows A in timestamp order without a gap or clock rewind.
    # Its two rows are not enough for a window; a tracker that leaked A's
    # history into B would incorrectly emit a mixed window for B.
    for session_id, height, indexes, offset in (
        ("session-a", 0.2, reversed(range(4)), 0.0),
        ("session-b", 0.8, reversed(range(2)), 0.4),
    ):
        for index in indexes:
            timestamp = offset + index / 10
            rows.append(
                {
                    "session_id": session_id,
                    "frame_index": str(index),
                    "timestamp": str(timestamp),
                    "left_wrist_height": str(height),
                    "right_wrist_height": "0.25",
                    "left_wrist_offset_x": "0.0",
                    "right_wrist_offset_x": "0.0",
                }
            )

    rebuilt = rebuild_windows(
        rows,
        window_seconds=0.3,
        max_gap=0.2,
        min_samples=4,
    )
    final_by_session = {
        item.session_id: item.features
        for item in rebuilt
        if (item.session_id == "session-a" and item.frame_index == 3)
        or (item.session_id == "session-b" and item.frame_index == 1)
    }

    assert set(final_by_session) == {"session-a", "session-b"}
    assert final_by_session["session-a"] is not None
    assert final_by_session["session-a"].left.mean_wrist_height == pytest.approx(0.2)
    assert final_by_session["session-b"] is None


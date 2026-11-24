"""Tests for reconstructing schema-2 rows from the recorded frame CSV."""

from __future__ import annotations

import math
from dataclasses import replace

from backend.dataset.windows import (
    WINDOW_FEATURE_COLUMNS,
    WINDOW_TRAINING_COLUMNS,
    build_window_input_row,
    build_window_rows,
)
from backend.features.features import FEATURE_NAMES, FrameFeatures
from backend.features.window import WindowFeatures


def _row(session: str, index: int, timestamp: float, height: float, label: str) -> dict:
    values = {name: 0.0 for name in FEATURE_NAMES}
    values.update(
        {
            "left_wrist_height": height,
            "right_wrist_height": 0.25,
            "left_wrist_offset_x": 0.0,
            "right_wrist_offset_x": 0.0,
            "left_wrist_above_shoulder": height > 0.5,
            "right_wrist_above_shoulder": False,
            "wrists_crossed": False,
        }
    )
    return {
        "session_id": session,
        "frame_index": str(index),
        "timestamp": str(timestamp),
        "label": label,
        **values,
    }


def test_rebuilds_sorted_session_windows_and_preserves_last_frame_label_and_static_features():
    fps = 40.0
    times = [index / fps for index in range(41)]
    wave_heights = [0.5 + 0.45 * math.sin(2 * math.pi * 3 * time) for time in times]
    raised_heights = [min(2.0 * time, 0.5) for time in times]
    wave_heights[-1] = raised_heights[-1]

    rows = []
    for index, timestamp in enumerate(times):
        rows.append(_row("wave-session", index, timestamp, wave_heights[index], "wave"))
        rows.append(
            _row("raise-session", index, timestamp, raised_heights[index], "arm_raised")
        )

    # Deliberately reverse the CSV order; reconstruction still sorts within
    # each session and must not join the two histories.
    rebuilt = build_window_rows(
        reversed(rows), window_seconds=1.0, max_gap=0.05, min_samples=4
    )

    assert len(rebuilt) == 2  # only the endpoint spans the full second
    by_session = {row["session_id"]: row for row in rebuilt}
    waving = by_session["wave-session"]
    raised = by_session["raise-session"]

    assert waving["label"] == "wave"
    assert raised["label"] == "arm_raised"
    assert waving["frame_index"] == raised["frame_index"] == 40
    assert waving["timestamp"] == raised["timestamp"] == 1.0
    assert [waving[name] for name in FEATURE_NAMES] == [raised[name] for name in FEATURE_NAMES]
    assert waving["left_window_direction_reversals"] >= 4
    assert raised["left_window_direction_reversals"] == 0
    assert waving["left_window_std_wrist_height"] > raised["left_window_std_wrist_height"]
    assert tuple(WINDOW_TRAINING_COLUMNS[: len(FEATURE_NAMES)]) == FEATURE_NAMES
    assert len(WINDOW_FEATURE_COLUMNS) == 14
    assert all(name in waving for name in WINDOW_TRAINING_COLUMNS)


def test_warmup_and_incomplete_wrist_windows_are_omitted_without_zero_filling():
    rows = [
        _row("short", index, index / 10, 0.4, "idle")
        for index in range(3)
    ]
    assert build_window_rows(
        rows, window_seconds=0.3, max_gap=0.2, min_samples=4
    ) == []


def test_training_and_live_input_use_the_same_complete_window_row_builder():
    rows = [_row("one", index, index / 10, 0.2 + index / 10, "wave") for index in range(4)]
    tracker = WindowFeatures(window_seconds=0.3, max_gap=0.2, min_samples=4)
    window = None
    for row in rows:
        window = tracker.push(float(row["timestamp"]), row)

    batch_row = build_window_rows(
        rows, window_seconds=0.3, max_gap=0.2, min_samples=4
    )[0]
    live_frame = FrameFeatures(
        frame_index=3,
        timestamp=0.3,
        **{name: rows[-1][name] for name in FEATURE_NAMES},
    )
    live_row = build_window_input_row(live_frame, window)

    assert live_row is not None
    assert {name: live_row[name] for name in WINDOW_TRAINING_COLUMNS} == {
        name: batch_row[name] for name in WINDOW_TRAINING_COLUMNS
    }
    assert build_window_input_row(live_frame, replace(window, right=None)) is None

    # A complete left wrist must not make an incomplete right-arm window look
    # like a measured zero.
    rows = [_row("one-wrist", index, index / 10, 0.4, "idle") for index in range(4)]
    for row in rows:
        row["right_wrist_height"] = ""
        row["right_wrist_offset_x"] = ""
    assert build_window_rows(
        rows, window_seconds=0.3, max_gap=0.2, min_samples=4
    ) == []

"""Gesture Engine tests — synthetic timelines, same spirit as test_motion.py.

No webcam, no model: the engine eats `FrameFeatures` + `MotionState`, so a
gesture is just a scripted sequence of postures with timestamps attached.
"""

from __future__ import annotations

import pytest

from backend.features import ArmMotion, Direction, FrameFeatures, MotionState
from backend.gestures import GestureConfig, GestureEngine

FPS = 30.0


def features(index: int, timestamp: float, **overrides) -> FrameFeatures:
    return FrameFeatures(frame_index=index, timestamp=timestamp, **overrides)


def motion(index: int, timestamp: float, left_direction: Direction) -> MotionState:
    return MotionState(
        frame_index=index,
        timestamp=timestamp,
        left=ArmMotion(direction=left_direction),
        right=ArmMotion(direction=Direction.UNKNOWN),
    )


def hold(seconds: float, **feature_overrides) -> tuple[float, dict, Direction]:
    """A phase where the features stay constant and the left arm is UNKNOWN."""
    return (seconds, feature_overrides, Direction.UNKNOWN)


def swing(seconds: float, direction: Direction) -> tuple[float, dict, Direction]:
    """A phase where only the left arm's classified direction is set."""
    return (seconds, {}, direction)


def run(engine: GestureEngine, phases, fps: float = FPS) -> list:
    """Feed `phases` back to back on one continuous clock; collect every event."""
    events = []
    t = 0.0
    step = 1 / fps
    frame = 0
    for seconds, feature_overrides, direction in phases:
        for _ in range(int(seconds * fps)):
            events += engine.update(
                features(frame, t, **feature_overrides), motion(frame, t, direction)
            )
            t += step
            frame += 1
    return events


# --- hold, edge, cooldown ------------------------------------------------


def test_a_quick_flicker_never_fires():
    engine = GestureEngine(GestureConfig(hold_seconds=0.6))
    assert run(engine, [hold(0.2, left_wrist_above_shoulder=True)]) == []


def test_sustained_raise_fires_once_after_the_hold_duration():
    engine = GestureEngine(GestureConfig(hold_seconds=0.5, cooldown_seconds=10))
    events = run(engine, [hold(1.0, left_wrist_above_shoulder=True)])
    assert [e.name for e in events] == ["arm_raised"]
    assert events[0].side == "left"
    assert events[0].timestamp == pytest.approx(0.5, abs=1 / FPS)


def test_cooldown_suppresses_a_second_raise_within_the_window():
    engine = GestureEngine(GestureConfig(hold_seconds=0.1, cooldown_seconds=1.0))
    events = run(
        engine,
        [
            hold(0.3, left_wrist_above_shoulder=True),
            hold(0.05, left_wrist_above_shoulder=False),
            hold(0.3, left_wrist_above_shoulder=True),
        ],
    )
    assert [e.name for e in events] == ["arm_raised"]


def test_lowering_and_raising_again_after_the_cooldown_fires_twice():
    engine = GestureEngine(GestureConfig(hold_seconds=0.1, cooldown_seconds=0.3))
    events = run(
        engine,
        [
            hold(0.4, left_wrist_above_shoulder=True),
            hold(0.5, left_wrist_above_shoulder=False),
            hold(0.4, left_wrist_above_shoulder=True),
        ],
    )
    assert [e.name for e in events] == ["arm_raised", "arm_raised"]


def test_missing_joint_never_fires_a_gesture():
    engine = GestureEngine(GestureConfig(hold_seconds=0.05))
    assert run(engine, [hold(0.5, left_wrist_above_shoulder=None)]) == []


def test_without_reset_a_stale_hold_fires_instantly_after_a_gap():
    """Documents the bug `reset()` exists to prevent: the process clock keeps
    advancing while the pipeline is idle and simply stops calling update()."""
    engine = GestureEngine(GestureConfig(hold_seconds=0.3, cooldown_seconds=5.0))
    run(engine, [hold(0.1, left_wrist_above_shoulder=True)])  # hold started, not stable yet
    resumed = engine.update(
        features(0, 100.0, left_wrist_above_shoulder=True), motion(0, 100.0, Direction.UNKNOWN)
    )
    assert [e.name for e in resumed] == ["arm_raised"]  # fires on the very first frame back


def test_reset_prevents_that_stale_hold_from_firing_instantly():
    engine = GestureEngine(GestureConfig(hold_seconds=0.3, cooldown_seconds=5.0))
    run(engine, [hold(0.1, left_wrist_above_shoulder=True)])
    engine.reset()
    resumed = engine.update(
        features(0, 100.0, left_wrist_above_shoulder=True), motion(0, 100.0, Direction.UNKNOWN)
    )
    assert resumed == []  # needs a fresh hold_seconds, same as any other first raise


# --- two-handed postures ---------------------------------------------------


def test_arms_crossed_fires_from_the_feature_flag():
    engine = GestureEngine(GestureConfig(hold_seconds=0.1))
    events = run(engine, [hold(0.3, wrists_crossed=True)])
    assert [e.name for e in events] == ["arms_crossed"]
    assert events[0].side is None


def test_arms_open_uses_the_wrist_distance_threshold():
    config = GestureConfig(hold_seconds=0.1, open_threshold=2.0)
    assert run(GestureEngine(config), [hold(0.3, wrist_distance=1.5)]) == []
    events = run(GestureEngine(config), [hold(0.3, wrist_distance=2.5)])
    assert [e.name for e in events] == ["arms_open"]


# --- wave ------------------------------------------------------------------


def test_wave_needs_the_minimum_number_of_reversals():
    config = GestureConfig(wave_window_seconds=2.0, wave_min_reversals=3, cooldown_seconds=10)
    events = run(
        GestureEngine(config),
        [swing(0.2, Direction.UP), swing(0.2, Direction.DOWN), swing(0.2, Direction.UP)],
    )
    assert events == []  # two reversals only


def test_enough_reversals_within_the_window_fires_a_wave():
    config = GestureConfig(wave_window_seconds=2.0, wave_min_reversals=3, cooldown_seconds=10)
    events = run(
        GestureEngine(config),
        [
            swing(0.2, Direction.UP),
            swing(0.2, Direction.DOWN),
            swing(0.2, Direction.UP),
            swing(0.2, Direction.DOWN),
        ],
    )
    assert [e.name for e in events] == ["wave"]
    assert events[0].side == "left"


def test_reversals_older_than_the_window_are_forgotten():
    config = GestureConfig(wave_window_seconds=0.3, wave_min_reversals=3, cooldown_seconds=10)
    events = run(
        GestureEngine(config),
        [
            swing(0.1, Direction.UP),
            swing(0.1, Direction.DOWN),
            swing(0.1, Direction.UP),
            swing(1.0, Direction.STILL),  # the first two reversals age out here
            swing(0.1, Direction.DOWN),
        ],
    )
    assert events == []


def test_a_pause_at_the_top_of_the_swing_does_not_reset_the_count():
    """STILL sits between reversals (velocity crosses zero at the peak) and
    must not zero the count the way a genuine change of mind should."""
    config = GestureConfig(wave_window_seconds=2.0, wave_min_reversals=3, cooldown_seconds=10)
    events = run(
        GestureEngine(config),
        [
            swing(0.2, Direction.UP),
            swing(0.05, Direction.STILL),
            swing(0.2, Direction.DOWN),
            swing(0.05, Direction.STILL),
            swing(0.2, Direction.UP),
            swing(0.05, Direction.STILL),
            swing(0.2, Direction.DOWN),
        ],
    )
    assert [e.name for e in events] == ["wave"]

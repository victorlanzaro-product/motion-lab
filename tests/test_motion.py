"""Motion Engine tests — synthetic timelines with a known slope.

No webcam, no model: the tracker eats feature rows, so a movement is just a
list of numbers with timestamps attached.
"""

from __future__ import annotations

import pytest

from backend.features import (
    Direction,
    FrameFeatures,
    MotionConfig,
    MotionTracker,
    SignalTrack,
    Trail,
    classify,
)
from backend.vision.landmarks import Point, Side

FPS = 30.0


def row(index: int, timestamp: float, **values) -> FrameFeatures:
    return FrameFeatures(frame_index=index, timestamp=timestamp, **values)


def ramp(track: SignalTrack, rate: float, seconds: float, fps: float = FPS, start: float = 0.0):
    """Feed a straight line rising at `rate` units per second."""
    steps = int(seconds * fps)
    for i in range(steps + 1):
        t = i / fps
        track.add(t, start + rate * t)


# --- SignalTrack: velocity is per second, not per frame -----------------


def test_constant_rise_reads_its_true_rate():
    track = SignalTrack()
    ramp(track, rate=1.5, seconds=1.0)
    assert track.velocity() == pytest.approx(1.5, abs=1e-6)


def test_same_movement_at_half_the_frame_rate_gives_the_same_velocity():
    """The whole point of dividing by elapsed time: the machine drops out."""
    fast, slow = SignalTrack(), SignalTrack()
    ramp(fast, rate=2.0, seconds=1.0, fps=60)
    ramp(slow, rate=2.0, seconds=1.0, fps=15)
    assert fast.velocity() == pytest.approx(slow.velocity(), abs=1e-6)


def test_falling_signal_has_negative_velocity():
    track = SignalTrack()
    ramp(track, rate=-0.8, seconds=1.0)
    assert track.velocity() == pytest.approx(-0.8, abs=1e-6)


def test_two_samples_are_not_enough_to_claim_a_velocity():
    track = SignalTrack()
    track.add(0.0, 0.0)
    track.add(0.1, 0.1)
    assert len(track) == 2
    assert track.velocity() is None


def test_a_burst_of_samples_in_no_time_reports_nothing():
    """Guard against dividing by an almost-zero time span."""
    track = SignalTrack()
    for i in range(10):
        track.add(i * 0.001, i * 0.5)
    assert track.span < track.config.min_span
    assert track.velocity() is None


def test_window_forgets_samples_older_than_it():
    track = SignalTrack(MotionConfig(window_seconds=0.4))
    ramp(track, rate=1.0, seconds=2.0)
    assert track.span <= 0.4 + 1 / FPS
    assert len(track) < 2 * FPS


def test_velocity_follows_the_recent_window_not_the_whole_history():
    """Arm goes up for a second, then comes down: the verdict is 'down'."""
    track = SignalTrack()
    ramp(track, rate=1.0, seconds=1.0)
    for i in range(1, 16):
        t = 1.0 + i / FPS
        track.add(t, 1.0 - 1.0 * (t - 1.0))
    assert track.velocity() == pytest.approx(-1.0, abs=1e-6)


# --- SignalTrack: gaps --------------------------------------------------


def test_one_missed_frame_does_not_destroy_the_history():
    track = SignalTrack()
    ramp(track, rate=1.0, seconds=0.5)
    before = len(track)
    track.add(0.5 + 1 / FPS, None)  # joint blinked out for a single frame
    assert len(track) == before
    assert track.velocity() == pytest.approx(1.0, abs=1e-6)


def test_a_long_blackout_wipes_the_history_instead_of_bridging_it():
    """We do not know what the arm did while it was hidden, so we say nothing."""
    track = SignalTrack()
    ramp(track, rate=1.0, seconds=0.5)
    track.add(3.0, None)  # 2.5 s with no wrist
    assert len(track) == 0
    assert track.velocity() is None


def test_a_sample_arriving_after_a_long_gap_starts_a_fresh_history():
    track = SignalTrack()
    ramp(track, rate=1.0, seconds=0.5)
    track.add(5.0, 0.0)
    assert len(track) == 1
    assert track.velocity() is None


def test_time_running_backwards_resets_the_track():
    track = SignalTrack()
    ramp(track, rate=1.0, seconds=0.5)
    track.add(0.2, 0.0)  # camera restarted, clock rewound
    assert len(track) == 1


# --- classify -----------------------------------------------------------


def test_classify_uses_the_dead_zone():
    assert classify(1.0, 0.35) is Direction.UP
    assert classify(-1.0, 0.35) is Direction.DOWN
    assert classify(0.05, 0.35) is Direction.STILL
    assert classify(-0.05, 0.35) is Direction.STILL
    assert classify(None, 0.35) is Direction.UNKNOWN


# --- MotionTracker: the acceptance criterion ----------------------------


def drive(tracker: MotionTracker, heights, offsets=None, elbows=None, fps: float = FPS):
    """Push a timeline of left-wrist heights through the tracker."""
    state = None
    for i, height in enumerate(heights):
        state = tracker.update(
            row(
                i,
                i / fps,
                left_wrist_height=height,
                left_wrist_offset_x=None if offsets is None else offsets[i],
                left_elbow_angle=None if elbows is None else elbows[i],
            )
        )
    return state


def test_raising_the_arm_reads_as_up():
    tracker = MotionTracker()
    heights = [i / FPS * 1.2 for i in range(20)]  # 1.2 shoulder-widths per second
    state = drive(tracker, heights, offsets=[0.0] * 20)
    assert state.left.direction is Direction.UP
    assert state.left.velocity_y == pytest.approx(1.2, abs=1e-6)
    assert state.left.moving is True


def test_lowering_the_arm_reads_as_down():
    tracker = MotionTracker()
    heights = [1.0 - i / FPS * 1.2 for i in range(20)]
    state = drive(tracker, heights, offsets=[0.0] * 20)
    assert state.left.direction is Direction.DOWN
    assert state.left.moving is True


def test_a_held_pose_reads_as_still_despite_landmark_jitter():
    """A real still arm never measures exactly zero — the dead zone earns its keep."""
    tracker = MotionTracker()
    jitter = [0.8 + (0.01 if i % 2 else -0.01) for i in range(20)]
    state = drive(tracker, jitter, offsets=[0.0] * 20)
    assert state.left.direction is Direction.STILL
    assert state.left.moving is False


def test_the_two_arms_are_tracked_independently():
    tracker = MotionTracker()
    state = None
    for i in range(20):
        t = i / FPS
        state = tracker.update(
            row(
                i,
                t,
                left_wrist_height=t * 1.2,  # rising
                right_wrist_height=-t * 1.2,  # falling
                left_wrist_offset_x=0.0,
                right_wrist_offset_x=0.0,
            )
        )
    assert state.left.direction is Direction.UP
    assert state.right.direction is Direction.DOWN
    assert state.arm(Side.LEFT) is state.left


def test_a_sideways_sweep_is_moving_but_not_rising():
    """`direction` answers up/down; `moving` answers whether anything happened."""
    tracker = MotionTracker()
    state = drive(
        tracker,
        heights=[0.5] * 20,
        offsets=[i / FPS * 1.5 for i in range(20)],
    )
    assert state.left.direction is Direction.STILL
    assert state.left.moving is True
    assert state.left.speed == pytest.approx(1.5, abs=1e-3)


def test_an_invisible_wrist_is_unknown_not_still():
    tracker = MotionTracker()
    state = drive(tracker, heights=[None] * 20)
    assert state.left.direction is Direction.UNKNOWN
    assert state.left.moving is None
    assert state.left.velocity_y is None


def test_the_first_frames_are_unknown_until_there_is_history():
    tracker = MotionTracker()
    first = tracker.update(row(0, 0.0, left_wrist_height=0.5, left_wrist_offset_x=0.0))
    assert first.left.direction is Direction.UNKNOWN


def test_elbow_velocity_is_reported_in_degrees_per_second():
    tracker = MotionTracker()
    state = drive(
        tracker,
        heights=[0.5] * 20,
        offsets=[0.0] * 20,
        elbows=[90.0 + i / FPS * 60.0 for i in range(20)],  # extending 60 deg/s
    )
    assert state.left.elbow_velocity == pytest.approx(60.0, abs=1e-4)


def test_reset_forgets_everything():
    tracker = MotionTracker()
    drive(tracker, [i / FPS * 1.2 for i in range(20)], offsets=[0.0] * 20)
    tracker.reset()
    state = tracker.update(row(99, 99.0, left_wrist_height=0.0, left_wrist_offset_x=0.0))
    assert state.left.direction is Direction.UNKNOWN


def test_payload_is_json_ready():
    tracker = MotionTracker()
    payload = drive(tracker, [i / FPS * 1.2 for i in range(20)], offsets=[0.0] * 20).to_dict()
    assert payload["left_direction"] == "up"  # plain string, not an Enum repr
    assert payload["right_direction"] == "unknown"
    assert payload["frame_index"] == 19


# --- Trail --------------------------------------------------------------


def p(x: float, y: float) -> Point:
    return Point(x=x, y=y, z=0.0, visibility=0.9, presence=0.9)


def test_trail_keeps_only_the_last_n_points():
    trail = Trail(length=4)
    for i in range(10):
        trail.add(p(i / 10, 0.5))
    assert len(trail) == 4


def test_trail_breaks_instead_of_teleporting_across_a_lost_joint():
    trail = Trail(length=10)
    for x in (0.1, 0.2, 0.3):
        trail.add(p(x, 0.5))
    trail.add(None)  # wrist left the frame
    for x in (0.8, 0.9):
        trail.add(p(x, 0.5))
    segments = trail.segments()
    assert len(segments) == 2
    assert len(segments[0]) == 3 and len(segments[1]) == 2


def test_a_lone_point_is_not_a_segment():
    trail = Trail(length=10)
    trail.add(p(0.1, 0.5))
    trail.add(None)
    trail.add(p(0.9, 0.5))
    assert trail.segments() == []

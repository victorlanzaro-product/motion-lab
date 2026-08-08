"""Gesture Engine — sustained postures and motion patterns become discrete events.

Everything upstream describes *now*: a feature row is one frame, `MotionState`
is one frame's velocity. A gesture is a claim about a stretch of time — "the
arm has been up for a while", "the wrist swung back and forth three times" —
so this file adds the last piece of temporal state in the pipeline.

Three decisions carry it:

1. **A gesture fires once, not every frame it is true.** `wrist_above_shoulder`
   stays `True` for as long as you hold your arm up; broadcasting it verbatim
   would produce dozens of identical "arm raised" events per second. Every
   gesture here is edge-triggered — it fires the instant the condition becomes
   true — and then serves a `cooldown_seconds` refractory period before it can
   fire again, the same shape Sprint 09's Event Engine will need for anything
   with a cooldown.
2. **A posture must hold before it counts.** A single noisy frame where
   `wrists_crossed` flickers `True` is not a gesture. `_Debounce` requires the
   raw condition to stay true for `hold_seconds` before it is considered
   stable — the same "dead zone" idea Sprint 04 used for velocity, applied to
   a boolean instead of a float.
3. **Missing data never fires a gesture.** A `None` from the feature layer
   means "we don't know", but a gesture is a positive claim: "the user did
   this". The engine folds `None` into "condition not met" rather than
   guessing, so an occluded wrist stays silent instead of triggering on a
   guess.

`GestureEngine` is the only stateful thing meant to be shared across frames
here, aside from the small rolling window each wave detector keeps — the same
shape `features/velocity.py` uses (`MotionTracker` plus `SignalTrack`). Rules
in `rules.py` stay pure.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

from backend.features.features import FrameFeatures
from backend.features.velocity import Direction, MotionState
from backend.gestures.rules import arm_raised, arms_crossed, arms_open
from backend.vision.landmarks import Side

#: Wire/attribute names for the two arms, in the order gestures report them.
SIDES: tuple[tuple[str, Side], ...] = (("left", Side.LEFT), ("right", Side.RIGHT))


@dataclass(frozen=True)
class GestureConfig:
    """Tuning for the temporal layer.

    Defaults favour a deliberate gesture over a twitch: raising an arm quickly
    and putting it straight back down does not count.
    """

    #: How long a posture must hold before it is considered real, not noise.
    hold_seconds: float = 0.6
    #: Minimum gap between two firings of the *same* gesture.
    cooldown_seconds: float = 1.5
    #: `wrist_distance` (shoulder widths) past which the arms count as "open".
    open_threshold: float = 2.4
    #: How far back a direction reversal still counts towards a wave.
    wave_window_seconds: float = 1.5
    #: Reversals needed inside the window to call it a wave.
    wave_min_reversals: int = 3


@dataclass(frozen=True)
class GestureEvent:
    """One fired gesture. `side` is `None` for two-handed postures."""

    name: str
    side: Optional[str]
    frame_index: int
    timestamp: float


class _Debounce:
    """Raw bool in, stable bool out: true only after holding for `seconds`.

    Falling is immediate — a moment of "arm down" cancels the attempt outright
    rather than being averaged away, so a real drop can never be smoothed into
    a still-active gesture.
    """

    def __init__(self, seconds: float) -> None:
        self._seconds = seconds
        self._true_since: Optional[float] = None

    def update(self, timestamp: float, value: bool) -> bool:
        if not value:
            self._true_since = None
            return False
        if self._true_since is None:
            self._true_since = timestamp
        return (timestamp - self._true_since) >= self._seconds

    def reset(self) -> None:
        self._true_since = None


class _EdgeCooldown:
    """Stable bool in, fire-this-frame bool out.

    Fires once per rising edge, then serves a `seconds` refractory period so
    the same gesture cannot repeat faster than that, no matter how choppy the
    input becomes.
    """

    def __init__(self, seconds: float) -> None:
        self._seconds = seconds
        self._previous = False
        self._last_fired: Optional[float] = None

    def update(self, timestamp: float, stable: bool) -> bool:
        rising = stable and not self._previous
        self._previous = stable
        if not rising:
            return False
        if self._last_fired is not None and timestamp - self._last_fired < self._seconds:
            return False
        self._last_fired = timestamp
        return True

    def reset(self) -> None:
        self._previous = False
        self._last_fired = None


class _WaveDetector:
    """Counts direction reversals in a rolling window — the swing in a wave.

    Deliberately blind to `STILL`/`UNKNOWN`: a brief pause mid-swing (the top
    or bottom of the arc, where velocity crosses zero) must not reset the
    count the way a genuine change of mind — stopping and never reversing —
    should.
    """

    def __init__(self, config: GestureConfig) -> None:
        self._config = config
        self._last_signal: Optional[Direction] = None
        self._reversals: Deque[float] = deque()

    def update(self, timestamp: float, direction: Direction) -> bool:
        if direction in (Direction.UP, Direction.DOWN):
            if self._last_signal is not None and direction is not self._last_signal:
                self._reversals.append(timestamp)
            self._last_signal = direction

        cutoff = timestamp - self._config.wave_window_seconds
        while self._reversals and self._reversals[0] < cutoff:
            self._reversals.popleft()
        return len(self._reversals) >= self._config.wave_min_reversals

    def reset(self) -> None:
        self._last_signal = None
        self._reversals.clear()


class GestureEngine:
    """Feed it `FrameFeatures` and `MotionState` in order; it answers what
    happened.

    Six gestures in V1: arm raised and wave, each per side, plus the
    two-handed postures arms crossed and arms open.
    """

    def __init__(self, config: GestureConfig = GestureConfig()) -> None:
        self.config = config
        self._holds = {
            "left_arm_raised": _Debounce(config.hold_seconds),
            "right_arm_raised": _Debounce(config.hold_seconds),
            "arms_crossed": _Debounce(config.hold_seconds),
            "arms_open": _Debounce(config.hold_seconds),
        }
        self._triggers = {
            key: _EdgeCooldown(config.cooldown_seconds)
            for key in (
                "left_arm_raised",
                "right_arm_raised",
                "arms_crossed",
                "arms_open",
                "left_wave",
                "right_wave",
            )
        }
        self._waves = {"left": _WaveDetector(config), "right": _WaveDetector(config)}

    def reset(self) -> None:
        """Drop all temporal state.

        Called when nobody is watching (mirrors `MotionTracker.reset()`): a
        gesture half-completed off-screen must not complete the instant a
        viewer reconnects, and — because the process clock keeps advancing
        while frames are not being fed in — a stale hold left un-reset would
        otherwise look like it had been held for the entire idle gap.
        """
        for hold in self._holds.values():
            hold.reset()
        for trigger in self._triggers.values():
            trigger.reset()
        for wave in self._waves.values():
            wave.reset()

    def update(self, features: FrameFeatures, motion: MotionState) -> list[GestureEvent]:
        t = features.timestamp
        events: list[GestureEvent] = []

        for name, side in SIDES:
            raise_key = f"{name}_arm_raised"
            # bool(None) -> False: an occluded wrist folds into "not raised".
            stable = self._holds[raise_key].update(t, bool(arm_raised(features, side)))
            if self._triggers[raise_key].update(t, stable):
                events.append(GestureEvent("arm_raised", name, features.frame_index, t))

            wave_key = f"{name}_wave"
            active = self._waves[name].update(t, motion.arm(side).direction)
            if self._triggers[wave_key].update(t, active):
                events.append(GestureEvent("wave", name, features.frame_index, t))

        crossed = self._holds["arms_crossed"].update(t, bool(arms_crossed(features)))
        if self._triggers["arms_crossed"].update(t, crossed):
            events.append(GestureEvent("arms_crossed", None, features.frame_index, t))

        open_ = self._holds["arms_open"].update(
            t, bool(arms_open(features, self.config.open_threshold))
        )
        if self._triggers["arms_open"].update(t, open_):
            events.append(GestureEvent("arms_open", None, features.frame_index, t))

        return events

"""Motion Engine — turns a stream of still frames into movement.

Everything before this file is memoryless: one frame in, one row of numbers out.
Here the pipeline grows a short-term memory, and that changes what it can say.
A single frame can only report "the wrist is above the shoulder"; a history can
report "the wrist is on its way up".

Three decisions carry the whole design:

1. **Velocity is per second, never per frame.** The webcam does not deliver a
   fixed frame rate (this machine measured ~22 FPS, and it dips when the CPU is
   busy). A per-frame delta would make the same physical gesture produce
   different numbers on a fast and a slow machine — and Sprint 08 would train a
   model on that artefact. Dividing by the real elapsed time removes the machine
   from the feature.

2. **The signal is measured against the body, not the image.** The tracked
   quantity is `wrist_height` (shoulder widths above your own shoulder), so
   walking towards the camera, or the laptop wobbling, does not read as arm
   movement. "Still" here means "still relative to your torso", which is what a
   gesture actually is.

3. **Landmarks jitter, so zero velocity does not exist.** A perfectly still arm
   still measures a few hundredths of a shoulder-width per second of noise.
   Without a dead zone the direction would flicker UP/DOWN every frame. The
   slope is fitted over a whole time window instead of two consecutive frames,
   and anything under `still_threshold` is reported as STILL.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Optional

from backend.features.features import FrameFeatures
from backend.vision.landmarks import Point, PoseSnapshot, Side


class Direction(str, Enum):
    """Vertical verdict for one arm — the Sprint 04 acceptance criterion.

    `str` mixin so it serialises straight to JSON in Sprint 05 without a
    conversion step.
    """

    UP = "up"
    DOWN = "down"
    STILL = "still"
    UNKNOWN = "unknown"  # joint not visible, or not enough history yet


@dataclass(frozen=True)
class MotionConfig:
    """Tuning for the temporal layer.

    Defaults are set for ~30 FPS: a 0.4 s window is ~12 frames, long enough to
    average out landmark jitter and short enough that a fast arm swing is not
    smeared into nothing.
    """

    #: How far back the slope is fitted. Longer = smoother but laggier.
    window_seconds: float = 0.4
    #: Samples needed before any velocity is reported. Two points would fit a
    #: line through pure noise.
    min_samples: int = 4
    #: Minimum time actually spanned by those samples. Guards against a burst of
    #: frames arriving with near-identical timestamps, where the slope explodes.
    min_span: float = 0.08
    #: If the signal vanishes for longer than this, the history is dropped
    #: instead of being joined across the blackout. A single missed frame is
    #: tolerated; a two-second occlusion is not.
    max_gap: float = 0.30
    #: Dead zone, in shoulder widths per second. Below it: STILL.
    still_threshold: float = 0.35
    #: How many wrist positions the on-screen trail keeps.
    trail_length: int = 32


class SignalTrack:
    """Rolling history of one scalar over time, with a least-squares slope.

    Deliberately generic: it does not know whether it is holding a wrist height,
    an elbow angle or, later, a classifier confidence.
    """

    def __init__(self, config: MotionConfig = MotionConfig()) -> None:
        self.config = config
        self._samples: Deque[tuple[float, float]] = deque()

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def samples(self) -> list[tuple[float, float]]:
        return list(self._samples)

    @property
    def latest(self) -> Optional[float]:
        return self._samples[-1][1] if self._samples else None

    @property
    def span(self) -> float:
        """Seconds covered by the retained samples."""
        if len(self._samples) < 2:
            return 0.0
        return self._samples[-1][0] - self._samples[0][0]

    def clear(self) -> None:
        self._samples.clear()

    def add(self, timestamp: float, value: Optional[float]) -> None:
        """Record one observation. `None` means the joint was not visible.

        A gap shorter than `max_gap` is simply skipped — the fit tolerates the
        hole. A longer one wipes the history: we do not know what the arm did
        while it was hidden, and inventing a straight line across the blackout
        would produce a confident, wrong velocity.
        """
        if value is None:
            if self._samples and timestamp - self._samples[-1][0] > self.config.max_gap:
                self._samples.clear()
            return

        if self._samples:
            if timestamp - self._samples[-1][0] > self.config.max_gap:
                self._samples.clear()
            elif timestamp < self._samples[-1][0]:
                # Time went backwards (camera restart). Old samples are unusable.
                self._samples.clear()

        self._samples.append((timestamp, float(value)))
        self._prune(timestamp)

    def _prune(self, now: float) -> None:
        cutoff = now - self.config.window_seconds
        while len(self._samples) > 2 and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def velocity(self) -> Optional[float]:
        """Units per second, from a least-squares fit over the window.

        The naive alternative — (last - first) / dt — throws away every sample
        in between and is therefore driven by the noise on exactly two of them.
        The fit uses all of them, which is the cheapest smoothing available.
        """
        if len(self._samples) < self.config.min_samples:
            return None
        if self.span < self.config.min_span:
            return None

        n = len(self._samples)
        mean_t = sum(t for t, _ in self._samples) / n
        mean_v = sum(v for _, v in self._samples) / n
        numerator = sum((t - mean_t) * (v - mean_v) for t, v in self._samples)
        denominator = sum((t - mean_t) ** 2 for t, _ in self._samples)
        if denominator <= 0.0:
            return None
        return numerator / denominator


def classify(velocity: Optional[float], threshold: float) -> Direction:
    """Signed velocity -> UP / DOWN / STILL.

    Positive is up: `wrist_height` was already flipped in `positions.py` so that
    a raised arm gives a bigger number, despite image y growing downwards.
    """
    if velocity is None:
        return Direction.UNKNOWN
    if velocity > threshold:
        return Direction.UP
    if velocity < -threshold:
        return Direction.DOWN
    return Direction.STILL


@dataclass(frozen=True)
class ArmMotion:
    """How one arm is moving, in body-relative units per second."""

    #: Shoulder widths per second, positive = rising.
    velocity_y: Optional[float] = None
    #: Shoulder widths per second, positive = towards the person's own left.
    velocity_x: Optional[float] = None
    #: Magnitude of the 2D wrist velocity — "moving at all", any direction.
    speed: Optional[float] = None
    #: Degrees per second at the elbow, positive = extending.
    elbow_velocity: Optional[float] = None
    #: Vertical verdict: parado / subindo / descendo.
    direction: Direction = Direction.UNKNOWN
    #: Moving at all, in any direction. Kept separate from `direction` on
    #: purpose: an arm sweeping sideways is `STILL` vertically but very much
    #: moving, and collapsing the two would make the HUD lie.
    moving: Optional[bool] = None

    @property
    def complete(self) -> bool:
        """True when every numeric signal is known -- not occluded, and the
        tracker has warmed up past `min_samples`. Same rule `FrameFeatures.complete`
        applies to static features (Sprint 03), extended to motion: a `None`
        here means "we don't know yet", never "not moving"."""
        return None not in (self.velocity_y, self.velocity_x, self.speed, self.elbow_velocity)


@dataclass(frozen=True)
class MotionState:
    """One frame's movement verdict, both arms."""

    frame_index: int
    timestamp: float
    left: ArmMotion = field(default_factory=ArmMotion)
    right: ArmMotion = field(default_factory=ArmMotion)

    def arm(self, side: Side) -> ArmMotion:
        return self.left if side is Side.LEFT else self.right

    @property
    def complete(self) -> bool:
        """True when both arms' motion is fully known.

        Anything that reads motion as a model input (`PipelineRunner._predict`)
        must check this the same way it already checks `FrameFeatures.complete`
        -- a frame or two after the camera starts, or right after an occlusion
        clears, the tracker has not warmed up yet (`MotionConfig.min_samples`)
        and every numeric field here is `None`, not `0`.
        """
        return self.left.complete and self.right.complete

    def to_dict(self) -> dict:
        """Flat payload for the Sprint 05 WebSocket."""
        out: dict = {"frame_index": self.frame_index, "timestamp": self.timestamp}
        for name, arm in (("left", self.left), ("right", self.right)):
            out[f"{name}_velocity_y"] = arm.velocity_y
            out[f"{name}_velocity_x"] = arm.velocity_x
            out[f"{name}_speed"] = arm.speed
            out[f"{name}_elbow_velocity"] = arm.elbow_velocity
            out[f"{name}_direction"] = arm.direction.value
            out[f"{name}_moving"] = arm.moving
        return out


class MotionTracker:
    """Feed it `FrameFeatures` in order; it answers what the arms are doing.

    Stateful on purpose, and the only stateful thing in `features/`. It is fed
    the *feature row*, not the pose snapshot, so it inherits the scale and
    aspect invariance already established upstream instead of re-deriving it.
    """

    def __init__(self, config: MotionConfig = MotionConfig()) -> None:
        self.config = config
        self._tracks = {
            side: {
                "height": SignalTrack(config),
                "offset": SignalTrack(config),
                "elbow": SignalTrack(config),
            }
            for side in (Side.LEFT, Side.RIGHT)
        }

    def reset(self) -> None:
        for tracks in self._tracks.values():
            for track in tracks.values():
                track.clear()

    def update(self, features: FrameFeatures) -> MotionState:
        t = features.timestamp
        arms = {}
        for side, prefix in ((Side.LEFT, "left"), (Side.RIGHT, "right")):
            tracks = self._tracks[side]
            tracks["height"].add(t, getattr(features, f"{prefix}_wrist_height"))
            tracks["offset"].add(t, getattr(features, f"{prefix}_wrist_offset_x"))
            tracks["elbow"].add(t, getattr(features, f"{prefix}_elbow_angle"))

            vy = tracks["height"].velocity()
            vx = tracks["offset"].velocity()
            speed = None if vy is None or vx is None else math.hypot(vx, vy)
            arms[side] = ArmMotion(
                velocity_y=vy,
                velocity_x=vx,
                speed=speed,
                elbow_velocity=tracks["elbow"].velocity(),
                direction=classify(vy, self.config.still_threshold),
                moving=None if speed is None else speed > self.config.still_threshold,
            )

        return MotionState(
            frame_index=features.frame_index,
            timestamp=features.timestamp,
            left=arms[Side.LEFT],
            right=arms[Side.RIGHT],
        )


class Trail:
    """Recent wrist positions in normalized image coords, for drawing the path.

    Display-only, which is why it takes the raw snapshot: the trajectory is
    drawn in *image* space, while everything the model sees stays in body space.
    `None` entries are kept, not dropped, so a lost joint breaks the line into
    two segments instead of teleporting across the gap.
    """

    def __init__(self, length: int = 32) -> None:
        self._points: Deque[Optional[Point]] = deque(maxlen=length)

    def __len__(self) -> int:
        return len(self._points)

    @property
    def points(self) -> list[Optional[Point]]:
        return list(self._points)

    def clear(self) -> None:
        self._points.clear()

    def add(self, point: Optional[Point]) -> None:
        self._points.append(point)

    def add_from(self, snapshot: PoseSnapshot, side: Side, threshold: float = 0.5) -> None:
        point = snapshot.point(side.wrist)
        self.add(point if point is not None and point.is_reliable(threshold) else None)

    def segments(self) -> list[list[Point]]:
        """Continuous runs of visible points — one polyline each."""
        runs: list[list[Point]] = []
        current: list[Point] = []
        for point in self._points:
            if point is None:
                if len(current) > 1:
                    runs.append(current)
                current = []
            else:
                current.append(point)
        if len(current) > 1:
            runs.append(current)
        return runs

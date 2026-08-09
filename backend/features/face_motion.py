"""Face Motion Engine — the temporal layer for facial signals.

Everything in `face_features.py` is memoryless: one snapshot in, one row of
numbers out. This file adds the same short-term memory `velocity.py` gave the
arms — "mudanças temporais desses sinais" (temporal change of these signals),
still strictly a rate/trend, never a state or an emotion. A rising
`smile_velocity` is a fact about how fast a blendshape score is changing; it
is not, and must never be presented as, "the person is starting to feel happy".

Two decisions:

1. **Trend signals reuse `SignalTrack` as-is.** Its own docstring already
   anticipates this ("does not know whether it is holding a wrist height, an
   elbow angle or, later, a classifier confidence") — smile/mouth-open/brow-
   raise are exactly that kind of scalar, so this file adds no new smoothing
   logic, just three more tracks fed from `FaceSignals` instead of
   `FrameFeatures`.
2. **Blink is a rate, not a velocity.** A blendshape score does not "trend"
   the way a continuous signal does — it is closer to a boolean (eye closed
   past a threshold, or not), so `_BlinkCounter` counts rising edges in a
   rolling window and reports blinks-per-minute, the same edge-counting shape
   `gestures/engine.py`'s `_WaveDetector` uses for reversals, applied to one
   eye at a time.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Optional

from backend.features.face_features import FaceSignals
from backend.features.velocity import MotionConfig, SignalTrack

#: One column set per trending signal, in payload/`to_dict()` order.
FACE_MOTION_COLUMNS: tuple[str, ...] = (
    "smile_velocity",
    "mouth_open_velocity",
    "brow_raise_velocity",
    "blink_rate_left",
    "blink_rate_right",
)


class _BlinkCounter:
    """Rising-edge count of one eye's blink blendshape, as a per-minute rate.

    `None` in, `None` out — a blendshape that was not available this frame
    (no face, or an occluded eye) is "unknown", never "not blinking".
    """

    def __init__(self, window_seconds: float = 60.0, threshold: float = 0.5) -> None:
        self._window = window_seconds
        self._threshold = threshold
        self._closed = False
        self._events: Deque[float] = deque()

    def update(self, timestamp: float, value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        closed = value >= self._threshold
        if closed and not self._closed:
            self._events.append(timestamp)
        self._closed = closed

        cutoff = timestamp - self._window
        while self._events and self._events[0] < cutoff:
            self._events.popleft()
        return len(self._events) * (60.0 / self._window)

    def reset(self) -> None:
        self._closed = False
        self._events.clear()


@dataclass(frozen=True)
class FaceMotionState:
    """How the face's signals are trending, one frame at a time."""

    frame_index: int
    timestamp: float
    smile_velocity: Optional[float] = None
    mouth_open_velocity: Optional[float] = None
    brow_raise_velocity: Optional[float] = None
    #: Blinks per minute, estimated from a rolling window — ramps up as more
    #: blinks land inside the window rather than being accurate from frame 1;
    #: document this the same way `SignalTrack.velocity()` documents needing
    #: `min_samples` before it reports anything.
    blink_rate_left: Optional[float] = None
    blink_rate_right: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "smile_velocity": self.smile_velocity,
            "mouth_open_velocity": self.mouth_open_velocity,
            "brow_raise_velocity": self.brow_raise_velocity,
            "blink_rate_left": self.blink_rate_left,
            "blink_rate_right": self.blink_rate_right,
        }


class FaceMotionTracker:
    """Feed it `FaceSignals` in order; it answers how they are trending.

    Stateful, same role `MotionTracker` plays for the arms — reset when
    nobody is watching, same as `MotionTracker.reset()`/`GestureEngine.reset()`.
    """

    def __init__(
        self,
        config: MotionConfig = MotionConfig(),
        blink_window_seconds: float = 60.0,
        blink_threshold: float = 0.5,
    ) -> None:
        self._tracks = {
            name: SignalTrack(config) for name in ("smile", "mouth_open", "brow_raise")
        }
        self._blinks = {
            "left": _BlinkCounter(blink_window_seconds, blink_threshold),
            "right": _BlinkCounter(blink_window_seconds, blink_threshold),
        }

    def reset(self) -> None:
        for track in self._tracks.values():
            track.clear()
        for blink in self._blinks.values():
            blink.reset()

    def update(self, signals: FaceSignals) -> FaceMotionState:
        t = signals.timestamp
        self._tracks["smile"].add(t, signals.smile)
        self._tracks["mouth_open"].add(t, signals.mouth_open)
        self._tracks["brow_raise"].add(t, signals.brow_raise)
        blink_left = self._blinks["left"].update(t, signals.eye_blink_left)
        blink_right = self._blinks["right"].update(t, signals.eye_blink_right)

        return FaceMotionState(
            frame_index=signals.frame_index,
            timestamp=signals.timestamp,
            smile_velocity=self._tracks["smile"].velocity(),
            mouth_open_velocity=self._tracks["mouth_open"].velocity(),
            brow_raise_velocity=self._tracks["brow_raise"].velocity(),
            blink_rate_left=blink_left,
            blink_rate_right=blink_right,
        )

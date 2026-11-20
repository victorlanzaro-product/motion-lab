"""Time-based temporal features over the recent wrist-height signal.

The window is measured in seconds, so changing the camera frame rate does not
change how much history is summarized. A long blackout, an incomplete wrist
measurement, or too little history makes the result ``None``: the missing
movement is unknown and must not be bridged or filled with zero.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Deque, Optional

from backend.features.features import FrameFeatures

DEFAULT_WINDOW_DEADBAND = 0.35


@dataclass(frozen=True)
class ArmWindowFeatures:
    """Summary for one wrist over a complete window.

    Heights and speeds are in shoulder widths and shoulder widths per second,
    respectively. ``moving_fraction`` is a fraction of elapsed time rather
    than frames.
    """

    mean_wrist_height: float
    std_wrist_height: float
    range_wrist_height: float
    direction_reversals: int
    max_speed: float
    max_vertical_speed: float
    moving_fraction: float


@dataclass(frozen=True)
class WindowState:
    """Per-arm temporal features; either side may be unavailable."""

    timestamp: float
    window_seconds: float
    left: Optional[ArmWindowFeatures]
    right: Optional[ArmWindowFeatures]


@dataclass(frozen=True)
class WindowRow:
    """One CSV row's reconstructed window; invalid warm-up rows have None."""

    session_id: str
    frame_index: int
    timestamp: float
    features: Optional[WindowState]


class WindowFeatures:
    """Build rolling temporal features from left/right wrist heights.

    ``push(timestamp, features)`` accepts a :class:`FrameFeatures` or a
    mapping such as one returned by ``csv.DictReader``. It reads wrist height
    and horizontal offset for both arms, deriving 2D speed from their changes.
    It returns a :class:`WindowState` only after a complete time window and
    ``min_samples`` observations are available for at least one arm. It
    returns ``None`` during warm-up while neither arm has a complete window.
    Each arm has independent continuity: a gap, out-of-order timestamp, or
    missing height/offset clears only that arm's history, and its result is
    ``None`` until coverage recovers. Missing joints are never interpreted as
    zero.

    2D speeds are computed from height and horizontal-offset slopes between
    observations in shoulder widths per second. Direction reversals use the
    signed vertical slope and ignore values at or below ``deadband``. The same
    threshold applied to 2D speed defines whether an interval counts as
    moving. Statistics and movement are calculated over the last
    ``window_seconds`` using elapsed time, not a fixed frame count.
    """

    def __init__(
        self,
        window_seconds: float = 1.0,
        max_gap: float = 0.30,
        min_samples: int = 4,
        deadband: float = DEFAULT_WINDOW_DEADBAND,
    ) -> None:
        if not math.isfinite(window_seconds) or window_seconds <= 0:
            raise ValueError("window_seconds must be a finite positive number")
        if not math.isfinite(max_gap) or max_gap <= 0:
            raise ValueError("max_gap must be a finite positive number")
        if isinstance(min_samples, bool) or not isinstance(min_samples, int) or min_samples < 2:
            raise ValueError("min_samples must be an integer of at least 2")
        if not math.isfinite(deadband) or deadband < 0:
            raise ValueError("deadband must be a finite non-negative number")

        self.window_seconds = float(window_seconds)
        self.max_gap = float(max_gap)
        self.min_samples = int(min_samples)
        self.deadband = float(deadband)
        self._samples: dict[str, Deque[tuple[float, float, float]]] = {
            "left": deque(),
            "right": deque(),
        }

    def reset(self) -> None:
        """Forget history after a session boundary or unusable observation."""
        for samples in self._samples.values():
            samples.clear()

    def push(
        self,
        timestamp: float,
        features: FrameFeatures | Mapping[str, Any],
    ) -> Optional[WindowState]:
        """Add one observation and return its valid trailing window, if ready.

        ``features`` provides each arm's ``wrist_height`` and ``wrist_offset_x``.
        A missing or non-finite value clears only that arm's history. The other
        arm continues independently, so a wrist outside the frame does not
        invalidate a visible wrist. Missing joints are never treated as zero.
        """
        try:
            now = float(timestamp)
        except (TypeError, ValueError, OverflowError):
            self.reset()
            return None

        if not math.isfinite(now):
            self.reset()
            return None

        cutoff = now - self.window_seconds
        left = self._push_arm("left", now, _read_pair(features, "left"), cutoff)
        right = self._push_arm("right", now, _read_pair(features, "right"), cutoff)
        if left is None and right is None:
            return None
        return WindowState(
            timestamp=now,
            window_seconds=self.window_seconds,
            left=left,
            right=right,
        )

    def _push_arm(
        self,
        side: str,
        now: float,
        measurement: Optional[tuple[float, float]],
        cutoff: float,
    ) -> Optional[ArmWindowFeatures]:
        samples = self._samples[side]
        if measurement is None:
            samples.clear()
            return None

        if samples:
            elapsed = now - samples[-1][0]
            if elapsed <= 0 or elapsed > self.max_gap:
                # Do not bridge a clock restart, duplicate timestamp, or
                # blackout. The unaffected arm keeps its own history.
                samples.clear()

        height, offset = measurement
        samples.append((now, height, offset))
        self._prune(samples, now)

        # Retain one point before the cutoff for interpolation. Requiring a
        # real observation at or before the cutoff prevents partial windows
        # from being reported as a full N-second summary.
        if (
            sum(sample[0] >= cutoff for sample in samples) < self.min_samples
            or samples[0][0] > cutoff
        ):
            return None
        return self._summarize(samples, cutoff, now)

    def _prune(self, samples: Deque[tuple[float, float, float]], now: float) -> None:
        cutoff = now - self.window_seconds
        while len(samples) > 2 and samples[1][0] < cutoff:
            samples.popleft()

    def _summarize(
        self,
        samples: Deque[tuple[float, float, float]],
        cutoff: float,
        end: float,
    ) -> Optional[ArmWindowFeatures]:
        points = list(samples)
        if len(points) < 2 or points[0][0] > cutoff or points[-1][0] < end:
            return None

        # Integrate the piecewise-linear signal. This makes mean and standard
        # deviation insensitive to uneven frame spacing while keeping range
        # based on the signal's observed/interpolated extrema.
        total = 0.0
        total_squared = 0.0
        max_speed = 0.0
        max_vertical_speed = 0.0
        moving_seconds = 0.0
        reversals = 0
        previous_direction = 0
        values: list[float] = []

        for point0, point1 in zip(points, points[1:]):
            t0, y0, x0 = point0
            t1, y1, x1 = point1
            overlap_start = max(t0, cutoff)
            overlap_end = min(t1, end)
            if overlap_end <= overlap_start:
                continue

            interval = t1 - t0
            if interval <= 0 or interval > self.max_gap:
                return None
            vertical_speed = (y1 - y0) / interval
            horizontal_speed = (x1 - x0) / interval
            speed = math.hypot(horizontal_speed, vertical_speed)
            start_ratio = (overlap_start - t0) / interval
            end_ratio = (overlap_end - t0) / interval
            clipped_start = y0 + (y1 - y0) * start_ratio
            clipped_end = y0 + (y1 - y0) * end_ratio
            duration = overlap_end - overlap_start

            total += duration * (clipped_start + clipped_end) / 2
            total_squared += duration * (
                clipped_start * clipped_start
                + clipped_start * clipped_end
                + clipped_end * clipped_end
            ) / 3
            values.extend((clipped_start, clipped_end))

            max_speed = max(max_speed, speed)
            max_vertical_speed = max(max_vertical_speed, abs(vertical_speed))
            if speed > self.deadband:
                moving_seconds += duration
            if abs(vertical_speed) > self.deadband:
                direction = 1 if vertical_speed > 0 else -1
                if previous_direction and direction != previous_direction:
                    reversals += 1
                previous_direction = direction

        if not values:
            return None

        duration = end - cutoff
        mean = total / duration
        variance = max(0.0, total_squared / duration - mean * mean)
        return ArmWindowFeatures(
            mean_wrist_height=mean,
            std_wrist_height=math.sqrt(variance),
            range_wrist_height=max(values) - min(values),
            direction_reversals=reversals,
            max_speed=max_speed,
            max_vertical_speed=max_vertical_speed,
            moving_fraction=min(1.0, moving_seconds / duration),
        )


def rebuild_windows(
    rows: Iterable[Mapping[str, Any]],
    window_seconds: float = 1.0,
    max_gap: float = 0.30,
    min_samples: int = 4,
    deadband: float = DEFAULT_WINDOW_DEADBAND,
) -> list[WindowRow]:
    """Rebuild rolling windows from existing dataset CSV rows.

    Rows are grouped by ``session_id`` and sorted by ``frame_index`` then
    ``timestamp`` before streaming through a fresh tracker for each session.
    No CSV columns are added or changed. A :class:`WindowRow` is returned for
    every source row; its ``features`` value is ``None`` during warm-up or
    whenever the underlying time window is invalid.
    """
    indexed_rows: list[tuple[str, int, float, int, Mapping[str, Any]]] = []
    for ordinal, row in enumerate(rows):
        try:
            raw_session_id = row["session_id"]
            session_id = "" if raw_session_id is None else str(raw_session_id).strip()
            frame_index = int(row["frame_index"])
            timestamp = float(row["timestamp"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "each dataset row needs session_id, frame_index, and timestamp"
            ) from exc
        if not session_id:
            raise ValueError("session_id must be non-empty")
        indexed_rows.append((session_id, frame_index, timestamp, ordinal, row))

    indexed_rows.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    output: list[WindowRow] = []
    active_session: Optional[str] = None
    tracker: Optional[WindowFeatures] = None

    for session_id, frame_index, timestamp, _ordinal, row in indexed_rows:
        if session_id != active_session:
            active_session = session_id
            tracker = WindowFeatures(
                window_seconds=window_seconds,
                max_gap=max_gap,
                min_samples=min_samples,
                deadband=deadband,
            )
        assert tracker is not None
        output.append(
            WindowRow(
                session_id=session_id,
                frame_index=frame_index,
                timestamp=timestamp,
                features=tracker.push(timestamp, row),
            )
        )
    return output


def _read_height(features: FrameFeatures | Mapping[str, Any], name: str) -> Optional[float]:
    if isinstance(features, Mapping):
        value = features.get(name)
    else:
        value = getattr(features, name)
    if value is None or value == "":
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _read_pair(
    features: FrameFeatures | Mapping[str, Any], side: str
) -> Optional[tuple[float, float]]:
    """Read one wrist's height and horizontal offset independently."""
    try:
        height = _read_height(features, f"{side}_wrist_height")
        offset = _read_height(features, f"{side}_wrist_offset_x")
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None
    if height is None or offset is None:
        return None
    return height, offset

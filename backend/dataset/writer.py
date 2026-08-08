"""Dataset Writer — turns labeled feature rows into the CSV Sprint 08 trains on.

Three decisions carry this file:

1. **Motion rides along with the static features.** Every `FEATURE_NAMES`
   column describes a single instant, but "wave" is not a posture — it *is*
   motion. A classifier trained on position alone has no way to tell a wrist
   mid-swing from a wrist that just happens to be there. `MOTION_COLUMNS`
   (velocity, direction) gives it the one signal that can actually separate
   the two, at the cost of a few extra columns Sprint 08 is free to ignore for
   the purely postural labels.
2. **An incomplete frame is not a training example.** `FrameFeatures.complete`
   already answers "is every ML column filled" (Sprint 03's rule: an occluded
   joint is `None`, never a guessed `0`). Writing a half-`None` row here would
   just move that guess downstream to Sprint 08. Incomplete frames are
   skipped and counted instead — `skipped` is what the live HUD shows so an
   occlusion is visible while it is happening, not after the session ends.
3. **One shared file, not one per session.** `session_id` is a column, not a
   filename, so `data/training/dataset.csv` accumulates across every
   recording session and Sprint 08 never has to glob and concatenate files to
   see the whole dataset.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

from backend.features.features import FEATURE_NAMES, FrameFeatures
from backend.features.velocity import MotionState

#: Default recording path: features + motion, labeled, never pixels.
DEFAULT_DATASET_PATH = Path("data/training/dataset.csv")

#: The five classes Sprint 07 collects by default. "idle" is not a gesture at
#: all — it is the negative example. Without it a classifier only ever sees
#: positive cases and never learns to say "nothing is happening".
DEFAULT_LABELS: tuple[str, ...] = ("arm_raised", "wave", "arms_crossed", "arms_open", "idle")

#: Order matches `MotionState.to_dict()` with the envelope (frame_index,
#: timestamp) stripped — those already have their own dataset columns.
MOTION_COLUMNS: tuple[str, ...] = (
    "left_velocity_y",
    "left_velocity_x",
    "left_speed",
    "left_elbow_velocity",
    "left_direction",
    "left_moving",
    "right_velocity_y",
    "right_velocity_x",
    "right_speed",
    "right_elbow_velocity",
    "right_direction",
    "right_moving",
)

#: Column order for data/training/dataset.csv. Append only, same rule as
#: `FEATURE_NAMES`: reordering silently invalidates every row written before.
DATASET_COLUMNS: tuple[str, ...] = (
    ("session_id", "frame_index", "timestamp", "label")
    + FEATURE_NAMES
    + ("shoulder_width",)
    + MOTION_COLUMNS
)


def _row(session_id: str, label: str, features: FrameFeatures, motion: MotionState) -> dict:
    row: dict = {
        "session_id": session_id,
        "frame_index": features.frame_index,
        "timestamp": features.timestamp,
        "label": label,
    }
    for name in FEATURE_NAMES:
        row[name] = getattr(features, name)
    row["shoulder_width"] = features.shoulder_width
    motion_flat = motion.to_dict()
    for name in MOTION_COLUMNS:
        row[name] = motion_flat.get(name)
    return row


class DatasetWriter:
    """Appends labeled rows to one CSV, shared across recording sessions.

    Follows the same open/close/context-manager shape as `CameraEngine` and
    `PoseDetector`: the file handle is held open for the life of a recording
    session, and every row is flushed immediately — a session killed with
    Ctrl+C must not lose the samples already written.
    """

    def __init__(self, path: Path = DEFAULT_DATASET_PATH, session_id: str = "") -> None:
        self.path = Path(path)
        self.session_id = session_id
        self.counts: dict[str, int] = {}
        self.skipped = 0
        self._file = None
        self._writer: Optional[csv.DictWriter] = None

    def open(self) -> "DatasetWriter":
        if self._file is not None:
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self.path.exists() or self.path.stat().st_size == 0
        self._file = self.path.open("a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=DATASET_COLUMNS)
        if is_new:
            self._writer.writeheader()
            self._file.flush()
        return self

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
        self._file = None
        self._writer = None

    def __enter__(self) -> "DatasetWriter":
        return self.open()

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def write(self, label: str, features: FrameFeatures, motion: MotionState) -> bool:
        """One frame, one row. Returns False (and counts a skip) if incomplete."""
        if self._writer is None:
            self.open()
        if not features.complete:
            self.skipped += 1
            return False
        self._writer.writerow(_row(self.session_id, label, features, motion))
        self._file.flush()
        self.counts[label] = self.counts.get(label, 0) + 1
        return True

    @property
    def total(self) -> int:
        return sum(self.counts.values())

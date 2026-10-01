"""Dataset Writer — turns labeled feature rows into the CSV Sprint 08 trains on.

Four decisions carry this file:

1. **Motion rides along with the static features.** `MOTION_COLUMNS`
   (velocity, direction) was added so a classifier could tell a wrist
   mid-swing from a wrist that just happens to be there — the one signal a
   static posture snapshot cannot give it. The motivating case, "wave",
   turned out to need more than one frame's instantaneous velocity to mean
   anything and was pulled from the trainable vocabulary entirely (decision
   4 below); the columns stay regardless, since they cost Sprint 08 nothing
   to ignore for the purely postural labels and a future label that
   genuinely separates on arm speed would need them again.
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
4. **"wave" is not a trainable label.** It is a class of *motion* (a
   direction reversal repeated a few times), not a posture, and this
   per-frame vector only ever carries one instant's velocity/direction — not
   the reversal window `GestureEngine`'s `_WaveDetector`
   (`backend/gestures/engine.py`) actually needs to tell a real wave from an
   arm rising without repeating. Recording "wave" here would silently train
   the Forest on a signal it cannot separate, and report a confident label
   for something it never really learned. `write()` refuses the label
   outright (`REMOVED_LABELS`); recognizing a wave stays `GestureEngine`'s
   job alone until the ML vector gets a real temporal window.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Optional

from backend.features.features import FEATURE_NAMES, FrameFeatures
from backend.features.velocity import MotionState

#: Default recording path: features + motion, labeled, never pixels.
DEFAULT_DATASET_PATH = Path("data/training/dataset.csv")

#: The four classes Sprint 07 collects by default. "idle" is not a gesture at
#: all — it is the negative example. Without it a classifier only ever sees
#: positive cases and never learns to say "nothing is happening". "wave" is
#: deliberately absent — see module docstring, decision 4, and `REMOVED_LABELS`.
DEFAULT_LABELS: tuple[str, ...] = ("arm_raised", "arms_crossed", "arms_open", "idle")

#: Labels this dataset used to collect and no longer does, mapped to why —
#: `write()` refuses them so a session cannot silently start re-accumulating
#: rows for a label the trainable vocabulary dropped. Keyed by label so a
#: second removed label in the future has one place to add to, not a second
#: bespoke check.
REMOVED_LABELS: dict[str, str] = {
    "wave": (
        "'wave' is out of the trainable ML vocabulary (module docstring, "
        "decision 4): it is motion, not a posture, and this per-frame vector "
        "cannot reproduce the reversal window GestureEngine's _WaveDetector "
        "needs to recognize one for real. Recognizing a wave is GestureEngine's "
        "job alone -- do not record it into dataset.csv."
    ),
}

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

#: Direction is excluded from ML training (backend/ml/train.py decision 2: a
#: lossy summary of the signed velocity columns, no extra separating power).
#: Defined here, not re-derived in train.py, so the two modules cannot drift.
NUMERIC_MOTION_COLUMNS: tuple[str, ...] = tuple(
    name for name in MOTION_COLUMNS if not name.endswith("_direction")
)

#: Column order for data/training/dataset.csv. Append only, same rule as
#: `FEATURE_NAMES`: reordering silently invalidates every row written before.
DATASET_COLUMNS: tuple[str, ...] = (
    ("session_id", "frame_index", "timestamp", "label")
    + FEATURE_NAMES
    + ("shoulder_width",)
    + MOTION_COLUMNS
)


class DatasetWriterError(RuntimeError):
    """Raised when a row or the file itself violates the dataset's own
    invariants — a caller bug (empty `session_id`, a `label` that is not
    actually a label, features paired with a different frame's motion, a
    non-finite value slipping through), never the data-quality gap
    `skipped`/`FrameFeatures.complete` already models (an occluded joint)."""


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
        if not is_new:
            self._check_header()
        self._file = self.path.open("a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=DATASET_COLUMNS)
        if is_new:
            self._writer.writeheader()
            self._file.flush()
        return self

    def _check_header(self) -> None:
        """Refuse to append onto a file written by a different column order.

        `load_dataset` already rejects a *missing* column at read time
        (`backend/ml/train.py`); this catches the same drift earlier, at
        write time, before a second, incompatible schema gets mixed into one
        file byte for byte.
        """
        with self.path.open(newline="") as handle:
            first_line = handle.readline()
        header = next(csv.reader([first_line]), [])
        if tuple(header) != DATASET_COLUMNS:
            raise DatasetWriterError(
                f"{self.path} already has a different header than the current "
                "backend/dataset/writer.py:DATASET_COLUMNS -- appending would "
                "silently mix two incompatible schemas in one file. Move/rename "
                "the old file, or point session_id-based training at a fresh path."
            )

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
        """One frame, one row. Returns False (and counts a skip) if incomplete.

        Raises `DatasetWriterError` for a caller-side invariant violation
        rather than writing a row nobody could train on: an empty
        `session_id` (the grouped train/test split in `backend/ml/train.py`
        needs a real one on every row), an empty `label`, `features`/`motion`
        from two different frames (a bug upstream, not an occlusion), or a
        non-finite value that slipped past `FrameFeatures`/`MotionState`.
        """
        if self._writer is None:
            self.open()
        if not self.session_id:
            raise DatasetWriterError(
                "DatasetWriter.session_id is empty. Every row needs one for the "
                "grouped train/test split (backend/ml/train.py) to mean anything "
                "-- set writer.session_id before writing."
            )
        label = (label or "").strip()
        if not label:
            raise DatasetWriterError("label must be a non-empty string.")
        if label in REMOVED_LABELS:
            raise DatasetWriterError(REMOVED_LABELS[label])
        if label not in DEFAULT_LABELS:
            casefolded = label.casefold()
            match = next((known for known in DEFAULT_LABELS if known.casefold() == casefolded), None)
            hint = f" Did you mean {match!r}? Labels are case-sensitive." if match else ""
            raise DatasetWriterError(
                f"{label!r} is not a recognized label -- DatasetWriter only accepts "
                f"{DEFAULT_LABELS} (whitespace already stripped, case must match "
                f"exactly).{hint} A genuinely new class needs a product decision to "
                "add it to backend/dataset/writer.py:DEFAULT_LABELS first."
            )
        if features.frame_index != motion.frame_index or features.timestamp != motion.timestamp:
            raise DatasetWriterError(
                f"features (frame {features.frame_index} @ {features.timestamp}) and "
                f"motion (frame {motion.frame_index} @ {motion.timestamp}) are not the "
                "same frame -- writing them as one row would silently pair a posture "
                "with the wrong instant's velocity."
            )
        if not features.complete:
            self.skipped += 1
            return False

        row = _row(self.session_id, label, features, motion)
        non_finite = [
            name
            for name, value in row.items()
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and not math.isfinite(value)
        ]
        if non_finite:
            raise DatasetWriterError(
                f"non-finite value(s) {non_finite} in row for label {label!r} at "
                f"frame {features.frame_index} -- refusing to write a NaN/Inf that "
                "would look like a real number to Sprint 08's Random Forest."
            )

        self._writer.writerow(row)
        self._file.flush()
        self.counts[label] = self.counts.get(label, 0) + 1
        return True

    @property
    def total(self) -> int:
        return sum(self.counts.values())

"""Inspect dataset coverage using the same row classifier as model training.

The report keeps malformed and unknown rows visible instead of making the
dashboard infer coverage from the CSV or quietly treating unusable rows as
training examples.
"""

from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path
from typing import Any

from backend.dataset.writer import DATASET_COLUMNS, DEFAULT_DATASET_PATH, DEFAULT_LABELS
from backend.ml.train import DatasetError, classify_dataset_row

MIN_SESSIONS_PER_LABEL = 2


def _empty_stats(path: Path) -> dict[str, Any]:
    """Return the exact empty-dataset response without asking training to load it."""
    return {
        "path": str(path),
        "exists": False,
        "total_rows": 0,
        "usable_rows": 0,
        "dropped": {},
        "labels": {
            label: {"rows": 0, "sessions": 0, "seconds": 0.0}
            for label in DEFAULT_LABELS
        },
        "sessions": [],
        "min_sessions_per_label": MIN_SESSIONS_PER_LABEL,
        "ready": False,
        "problems": ["dataset ausente"],
    }


def _timestamp(row: dict[str, Any], row_number: int, path: Path) -> float:
    """Read a usable row's timestamp or fail visibly on a corrupt CSV value.

    The writer guarantees finite numeric timestamps. Replacing a damaged or
    missing value with zero would understate session duration without evidence.
    """
    raw = row.get("timestamp")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise DatasetError(
            f"{path} row {row_number} has an invalid timestamp {raw!r}; "
            "dataset coverage cannot measure session duration."
        ) from exc
    if not math.isfinite(value):
        raise DatasetError(
            f"{path} row {row_number} has a non-finite timestamp {raw!r}; "
            "dataset coverage cannot measure session duration."
        )
    return value


def dataset_stats(path: Path = DEFAULT_DATASET_PATH) -> dict[str, Any]:
    """Return row, label, session, and readiness statistics for ``path``.

    ``load_dataset`` intentionally refuses unknown labels so they can never
    train a model. Coverage uses its shared row classifier to count those rows
    as ``unknown_label`` and exclude them, making the problem inspectable before
    a training attempt. Other drop reasons remain exactly the loader's reasons.
    """
    path = Path(path)
    if not path.exists():
        return _empty_stats(path)

    labels = {
        label: {"rows": 0, "sessions": 0, "seconds": 0.0}
        for label in DEFAULT_LABELS
    }
    dropped: Counter[str] = Counter()
    # Keep one interval per recording session and label. The duration of each
    # interval is summed into its label after usable rows have been classified.
    session_rows: dict[tuple[str, str], dict[str, float | int]] = {}
    total_rows = 0
    usable_rows = 0

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        header = set(reader.fieldnames or ())
        missing_columns = set(DATASET_COLUMNS) - header
        if missing_columns:
            raise DatasetError(
                f"{path} is missing column(s) {sorted(missing_columns)} — its header "
                "does not match backend/dataset/writer.py:DATASET_COLUMNS. Re-record "
                "with the current hello_dataset.py rather than hand-editing the CSV."
            )

        for row_number, row in enumerate(reader, start=2):
            total_rows += 1
            reason, _vector = classify_dataset_row(row)
            if reason is not None:
                dropped[reason] += 1
                continue

            # ``classify_dataset_row`` only returns a vector for known labels
            # with all training features present and finite.
            label = (row.get("label") or "").strip()
            session_id = row.get("session_id") or ""
            timestamp = _timestamp(row, row_number, path)
            usable_rows += 1
            labels[label]["rows"] += 1

            key = (session_id, label)
            interval = session_rows.get(key)
            if interval is None:
                session_rows[key] = {"rows": 1, "min": timestamp, "max": timestamp}
            else:
                interval["rows"] += 1
                interval["min"] = min(float(interval["min"]), timestamp)
                interval["max"] = max(float(interval["max"]), timestamp)

    sessions: list[dict[str, Any]] = []
    for (session_id, label), interval in sorted(session_rows.items()):
        exact_seconds = float(interval["max"]) - float(interval["min"])
        seconds = round(exact_seconds, 1)
        labels[label]["sessions"] += 1
        labels[label]["seconds"] += exact_seconds
        sessions.append(
            {
                "session_id": session_id,
                "label": label,
                "rows": int(interval["rows"]),
                "seconds": seconds,
            }
        )
    for label in DEFAULT_LABELS:
        labels[label]["seconds"] = round(float(labels[label]["seconds"]), 1)

    problems = [
        (
            f"{label}: {labels[label]['sessions']} sessão "
            f"(mínimo {MIN_SESSIONS_PER_LABEL})"
            if labels[label]["sessions"] == 1
            else f"{label}: {labels[label]['sessions']} sessões (mínimo {MIN_SESSIONS_PER_LABEL})"
        )
        for label in DEFAULT_LABELS
        if labels[label]["sessions"] < MIN_SESSIONS_PER_LABEL
    ]
    unknown_count = dropped.get("unknown_label", 0)
    if unknown_count:
        problems.append(
            f"{unknown_count} linhas com rótulo desconhecido (treino vai recusar)"
        )

    return {
        "path": str(path),
        "exists": True,
        "total_rows": total_rows,
        "usable_rows": usable_rows,
        "dropped": {reason: count for reason, count in dropped.items() if count > 0},
        "labels": labels,
        "sessions": sessions,
        "min_sessions_per_label": MIN_SESSIONS_PER_LABEL,
        "ready": not problems,
        "problems": problems,
    }

"""Build one training row per complete temporal window in a dataset CSV.

The source CSV stays frame-based and unchanged. For schema 2, each usable
window keeps the static features from its final frame and adds time-based
summaries reconstructed from that same session's wrist history.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from backend.features.features import FEATURE_NAMES, FrameFeatures
from backend.features.window import (
    DEFAULT_WINDOW_DEADBAND,
    ArmWindowFeatures,
    WindowState,
    rebuild_windows,
)

_WINDOW_FIELDS: tuple[str, ...] = (
    "mean_wrist_height",
    "std_wrist_height",
    "range_wrist_height",
    "direction_reversals",
    "max_speed",
    "max_vertical_speed",
    "moving_fraction",
)

WINDOW_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    f"{side}_window_{field}"
    for side in ("left", "right")
    for field in _WINDOW_FIELDS
)

# The feature vector preserves the v1 static columns and appends temporal
# columns in a stable order. Bundles store this exact tuple.
WINDOW_TRAINING_COLUMNS: tuple[str, ...] = FEATURE_NAMES + WINDOW_FEATURE_COLUMNS


def build_window_rows(
    rows: Iterable[Mapping[str, Any]],
    window_seconds: float = 1.0,
    max_gap: float = 0.30,
    min_samples: int = 4,
    deadband: float = DEFAULT_WINDOW_DEADBAND,
) -> list[dict[str, Any]]:
    """Return complete window rows rebuilt from existing frame rows.

    Each output row contains the source row's ``label``, session/frame/time
    metadata, all ``FEATURE_NAMES`` static values from the window's final
    frame, and the 14 numeric left/right window summaries. Sessions are
    independently sorted and streamed through :func:`rebuild_windows`; no
    history crosses a session boundary. Warm-up windows, invalid gaps, or a
    window missing either wrist are omitted instead of filling unknowns with
    zero.
    """
    source_rows = list(rows)
    ordered_rows = sorted(enumerate(source_rows), key=_sort_key)
    sorted_rows = [row for _ordinal, row in ordered_rows]
    rebuilt = rebuild_windows(
        sorted_rows,
        window_seconds=window_seconds,
        max_gap=max_gap,
        min_samples=min_samples,
        deadband=deadband,
    )

    output: list[dict[str, Any]] = []
    for row, window in zip(sorted_rows, rebuilt):
        result = build_window_input_row(row, window.features)
        if result is not None:
            result["session_id"] = window.session_id
            result["frame_index"] = window.frame_index
            result["timestamp"] = window.timestamp
            output.append(result)
    return output


def build_window_input_row(
    features: FrameFeatures | Mapping[str, Any],
    window: WindowState | None,
) -> dict[str, Any] | None:
    """Combine a frame's static values with a complete temporal window.

    This is the shared training/inference row builder. Returning ``None`` for
    either missing arm keeps an incomplete temporal vector out of the model;
    absent measurements are never filled with zero.
    """
    if window is None or window.left is None or window.right is None:
        return None

    source = features.to_dict() if isinstance(features, FrameFeatures) else features
    row: dict[str, Any] = {name: source.get(name) for name in FEATURE_NAMES}
    for name in ("session_id", "label"):
        if name in source:
            row[name] = _label(source[name]) if name == "label" else source[name]
    row["frame_index"] = source.get("frame_index")
    row["timestamp"] = window.timestamp
    row.update(_arm_values("left", window.left))
    row.update(_arm_values("right", window.right))
    return row


def _sort_key(item: tuple[int, Mapping[str, Any]]) -> tuple[str, int, float, int]:
    ordinal, row = item
    try:
        session_id = "" if row["session_id"] is None else str(row["session_id"]).strip()
        frame_index = int(row["frame_index"])
        timestamp = float(row["timestamp"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "each dataset row needs session_id, frame_index, and timestamp"
        ) from exc
    if not session_id:
        raise ValueError("session_id must be non-empty")
    return session_id, frame_index, timestamp, ordinal


def _label(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _arm_values(side: str, values: ArmWindowFeatures) -> dict[str, float | int]:
    return {
        f"{side}_window_mean_wrist_height": values.mean_wrist_height,
        f"{side}_window_std_wrist_height": values.std_wrist_height,
        f"{side}_window_range_wrist_height": values.range_wrist_height,
        f"{side}_window_direction_reversals": values.direction_reversals,
        f"{side}_window_max_speed": values.max_speed,
        f"{side}_window_max_vertical_speed": values.max_vertical_speed,
        f"{side}_window_moving_fraction": values.moving_fraction,
    }

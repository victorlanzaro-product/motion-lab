"""Dataset coverage tests: local CSVs only, no camera or model training."""

from __future__ import annotations

import csv
import json
import sys

import pytest

from backend.dataset.stats import MIN_SESSIONS_PER_LABEL, dataset_stats
from backend.dataset.writer import (
    DATASET_COLUMNS,
    DEFAULT_LABELS,
    DEFAULT_V2_DATASET_PATH,
    V2_LABELS,
)
from backend.ml.train import DatasetError, MIN_SAMPLES_PER_CLASS, load_dataset
from scripts.dataset_stats import main as dataset_stats_main


def _row(label: str, session_id: str, frame_index: int, timestamp: float) -> dict[str, str]:
    """Build a complete dataset-shaped row so stats exercises the real loader rules."""
    row = {column: "0" for column in DATASET_COLUMNS}
    row.update(
        {
            "session_id": session_id,
            "frame_index": str(frame_index),
            "timestamp": str(timestamp),
            "label": label,
        }
    )
    return row


def _write(path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DATASET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def test_missing_dataset_returns_the_contract_empty_shape(tmp_path):
    path = tmp_path / "missing.csv"

    stats = dataset_stats(path)

    assert stats == {
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


def test_missing_v2_dataset_includes_an_empty_wave_coverage_entry(tmp_path):
    path = tmp_path / "dataset_v2.csv"

    stats = dataset_stats(path, allowed_labels=V2_LABELS)

    assert DEFAULT_V2_DATASET_PATH.as_posix() == "data/training/dataset_v2.csv"
    assert list(stats["labels"]) == list(V2_LABELS)
    assert stats["labels"]["wave"] == {"rows": 0, "sessions": 0, "seconds": 0.0}
    assert stats["ready"] is False
    assert stats["problems"] == ["dataset ausente"]


def test_json_cli_prints_the_contract_and_exits_one_when_not_ready(tmp_path, monkeypatch, capsys):
    path = tmp_path / "missing.csv"
    monkeypatch.setattr(sys, "argv", ["dataset_stats.py", "--dataset", str(path), "--json"])

    exit_code = dataset_stats_main()

    assert exit_code == 1
    assert json.loads(capsys.readouterr().out) == dataset_stats(path)


def test_table_cli_exits_zero_when_every_label_has_enough_sessions(tmp_path, monkeypatch, capsys):
    path = tmp_path / "dataset.csv"
    rows = [
        _row(label, f"s{session}", index, index / 30)
        for index, (label, session) in enumerate(
            (label, session)
            for label in DEFAULT_LABELS
            for session in range(MIN_SESSIONS_PER_LABEL)
        )
    ]
    _write(path, rows)
    monkeypatch.setattr(sys, "argv", ["dataset_stats.py", "--dataset", str(path)])

    exit_code = dataset_stats_main()

    assert exit_code == 0
    assert "Cobertura por sessões pronta: sim" in capsys.readouterr().out


def test_stats_counts_rows_sessions_seconds_and_preserves_default_label_order(tmp_path):
    path = tmp_path / "dataset.csv"
    rows = []
    index = 0
    for label in DEFAULT_LABELS:
        for session in range(3):
            for offset in (0.0, 0.26):
                rows.append(
                    _row(label, f"s{session}", index, session * 10.0 + offset)
                )
                index += 1
    _write(path, rows)

    stats = dataset_stats(path)

    assert stats["exists"] is True
    assert stats["path"] == str(path)
    assert stats["total_rows"] == stats["usable_rows"] == 24
    assert stats["dropped"] == {}
    assert list(stats["labels"]) == list(DEFAULT_LABELS)
    assert all(
        stats["labels"][label] == {"rows": 6, "sessions": 3, "seconds": 0.8}
        for label in DEFAULT_LABELS
    )
    assert len(stats["sessions"]) == 12
    assert stats["sessions"][0] == {
        "session_id": "s0",
        "label": "arm_raised",
        "rows": 2,
        "seconds": 0.3,
    }
    assert stats["ready"] is True
    assert stats["problems"] == []


def test_v2_stats_counts_wave_while_v1_stats_keeps_it_unknown(tmp_path):
    path = tmp_path / "dataset_v2.csv"
    rows = []
    index = 0
    for label in V2_LABELS:
        for session in range(2):
            for offset in (0.0, 0.5):
                rows.append(
                    _row(label, f"{label}-s{session}", index, session * 10 + offset)
                )
                index += 1
    _write(path, rows)

    v2_stats = dataset_stats(path, allowed_labels=V2_LABELS)
    v1_stats = dataset_stats(path)

    assert v2_stats["labels"]["wave"] == {"rows": 4, "sessions": 2, "seconds": 1.0}
    assert v2_stats["ready"] is True
    assert list(v2_stats["labels"]) == list(V2_LABELS)
    assert "wave" not in v1_stats["labels"]
    assert v1_stats["dropped"] == {"unknown_label": 4}
    assert v1_stats["ready"] is False


def test_one_session_for_a_label_is_not_ready_and_has_a_readable_problem(tmp_path):
    path = tmp_path / "dataset.csv"
    rows = []
    index = 0
    for label in DEFAULT_LABELS:
        sessions = 1 if label == "idle" else MIN_SESSIONS_PER_LABEL
        for session in range(sessions):
            rows.append(_row(label, f"{label}-s{session}", index, float(index)))
            index += 1
    _write(path, rows)

    stats = dataset_stats(path)

    assert stats["labels"]["idle"]["sessions"] == 1
    assert stats["ready"] is False
    assert "idle: 1 sessão (mínimo 2)" in stats["problems"]


def test_drop_reasons_match_the_training_loader(tmp_path):
    path = tmp_path / "dataset.csv"
    rows = []
    index = 0
    for label in DEFAULT_LABELS:
        for session in range(2):
            for sample in range(MIN_SAMPLES_PER_CLASS // 2 + 2):
                rows.append(_row(label, f"{label}-s{session}", index, index / 30))
                index += 1

    missing_feature = _row("idle", "idle-s0", index, index / 30)
    missing_feature["left_elbow_angle"] = ""
    rows.append(missing_feature)
    index += 1

    non_finite = _row("idle", "idle-s0", index, index / 30)
    non_finite["left_elbow_angle"] = "nan"
    rows.append(non_finite)

    missing_label = _row("", "ignored-session", index + 1, (index + 1) / 30)
    rows.append(missing_label)
    _write(path, rows)

    stats = dataset_stats(path)
    loaded = load_dataset(path)

    assert stats["dropped"] == loaded.dropped == {
        "missing_label": 1,
        "missing_feature": 1,
        "non_finite": 1,
    }
    assert stats["total_rows"] == len(rows)
    assert stats["usable_rows"] == len(rows) - 3


def test_unknown_labels_are_counted_and_excluded_but_training_still_rejects_them(tmp_path):
    path = tmp_path / "dataset.csv"
    rows = [
        _row(label, f"{label}-s{session}", index, index / 30)
        for index, (label, session) in enumerate(
            (label, session)
            for label in DEFAULT_LABELS
            for session in range(MIN_SESSIONS_PER_LABEL)
        )
    ]
    unknown = _row("not-a-real-label", "must-not-appear", 100, 9.0)
    rows.append(unknown)
    _write(path, rows)

    stats = dataset_stats(path)

    assert stats["total_rows"] == len(rows)
    assert stats["usable_rows"] == len(rows) - 1
    assert stats["dropped"] == {"unknown_label": 1}
    assert "must-not-appear" not in {session["session_id"] for session in stats["sessions"]}
    assert stats["ready"] is False
    assert "1 linhas com rótulo desconhecido (treino vai recusar)" in stats["problems"]

    # Coverage helps diagnose the row, but the training contract stays strict.
    with pytest.raises(DatasetError, match="not one of the current trainable classes"):
        load_dataset(path)

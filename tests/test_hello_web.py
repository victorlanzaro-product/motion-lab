"""CLI configuration tests for selecting the live model/data schema."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from backend.dataset.writer import DEFAULT_DATASET_PATH, DEFAULT_V2_DATASET_PATH
from backend.ml.train import DEFAULT_MODEL_PATH, DEFAULT_V2_MODEL_PATH
from scripts import hello_web


@pytest.mark.parametrize(
    ("schema", "model_path", "dataset_path"),
    [
        (1, DEFAULT_MODEL_PATH, DEFAULT_DATASET_PATH),
        (2, DEFAULT_V2_MODEL_PATH, DEFAULT_V2_DATASET_PATH),
    ],
)
def test_hello_web_cli_resolves_schema_specific_default_paths(
    monkeypatch, schema, model_path, dataset_path
):
    captured = {}
    monkeypatch.setattr(sys, "argv", ["hello_web.py", "--model-schema", str(schema)])
    monkeypatch.setattr(hello_web, "ensure_model", lambda _path: None)
    monkeypatch.setattr(
        hello_web.uvicorn,
        "run",
        lambda app, **_kwargs: captured.setdefault("config", app.state.runner.config),
    )

    assert hello_web.main() == 0
    config = captured["config"]
    assert config.model_schema == schema
    assert config.model_path == model_path
    assert config.dataset_path == dataset_path


def test_hello_web_cli_allows_model_and_dataset_overrides(monkeypatch, tmp_path):
    model_path = tmp_path / "custom.joblib"
    dataset_path = tmp_path / "custom.csv"
    captured = {}
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hello_web.py",
            "--model-schema",
            "2",
            "--model",
            str(model_path),
            "--dataset",
            str(dataset_path),
        ],
    )
    monkeypatch.setattr(hello_web, "ensure_model", lambda _path: None)
    monkeypatch.setattr(
        hello_web.uvicorn,
        "run",
        lambda app, **_kwargs: captured.setdefault("config", app.state.runner.config),
    )

    assert hello_web.main() == 0
    config = captured["config"]
    assert config.model_path == Path(model_path)
    assert config.dataset_path == Path(dataset_path)

"""Motion Lab Report tests — pure functions, synthetic data, no camera."""

from __future__ import annotations

import json

from backend.ml import compute_agreement, render_report, rule_label

TRAINING = {
    "trained_at": "2026-01-01T00:00:00",
    "n_samples": 100,
    "counts": {"idle": 50, "arms_open": 50},
    "accuracy": 0.97,
    "feature_importances": {"wrist_distance": 0.6, "left_wrist_height": 0.4},
    "class_labels": ["arms_open", "idle"],
    "confusion_matrix": [[24, 1], [0, 25]],
    "per_class": {
        "arms_open": {"precision": 1.0, "recall": 0.96, "f1-score": 0.98, "support": 25},
        "idle": {"precision": 0.96, "recall": 1.0, "f1-score": 0.98, "support": 25},
    },
}


# --- rule_label / compute_agreement --------------------------------------


def test_rule_label_reads_the_three_static_predicates():
    quiet = {
        "wrists_crossed": False,
        "wrist_distance": 1.0,
        "left_wrist_above_shoulder": False,
        "right_wrist_above_shoulder": False,
    }
    assert (
        rule_label({**quiet, "wrists_crossed": True}, open_threshold=2.4)
        == "arms_crossed"
    )
    assert (
        rule_label({**quiet, "wrist_distance": 3.0}, open_threshold=2.4)
        == "arms_open"
    )
    assert (
        rule_label({**quiet, "left_wrist_above_shoulder": True}, open_threshold=2.4)
        == "arm_raised"
    )
    assert rule_label(quiet, open_threshold=2.4) == "idle"
    assert rule_label({}, open_threshold=2.4) is None


def test_rule_label_treats_a_missing_wrist_distance_as_not_open():
    # Without the other predicate inputs this is not evidence for "idle".
    assert rule_label({"wrist_distance": None}, open_threshold=2.4) is None


def test_rule_label_only_returns_idle_when_every_predicate_is_observable():
    features = {
        "wrists_crossed": False,
        "wrist_distance": 1.0,
        "left_wrist_above_shoulder": False,
        "right_wrist_above_shoulder": False,
    }
    assert rule_label(features, open_threshold=2.4) == "idle"


def test_rule_label_returns_unknown_when_a_required_predicate_feature_is_missing():
    features = {
        "wrists_crossed": False,
        "wrist_distance": 1.0,
        "left_wrist_above_shoulder": False,
        # The missing right wrist could have been raised, so "idle" is not
        # supported by this frame.
    }
    assert rule_label(features, open_threshold=2.4) is None


def test_compute_agreement_counts_matches_and_mismatches():
    frames = [
        {
            "features": {
                "wrists_crossed": True,
                "wrist_distance": 1.0,
                "left_wrist_above_shoulder": False,
                "right_wrist_above_shoulder": False,
            },
            "ml": {"label": "arms_crossed"},
        },  # match
        {
            "features": {
                "wrists_crossed": False,
                "wrist_distance": 3.0,
                "left_wrist_above_shoulder": False,
                "right_wrist_above_shoulder": False,
            },
            "ml": {"label": "idle"},
        },  # mismatch
        {"features": {}, "ml": None},  # no signal, not counted
    ]
    result = compute_agreement(frames, open_threshold=2.4)
    assert result == {"compared": 2, "matches": 1, "excluded_wave": 0, "rate": 0.5}


def test_compute_agreement_excludes_wave_instead_of_counting_it_as_a_mismatch():
    """Wave has no static rule equivalent — it must not drag the rate down."""
    frames = [{"features": {}, "ml": {"label": "wave"}}]
    result = compute_agreement(frames, open_threshold=2.4)
    assert result == {"compared": 0, "matches": 0, "excluded_wave": 1, "rate": None}


def test_compute_agreement_denominator_contains_only_frames_with_both_labels():
    quiet = {
        "wrists_crossed": False,
        "wrist_distance": 1.0,
        "left_wrist_above_shoulder": False,
        "right_wrist_above_shoulder": False,
    }
    frames = [
        {"features": {**quiet, "wrists_crossed": True}, "ml": {"label": "arms_crossed"}},
        {"features": {**quiet, "wrist_distance": 3.0}, "ml": {"label": "idle"}},
        {"features": quiet, "ml": None},
        {"features": {}, "ml": {"label": "wave"}},
        {
            "features": {**quiet, "right_wrist_above_shoulder": None},
            "ml": {"label": "idle"},
        },
    ]

    assert compute_agreement(frames, open_threshold=2.4) == {
        "compared": 2,
        "matches": 1,
        "excluded_wave": 1,
        "rate": 0.5,
    }


def test_compute_agreement_keeps_zero_rate_distinct_from_no_comparisons():
    quiet = {
        "wrists_crossed": False,
        "wrist_distance": 1.0,
        "left_wrist_above_shoulder": False,
        "right_wrist_above_shoulder": False,
    }
    zero_matches = compute_agreement(
        [
            {"features": quiet, "ml": {"label": "arms_open"}},
            {"features": quiet, "ml": {"label": "arm_raised"}},
        ],
        open_threshold=2.4,
    )
    no_comparisons = compute_agreement(
        [
            {"features": quiet, "ml": None},
            {"features": quiet, "ml": {"label": "wave"}},
        ],
        open_threshold=2.4,
    )

    assert zero_matches["compared"] == 2
    assert zero_matches["matches"] == 0
    assert zero_matches["rate"] == 0.0
    assert no_comparisons["compared"] == 0
    assert no_comparisons["rate"] is None
    assert "rate" in no_comparisons
    decoded = json.loads(json.dumps({"zero": zero_matches["rate"], "none": no_comparisons["rate"]}))
    assert decoded == {"zero": 0.0, "none": None}


def test_rule_label_distinguishes_observed_zero_from_missing_feature():
    observed = {
        "wrists_crossed": False,
        "wrist_distance": 0.0,
        "left_wrist_above_shoulder": False,
        "right_wrist_above_shoulder": False,
    }
    missing = {**observed, "wrist_distance": None}
    absent = {key: value for key, value in observed.items() if key != "right_wrist_above_shoulder"}

    assert rule_label(observed, open_threshold=2.4) == "idle"
    assert rule_label(missing, open_threshold=2.4) is None
    assert rule_label(absent, open_threshold=2.4) is None


# --- render_report -----------------------------------------------------


def test_report_renders_without_a_live_session():
    html = render_report(TRAINING, live=None)
    assert "MOTION LAB" in html
    assert "wrist_distance" in html  # feature importance bar
    assert "arms_open" in html and "idle" in html  # confusion matrix labels
    assert "97.0%" in html  # accuracy
    assert "nenhuma sessão ao vivo" in html


def test_report_renders_the_live_section_when_provided():
    live = {
        "frames": 240,
        "duration_s": 8.0,
        "fps_mean": 29.5,
        "inference_ms_mean": 14.2,
        "inference_ms_p95": 22.0,
        "detection_rate": 0.95,
        "ml_rate": 0.90,
        "agreement": {"compared": 200, "matches": 180, "excluded_wave": 10, "rate": 0.9},
    }
    html = render_report(TRAINING, live=live)
    assert "240 frames" in html
    assert "90%" in html  # agreement rate, rendered as a whole percent
    assert "nenhuma sessão ao vivo" not in html


def test_report_handles_a_zero_compared_agreement_without_crashing():
    live = {
        "frames": 10,
        "duration_s": 1.0,
        "fps_mean": 30.0,
        "inference_ms_mean": 10.0,
        "inference_ms_p95": 12.0,
        "detection_rate": 1.0,
        "ml_rate": 0.0,
        "agreement": {"compared": 0, "matches": 0, "excluded_wave": 0, "rate": None},
    }
    html = render_report(TRAINING, live=live)
    assert "sem sinal de ML suficiente" in html


def test_report_renders_dataset_coverage_and_problems():
    dataset = {
        "path": "data/training/dataset.csv",
        "exists": True,
        "total_rows": 12,
        "usable_rows": 10,
        "dropped": {"unknown_label": 2},
        "labels": {
            "arms_open": {"rows": 10, "sessions": 2, "seconds": 4.2},
            "idle": {"rows": 0, "sessions": 0, "seconds": 0.0},
        },
        "sessions": [
            {"session_id": "s<&1", "label": "arms_open", "rows": 10, "seconds": 4.2}
        ],
        "min_sessions_per_label": 2,
        "ready": False,
        "problems": ["idle: 0 sessões (mínimo 2)", "2 linhas com rótulo desconhecido (treino vai recusar)"],
    }
    html = render_report(TRAINING, dataset=dataset)

    assert "<h2>Dataset</h2>" in html
    assert "arms_open" in html and "10</td><td>2</td><td>4.2s" in html
    assert "idle: 0 sessões (mínimo 2)" in html
    assert "2 linhas com rótulo desconhecido (treino vai recusar)" in html
    assert "s&lt;&amp;1" in html
    assert "s<&1" not in html


def test_report_handles_an_absent_dataset_without_fabricating_coverage():
    dataset = {
        "path": "missing.csv",
        "exists": False,
        "total_rows": 0,
        "usable_rows": 0,
        "dropped": {},
        "labels": {},
        "sessions": [],
        "ready": False,
        "problems": ["dataset ausente"],
    }
    html = render_report(TRAINING, dataset=dataset)

    assert "Dataset ausente" in html
    assert "missing.csv" in html
    assert "Nenhuma sessão" not in html


def test_report_renders_cv_summary_and_invalid_folds():
    training = {
        **TRAINING,
        "cv": {
            "n_splits": 3,
            "folds_used": 2,
            "folds_skipped": 1,
            "macro_f1_mean": 0.81,
            "macro_f1_std": 0.04,
            "per_fold": [
                {"fold": 1, "valid": True, "macro_f1": 0.85},
                {
                    "fold": 2,
                    "valid": False,
                    "macro_f1": None,
                    "reason": "faltou arms_open no treino",
                },
                {"fold": 3, "valid": True, "macro_f1": 0.77},
            ],
        },
    }
    html = render_report(training)

    assert "<h2>Validação cruzada</h2>" in html
    assert "0.810 ± 0.040" in html
    assert "1</td><td>0.850</td><td>incluída na média" in html
    assert "2</td><td>—</td><td>ignorada" in html
    assert "faltou arms_open no treino" in html
    assert "2 dobras incluídas e" in html and "1 ignoradas" in html


def test_report_compares_v1_and_v2_grouped_cv_side_by_side():
    comparison = {
        "v1": {
            "schema_version": 1,
            "n_samples": 80,
            "classes": ["arm_raised", "idle"],
            "cv": {
                "n_splits": 4,
                "folds_used": 4,
                "folds_skipped": 0,
                "macro_f1_mean": 0.52,
                "macro_f1_std": 0.08,
            },
        },
        "v2": {
            "schema_version": 2,
            "n_samples": 60,
            "classes": ["arm_raised", "wave"],
            "window_seconds": 1.0,
            "max_gap": 0.30,
            "min_samples": 4,
            "deadband": 0.35,
            "cv": {
                "n_splits": 4,
                "folds_used": 3,
                "folds_skipped": 1,
                "macro_f1_mean": 0.91,
                "macro_f1_std": 0.03,
            },
        },
    }

    html = render_report(TRAINING, schema_comparison=comparison)

    assert "Comparação S14 · v1 × v2" in html
    assert "0.520 ± 0.080" in html
    assert "0.910 ± 0.030" in html
    assert "wave" in html and "1s · deadband 0.35" in html
    assert "Decisões S14 prontas para o README" in html
    assert "Inferência ao vivo v2 implementada" in html
    assert "hello.model_schema" in html


def test_report_marks_the_missing_schema_in_a_partial_comparison():
    html = render_report(
        TRAINING,
        schema_comparison={
            "v1": None,
            "v2": {
                "schema_version": 2,
                "n_samples": 24,
                "classes": ["arm_raised", "wave"],
                "window_seconds": 1.0,
                "deadband": 0.35,
                "cv": {"macro_f1_mean": 0.8, "macro_f1_std": 0.1},
            },
        },
    )

    assert "v1 · por frame</td><td>indisponível" in html
    assert "v2 · janela</td><td>24" in html


def test_report_distinguishes_pooled_loso_from_fold_mean():
    training = {
        **TRAINING,
        "cv": {
            "mode": "loso",
            "macro_f1_oof": 0.9,
            "macro_f1_mean": 0.24,
            "macro_f1_std": 0.01,
            "n_splits": 12,
            "folds_used": 12,
            "folds_skipped": 0,
        },
    }

    html = render_report(training)

    assert "macro-F1 LOSO: <strong>0.900</strong>" in html
    assert "uma sessão por dobra" in html
    assert "macro-F1 médio: <strong>0.240 ± 0.010</strong>" in html
    assert "12 dobras incluídas" in html

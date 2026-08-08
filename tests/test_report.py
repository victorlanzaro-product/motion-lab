"""Motion Lab Report tests — pure functions, synthetic data, no camera."""

from __future__ import annotations

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
    assert rule_label({"wrists_crossed": True}, open_threshold=2.4) == "arms_crossed"
    assert rule_label({"wrist_distance": 3.0}, open_threshold=2.4) == "arms_open"
    assert rule_label({"left_wrist_above_shoulder": True}, open_threshold=2.4) == "arm_raised"
    assert rule_label({}, open_threshold=2.4) == "idle"


def test_rule_label_treats_a_missing_wrist_distance_as_not_open():
    assert rule_label({"wrist_distance": None}, open_threshold=2.4) == "idle"


def test_compute_agreement_counts_matches_and_mismatches():
    frames = [
        {"features": {"wrists_crossed": True}, "ml": {"label": "arms_crossed"}},  # match
        {"features": {"wrist_distance": 3.0}, "ml": {"label": "idle"}},  # mismatch
        {"features": {}, "ml": None},  # no signal, not counted
    ]
    result = compute_agreement(frames, open_threshold=2.4)
    assert result == {"compared": 2, "matches": 1, "excluded_wave": 0, "rate": 0.5}


def test_compute_agreement_excludes_wave_instead_of_counting_it_as_a_mismatch():
    """Wave has no static rule equivalent — it must not drag the rate down."""
    frames = [{"features": {}, "ml": {"label": "wave"}}]
    result = compute_agreement(frames, open_threshold=2.4)
    assert result == {"compared": 0, "matches": 0, "excluded_wave": 1, "rate": None}


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

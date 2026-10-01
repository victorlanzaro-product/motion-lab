"""Motion Lab Report — the model's numbers, in one file you can just open.

Two pieces, independently useful:

1. **Training-time quality + explainability** — confusion matrix, per-class
   precision/recall, feature importance. All of it is already sitting in the
   model bundle `train()` saves (Sprint 08); this file only renders it. No new
   explanation method is invented here on purpose: feature importance is the
   one explanation a Random Forest gives you for free and actually means what
   it says, unlike a post-hoc method bolted on afterwards.
2. **Live session health + rule-vs-ML agreement** — optional, because it needs
   an actual camera session to measure (`scripts/report.py` gathers it, this
   file just lays it out). `compute_agreement` reconstructs the rule engine's
   verdict from the same three static booleans `GestureEngine` already checks
   (`arm_raised`, `arms_crossed`, `arms_open` — see `backend/gestures/rules.py`)
   directly off the wire payload, so the comparison uses the identical
   thresholds the live app itself uses, not a re-guessed copy. A `ml.label`
   of `"wave"` is excluded from the comparison rather than counted as a
   disagreement — not because the current ML model recognizes wave (it
   cannot: "wave" is out of the trainable vocabulary, see
   `backend/dataset/writer.py:REMOVED_LABELS`), but so this report does not
   crash or silently misreport agreement against a `gesture_classifier.joblib`
   trained before that decision, which may still emit the class it was
   trained on.
"""

from __future__ import annotations

import time
from typing import Any, Optional


def rule_label(features: dict[str, Any], open_threshold: float) -> str:
    """The same three static predicates `rules.py` exposes, read off one
    frame's wire `features` dict instead of a `FrameFeatures` object."""
    if features.get("wrists_crossed"):
        return "arms_crossed"
    wrist_distance = features.get("wrist_distance")
    if wrist_distance is not None and wrist_distance > open_threshold:
        return "arms_open"
    if features.get("left_wrist_above_shoulder") or features.get("right_wrist_above_shoulder"):
        return "arm_raised"
    return "idle"


def compute_agreement(frames: list[dict[str, Any]], open_threshold: float) -> dict[str, Any]:
    """How often the rule-based label and the live ML label agree.

    `frames` is the raw sequence of `type: "frame"` WebSocket messages
    (`backend/web/payload.py:frame_message`) pulled from a live session.
    """
    compared = 0
    matches = 0
    excluded_wave = 0
    for frame in frames:
        ml = frame.get("ml")
        if not ml:
            continue
        if ml["label"] == "wave":
            # Not something the current model can predict (out of the
            # trainable vocabulary) -- this only fires against a bundle
            # trained before that decision. No static rule equivalent to
            # compare it against either way.
            excluded_wave += 1
            continue
        compared += 1
        if rule_label(frame["features"], open_threshold) == ml["label"]:
            matches += 1

    return {
        "compared": compared,
        "matches": matches,
        "excluded_wave": excluded_wave,
        "rate": (matches / compared) if compared else None,
    }


def _bar(label: str, value: float, max_value: float) -> str:
    width = 0 if max_value <= 0 else round(100 * value / max_value)
    return (
        '<div class="bar-row">'
        f'<span class="bar-label">{label}</span>'
        f'<div class="bar-track"><div class="bar-fill" style="width:{width}%"></div></div>'
        f'<span class="bar-value">{value:.3f}</span>'
        "</div>"
    )


def _confusion_table(class_labels: list[str], matrix: list[list[int]]) -> str:
    header = "<th></th>" + "".join(f"<th>{label}</th>" for label in class_labels)
    rows = []
    for true_label, row in zip(class_labels, matrix):
        total = sum(row) or 1
        cells = []
        for col_label, count in zip(class_labels, row):
            hit = "hit" if col_label == true_label else ""
            intensity = min(1.0, count / total)
            cells.append(f'<td class="cm-cell {hit}" style="--i:{intensity:.2f}">{count}</td>')
        rows.append(f"<tr><th>{true_label}</th>{''.join(cells)}</tr>")
    return (
        f"<table class='confusion'><thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _per_class_table(class_labels: list[str], per_class: dict[str, dict[str, float]]) -> str:
    rows = []
    for label in class_labels:
        metrics = per_class.get(label, {})
        rows.append(
            "<tr><td>{label}</td><td>{p:.2f}</td><td>{r:.2f}</td>"
            "<td>{f1:.2f}</td><td>{n:.0f}</td></tr>".format(
                label=label,
                p=metrics.get("precision", 0.0),
                r=metrics.get("recall", 0.0),
                f1=metrics.get("f1-score", 0.0),
                n=metrics.get("support", 0.0),
            )
        )
    return (
        "<table class='per-class'><thead><tr>"
        "<th>classe</th><th>precision</th><th>recall</th><th>f1</th><th>amostras</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


def _live_section(live: Optional[dict[str, Any]]) -> str:
    if live is None:
        return (
            "<section><h2>sessão ao vivo</h2>"
            "<p class='muted'>nenhuma sessão ao vivo neste relatório — rode "
            "<code>scripts/report.py</code> com a câmera disponível pra incluir "
            "saúde do pipeline e concordância regra × ML.</p></section>"
        )

    agreement = live.get("agreement")
    if agreement and agreement["rate"] is not None:
        agreement_html = (
            f"<p>concordância regra × ML: <strong>{agreement['rate'] * 100:.0f}%</strong> "
            f"({agreement['matches']}/{agreement['compared']} frames comparados; "
            f"{agreement['excluded_wave']} excluídos por serem 'wave' -- classe fora do "
            "vocabulário treinável desde este sprint; só aparece aqui se o modelo "
            "carregado for de antes dessa decisão)</p>"
        )
    else:
        agreement_html = "<p class='muted'>sem sinal de ML suficiente pra comparar com a regra.</p>"

    return f"""
    <section>
      <h2>sessão ao vivo</h2>
      <p class="muted">{live['frames']} frames em {live['duration_s']:.1f}s</p>
      <dl class="stats">
        <dt>fps médio</dt><dd>{live['fps_mean']:.1f}</dd>
        <dt>inferência média</dt><dd>{live['inference_ms_mean']:.1f} ms</dd>
        <dt>inferência p95</dt><dd>{live['inference_ms_p95']:.1f} ms</dd>
        <dt>taxa de detecção</dt><dd>{live['detection_rate'] * 100:.0f}%</dd>
        <dt>taxa de sinal ML</dt><dd>{live['ml_rate'] * 100:.0f}%</dd>
      </dl>
      {agreement_html}
    </section>
    """


def render_report(training: dict[str, Any], live: Optional[dict[str, Any]] = None) -> str:
    """Training-time quality + explainability, plus an optional live section."""
    importances = sorted(
        training["feature_importances"].items(), key=lambda kv: kv[1], reverse=True
    )
    max_importance = importances[0][1] if importances else 1.0
    bars = "".join(_bar(name, value, max_importance) for name, value in importances)
    counts = ", ".join(f"{label}={n}" for label, n in training["counts"].items())

    return f"""<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8" />
<title>Motion Lab — relatório do modelo</title>
<style>
  :root {{
    --bg: #0b0f14; --panel: #121821; --line: #1e2733; --text: #d7e0ea;
    --dim: #7a8899; --accent: #78ff78; --alert: #ff6b6b;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 24px; background: var(--bg); color: var(--text);
    font: 14px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  }}
  h1 {{ font-size: 16px; letter-spacing: 0.08em; margin: 0 0 4px; }}
  h2 {{ font-size: 12px; letter-spacing: 0.14em; text-transform: uppercase; color: var(--accent);
        margin: 28px 0 10px; }}
  .muted {{ color: var(--dim); }}
  section {{ max-width: 760px; }}
  .bar-row {{ display: grid; grid-template-columns: 210px 1fr 60px; gap: 10px; align-items: center;
              margin: 4px 0; }}
  .bar-label {{ color: var(--dim); text-align: right; white-space: nowrap; overflow: hidden;
                text-overflow: ellipsis; }}
  .bar-track {{ background: var(--panel); border: 1px solid var(--line); border-radius: 3px; height: 14px; }}
  .bar-fill {{ background: var(--accent); height: 100%; border-radius: 2px; }}
  .bar-value {{ font-variant-numeric: tabular-nums; }}
  table {{ border-collapse: collapse; margin: 8px 0 16px; }}
  th, td {{ padding: 4px 10px; border: 1px solid var(--line); text-align: right; }}
  th {{ color: var(--dim); font-weight: 400; }}
  td:first-child, th:first-child {{ text-align: left; }}
  .confusion td.cm-cell {{ background: rgba(120, 255, 120, calc(var(--i) * 0.5)); }}
  .confusion td.hit {{ font-weight: 700; border-color: var(--accent); }}
  dl.stats {{ display: grid; grid-template-columns: 200px auto; gap: 4px 12px; margin: 8px 0; }}
  dl.stats dt {{ color: var(--dim); }}
  code {{ color: var(--accent); }}
</style>
</head>
<body>
  <h1>MOTION LAB — relatório do modelo</h1>
  <p class="muted">treinado em {training['trained_at']} · {training['n_samples']} amostras
    ({counts}) · acurácia no teste: {training['accuracy'] * 100:.1f}%
    · macro-F1: {training.get('macro_f1', 0.0):.3f}
    · acurácia balanceada: {training.get('balanced_accuracy', 0.0) * 100:.1f}%</p>
  <p class="muted">acurácia simples pode parecer boa só por acertar a classe majoritária
    (<code>idle</code> costuma dominar uma sessão real); macro-F1 e acurácia balanceada
    pesam cada classe igual, sem deixar uma classe rara se esconder atrás da média.</p>

  <section>
    <h2>importância de feature</h2>
    <p class="muted">quanto o Random Forest realmente usa cada coluna pra decidir — não
      um método de explicação separado, é o que o próprio modelo relata.</p>
    {bars}
  </section>

  <section>
    <h2>matriz de confusão</h2>
    <p class="muted">linha = classe real, coluna = classe prevista, no split de teste separado.</p>
    {_confusion_table(training['class_labels'], training['confusion_matrix'])}
  </section>

  <section>
    <h2>precision / recall por classe</h2>
    {_per_class_table(training['class_labels'], training['per_class'])}
  </section>

  {_live_section(live)}

  <p class="muted">gerado em {time.strftime('%Y-%m-%dT%H:%M:%S')} · nenhuma imagem foi usada
    pra gerar este relatório, só features e números.</p>
</body>
</html>
"""

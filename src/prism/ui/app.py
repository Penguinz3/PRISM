import html as html_lib

import gradio as gr

from prism import config
from prism.model import LLMWrapper
from prism.markers.mean_logprob import MeanLogProbMarker
from prism.markers.self_consistency import SelfConsistencyMarker
from prism.markers.semantic_entropy import SemanticEntropyMarker
from prism.pipeline import PRISMPipeline

# ── Constructed once at module load ──────────────────────────────────────────
PIPELINE = PRISMPipeline(
    model=LLMWrapper(config.DEFAULT_MODEL_NAME),
    markers=[
        MeanLogProbMarker(threshold=config.MARKER_THRESHOLDS["mean_logprob"]),
        SelfConsistencyMarker(threshold=config.MARKER_THRESHOLDS["self_consistency"]),
        SemanticEntropyMarker(threshold=config.MARKER_THRESHOLDS["semantic_entropy"]),
    ],
)


# ── Rendering helpers ─────────────────────────────────────────────────────────

def _badge(triggered: bool) -> str:
    if triggered:
        return (
            '<span style="background:#ef4444;color:white;padding:2px 10px;'
            'border-radius:4px;font-weight:bold;font-size:0.85em">TRIGGERED</span>'
        )
    return (
        '<span style="background:#22c55e;color:white;padding:2px 10px;'
        'border-radius:4px;font-weight:bold;font-size:0.85em">OK</span>'
    )


def _marker_card_html(result) -> str:
    direction_note = "lower is worse" if result.direction == "lower_is_worse" else "higher is worse"
    expl = (
        f'<div style="color:#6b7280;font-size:0.88em;margin-top:6px">'
        f'{html_lib.escape(result.explanation)}</div>'
        if result.explanation
        else ""
    )
    return (
        f'<div style="border:1px solid #e5e7eb;border-radius:8px;padding:12px 16px;'
        f'background:#f9fafb;margin-bottom:4px">'
        f'<div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap">'
        f'<strong style="font-size:1.05em;font-family:monospace">{result.name}</strong>'
        f'{_badge(result.triggered)}'
        f'<span>Score: <strong>{result.score:.3f}</strong></span>'
        f'<span style="color:#9ca3af">threshold {result.threshold} &mdash; {direction_note}</span>'
        f"</div>{expl}</div>"
    )


def _token_html(tokens: list[str], logprobs: list[float]) -> str:
    if not tokens:
        return "<em>No tokens to display.</em>"

    CLIP_LOW = -10.0
    clamped = [max(lp, CLIP_LOW) for lp in logprobs]
    lo, hi = min(clamped), max(clamped)

    spans = []
    for token, lp, clp in zip(tokens, logprobs, clamped):
        t = (clp - lo) / (hi - lo) if hi != lo else 1.0
        # t=0 → red, t=1 → green
        r = int(248 + (74 - 248) * t)
        g = int(113 + (222 - 113) * t)
        b = int(113 + (128 - 113) * t)
        bg = f"rgba({r},{g},{b},0.45)"
        escaped = html_lib.escape(token).replace(" ", "&nbsp;")
        tip = html_lib.escape(f"logprob: {lp:.4f}")
        spans.append(
            f'<span style="background:{bg};padding:1px 3px;border-radius:3px;'
            f'font-family:monospace" title="{tip}">{escaped}</span>'
        )

    return (
        '<div style="line-height:2.2;padding:10px 12px;border:1px solid #e5e7eb;'
        'border-radius:8px;background:#ffffff">'
        + "".join(spans)
        + "</div>"
    )


def _samples_text(sample_texts: list[str]) -> str:
    return "\n\n---\n\n".join(
        f"[{i + 1}] {s}" for i, s in enumerate(sample_texts)
    )


def _cluster_html(clusters: list[list[int]], all_texts: list[str]) -> str:
    lines = []
    for k, cluster in enumerate(clusters):
        lines.append(f"<strong>Cluster {k + 1}</strong> ({len(cluster)} member(s)):")
        for idx in cluster:
            label = "primary" if idx == 0 else f"sample {idx}"
            text = html_lib.escape(all_texts[idx][:200])
            lines.append(f'&nbsp;&nbsp;<em>[{label}]</em> {text}')
    return "<br>".join(lines)


# ── Handler ───────────────────────────────────────────────────────────────────

def run_analysis(prompt: str, num_samples: int):
    if not prompt.strip():
        empty = ""
        return empty, empty, empty, empty, empty, empty, empty

    result = PIPELINE.analyze(prompt, num_samples=int(num_samples))
    by_name = {r.name: r for r in result.marker_results}

    # Primary response
    primary_text = result.primary_text

    # Mean log-prob
    mlp = by_name["mean_logprob"]
    mlp_card = _marker_card_html(mlp)
    tok_html = _token_html(mlp.details["tokens"], mlp.details["per_token_logprobs"])

    # Self-consistency
    sc = by_name["self_consistency"]
    sc_card = _marker_card_html(sc)
    samples_text = _samples_text(result.sample_texts)

    # Semantic entropy
    se = by_name["semantic_entropy"]
    se_card = _marker_card_html(se)
    all_texts = [result.primary_text] + result.sample_texts
    cluster_html = _cluster_html(se.details["clusters"], all_texts)
    se_details_html = (
        f'<div style="padding:8px 0">'
        f'<strong>Clusters: {se.details["cluster_count"]}</strong><br><br>'
        f'{cluster_html}</div>'
    )

    return primary_text, mlp_card, tok_html, sc_card, samples_text, se_card, se_details_html


# ── UI layout ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="PRISM — Hallucination Markers") as demo:
    gr.Markdown(
        "# PRISM\n"
        "**Hallucination marker demo** — generates a response and scores it with "
        "three uncertainty markers."
    )

    with gr.Row():
        with gr.Column(scale=3):
            prompt_box = gr.Textbox(
                label="Prompt",
                lines=4,
                placeholder="e.g. What is the capital of France?",
            )
        with gr.Column(scale=1):
            num_samples = gr.Slider(
                minimum=3, maximum=10, value=5, step=1, label="Number of samples"
            )
            generate_btn = gr.Button("Generate & analyze", variant="primary", size="lg")

    primary_out = gr.Textbox(
        label="Primary response (temperature = 0)", lines=6, interactive=False
    )

    gr.Markdown("---")

    # ── Mean log-prob ─────────────────────────────────────────────────────────
    gr.Markdown("### Mean log-probability")
    mlp_card_out = gr.HTML()
    with gr.Accordion("Per-token confidence highlighting", open=True):
        token_html_out = gr.HTML()

    gr.Markdown("---")

    # ── Self-consistency ──────────────────────────────────────────────────────
    gr.Markdown("### Self-consistency")
    sc_card_out = gr.HTML()
    with gr.Accordion("Sampled responses", open=False):
        sc_samples_out = gr.Textbox(label="", lines=10, interactive=False)

    gr.Markdown("---")

    # ── Semantic entropy ──────────────────────────────────────────────────────
    gr.Markdown("### Semantic entropy")
    se_card_out = gr.HTML()
    with gr.Accordion("Cluster assignments", open=False):
        se_details_out = gr.HTML()

    # ── Wire up ───────────────────────────────────────────────────────────────
    generate_btn.click(
        fn=run_analysis,
        inputs=[prompt_box, num_samples],
        outputs=[
            primary_out,
            mlp_card_out,
            token_html_out,
            sc_card_out,
            sc_samples_out,
            se_card_out,
            se_details_out,
        ],
    )


if __name__ == "__main__":
    demo.launch()

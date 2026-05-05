# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Python 3.10+ required. The venv lives at `.venv/` and is managed with `uv`.

```bash
uv venv .venv
uv pip install -e ".[dev]"
# also needed at runtime (not yet in pyproject.toml):
uv pip install accelerate
```

Use `.venv/bin/python` directly, or activate with `source .venv/bin/activate`. The system `pip` is externally managed (Homebrew) and will refuse installs — always go through `uv pip` or the venv.

## Commands

```bash
# Run the UI
python -m prism.ui.app          # opens http://127.0.0.1:7860

# Run all tests (slow — model tests download/load the LLM)
.venv/bin/python -m pytest tests/

# Run a single test file
.venv/bin/python -m pytest tests/test_markers.py -v

# Run a single test by name
.venv/bin/python -m pytest tests/test_markers.py::test_identical_responses_high_consistency_low_entropy -v
```

## Architecture

The data flow is linear: **prompt → `LLMWrapper` → `MarkerInput` → markers → `AnalysisResult` → Gradio UI**.

### Key types (`src/prism/`)

- **`config.py`** — all tuneable constants (model names, generation params, marker thresholds). Change `DEFAULT_MODEL_NAME` here to swap the LLM.
- **`model.py` — `LLMWrapper`** — wraps a HuggingFace causal LM. `generate()` returns `list[GenerationOutput]`, each containing `text`, `token_ids`, and `token_logprobs` (log-prob of the chosen token at each step, captured via `output_scores=True`). `temperature=0` forces `do_sample=False` and `num_return_sequences=1`.
- **`markers/base.py`** — defines the three shared types: `MarkerInput` (prompt + primary + samples), `MarkerResult` (name, score, threshold, triggered, direction, details), and the `BaseMarker` Protocol.
- **`markers/`** — three concrete markers, each a plain class with a `compute(MarkerInput) -> MarkerResult` method:
  - `MeanLogProbMarker` — mean of `primary_response.token_logprobs`. Needs a tokenizer (injected by the pipeline; pass `tokenizer=None` in the constructor and the pipeline fills it in).
  - `SelfConsistencyMarker` — cosine similarity between primary and sample embeddings via `all-MiniLM-L6-v2`.
  - `SemanticEntropyMarker` — Shannon entropy over NLI-entailment clusters using `cross-encoder/nli-deberta-v3-small`. Entailment label index is 1.
- **`pipeline.py` — `PRISMPipeline`** — holds one `LLMWrapper` and a list of markers. `analyze()` makes exactly two `generate()` calls (primary at temp=0, N samples at temp=0.7 in a single call with `num_return_sequences=N`), then runs all markers. Also injects the model tokenizer into any `MeanLogProbMarker` that has `tokenizer=None`.
- **`ui/app.py`** — constructs `PIPELINE` once at module load (not per request). The Gradio handler calls `PIPELINE.analyze()` and renders results; per-token log-prob coloring is done in `_token_html()`.

### Adding a new marker

1. Create `src/prism/markers/your_marker.py` with a class exposing `name: str` and `compute(MarkerInput) -> MarkerResult`.
2. Set `direction` to `"lower_is_worse"` or `"higher_is_worse"` — the UI uses this for badge logic.
3. Add it to the `markers=[...]` list in `ui/app.py`.
4. Add a threshold to `MARKER_THRESHOLDS` in `config.py`.

### Test structure

- `test_model.py` — loads the real LLM, generates one response, asserts logprob/token_id alignment. Slow (~2 min first run).
- `test_markers.py` — uses fake `GenerationOutput` objects (no LLM needed) but loads the two sub-models. Tests identical vs. varied responses. Uses `scope="module"` fixtures to load sub-models once per session.
- `test_pipeline.py` — end-to-end with the real LLM and all three markers. Slowest test.

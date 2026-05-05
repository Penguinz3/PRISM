# PRISM MVP — Build Plan for an Agentic Coding Copilot

This is a build spec for an AI coding agent (Claude Code, Cursor agent, Aider, etc.). It describes a minimum viable PRISM: load a small HF model, take a prompt, generate a response, and show how each hallucination marker scores it in a web UI.

The plan is broken into phases. Each phase ends with a runnable checkpoint — finish the phase, run the checkpoint command, confirm output, then move on. Do not skip ahead; later phases assume earlier ones work.

## Stack and constraints

- Python 3.10+
- `transformers` and `torch` for the model
- `gradio` for the UI (chosen over Streamlit because it ships with text-input and JSON-display components that match this use case exactly, and it has zero boilerplate)
- Default model: `Qwen/Qwen2.5-1.5B-Instruct` (ungated, ~3GB, runs on CPU acceptably and on any GPU comfortably). The user can swap in `google/gemma-2-2b-it` via a config flag, but Gemma is gated and requires an HF auth token, so it is not the default.
- Secondary model for NLI (used by semantic entropy): `cross-encoder/nli-deberta-v3-small` (~180MB).
- All paths relative to the repo root.

## Target file layout

```
prism/
├── pyproject.toml
├── README.md
├── src/prism/
│   ├── __init__.py
│   ├── config.py            # Model name, generation params, marker thresholds
│   ├── model.py             # HF model wrapper
│   ├── markers/
│   │   ├── __init__.py
│   │   ├── base.py          # BaseMarker, MarkerInput, MarkerResult
│   │   ├── mean_logprob.py
│   │   ├── self_consistency.py
│   │   └── semantic_entropy.py
│   ├── pipeline.py          # Orchestrates generation + all markers
│   └── ui/
│       └── app.py           # Gradio app
└── tests/
    ├── test_model.py
    ├── test_markers.py
    └── test_pipeline.py
```

## Phase 1 — Project scaffold

Create the directory structure above. Create `pyproject.toml` with dependencies: `transformers>=4.44`, `torch>=2.2`, `gradio>=4.40`, `numpy`, `sentence-transformers` (for NLI cross-encoder), `pytest` (dev). Use `setuptools` or `hatchling` as the build backend, whichever is simpler.

Create empty `__init__.py` files. Create a stub `config.py` exposing module-level constants:

```python
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
DEFAULT_MAX_NEW_TOKENS = 200
DEFAULT_TEMPERATURE = 0.7
NUM_SAMPLES_FOR_CONSISTENCY = 5
MARKER_THRESHOLDS = {
    "mean_logprob": -1.5,        # Below this = triggered (low confidence)
    "self_consistency": 0.6,     # Below this = triggered
    "semantic_entropy": 1.0,     # Above this = triggered
}
```

Thresholds will be tunable in the UI — these are starting defaults.

**Checkpoint:** `pip install -e .` succeeds; `python -c "from prism import config"` runs without error.

## Phase 2 — Model wrapper

Implement `src/prism/model.py` with a single class `LLMWrapper`. It must:

1. Load model and tokenizer in `__init__`, accepting a model name.
2. Expose `generate(prompt: str, temperature: float, max_new_tokens: int, num_return_sequences: int) -> list[GenerationOutput]`.
3. Each `GenerationOutput` is a dataclass containing: `text` (decoded response), `token_ids` (list of generated token ids), `token_logprobs` (list of float, one per generated token, the log-prob of the chosen token at that step).

Implementation notes for the agent:

- Use `model.generate(..., return_dict_in_generate=True, output_scores=True)` to capture per-step logits.
- For each step, apply `log_softmax` over the vocab dim, then gather the log-prob of the actually-chosen token. Be careful with batch indexing when `num_return_sequences > 1`.
- The chat template should be applied via `tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)`. Then tokenize separately.
- When `temperature == 0`, set `do_sample=False` and `num_return_sequences=1` regardless of the requested count, otherwise generation will error out.
- Cache the loaded model on the class instance — do not reload per call.
- Support both CPU and CUDA via `device_map="auto"`.

**Checkpoint:** Write `tests/test_model.py` with a single test that loads the model, generates one response to "What is 2+2?", and asserts that `token_logprobs` has the same length as `token_ids` and all values are non-positive floats.

## Phase 3 — Marker base interface

Implement `src/prism/markers/base.py`:

```python
from dataclasses import dataclass, field
from typing import Protocol

@dataclass
class MarkerInput:
    prompt: str
    primary_response: GenerationOutput        # temperature=0 generation
    sampled_responses: list[GenerationOutput] # N samples at temperature=0.7

@dataclass
class MarkerResult:
    name: str
    score: float
    threshold: float
    triggered: bool
    direction: str                # "lower_is_worse" or "higher_is_worse"
    details: dict = field(default_factory=dict)  # marker-specific extras for UI
    explanation: str = ""

class BaseMarker(Protocol):
    name: str
    def compute(self, input: MarkerInput) -> MarkerResult: ...
```

The `direction` field tells the UI how to interpret the score relative to the threshold. The `details` dict is for marker-specific structured output that the UI can render (e.g., per-token logprobs for highlighting, cluster assignments for semantic entropy).

**Checkpoint:** `from prism.markers.base import BaseMarker, MarkerInput, MarkerResult` works.

## Phase 4 — Implement three markers

### 4a. Mean log-probability (`mean_logprob.py`)

The score is the arithmetic mean of `primary_response.token_logprobs`. Direction is `lower_is_worse`. In `details`, return `{"per_token_logprobs": [...], "tokens": [...]}` so the UI can render token-level highlighting. Decode tokens individually with `tokenizer.decode([tid])` — this requires passing the tokenizer in. Easiest path: the marker takes the tokenizer as a constructor arg.

### 4b. Self-consistency (`self_consistency.py`)

Compute pairwise similarity between `primary_response.text` and each `sampled_responses[i].text`. Use sentence embeddings: load `sentence-transformers/all-MiniLM-L6-v2` (small, ~80MB), embed all responses, compute cosine similarity between primary and each sample, take the mean. Score = mean similarity. Direction is `lower_is_worse`. In `details`, return `{"sample_similarities": [...], "samples": [...]}`.

Cache the sentence-transformer model on the marker instance.

### 4c. Semantic entropy (`semantic_entropy.py`)

The full Kuhn et al. algorithm:

1. Take the primary response plus all sampled responses (N+1 strings total).
2. Cluster them by bidirectional NLI entailment: two responses are in the same cluster iff each entails the other (according to the NLI cross-encoder). Use a greedy clustering — for each response, check entailment against each existing cluster's representative; if both directions entail, add to that cluster; otherwise start a new cluster.
3. For each cluster, count members. Convert to probabilities by dividing by N+1.
4. Score = Shannon entropy over cluster probabilities (in nats).

Direction is `higher_is_worse`. In `details`, return `{"clusters": [[response_idx, ...], ...], "cluster_count": K}`.

The NLI cross-encoder API: pass pairs `[(premise, hypothesis), ...]` and get back logits. The model outputs three classes (contradiction, entailment, neutral) — entailment is the index to check. A pair is bidirectionally entailing iff entailment is the argmax in both directions.

**Checkpoint:** `tests/test_markers.py` should construct a fake `MarkerInput` with hand-crafted text and verify each marker produces a `MarkerResult` with sensible values. Specifically: a test where the primary response and all samples are identical should give high self-consistency and low semantic entropy; a test where they are all different should give low self-consistency and high semantic entropy.

## Phase 5 — Pipeline orchestrator

Implement `src/prism/pipeline.py` with a `PRISMPipeline` class that holds an `LLMWrapper` and a list of marker instances.

The main method is `analyze(prompt: str, num_samples: int = 5) -> AnalysisResult`. It:

1. Generates the primary response with temperature=0, num_return_sequences=1.
2. Generates N samples with temperature=0.7, num_return_sequences=N (one model call, not N).
3. Builds a `MarkerInput` and runs every marker.
4. Returns an `AnalysisResult` dataclass containing the prompt, primary response text, sample texts, and a list of `MarkerResult`.

The pipeline should be constructed once and reused across UI calls. Initialization is slow (model loading) — the UI must not reconstruct the pipeline per request.

**Checkpoint:** `tests/test_pipeline.py` instantiates the pipeline once, runs `analyze("What is the capital of France?")`, and asserts the returned `AnalysisResult` has three `MarkerResult` entries and a non-empty primary response.

## Phase 6 — Gradio UI

Implement `src/prism/ui/app.py`. The layout:

- A `gr.Textbox` for the prompt (multiline).
- A `gr.Slider` for "number of samples" (default 5, range 3–10). More samples means slower but more reliable consistency/entropy estimates.
- A "Generate & analyze" button.
- A `gr.Textbox` (large, readonly) showing the primary response.
- A panel of marker cards. For each marker, render: marker name, score (formatted to 3 decimals), threshold, a colored "TRIGGERED" or "OK" badge, and a collapsible section with the marker's `details` and `explanation`.
- For the mean-logprob marker specifically, render the response text with per-token background coloring — green for high log-prob (confident), red for low log-prob (surprising). This is the most visually rewarding part of the demo and worth the extra effort. Use HTML output via `gr.HTML`.
- For self-consistency and semantic-entropy, show the sampled responses in an expandable accordion so the user can see what the model is varying on.

Top of the file should construct the pipeline once at module load:

```python
PIPELINE = PRISMPipeline(
    model=LLMWrapper(config.DEFAULT_MODEL_NAME),
    markers=[
        MeanLogProbMarker(threshold=config.MARKER_THRESHOLDS["mean_logprob"]),
        SelfConsistencyMarker(threshold=config.MARKER_THRESHOLDS["self_consistency"]),
        SemanticEntropyMarker(threshold=config.MARKER_THRESHOLDS["semantic_entropy"]),
    ],
)
```

The Gradio handler then just calls `PIPELINE.analyze(prompt, num_samples)` and renders the result.

Add a `if __name__ == "__main__":` block that calls `demo.launch()`.

**Checkpoint:** `python -m prism.ui.app` launches the UI, the user enters a prompt, clicks the button, and sees the response plus three marker cards. Manual verification only — no automated test for the UI.

## Phase 7 — README and polish

Write a `README.md` covering: what PRISM is in two sentences, install steps (`pip install -e .`), how to run the UI, how to swap the model, what each marker means in plain English, and a "known limitations" section (only three markers, MVP-scope, etc.).

Add a `examples.md` with three suggested prompts that should produce visibly different marker behavior:
- A clearly factual prompt the model knows ("What is the capital of France?") — markers should mostly say OK.
- A long-tail factual prompt ("Who composed the score for the 1973 film *The Mackintosh Man*?") — sampling consistency and semantic entropy should fire because the model will guess differently each time.
- An ambiguous prompt ("What's the best programming language?") — high semantic entropy because the model legitimately samples different opinions, but each response will be internally confident; this is a good demo of why one marker isn't enough.

## Out of scope for MVP — explicitly do not implement

- SAPLMA-style hidden-state probes. These need labeled training data and a separate training pipeline. Phase 2 of the project.
- The PRISM contradiction probe. Same reason.
- Retrieval augmentation. The MVP runs closed-book.
- Benchmark evaluation, AUROC computation, prompt taxonomy. Those belong to the research-paper pipeline, not this tool.
- Multiple-model comparison. One model at a time.

## Notes for the agent

- Do not invent dependencies. If a package is not in the pyproject.toml, do not import it. Add it explicitly first.
- After each phase, run the relevant tests and report results before moving on. If a checkpoint fails, fix before proceeding.
- Prefer simple, readable implementations over clever ones. This codebase will be extended; legibility matters more than micro-optimization.
- The first end-to-end run will be slow (model download). Cache locations matter — respect `HF_HOME` if set.
- Do not commit model files. Add a sensible `.gitignore` (Python defaults plus `*.pt`, `*.bin`, `models/`, `.cache/`).

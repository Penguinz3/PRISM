# PRISM

PRISM is a local web UI that runs a small language model, generates a response to your prompt, and scores it with three hallucination markers — giving you a live read on how confident and consistent the model is before you trust the output.

## Install

Python 3.10+ required. [uv](https://github.com/astral-sh/uv) is recommended; plain pip works too.

```bash
git clone <repo-url>
cd prism

# with uv (recommended)
uv venv .venv
uv pip install -e ".[dev]"

# or with pip inside a venv
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The first run downloads model weights (~3 GB for the default model, ~260 MB for the two marker sub-models). They are cached in the Hugging Face cache directory (`~/.cache/huggingface` by default, or wherever `HF_HOME` points).

## Run the UI

```bash
source .venv/bin/activate      # if not already active
python -m prism.ui.app
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860) in your browser.

Enter a prompt, adjust the number of samples if you like (more = slower but more reliable consistency and entropy estimates), and click **Generate & analyze**.

## Swap the model

The default model is `Qwen/Qwen2.5-1.5B-Instruct` (ungated, ~3 GB, runs on CPU and any GPU). To use a different model, edit `src/prism/config.py`:

```python
DEFAULT_MODEL_NAME = "google/gemma-2-2b-it"   # example
```

> **Note:** Gemma and some other models are gated on Hugging Face. You will need to accept the model license on hf.co and set `HF_TOKEN` in your environment before running:
> ```bash
> export HF_TOKEN=hf_...
> python -m prism.ui.app
> ```

## Run tests

```bash
pytest tests/
```

The model tests load the full LLM and take a few minutes. The marker tests load only the two small sub-models and run in under a minute.

## What each marker means

### Mean log-probability
At each generation step the model produces a probability distribution over its vocabulary and picks the most likely token (at temperature 0) or samples (at temperature > 0). The log-probability of the chosen token measures how peaked that distribution was — how certain the model was about that word.

**Mean log-prob** averages these values across the whole response. A score near 0 means the model was confident throughout. A very negative score means it was frequently uncertain — often a sign that it is generating text outside its training distribution.

The UI shows per-token background coloring: **green** tokens were high-confidence, **red** tokens were surprising to the model. Hover over a token to see its exact log-prob.

*Triggered when score is below −1.5 (default threshold).*

### Self-consistency
The model generates N additional responses at temperature 0.7 (sampling mode). PRISM embeds all of them plus the primary response using a small sentence embedding model (`all-MiniLM-L6-v2`) and computes the average cosine similarity between the primary response and each sample.

A high score means the model returns roughly the same answer every time — a strong signal of factual confidence. A low score means the model is unsure and exploring different answers each time.

*Triggered when score is below 0.6 (default threshold).*

### Semantic entropy
This marker implements the Kuhn et al. (2023) algorithm. It takes all N+1 responses and clusters them by meaning: two responses land in the same cluster only if each one logically entails the other, as judged by a cross-encoder NLI model (`nli-deberta-v3-small`).

Shannon entropy is then computed over the cluster size distribution. If all responses say the same thing (one cluster), entropy is 0. If every response makes a different claim (N+1 singleton clusters), entropy is maximised.

Semantic entropy catches cases where self-consistency would miss — for example, when the model is paraphrasing the same wrong fact across samples. Because it works at the meaning level, not the word level, it is harder to fool.

*Triggered when score is above 1.0 nat (default threshold).*

## Known limitations

- **Three markers only.** This is an MVP. SAPLMA-style hidden-state probes and the PRISM contradiction probe are not implemented; both require labeled training data and a separate training pipeline.
- **No retrieval.** The model runs closed-book. There is no mechanism to ground responses in retrieved documents.
- **Thresholds are heuristic.** The default trigger thresholds were chosen as reasonable starting points, not calibrated on a benchmark. Use the UI sliders to tune them for your use case.
- **One model at a time.** Multi-model comparison is out of scope for this version.
- **No AUROC or benchmark evaluation.** Measuring marker quality against a labeled hallucination dataset is a research-pipeline task, not implemented here.
- **CPU is slow.** On a machine without a GPU, each full analysis (5 samples, 200 tokens each) takes several minutes.

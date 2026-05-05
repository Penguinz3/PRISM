# Suggested prompts

Three prompts chosen to show visibly different marker behavior, useful for demoing or calibrating thresholds.

---

## 1. Clearly factual — markers mostly OK

**Prompt:**
> What is the capital of France?

**Expected behavior:**

| Marker | Expected | Why |
|--------|----------|-----|
| Mean log-prob | High (close to 0), not triggered | The model has seen "Paris is the capital of France" many times; token probabilities will be peaked. |
| Self-consistency | High (> 0.9), not triggered | Every sample will say "Paris." The embedding similarity will be very close to 1. |
| Semantic entropy | Low (near 0), not triggered | All responses entail each other — one cluster. Entropy = 0. |

This is the baseline "nothing fires" case. Good for confirming the pipeline is working.

---

## 2. Long-tail factual — consistency and entropy fire

**Prompt:**
> Who composed the score for the 1973 film *The Mackintosh Man*?

**Expected behavior:**

| Marker | Expected | Why |
|--------|----------|-----|
| Mean log-prob | Moderate to low, may trigger | The model may know this film marginally; token confidence will be uneven. |
| Self-consistency | Low (< 0.6), triggered | The model will guess different composers across samples — Maurice Jarre, John Barry, others — because it is uncertain. |
| Semantic entropy | High (> 1.0), triggered | Different composer names produce semantically distinct claims that don't entail each other. Multiple clusters. |

This is the canonical hallucination-prone case: the model produces a confident-sounding answer but the sampling distribution reveals it is guessing. Semantic entropy and self-consistency catch it even when mean log-prob looks acceptable.

---

## 3. Genuinely ambiguous — entropy fires, consistency may not

**Prompt:**
> What's the best programming language?

**Expected behavior:**

| Marker | Expected | Why |
|--------|----------|-----|
| Mean log-prob | High, not triggered | The model generates fluent opinionated text confidently; token probabilities are high regardless of which language it picks. |
| Self-consistency | Moderate to low, may trigger | The model may sample Python in some runs, Rust or JavaScript in others. Embedding similarity depends on how different the chosen languages are. |
| Semantic entropy | High, triggered | "Python is the best" and "Rust is the best" do not entail each other. Each sampled opinion forms its own cluster. |

This case demonstrates why mean log-prob alone is insufficient: the model is internally confident while legitimately exploring different positions. High semantic entropy with low mean log-prob signals opinion or ambiguity rather than factual uncertainty — a qualitatively different failure mode from prompt 2.

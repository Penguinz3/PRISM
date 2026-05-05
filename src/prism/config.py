DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
DEFAULT_MAX_NEW_TOKENS = 200
DEFAULT_TEMPERATURE = 0.7
NUM_SAMPLES_FOR_CONSISTENCY = 5
MARKER_THRESHOLDS = {
    "mean_logprob": -1.5,
    "self_consistency": 0.6,
    "semantic_entropy": 1.0,
}

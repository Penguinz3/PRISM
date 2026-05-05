from prism.model import LLMWrapper
from prism import config


def test_generate_logprobs():
    wrapper = LLMWrapper(config.DEFAULT_MODEL_NAME)
    outputs = wrapper.generate("What is 2+2?", temperature=0, max_new_tokens=50, num_return_sequences=1)

    assert len(outputs) == 1
    out = outputs[0]

    assert len(out.token_logprobs) == len(out.token_ids), (
        f"token_logprobs length {len(out.token_logprobs)} != "
        f"token_ids length {len(out.token_ids)}"
    )
    assert all(lp <= 0.0 for lp in out.token_logprobs), (
        f"Found positive log-prob: {[lp for lp in out.token_logprobs if lp > 0]}"
    )
    assert len(out.token_ids) > 0, "Expected non-empty generation"
    assert isinstance(out.text, str) and len(out.text) > 0

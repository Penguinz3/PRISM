from dataclasses import dataclass
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class GenerationOutput:
    text: str
    token_ids: list[int]
    token_logprobs: list[float]


class LLMWrapper:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto"
        )
        self.model.eval()

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_new_tokens: int = 200,
        num_return_sequences: int = 1,
    ) -> list[GenerationOutput]:
        chat = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self.tokenizer(chat, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        if temperature == 0:
            do_sample = False
            num_return_sequences = 1
            gen_kwargs = {}
        else:
            do_sample = True
            gen_kwargs = {"temperature": temperature}

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_return_sequences,
                do_sample=do_sample,
                return_dict_in_generate=True,
                output_scores=True,
                **gen_kwargs,
            )

        # out.sequences shape: (num_return_sequences, input_len + generated_len)
        # out.scores: tuple of (num_return_sequences, vocab_size) tensors, one per step
        results = []
        for seq_idx in range(num_return_sequences):
            generated_ids = out.sequences[seq_idx, input_len:]
            token_ids = generated_ids.tolist()

            # Gather log-prob of chosen token at each step
            token_logprobs = []
            for step, scores in enumerate(out.scores):
                # scores shape: (num_return_sequences, vocab_size)
                log_probs = F.log_softmax(scores[seq_idx], dim=-1)
                chosen_token_id = generated_ids[step].item()
                token_logprobs.append(log_probs[chosen_token_id].item())

            text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            results.append(GenerationOutput(
                text=text,
                token_ids=token_ids,
                token_logprobs=token_logprobs,
            ))

        return results

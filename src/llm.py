"""Load a small open-source instruction-tuned language model and generate text.

Runs on a Colab GPU. Generation is greedy (no random sampling), so the same
input always gives the same output: important for reproducibility.
"""

import re

# Tried in order: the first one that loads is used.
#   Qwen3-4B-Instruct-2507 - 4B parameters, Apache-2.0 license, fits a T4 GPU in float16
#   Qwen2.5-1.5B-Instruct  - smaller fallback, Apache-2.0 license
DEFAULT_MODELS = ["Qwen/Qwen3-4B-Instruct-2507", "Qwen/Qwen2.5-1.5B-Instruct"]


def load_llm(model_ids: list[str] = DEFAULT_MODELS, max_new_tokens: int = 200):
    """Return (generate_fn, model_id). generate_fn(messages) -> generated text.

    `messages` is a chat: [{"role": "system", "content": ...}, {"role": "user", "content": ...}].
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    last_error = None
    for model_id in model_ids:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            try:
                model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map="auto")
            except TypeError:  # older transformers versions call it torch_dtype
                model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map="auto")
            model.eval()
            break
        except Exception as error:  # model not found, out of memory, unsupported architecture ...
            print(f"Could not load {model_id}: {type(error).__name__}: {error}")
            last_error = error
    else:
        raise RuntimeError("No language model could be loaded") from last_error

    def generate(messages: list[dict]) -> str:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                    pad_token_id=tokenizer.eos_token_id)
        new_tokens = output[0][inputs["input_ids"].shape[1]:]          # only the generated part
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    return generate, model_id

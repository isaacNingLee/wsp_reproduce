"""Benchmarks + grading for the Qwen math experiments, following the blog's exp8 protocol
(lm_understand-logs/exp8_math/code/{mathqa,profiles}.py): user message = problem + INSTRUCTION, chat template,
last \\boxed{} graded with math-verify. Problem subsets are drawn once with a fixed seed so every condition sees
the same problems."""
import random
import re

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."

# Sampling profiles from exp8_math/code/profiles.py (both are from the Qwen3.5-0.8B model card).
PROFILES = {
    "think": dict(thinking=True, temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5, max_tokens=32768),
    "nothink": dict(thinking=False, temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5, max_tokens=8192),
}


def load_bench(name, n=None, seed=0):
    from datasets import load_dataset
    if name == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        items = [{"id": r["unique_id"], "problem": r["problem"], "answer": r["answer"]} for r in ds]
    elif name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        items = [{"id": f"gsm8k/test/{i}", "problem": r["question"], "answer": r["answer"].split("####")[-1].strip()}
                 for i, r in enumerate(ds)]
    elif name == "gsm8k_train":  # practice pool for the agent (never the test set)
        ds = load_dataset("openai/gsm8k", "main", split="train")
        items = [{"id": f"gsm8k/train/{i}", "problem": r["question"], "answer": r["answer"].split("####")[-1].strip()}
                 for i, r in enumerate(ds)]
    else:
        raise ValueError(name)
    if n is not None and n < len(items):
        items = random.Random(seed).sample(items, n)
    return items


def last_boxed(text):
    """Content of the last \\boxed{...} / \\fbox{...} (brace matched), or the '\\boxed 5' form; None if absent."""
    idx = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if idx < 0:
        return None
    key = "\\boxed" if text.startswith("\\boxed", idx) else "\\fbox"
    rest = text[idx + len(key):]
    if rest.startswith(" ") and key == "\\boxed" and not rest.lstrip().startswith("{"):
        return re.split(r"[$\n]", rest.strip())[0].strip() or None
    rest = rest.lstrip()
    if not rest.startswith("{"):
        return None
    depth = 0
    for i, ch in enumerate(rest):
        depth += ch == "{"
        depth -= ch == "}"
        if depth == 0:
            return rest[1:i]
    return None


def grade(pred, gold):
    if pred is None:
        return False
    from math_verify import parse, verify
    try:
        return bool(verify(parse("\\boxed{" + gold + "}"), parse("\\boxed{" + pred + "}")))
    except Exception:
        return False


def answer_part(text):
    """Graded text: what follows </think> when the model closed its thinking, else the whole text (as exp8)."""
    return text.split("</think>")[-1] if "</think>" in text else text


def build_prompt(tok, problem, thinking):
    return tok.apply_chat_template([{"role": "user", "content": problem + "\n" + INSTRUCTION}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=thinking)

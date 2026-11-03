"""Qwen3.5-0.8B checkpoint helpers shared by the controls, the qwen_math kernel and its final evaluation."""
import json
import os

from wsp import paths

FROZEN = ("model.language_model.norm.weight", "model.language_model.embed_tokens.weight", "lm_head.weight")


def base_dir():
    return paths.snapshot(paths.QWEN_BASE, paths.QWEN_BASE_REV)


def st_name(d):
    st = [f for f in os.listdir(d) if f.endswith(".safetensors")]
    assert len(st) == 1, st
    return st[0]


def load_base():
    from wsp.checkpoints import load_tensors
    return load_tensors(base_dir())


def write_ckpt(base, edits, out_dir, meta=None):
    """Base tensors with `edits` (cast to bf16) as a loadable checkpoint dir; non-weight files are symlinked."""
    import torch
    from safetensors.torch import save_file
    d = base_dir()
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(d):
        dst = os.path.join(out_dir, f)
        if not f.endswith((".safetensors", ".safetensors.index.json")) and not os.path.lexists(dst):
            src = os.path.realpath(os.path.join(d, f))
            if os.path.isfile(src):
                os.symlink(src, dst)
    t = dict(base)
    t.update({k: v.to(torch.bfloat16).contiguous() for k, v in edits.items()})
    save_file(t, os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"})
    if meta is not None:
        json.dump(meta, open(os.path.join(out_dir, "control_meta.json"), "w"), indent=1)
    return out_dir


def practice_set():
    """GSM8K train + MATH train (never the test sets), cached as JSON: [{id, source, problem, answer}]."""
    fn = os.path.join(paths.SCRATCH, "cache", "qwen_practice.json")
    if os.path.exists(fn):
        return json.load(open(fn))
    from datasets import load_dataset

    from wsp.mathbench import last_boxed
    items = []
    for i, r in enumerate(load_dataset("openai/gsm8k", "main", split="train")):
        items.append({"id": f"gsm8k/train/{i}", "source": "gsm8k", "problem": r["question"],
                      "answer": r["answer"].split("####")[-1].strip()})
    for i, r in enumerate(load_dataset("DigitalLearningGmbH/MATH-lighteval", split="train")):
        a = last_boxed(r["solution"])
        if a:
            items.append({"id": f"math/train/{i}", "source": "math", "level": r.get("level"), "type": r.get("type"),
                          "problem": r["problem"], "answer": a})
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    json.dump(items, open(fn, "w"))
    return items

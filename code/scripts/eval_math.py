"""A2: score a checkpoint on GSM8K / MATH-500 subsets with the exp8 protocol (vLLM).

    python scripts/eval_math.py --ckpt base --profile think --bench math500 --n 100 --samples 2

--ckpt is "base", "edited", or a local dir (e.g. one written by make_control_edits.py).
Writes OUT/math/<tag>/<bench>_<profile>.json with accuracy, cut share (hit max_tokens), mean tokens and per-sample
records, so conditions can be compared problem by problem.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from wsp import paths
from wsp.mathbench import PROFILES, answer_part, build_prompt, grade, last_boxed, load_bench


def resolve(ckpt):
    if ckpt == "base":
        return paths.snapshot(paths.QWEN_BASE, paths.QWEN_BASE_REV), "base"
    if ckpt == "edited":
        return paths.snapshot(paths.QWEN_EDIT, paths.QWEN_EDIT_REV), "edited"
    return ckpt, os.path.basename(ckpt.rstrip("/"))


# Portable defaults: vLLM subprocess startup and sampling without an nvcc dependency.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--profile", choices=list(PROFILES), default="think")
    ap.add_argument("--bench", choices=["math500", "gsm8k"], required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--max_tokens", type=int, default=None, help="override the profile's limit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--enforce_eager", action="store_true", help="skip graph compilation for quick startup checks")
    ap.add_argument("--kv_cache_mb", type=int, help="explicit KV-cache MiB; bypass automatic free-memory estimation")
    ap.add_argument("--max_num_seqs", type=int, help="override maximum concurrent sequences")
    ap.add_argument("--max_num_batched_tokens", type=int, help="override scheduler token batch size")
    args = ap.parse_args()
    if args.n <= 0 or args.samples <= 0:
        ap.error("--n and --samples must be positive")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    path, tag = resolve(args.ckpt)
    tag = args.tag or tag
    prof = dict(PROFILES[args.profile])
    if args.max_tokens:
        prof["max_tokens"] = args.max_tokens
    items = load_bench(args.bench, args.n, seed=0)
    tok = AutoTokenizer.from_pretrained(path)
    prompts = [build_prompt(tok, it["problem"], prof["thinking"]) for it in items]
    sp = SamplingParams(n=args.samples, temperature=prof["temperature"], top_p=prof["top_p"], top_k=prof["top_k"],
                        presence_penalty=prof["presence_penalty"], max_tokens=prof["max_tokens"], seed=args.seed)
    runtime = {}
    if args.kv_cache_mb is not None:
        if args.kv_cache_mb <= 0:
            ap.error("--kv_cache_mb must be positive")
        runtime["kv_cache_memory_bytes"] = args.kv_cache_mb * 1024 * 1024
    for key in ("max_num_seqs", "max_num_batched_tokens"):
        if getattr(args, key) is not None:
            if getattr(args, key) <= 0:
                ap.error(f"--{key} must be positive")
            runtime[key] = getattr(args, key)
    llm = LLM(model=path, tokenizer=path, dtype="bfloat16", enforce_eager=args.enforce_eager, max_model_len=prof["max_tokens"] + 2048,
              gpu_memory_utilization=args.gpu_mem, seed=args.seed, limit_mm_per_prompt={"image": 0, "video": 0}, **runtime)
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    secs = time.time() - t0

    recs, ntok = [], 0
    for it, o in zip(items, outs):
        for s in o.outputs:
            text = s.text
            pred = last_boxed(answer_part(text))
            n = len(s.token_ids)
            ntok += n
            recs.append({"id": it["id"], "gold": it["answer"], "pred": pred, "correct": grade(pred, it["answer"]),
                         "cut": s.finish_reason == "length", "tokens": n, "closed_think": "</think>" in text,
                         "tail": text[-300:]})
    k = len(recs)
    summ = {"ckpt": tag, "path": path, "bench": args.bench, "profile": args.profile, "sampling": prof, "runtime": dict(runtime, enforce_eager=args.enforce_eager, gpu_memory_utilization=args.gpu_mem),
            "n_problems": len(items), "samples": args.samples, "n": k,
            "acc": sum(r["correct"] for r in recs) / k, "cut_share": sum(r["cut"] for r in recs) / k,
            "mean_tokens": ntok / k, "seconds": round(secs), "tok_per_s": round(ntok / secs)}
    out = os.path.join(paths.OUT, "math", tag)
    os.makedirs(out, exist_ok=True)
    fn = os.path.join(out, f"{args.bench}_{args.profile}" + (f"_mt{args.max_tokens}" if args.max_tokens else "") + ".json")
    json.dump({"summary": summ, "records": recs}, open(fn, "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()

"""Subprocess used by the qwen_math kernel's generate(): run vLLM on one checkpoint dir over a list of problems.
Input/output are JSON files so the kernel never imports vLLM itself (vLLM wants the GPU and its own process)."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# Portable defaults: vLLM subprocess startup and sampling without an nvcc dependency.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

def main():
    spec = json.load(open(sys.argv[1]))
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from wsp.mathbench import answer_part, build_prompt, grade, last_boxed
    tok = AutoTokenizer.from_pretrained(spec["ckpt"])
    s = spec["sampling"]
    prompts = [build_prompt(tok, p["problem"], s["thinking"]) for p in spec["problems"]]
    llm = LLM(model=spec["ckpt"], dtype="bfloat16", max_model_len=s["max_tokens"] + 2048,
              gpu_memory_utilization=spec.get("gpu_mem", 0.45), seed=s.get("seed", 0), enforce_eager=False,
              limit_mm_per_prompt={"image": 0, "video": 0})
    sp = SamplingParams(n=1, temperature=s["temperature"], top_p=s["top_p"], top_k=s["top_k"],
                        presence_penalty=s["presence_penalty"], max_tokens=s["max_tokens"], seed=s.get("seed", 0))
    outs = llm.generate(prompts, sp)
    recs = []
    for p, o in zip(spec["problems"], outs):
        t = o.outputs[0]
        pred = last_boxed(answer_part(t.text))
        recs.append({"id": p["id"], "correct": grade(pred, p["answer"]), "pred": pred, "gold": p["answer"],
                     "cut": t.finish_reason == "length", "tokens": len(t.token_ids), "text": t.text})
    json.dump(recs, open(spec["out"], "w"))


if __name__ == "__main__":
    main()

"""T3 (the blog's Exp. 8, re-run with our editor LLMs): make Qwen3.5-0.8B (thinking on) better at math by editing
its weights directly, with a forward-pass budget, no training.

Harness, after exp8_math/README.md: a persistent kernel over the weights; `save_edit` (final norm, embeddings and
output layer frozen; whole-tensor rescales refused); forward passes = either a generation pass on practice
problems (GSM8K/MATH train, never test) or a teacher-forced read (log-probs and activations). Differences from the
blog, for cost: up to 128 problems per generation pass (blog: 512) and practice generations capped at 8,192 tokens
(blog: 32,768); the final evaluation uses the full 32,768-token protocol (scripts/eval_math.py).
"""
import json
import os
import subprocess
import sys

GEN_MAX = 128
READ_MAX_TEXTS, READ_MAX_TOKENS = 64, 4096


def system_prompt(cfg):
    from wsp.qwen import practice_set
    ps = practice_set()
    n_g = sum(p["source"] == "gsm8k" for p in ps)
    return f"""You are an expert in transformer internals who edits model weights directly, by reasoning, without training.

## The model
Qwen3.5-0.8B (text part): 24 decoder layers of width 1,024 with a hybrid layout -- layers 3, 7, 11, 15, 19, 23 are
full softmax attention (`self_attn`: q/k/v/o projections, gated), the other 18 are Gated DeltaNet linear attention
(`linear_attn`); every layer has a SwiGLU MLP (`mlp.gate_proj`, `mlp.up_proj`: 1,024 -> 3,584; `mlp.down_proj`:
3,584 -> 1,024); RMSNorms; tied input/output embeddings (vocab 248,320). It is run in THINKING mode: the chat
template opens a <think> block, the model reasons, closes it with </think>, then answers.

## Goal
Make the model better at math by editing its weights. You are scored afterwards on math benchmarks you never see
(GSM8K-style word problems and MATH-style competition problems), with the same prompt and sampling as your
generation passes but up to 32,768 new tokens: user message = problem + "\\nPlease reason step by step, and put your
final answer within \\boxed{{}}."; temperature 0.6, top_p 0.95, top_k 20, presence penalty 1.5; the last \\boxed{{}}
after </think> is graded. Higher accuracy is better. Keep the model a working language model.

## Tools in the kernel
- `sd`: dict name -> torch.bfloat16 CPU tensor, the ORIGINAL language-model weights (names like
  `model.language_model.layers.15.mlp.down_proj.weight`, shape (1024, 3584) = (out, in)). Read-only.
- `save_edit(name, tensor)`: replace one tensor with your version (same shape; stored as bfloat16). Saved to disk
  at once; the latest saved version of each tensor is your submission. `clear_edit(name)` / `clear_edit()` revert,
  `list_edits()` lists. Refused: the final norm, the embeddings / output layer, and edits that only multiply a
  whole tensor by a constant.
- `practice`: list of {len(ps)} practice problems with answers ({n_g} GSM8K train, {len(ps) - n_g} MATH train), dicts with
  id, source, problem, answer (MATH ones also level, type).
- `generate(problems, which="edited", max_tokens=8192, seed=0)` -- COSTS ONE FORWARD PASS. `problems` = list of
  indices into `practice` (at most {GEN_MAX}). The model (original weights + your saved edits, or "original", or
  "both" = both models on the same problems, still one pass) answers them with the evaluation sampling, at most
  `max_tokens` new tokens (<= 8192 here, to save time). Returns a list of dicts: id, correct, pred, gold, cut
  (True if it hit max_tokens), tokens, text (the full response). For "both": {{"original": [...], "edited": [...]}}.
  Print summaries, not whole texts.
- `read(texts, layers, kind="resid", which="edited", chat=True)` -- COSTS ONE FORWARD PASS. Teacher-forced pass over
  up to {READ_MAX_TEXTS} texts (each cut to {READ_MAX_TOKENS} tokens). If chat=True each text is a (problem, response) pair and
  is wrapped in the chat template as user message + assistant response (response should start with your own or
  the model's thinking text; the template already opens <think>); if chat=False texts are raw strings. Returns
  {{"tokens": [int arrays], "token_strs": [lists], "logprobs": [float arrays, log p(token_t | <t), first entry 0],
  "acts": {{layer: [arrays (T, d) float16]}}}} where kind is "resid" (layer output, d=1024), "mlp_out"
  (down_proj output, 1024), "mlp_act" (down_proj input, 3584) or "attn_out" (attention/linear-attention block
  output, 1024).
- `tokenizer`, numpy as np, torch.

## Budget
{cfg['passes']} forward passes in total (generation or read). A failed call (error) does not count.

## Rules
No gradient descent or training of any kind (no backward passes, autograd, optimizers, iterative fitting of weights
to a loss); do not run the model yourself in the kernel (only generate/read run it); no internet. You may compute
anything from weights and from the activations/log-probs the tools return, and design edits by reasoning.

## Answer format
Your final answer: say what you changed and why, and what you measured. The saved edits at that time are submitted.
"""


def first_message(cfg):
    return "The kernel is ready. Begin."


def _find_layers(m):
    for path in ("model.language_model.layers", "model.layers", "language_model.model.layers"):
        o = m
        try:
            for p in path.split("."):
                o = getattr(o, p)
            return o
        except AttributeError:
            continue
    raise AttributeError("no decoder layers found")


def kernel_setup(cfg):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from wsp import qwen
    gpu = cfg.get("gpu", 0)
    torch.cuda.set_device(gpu)
    run_dir = cfg["run_dir"]
    edit_dir, gen_dir = os.path.join(run_dir, "edit"), os.path.join(run_dir, "gens")
    os.makedirs(edit_dir, exist_ok=True)
    os.makedirs(gen_dir, exist_ok=True)
    state_fn = os.path.join(run_dir, "budget.json")
    state = json.load(open(state_fn)) if os.path.exists(state_fn) else {"used": 0, "log": []}

    base = qwen.load_base()
    sd = {k: v for k, v in base.items() if "language_model" in k}
    bdir = qwen.base_dir()
    tok = AutoTokenizer.from_pretrained(bdir)
    hf = AutoModelForCausalLM.from_pretrained(bdir, dtype=torch.bfloat16).cuda().eval()
    hf_params = dict(hf.named_parameters())
    key2hf = {}
    for k in sd:
        suf = k.split("language_model.", 1)[1]
        cands = [n for n in hf_params if n.endswith(suf) and "visual" not in n]
        if len(cands) == 1:
            key2hf[k] = cands[0]
    layers = _find_layers(hf)
    practice = qwen.practice_set()

    def _edits():
        return {f[:-3]: torch.load(os.path.join(edit_dir, f)) for f in sorted(os.listdir(edit_dir))}

    def save_edit(name, tensor):
        if name not in sd:
            raise KeyError(f"unknown tensor {name!r}")
        if name in qwen.FROZEN:
            raise PermissionError(f"{name} is frozen (final norm / embeddings / output layer)")
        t = torch.as_tensor(np.asarray(tensor) if not torch.is_tensor(tensor) else tensor).detach()
        t = t.to(torch.bfloat16).cpu()
        if t.shape != sd[name].shape:
            raise ValueError(f"shape {tuple(t.shape)} != {tuple(sd[name].shape)}")
        if not torch.isfinite(t.float()).all():
            raise ValueError("non-finite values")
        w0, w1 = sd[name].float().flatten(), t.float().flatten()
        c = float(w1 @ w0 / (w0 @ w0))
        if (w1 - c * w0).abs().max() < 1e-6 * max(1.0, w0.abs().max().item()) and abs(c - 1) > 1e-6:
            raise PermissionError("edits that only multiply a whole tensor by a constant are refused")
        torch.save(t.clone(), os.path.join(edit_dir, name + ".pt"))
        d = t.float() - sd[name].float()
        return f"saved {name}: {int((d != 0).sum())} entries changed, ||delta||/||W|| = {float(d.norm() / sd[name].float().norm()):.4g}"

    def clear_edit(name=None):
        for f in os.listdir(edit_dir):
            if name is None or f == name + ".pt":
                os.remove(os.path.join(edit_dir, f))
        return list_edits()

    def list_edits():
        return sorted(_edits())

    def _charge(op, detail):
        if state["used"] >= cfg["passes"]:
            raise RuntimeError("forward-pass budget exhausted")
        entry = dict(detail, op=op, edits=list_edits())  # built before charging: a bad entry can never eat a pass
        state["used"] += 1
        state["log"].append(entry)
        json.dump(state, open(state_fn, "w"))

    def _gen_one(which, probs, max_tokens, seed):
        if which == "original":
            ckpt = bdir
        else:
            ckpt = qwen.write_ckpt(base, _edits(), os.path.join(run_dir, "ckpt_edited"))
        k = len(state["log"])
        spec = {"ckpt": ckpt, "problems": probs, "out": os.path.join(gen_dir, f"pass{k:02d}_{which}.json"),
                "sampling": {"thinking": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20,
                             "presence_penalty": 1.5, "max_tokens": max_tokens, "seed": seed}}
        sfn = spec["out"].replace(".json", "_spec.json")
        json.dump(spec, open(sfn, "w"))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        worker = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "gen_worker.py")
        r = subprocess.run([sys.executable, worker, sfn], env=env, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("generation failed:\n" + r.stderr[-3000:])
        return json.load(open(spec["out"]))

    def generate(problems, which="edited", max_tokens=8192, seed=0):
        problems = [int(i) for i in problems]
        if not 0 < len(problems) <= GEN_MAX:
            raise ValueError(f"1..{GEN_MAX} problems per pass")
        if not 0 < max_tokens <= 8192:
            raise ValueError("max_tokens must be in 1..8192")
        if which not in ("edited", "original", "both"):
            raise ValueError("which must be 'edited', 'original' or 'both'")
        probs = [practice[i] for i in problems]
        _charge("generate", {"which": which, "n": len(problems), "max_tokens": max_tokens})
        try:
            if which == "both":
                return {"original": _gen_one("original", probs, max_tokens, seed),
                        "edited": _gen_one("edited", probs, max_tokens, seed)}
            return _gen_one(which, probs, max_tokens, seed)
        except Exception:
            state["used"] -= 1  # a failed request does not count
            json.dump(state, open(state_fn, "w"))
            raise

    @torch.no_grad()
    def read(texts, layers=(), kind="resid", which="edited", chat=True):
        if not 0 < len(texts) <= READ_MAX_TEXTS:
            raise ValueError(f"1..{READ_MAX_TEXTS} texts per pass")
        if kind not in ("resid", "mlp_out", "mlp_act", "attn_out"):
            raise ValueError("bad kind")
        layers = [int(l) for l in layers]
        if chat:
            strs = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=True) + r for p, r in texts]
        else:
            strs = list(texts)
        _charge("read", {"which": which, "n": len(texts), "act": kind, "layers": layers})
        edits = _edits() if which == "edited" else {}
        saved = {}
        for k, v in edits.items():
            p = hf_params[key2hf[k]]
            saved[k] = p.data.clone()
            p.data.copy_(v.to(p.dtype))
        acts = {l: [] for l in layers}
        cur = {}
        hooks = []
        for l in layers:
            L = layers_mod[l]
            if kind == "resid":
                hooks.append(L.register_forward_hook(lambda m, i, o, l=l: cur.__setitem__(l, (o[0] if isinstance(o, tuple) else o))))
            elif kind == "mlp_out":
                hooks.append(L.mlp.down_proj.register_forward_hook(lambda m, i, o, l=l: cur.__setitem__(l, o)))
            elif kind == "mlp_act":
                hooks.append(L.mlp.down_proj.register_forward_hook(lambda m, i, o, l=l: cur.__setitem__(l, i[0])))
            else:
                att = getattr(L, "self_attn", None) or getattr(L, "linear_attn")
                hooks.append(att.register_forward_hook(lambda m, i, o, l=l: cur.__setitem__(l, (o[0] if isinstance(o, tuple) else o))))
        out = {"tokens": [], "token_strs": [], "logprobs": [], "acts": acts}
        try:
            for s in strs:
                ids = tok(s, return_tensors="pt", add_special_tokens=False).input_ids[:, :READ_MAX_TOKENS].cuda()
                cur.clear()
                logits = hf(input_ids=ids).logits[0].float()
                lp = torch.log_softmax(logits[:-1], -1).gather(1, ids[0, 1:, None])[:, 0]
                out["tokens"].append(ids[0].cpu().numpy())
                out["token_strs"].append(tok.convert_ids_to_tokens(ids[0].tolist()))
                out["logprobs"].append(np.concatenate([[0.0], lp.cpu().numpy()]))
                for l in layers:
                    acts[l].append(cur[l][0].float().cpu().numpy().astype(np.float16))
        except Exception:
            state["used"] -= 1
            json.dump(state, open(state_fn, "w"))
            raise
        finally:
            for h in hooks:
                h.remove()
            for k, v in saved.items():
                hf_params[key2hf[k]].data.copy_(v)
        return out

    layers_mod = layers

    def _wsp_status():
        return f"forward passes left: {cfg['passes'] - state['used']}/{cfg['passes']}; saved edits: {list_edits()}"

    return {"sd": sd, "save_edit": save_edit, "clear_edit": clear_edit, "list_edits": list_edits,
            "practice": practice, "generate": generate, "read": read, "tokenizer": tok, "np": np, "torch": torch,
            "_wsp_status": _wsp_status, "_wsp_finalize": lambda: print(list_edits())}


def score(final, cfg, run_dir):
    """Build the submitted checkpoint; the 32k-token benchmark eval is run separately (scripts/eval_math.py), as it
    takes ~an hour per condition."""
    import torch

    from wsp import qwen
    edit_dir = os.path.join(run_dir, "edit")
    edits = {f[:-3]: torch.load(os.path.join(edit_dir, f)) for f in sorted(os.listdir(edit_dir))} if os.path.isdir(edit_dir) else {}
    res = {"task": "qwen_math", "edited_tensors": sorted(edits), "final_answer": (final or "")[-6000:]}
    if edits:
        base = qwen.load_base()
        res["n_weights_changed"] = int(sum(((v.float() - base[k].float()) != 0).sum() for k, v in edits.items()))
        ck = qwen.write_ckpt(base, edits, os.path.join(run_dir, "ckpt_final"))
        res["ckpt"] = ck
    else:
        res["n_weights_changed"] = 0
    return res

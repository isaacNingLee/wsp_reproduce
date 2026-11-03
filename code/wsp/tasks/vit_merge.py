"""T4 (not in the blog): MERGE two fine-tuned models into one by reasoning over weights, no training.

The local ImageNet ViT-B/16 (parent) was fine-tuned separately on CIFAR-100 and on CUB-200. The editor must write a
single backbone (everything except the classification heads) that serves BOTH tasks: the CIFAR-100 head and the CUB
head are kept fixed and both read the same merged backbone. It may evaluate candidate backbones on PRACTICE images
(the practice halves from scripts/prep_vit_fix.py) with a small budget; the score is mean HOLDOUT accuracy.
Reference points (scripts/merge_baselines.py): each child alone, plain averaging, task arithmetic with lambda picked
on practice. Fast to score: one evaluation is ~30 s.
"""
import json
import os

DATASETS = ("cifar100", "cub")
HEAD = ("heads.head.weight", "heads.head.bias")


def system_prompt(cfg):
    base = _baselines()
    ref = ""
    if base:
        p = base["practice"]
        ref = ("\nReference practice accuracies (CIFAR-100 / CUB / mean):\n" +
               "\n".join(f"- {k}: {100 * v['cifar100']:.2f} / {100 * v['cub']:.2f} / {100 * v['mean']:.2f}"
                         for k, v in p.items() if k in ("cifar_child", "cub_child", "average")) + "\n")
    return f"""You are an expert in neural-network weights who builds models by reasoning, without training.

## The models
Three torchvision `vit_b_16` checkpoints (12 pre-LN encoder blocks, width 768, 12 heads, MLP 768->3072->768):
- `parent`: supervised ImageNet-1k model (torchvision IMAGENET1K_V1);
- `cifar`: `parent` fine-tuned (all weights, 40 epochs, AdamW, backbone lr 1e-4, wd 0.05) on CIFAR-100
  (100 classes; 32x32 images upsampled to 224); 88.4% test accuracy;
- `cub`: `parent` fine-tuned the same way on CUB-200 birds (200 species); 83.8% test accuracy.
Inputs are normalized with mean=std=0.5 for both tasks.

## Goal
Write ONE backbone -- every tensor except `heads.head.weight` / `heads.head.bias` -- such that the CIFAR-100 head
(from `cifar`) and the CUB head (from `cub`), both kept FIXED, work as well as possible on top of it. Score = mean
of CIFAR-100 and CUB accuracy on held-out images you never see (same distribution as the practice images).
Using one child's backbone alone fails the other task badly; the question is how well you can combine them.
{ref}
## Tools in the kernel
- `parent`, `cifar`, `cub`: dicts name -> torch.float32 CPU tensor (backbone only, heads removed). Read-only.
- `heads`: {{"cifar100": (W, b), "cub": (W, b)}} -- the fixed heads, for analysis.
- `save_merge(sd)`: submit a full backbone dict (same keys/shapes as `parent`). Saved to disk at once; the latest
  saved one is your submission. If you never save, the submission is the parent backbone.
- `evaluate(sd, datasets=("cifar100","cub"), per_class=False)` -- COSTS ONE EVALUATION (budget: {cfg['passes']}).
  Runs the backbone `sd` (any dict with all backbone keys) with each fixed head on that dataset's PRACTICE images
  (5,000 CIFAR-100 and 2,875 CUB images) and returns {{"cifar100": acc, "cub": acc, "mean": ...}} (plus per-class
  accuracies if per_class=True). Evaluating only one dataset still costs one evaluation.
- numpy as np, torch.

## Rules
No gradient descent or training of any kind (no backward passes, autograd, optimizers, iterative fitting of
weights to a loss), no fitting of weights to labels by regression. You may compute anything from the weights
(differences, norms, SVDs, sign agreement, per-layer statistics), design the merge by reasoning, and check
candidates with `evaluate` within the budget. The heads cannot be changed.

## Answer format
Your final answer: describe the merge rule you used and why, and the practice accuracies you measured.
"""


def first_message(cfg):
    return "The kernel is ready. Begin."


def _baselines():
    from wsp import paths
    fn = os.path.join(paths.SCRATCH, "cache", "merge_baselines.json")
    return json.load(open(fn)) if os.path.exists(fn) else None


def load_parts():
    """parent/cifar/cub backbones + heads, float32 CPU."""
    import torch
    from wsp import paths
    sds = {ds: torch.load(paths.VIT_CKPTS[ds][0], map_location="cpu") for ds in ("imagenet", "cifar100", "cub")}
    bb = {name: {k: v.float() for k, v in sds[ds].items() if k not in HEAD}
          for name, ds in (("parent", "imagenet"), ("cifar", "cifar100"), ("cub", "cub"))}
    heads = {ds: (sds[ds][HEAD[0]].float(), sds[ds][HEAD[1]].float()) for ds in DATASETS}
    return bb, heads


class Evaluator:
    """Runs a backbone + each fixed head on cached images of one split (practice or holdout)."""

    def __init__(self, split, device="cuda"):
        import torch

        from wsp import paths, vit
        self.device, self.models, self.data = device, {}, {}
        for ds in DATASETS:
            cache = torch.load(vit.cache_path(ds))
            prep = torch.load(os.path.join(paths.SCRATCH, "cache", f"{ds}_fix_prep.pt"))
            idx = prep[split]
            self.data[ds] = (cache["x"][idx].to(device), cache["y"][idx])
            self.models[ds] = vit.load_vit(ds, device)

    def __call__(self, sd, heads, datasets=DATASETS, per_class=False):
        import torch

        from wsp import vit
        out = {}
        for ds in datasets:
            m = self.models[ds]
            full = dict(sd)
            full[HEAD[0]], full[HEAD[1]] = heads[ds]
            m.load_state_dict(full)
            x, y = self.data[ds]
            logits, _ = vit.run(m, x, ds, device=self.device)
            ok = (logits.argmax(1) == y)
            out[ds] = float(ok.float().mean())
            if per_class:
                out[ds + "_per_class"] = [float(ok[y == k].float().mean()) for k in range(int(y.max()) + 1)]
        if len(datasets) == 2:
            out["mean"] = (out["cifar100"] + out["cub"]) / 2
        return out


def kernel_setup(cfg):
    import numpy as np
    import torch
    torch.cuda.set_device(cfg.get("gpu", 0))
    run_dir = cfg["run_dir"]
    bb, heads = load_parts()
    ev = Evaluator("practice")
    keys = {k: v.shape for k, v in bb["parent"].items()}
    state_fn = os.path.join(run_dir, "budget.json")
    state = json.load(open(state_fn)) if os.path.exists(state_fn) else {"used": 0, "log": []}

    def _check(sd):
        missing = set(keys) - set(sd)
        if missing:
            raise KeyError(f"missing {len(missing)} keys, e.g. {sorted(missing)[:3]}")
        clean = {}
        for k, shp in keys.items():
            t = torch.as_tensor(np.asarray(sd[k]) if not torch.is_tensor(sd[k]) else sd[k]).detach().float().cpu()
            if t.shape != shp:
                raise ValueError(f"{k}: shape {tuple(t.shape)} != {tuple(shp)}")
            if not torch.isfinite(t).all():
                raise ValueError(f"{k}: non-finite values")
            clean[k] = t
        return clean

    def save_merge(sd):
        clean = _check(sd)
        torch.save(clean, os.path.join(run_dir, "merge.pt"))
        d = sum(float((clean[k] - bb["parent"][k]).norm() ** 2) for k in keys) ** 0.5
        return f"saved merged backbone ({len(clean)} tensors); ||merged - parent|| = {d:.3f}"

    def evaluate(sd, datasets=DATASETS, per_class=False):
        datasets = tuple(datasets) if not isinstance(datasets, str) else (datasets,)
        if not set(datasets) <= set(DATASETS):
            raise ValueError(f"datasets must be from {DATASETS}")
        if state["used"] >= cfg["passes"]:
            raise RuntimeError("evaluation budget exhausted")
        clean = _check(sd)
        entry = {"datasets": list(datasets)}
        state["used"] += 1
        state["log"].append(entry)
        json.dump(state, open(state_fn, "w"))
        try:
            r = ev(clean, heads, datasets, per_class)
        except Exception:
            state["used"] -= 1
            json.dump(state, open(state_fn, "w"))
            raise
        entry["result"] = {k: v for k, v in r.items() if not k.endswith("per_class")}
        json.dump(state, open(state_fn, "w"))
        torch.cuda.empty_cache()
        return r

    def _wsp_status():
        return (f"evaluations left: {cfg['passes'] - state['used']}/{cfg['passes']}; "
                f"merge saved: {os.path.exists(os.path.join(run_dir, 'merge.pt'))}")

    return {"parent": bb["parent"], "cifar": bb["cifar"], "cub": bb["cub"], "heads": heads,
            "save_merge": save_merge, "evaluate": evaluate, "np": np, "torch": torch,
            "_wsp_status": _wsp_status, "_wsp_finalize": lambda: print(os.path.exists(os.path.join(run_dir, "merge.pt")))}


def score(final, cfg, run_dir):
    import torch
    fn = os.path.join(run_dir, "merge.pt")
    bb, heads = load_parts()
    sd = torch.load(fn) if os.path.exists(fn) else bb["parent"]
    dev = f"cuda:{cfg.get('gpu', 0)}"
    res = {"task": "vit_merge", "submitted": os.path.exists(fn),
           "holdout": Evaluator("holdout", dev)(sd, heads), "practice": Evaluator("practice", dev)(sd, heads)}
    base = _baselines()
    if base:
        h = base["holdout"]
        res["baselines_holdout"] = h
        res["beats_average"] = res["holdout"]["mean"] > h["average"]["mean"]
        res["beats_task_arithmetic"] = res["holdout"]["mean"] > h["task_arithmetic"]["mean"]
    res["final_answer"] = (final or "")[-4000:]
    return res


def sandbox_visible(cfg):
    """Hide configured private roots except the interpreter, harness code, the three checkpoints and the cached
    images/splits/baselines this task reads (so parallel runs cannot see each other's work)."""
    import sys

    from wsp import paths, vit
    from wsp.kernel import REPO
    cache = os.path.join(paths.SCRATCH, "cache")
    files = [paths.VIT_CKPTS[ds][0] for ds in ("imagenet", "cifar100", "cub")]
    files += [vit.cache_path(ds) for ds in DATASETS] + [os.path.join(cache, f"{ds}_fix_prep.pt") for ds in DATASETS]
    files += [os.path.join(cache, "merge_baselines.json")]
    return [sys.prefix, os.path.join(REPO, "wsp")] + files

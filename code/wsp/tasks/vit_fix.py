"""T2 (analog of blog Exps. 4 / 8: fix a failing behaviour by editing weights, with a forward-pass budget).

A local fine-tuned ViT-B/16 (CIFAR-100 or CUB) confuses one pair of classes most (picked on the practice half of
the test split by scripts/prep_vit_fix.py). The editor must raise accuracy on that pair without hurting the rest,
by editing weights directly. It gets the weights, a budgeted forward-pass tool on PRACTICE images (logits and CLS
features), and `save_edit`. Scoring is on the HOLDOUT half, never seen by the editor.

Success = holdout pair accuracy up AND holdout overall accuracy down by at most 0.3 points.
"""
import os

TOL = 0.003


def _prep(cfg):
    import torch
    from wsp import paths
    return torch.load(os.path.join(paths.SCRATCH, "cache", f"{cfg['dataset']}_fix_prep.pt"))


def system_prompt(cfg):
    import torch  # noqa: F401
    info = _prep(cfg)["info"]
    a, b = info["pair_names"]
    dsdesc = {"cifar100": "CIFAR-100 (100 classes; 32x32 images upsampled to 224x224)",
              "cub": "CUB-200-2011 (200 bird species; images resized to 256 and center-cropped to 224)"}[cfg["dataset"]]
    return f"""You are an expert in neural-network internals who edits model weights directly, by reasoning, without training.

## The model
A torchvision `vit_b_16` image classifier (12 pre-LN encoder blocks, width 768, 12 heads, MLP 768->3072->768,
GELU, class token + 196 patch tokens) fine-tuned on {dsdesc}. Inputs are normalized with mean=std=0.5.
Its last layer: `heads.head.weight` ({len(_names(cfg))}, 768) and `heads.head.bias`, applied to the final-LayerNorm
class-token vector.

## The problem
The model most often confuses class {info['pair'][0]} "{a}" and class {info['pair'][1]} "{b}". On the practice
images: accuracy on these two classes is {100 * info['pair_acc_practice']:.1f}% (confusions: {info['confusions_practice']}),
overall accuracy {100 * info['acc_practice']:.2f}%.

## Goal
Edit the weights so that the model classifies "{a}" vs "{b}" images better, WITHOUT hurting other classes. You are
scored on held-out images you never see (same distribution as the practice images):
- success = held-out accuracy on images of these two classes goes UP, and held-out overall accuracy over all
  classes drops by at most 0.3 points;
- among successes, a larger gain on the pair is better.

## Tools in the kernel
- `sd`: dict name -> torch.float32 CPU tensor, the ORIGINAL weights (torchvision names, e.g.
  `encoder.layers.encoder_layer_11.mlp.0.weight`, `encoder.ln.weight`, `heads.head.weight`). Read freely; treat as
  read-only (edits only count through save_edit).
- `save_edit(name, tensor)`: replace one tensor of the model with your version (same shape). Saved to disk
  immediately; the latest saved version of each tensor is your submission. `clear_edit(name)` reverts one,
  `clear_edit()` reverts all, `list_edits()` shows them.
- `forward(which="edited", classes=None, features=())` -- COSTS ONE FORWARD PASS (budget: {cfg['passes']}).
  Runs the model on the practice images ({info['n_practice']} images, ~{info['n_practice'] // len(_names(cfg))} per
  class), or only on those whose true class is in `classes` (list of class indices). `which` is "edited" (original
  + your saved edits), "original", or "both" (both models on the same images; still one pass). Returns a dict (or
  {{"original": ..., "edited": ...}} for "both") with numpy arrays: `labels` (N,), `logits` (N, C), `pred` (N,),
  `acc` (float), and, for each requested feature, `feats[f]` (N, 768): f = "head_in" (final-LN class-token vector
  the head reads) or "block{{i}}" (class-token residual stream after encoder block i, i = 0..11).
- `classes` (list of class names), `pair` = {info['pair']}, `practice_labels` (numpy, true labels of all practice
  images), and numpy/torch.

## Rules
No gradient descent or training of any kind (no backward passes, autograd, optimizers, iterative fitting of
weights to a loss), and no fitting of weights to the labels by regression (least squares, ridge, logistic
regression) either. You may compute statistics of weights and activations (means, differences, covariances, SVDs,
projections) and design edits from them by reasoning, then check them with forward passes. Edits must be weight
changes (no wrapping the model in new code).

## Answer format
Your final answer: explain what you changed and why, and what you measured on the practice images.
"""


def _names(cfg):
    import torch
    from wsp import paths, vit  # noqa: F401
    return torch.load(vit.cache_path(cfg["dataset"]))["names"]


def first_message(cfg):
    return "The kernel is ready. Begin."


def kernel_setup(cfg):
    import json

    import numpy as np
    import torch

    from wsp import vit
    torch.cuda.set_device(cfg.get("gpu", 0))
    ds, run_dir = cfg["dataset"], cfg["run_dir"]
    cache, prep = torch.load(vit.cache_path(ds)), _prep(cfg)
    prac = prep["practice"]
    x, y = cache["x"][prac].cuda(), cache["y"][prac]
    model = vit.load_vit(ds)
    sd0 = {k: v.detach().float().cpu().clone() for k, v in model.state_dict().items()}
    edit_dir = os.path.join(run_dir, "edit")
    os.makedirs(edit_dir, exist_ok=True)
    state_fn = os.path.join(run_dir, "budget.json")
    state = json.load(open(state_fn)) if os.path.exists(state_fn) else {"used": 0, "log": []}

    def _edits():
        return {f[:-3]: torch.load(os.path.join(edit_dir, f)) for f in sorted(os.listdir(edit_dir))}

    def save_edit(name, tensor):
        if name not in sd0:
            raise KeyError(f"unknown tensor {name!r}")
        t = torch.as_tensor(np.asarray(tensor) if not torch.is_tensor(tensor) else tensor).detach().float().cpu()
        if t.shape != sd0[name].shape:
            raise ValueError(f"shape {tuple(t.shape)} != {tuple(sd0[name].shape)}")
        if not torch.isfinite(t).all():
            raise ValueError("non-finite values")
        torch.save(t.clone(), os.path.join(edit_dir, name + ".pt"))
        d = t - sd0[name]
        return f"saved {name}: {int((d != 0).sum())} entries changed, ||delta||/||W|| = {float(d.norm() / sd0[name].norm()):.4g}"

    def clear_edit(name=None):
        for f in os.listdir(edit_dir):
            if name is None or f == name + ".pt":
                os.remove(os.path.join(edit_dir, f))
        return list_edits()

    def list_edits():
        return sorted(_edits())

    def _run(which, sel, features):
        sdx = dict(sd0)
        if which == "edited":
            sdx.update(_edits())
        model.load_state_dict(sdx)
        logits, feats = vit.run(model, x[sel], ds, features=tuple(features))
        lab = y[sel]
        pred = logits.argmax(1)
        return {"labels": lab.numpy(), "logits": logits.numpy(), "pred": pred.numpy(),
                "acc": float((pred == lab).float().mean()), "feats": {k: v.numpy() for k, v in feats.items()}}

    def forward(which="edited", classes=None, features=()):
        if which not in ("edited", "original", "both"):
            raise ValueError("which must be 'edited', 'original' or 'both'")
        if isinstance(features, str):
            features = (features,)
        for f in features:
            if not (f == "head_in" or (f.startswith("block") and f[5:].isdigit() and 0 <= int(f[5:]) < 12)):
                raise ValueError(f"unknown feature {f!r}")
        if state["used"] >= cfg["passes"]:
            raise RuntimeError("forward-pass budget exhausted")
        sel = torch.arange(len(y)) if classes is None else torch.isin(y, torch.tensor(list(classes))).nonzero().flatten()
        state["used"] += 1
        state["log"].append({"which": which, "classes": None if classes is None else list(classes),
                             "features": list(features), "edits": list_edits()})
        json.dump(state, open(state_fn, "w"))
        if which == "both":
            return {"original": _run("original", sel, features), "edited": _run("edited", sel, features)}
        return _run(which, sel, features)

    def _wsp_status():
        return f"forward passes left: {cfg['passes'] - state['used']}/{cfg['passes']}; saved edits: {list_edits()}"

    return {"sd": {k: v.clone() for k, v in sd0.items()}, "save_edit": save_edit, "clear_edit": clear_edit, "list_edits": list_edits,
            "forward": forward, "classes": list(cache["names"]), "pair": prep["info"]["pair"],
            "practice_labels": y.numpy(), "np": np, "torch": torch,
            "_wsp_status": _wsp_status, "_wsp_finalize": lambda: print(list_edits())}


def evaluate_edits(ds, edits, prep=None, device="cuda"):
    """Accuracy of original+edits on practice and holdout halves (overall, pair, per class)."""
    import torch
    from wsp import vit
    prep = prep or torch.load(os.path.join(os.path.dirname(vit.cache_path(ds)), f"{ds}_fix_prep.pt"))
    cache = torch.load(vit.cache_path(ds))
    sd = torch.load(__import__("wsp.paths", fromlist=["x"]).VIT_CKPTS[ds][0], map_location="cpu")
    sd = {k: v.float() for k, v in sd.items()}
    sd.update(edits)
    model = vit.load_vit(ds, device, sd=sd)
    a, b = prep["info"]["pair"]
    out = {}
    for split in ("practice", "holdout"):
        idx = prep[split]
        logits, _ = vit.run(model, cache["x"][idx], ds, device=device)
        y = cache["y"][idx]
        pred = logits.argmax(1)
        ok = pred == y
        inpair = (y == a) | (y == b)
        out[split] = {"acc": float(ok.float().mean()), "pair_acc": float(ok[inpair].float().mean()),
                      "acc_a": float(ok[y == a].float().mean()), "acc_b": float(ok[y == b].float().mean()),
                      "acc_nonpair": float(ok[~inpair].float().mean())}
    return out


def score(final, cfg, run_dir):
    import torch
    edit_dir = os.path.join(run_dir, "edit")
    edits = {f[:-3]: torch.load(os.path.join(edit_dir, f)) for f in sorted(os.listdir(edit_dir))} if os.path.isdir(edit_dir) else {}
    prep = _prep(cfg)
    dev = f"cuda:{cfg.get('gpu', 0)}"  # the run's GPU, not the default one (may be full)
    ed = evaluate_edits(cfg["dataset"], edits, prep, device=dev)
    base = evaluate_edits(cfg["dataset"], {}, prep, device=dev)
    h, h0 = ed["holdout"], base["holdout"]
    n_changed = 0
    if edits:
        from wsp import paths
        sd0 = torch.load(paths.VIT_CKPTS[cfg["dataset"]][0], map_location="cpu")
        n_changed = int(sum(((v - sd0[k].float()) != 0).sum() for k, v in edits.items()))
    return {"task": "vit_fix", "dataset": cfg["dataset"], "pair": prep["info"]["pair_names"],
            "edited_tensors": sorted(edits), "n_weights_changed": n_changed,
            "base": base, "edited": ed,
            "holdout_pair_gain": h["pair_acc"] - h0["pair_acc"], "holdout_overall_change": h["acc"] - h0["acc"],
            "success": bool(h["pair_acc"] > h0["pair_acc"] and h["acc"] >= h0["acc"] - TOL),
            "final_answer": (final or "")[-4000:]}

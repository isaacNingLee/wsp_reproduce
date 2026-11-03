"""A2 controls: rebuild the released edit from its audited (u, columns) and build matched controls.

The audit (audit_edit.py) shows the released edit is exactly  W[:,S] <- W[:,S] - c * u u^T W[:,S]  with c = 6 on 64
columns S of layers 15 and 19 down_proj. Controls (same two layers):
  rebuilt      - our re-application of the fitted (u, S, c); must reproduce the release up to bf16 rounding
  c1           - same u, S but c = 1: literally "remove the direction" (what the blog text describes)
  randdir_s<k> - random unit u_r on the same S, c scaled so ||delta||_F equals the real edit's (norm-matched)
  randcols_s<k>- the real u and c on 64 random columns (not the 64 that write u most)
Each control is a directory with the base's config/tokenizer symlinked and one rewritten safetensors file.
"""
import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from safetensors.torch import load_file, save_file

from wsp import paths
from wsp.checkpoints import load_tensors

LAYERS = (15, 19)
KEY = "model.language_model.layers.{}.mlp.down_proj.weight"


def edit(W, u, cols, c):
    W = W.float().clone()
    Ws = W[:, cols]
    W[:, cols] = Ws - c * torch.outer(u, u @ Ws)
    return W.to(torch.bfloat16)


def write_ckpt(base_dir, tensors, new, out_dir, meta):
    os.makedirs(out_dir, exist_ok=True)
    weight_file = "model.safetensors"
    for f in os.listdir(base_dir):
        if not f.endswith((".safetensors", ".safetensors.index.json")) and not os.path.exists(os.path.join(out_dir, f)):
            src = os.path.realpath(os.path.join(base_dir, f))
            if os.path.isfile(src):
                os.symlink(src, os.path.join(out_dir, f))
    t = dict(tensors)
    t.update(new)
    save_file(t, os.path.join(out_dir, weight_file), metadata={"format": "pt"})
    with open(os.path.join(out_dir, "control_meta.json"), "w") as handle:
        json.dump(meta, handle, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = ap.parse_args()

    base_dir = paths.snapshot(paths.QWEN_BASE, paths.QWEN_BASE_REV)
    edit_dir = paths.snapshot(paths.QWEN_EDIT, paths.QWEN_EDIT_REV)
    base = load_tensors(base_dir)
    released = load_tensors(edit_dir)
    dirs = torch.load(os.path.join(paths.OUT, "audit/directions.pt"))
    root = os.path.join(paths.SCRATCH, "ckpts")

    fit = {}
    for L in LAYERS:
        k = KEY.format(L)
        W, u, cols = base[k].float(), dirs[k]["u"].float(), dirs[k]["cols"]
        d = released[k].float() - W
        p = torch.outer(u, u @ W[:, cols])
        c = float(-(d[:, cols] * p).sum() / (p ** 2).sum())
        fit[L] = (W, u, cols, c, d.norm().item())
    print({L: round(v[3], 4) for L, v in fit.items()})

    specs = {"rebuilt": {L: (u, cols, c) for L, (W, u, cols, c, _) in fit.items()},
             "c1": {L: (u, cols, 1.0) for L, (W, u, cols, c, _) in fit.items()}}
    for s in args.seeds:
        g = torch.Generator().manual_seed(1000 + s)
        rd, rc = {}, {}
        for L, (W, u, cols, c, dnorm) in fit.items():
            ur = torch.randn(W.shape[0], generator=g)
            ur /= ur.norm()
            cr = dnorm / (ur @ W[:, cols]).norm().item()      # ||c u u^T W_S|| = c ||u^T W_S||
            rd[L] = (ur, cols, cr)
            rcols = torch.randperm(W.shape[1], generator=g)[: len(cols)]
            rc[L] = (u, rcols, c)
        specs[f"randdir_s{s}"] = rd
        specs[f"randcols_s{s}"] = rc

    for name, spec in specs.items():
        new, meta = {}, {"name": name, "layers": {}}
        for L, (u, cols, c) in spec.items():
            k = KEY.format(L)
            new[k] = edit(base[k], u, cols, c)
            dn = (new[k].float() - base[k].float()).norm().item()
            meta["layers"][L] = {"c": c, "n_cols": len(cols), "delta_fro": dn, "released_delta_fro": fit[L][4],
                                 "cos_u_released": float(u @ fit[L][1])}
        if name == "rebuilt":
            for k in new:
                meta.setdefault("max_abs_diff_vs_released", {})[k] = float((new[k].float() - released[k].float()).abs().max())
                meta.setdefault("frac_bit_exact_vs_released", {})[k] = float((new[k] == released[k]).float().mean())
        out = os.path.join(root, f"qwen35_{name}")
        if os.path.exists(out):
            shutil.rmtree(out)
        write_ckpt(base_dir, base, new, out, meta)
        print(name, json.dumps(meta["layers"]), meta.get("frac_bit_exact_vs_released", ""))


if __name__ == "__main__":
    main()

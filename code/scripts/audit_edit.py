"""A1: audit the released edit against its base, with no forward pass.

Claim checked (blog Exp. 8 / model card): only layers 15 and 19 `mlp.down_proj` differ; 64 output
columns per layer; 131,072 weights total; the change is "subtract one direction" from those columns.

For every differing tensor we report: #changed entries, which columns, the rank of the delta, how
well the delta is explained by a rank-1 projection removal W[:,S] -= u u^T W[:,S] and whether both
layers use the same u. We then read u through the tied unembedding (logit lens) to see which tokens
the removed direction was promoting.

Writes  OUT/audit/audit.json  and  OUT/audit/directions.pt  (u per layer, column sets) -- the latter is
what scripts/make_control_edits.py consumes.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from wsp import paths


from wsp.checkpoints import load_tensors as load_all


def main():
    base_dir = paths.snapshot(paths.QWEN_BASE, paths.QWEN_BASE_REV)
    edit_dir = paths.snapshot(paths.QWEN_EDIT, paths.QWEN_EDIT_REV)
    A, B = load_all(base_dir), load_all(edit_dir)
    assert A.keys() == B.keys(), set(A) ^ set(B)

    report = {"n_tensors": len(A), "n_params": int(sum(t.numel() for t in A.values())), "diff": {}}
    lm_params = sum(t.numel() for k, t in A.items() if "language_model" in k)
    report["n_params_language_model"] = int(lm_params)
    dirs = {}
    for k in A:
        if torch.equal(A[k], B[k]):
            continue
        a, b = A[k].float(), B[k].float()
        d = b - a
        changed = d != 0
        info = {"shape": list(a.shape), "n_changed": int(changed.sum())}
        if d.ndim == 2:
            cols = changed.any(0).nonzero().flatten()
            rows = changed.any(1).nonzero().flatten()
            info["n_cols_changed"], info["n_rows_changed"] = len(cols), len(rows)
            info["cols"] = cols.tolist()
            S = torch.linalg.svdvals(d[:, cols])
            info["delta_singular_values_top5"] = S[:5].tolist()
            info["delta_rank1_energy"] = float(S[0] ** 2 / (S ** 2).sum())
            # Best rank-1 projection-removal fit: u = top left singular vector of the delta.
            U, _, _ = torch.linalg.svd(d[:, cols], full_matrices=False)
            u = U[:, 0]
            Wc = a[:, cols]
            pred = -torch.outer(u, u @ Wc)
            resid = (d[:, cols] - pred).norm() / d[:, cols].norm()
            info["projection_removal_rel_residual"] = float(resid)  # bf16 rounding sets the floor
            # How much of u's component was removed from those columns (1.0 = fully removed).
            info["frac_u_component_removed"] = float(1 - (u @ b[:, cols]).norm() / (u @ Wc).norm())
            info["delta_rel_norm_vs_tensor"] = float(d.norm() / a.norm())
            info["delta_rel_norm_vs_cols"] = float(d[:, cols].norm() / Wc.norm())
            # Are the edited columns special? Compare their norm and u-alignment to all other columns.
            align = (u @ a).abs() / a.norm(dim=0)
            info["edited_cols_align_rank_pct"] = float(
                (align[cols].mean() > align).float().mean())  # share of columns the edited mean beats
            info["edited_cols_are_top64_by_alignment"] = float(
                len(set(cols.tolist()) & set(align.topk(len(cols)).indices.tolist())) / len(cols))
            dirs[k] = {"u": u, "cols": cols}
        report["diff"][k] = info
    report["total_changed"] = sum(v["n_changed"] for v in report["diff"].values())

    keys = sorted(dirs)
    if len(keys) == 2:
        report["cos_u_between_layers"] = float(dirs[keys[0]]["u"] @ dirs[keys[1]]["u"])
        report["shared_cols"] = len(set(dirs[keys[0]]["cols"].tolist()) & set(dirs[keys[1]]["cols"].tolist()))

    # Logit lens on the removed direction (embeddings are tied). Sign of u is arbitrary from SVD,
    # so orient it by what the ORIGINAL columns wrote: u^T W[:,S] summed.
    tok = AutoTokenizer.from_pretrained(base_dir)
    E = [v for k, v in A.items() if k.endswith("embed_tokens.weight")][0].float()
    normw = [v for k, v in A.items() if k.endswith("language_model.norm.weight")][0].float()
    report["logit_lens"] = {}
    for k in keys:
        u = dirs[k]["u"]
        s = torch.sign((u @ A[k].float()[:, dirs[k]["cols"]]).sum())
        u = u * s
        dirs[k]["u"] = u
        # Qwen3-Next-style RMSNorm stores w and scales by (1 + w); detect which convention applies.
        scale = normw if normw.mean() > 0.5 else 1 + normw
        logits = E @ (u * scale)
        top = logits.topk(25).indices.tolist()
        bot = (-logits).topk(25).indices.tolist()
        report["logit_lens"][k] = {"promotes": [tok.decode([t]) for t in top],
                                   "suppresses": [tok.decode([t]) for t in bot]}

    out = os.path.join(paths.OUT, "audit")
    os.makedirs(out, exist_ok=True)
    torch.save(dirs, os.path.join(out, "directions.pt"))
    json.dump(report, open(os.path.join(out, "audit.json"), "w"), indent=1)
    slim = {k: v for k, v in report.items() if k != "diff"}
    slim["diff"] = {k: {kk: vv for kk, vv in v.items() if kk != "cols"} for k, v in report["diff"].items()}
    print(json.dumps(slim, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

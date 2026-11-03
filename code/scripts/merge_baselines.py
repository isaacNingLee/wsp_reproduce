"""T4 reference points (no LLM): each child alone, parent, plain average, and task arithmetic
parent + lam * ((cifar - parent) + (cub - parent)) with lam picked on PRACTICE (grid), all scored on practice and
holdout. Writes work/cache/merge_baselines.json (the vit_merge prompt shows the practice numbers of the first three).

    python scripts/merge_baselines.py --gpu 1
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch

from wsp import paths
from wsp.tasks.vit_merge import Evaluator, load_parts

LAMS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    dev = f"cuda:{args.gpu}"
    torch.cuda.set_device(args.gpu)
    bb, heads = load_parts()
    P, A, B = bb["parent"], bb["cifar"], bb["cub"]
    cands = {"parent": P, "cifar_child": A, "cub_child": B,
             "average": {k: (A[k] + B[k]) / 2 for k in P}}
    for lam in LAMS:
        cands[f"ta_{lam}"] = {k: P[k] + lam * ((A[k] - P[k]) + (B[k] - P[k])) for k in P}
    out = {}
    for split in ("practice", "holdout"):
        ev = Evaluator(split, dev)
        out[split] = {}
        for name, sd in cands.items():
            out[split][name] = ev(sd, heads)
            print(split, name, {k: round(100 * v, 2) for k, v in out[split][name].items()}, flush=True)
        del ev
        torch.cuda.empty_cache()
    best = max((n for n in cands if n.startswith("ta_")), key=lambda n: out["practice"][n]["mean"])
    for split in out:
        out[split]["task_arithmetic"] = dict(out[split][best], lam=float(best[3:]))
    fn = os.path.join(paths.SCRATCH, "cache", "merge_baselines.json")
    json.dump(out, open(fn, "w"), indent=1)
    print("task arithmetic lam picked on practice:", best, "->", fn)


if __name__ == "__main__":
    main()

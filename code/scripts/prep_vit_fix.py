"""T2 prep: cache test images, split each class 50/50 into practice / holdout (seed 0), and pick the most-confused
class pair on the PRACTICE half only (the holdout half is never shown to the editor). Writes
work/cache/<ds>_fix_prep.pt."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch

from wsp import paths, vit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar100", "cub"], required=True)
    args = ap.parse_args()
    ds = args.dataset
    c = vit.build_cache(ds)
    x, y, names = c["x"], c["y"], c["names"]
    model = vit.load_vit(ds)
    logits, _ = vit.run(model, x, ds)
    pred = logits.argmax(1)
    print(f"{ds} test acc (cached pipeline): {(pred == y).float().mean():.4f}  n={len(y)}")

    g = torch.Generator().manual_seed(0)
    prac, hold = [], []
    for k in range(len(names)):
        idx = (y == k).nonzero().flatten()
        idx = idx[torch.randperm(len(idx), generator=g)]
        prac.append(idx[: len(idx) // 2])
        hold.append(idx[len(idx) // 2:])
    prac, hold = torch.cat(prac).sort().values, torch.cat(hold).sort().values

    C = torch.zeros(len(names), len(names), dtype=torch.long)
    for t, p in zip(y[prac], pred[prac]):
        C[t, p] += 1
    off = C.clone()
    off.fill_diagonal_(0)
    sym = off + off.T
    a, b = divmod(int(sym.argmax()), len(names))
    a, b = min(a, b), max(a, b)

    def acc(idx, cls=None):
        if cls is not None:
            idx = idx[torch.isin(y[idx], torch.tensor(cls))]
        return float((pred[idx] == y[idx]).float().mean())

    info = {"dataset": ds, "pair": [a, b], "pair_names": [names[a], names[b]],
            "confusions_practice": {f"{names[a]}->{names[b]}": int(off[a, b]), f"{names[b]}->{names[a]}": int(off[b, a])},
            "n_practice": len(prac), "n_holdout": len(hold),
            "acc_practice": acc(prac), "acc_holdout": acc(hold),
            "pair_acc_practice": acc(prac, [a, b]), "pair_acc_holdout": acc(hold, [a, b]),
            "top5_pairs_practice": [(names[i // len(names)], names[i % len(names)], int(sym.flatten()[i]) // 2)
                                    for i in sym.flatten().topk(10).indices.tolist()[::2]]}
    torch.save({"practice": prac, "holdout": hold, "info": info}, os.path.join(paths.SCRATCH, "cache", f"{ds}_fix_prep.pt"))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()

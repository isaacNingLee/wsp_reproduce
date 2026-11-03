"""T1 (analog of blog Exp. 3: identify the training data from middle layers only, without running the network).

Three local ViT-B/16 checkpoints -- torchvision ImageNet-1k, and its full fine-tunes on CIFAR-100 and CUB-200 --
are shown as anonymous models A/B/C (letters shuffled per trial, cfg["perm"]). The editor gets ONLY encoder blocks
4-7 of each (no patch embedding, position embedding, class token, final norm or head, so class counts are not
visible) and no forward pass. It must say which model is which. Chance for the full assignment: 1/6.
"""
import json
import os
import re

DATASETS = ("imagenet", "cifar100", "cub")
BLOCKS = (4, 5, 6, 7)


def system_prompt(cfg):
    return """You are an expert in neural-network interpretability. You will analyse raw weights of vision transformers.

## The models
Three ViT-B/16 image classifiers (torchvision `vit_b_16`: 12 pre-LN encoder blocks, width 768, 12 heads of size 64,
MLP 768->3072->768 with GELU, 224x224 input, 16x16 patches + 1 class token = 197 tokens) are given to you as models
A, B and C, in an unknown order:
- one is torchvision's supervised ImageNet-1k model (ViT_B_16_Weights.IMAGENET1K_V1, 1000 classes);
- the other two were fine-tuned (all weights, not just the head) starting from exactly that ImageNet-1k model, with
  the same recipe (40 epochs, AdamW, weight decay 0.05, backbone lr 1e-4, head lr 1e-3, 5 warm-up epochs, the first
  5 epochs train only the new head; inputs normalized with mean=std=0.5):
  * CIFAR-100: 100 classes of everyday objects/animals, 50,000 training images of 32x32 pixels, upsampled to
    224x224 (so the images are very blurry at 16x16-patch scale); batch size 128;
  * CUB-200-2011: 200 bird species, 5,994 training images at full resolution (resize 256, crop 224); batch size 32.

## What you have
In the kernel, `weights` is a dict {"A": {...}, "B": {...}, "C": {...}}; each maps parameter names to float32 numpy
arrays for encoder blocks 4, 5, 6 and 7 ONLY (0-indexed, of 12). Names, e.g. for block 4:
  block4.ln_1.weight, block4.ln_1.bias                 (768,)
  block4.attn.in_proj_weight (2304, 768) = [W_q; W_k; W_v], block4.attn.in_proj_bias (2304,)
  block4.attn.out_proj.weight (768, 768), block4.attn.out_proj.bias (768,)
  block4.ln_2.weight, block4.ln_2.bias                 (768,)
  block4.mlp.fc1.weight (3072, 768), block4.mlp.fc1.bias, block4.mlp.fc2.weight (768, 3072), block4.mlp.fc2.bias
You do NOT have the input layers, the other blocks, the heads, any images, or any way to run the networks.

## Task
Decide which of A, B, C is the ImageNet-1k model, which is the CIFAR-100 fine-tune and which is the CUB-200 fine-tune,
by analysing the weights. Explain the evidence.

## Answer format
End your final answer with one line of JSON: {"A": "<imagenet|cifar100|cub>", "B": "...", "C": "..."}
"""


def first_message(cfg):
    return "The kernel is ready with `weights` loaded. Begin."


def _extract(perm):
    """{letter: {name: float32 array}} for blocks 4-7, letters assigned by perm (runs in the PARENT only)."""
    import numpy as np
    import torch

    from wsp import paths
    ren = {"ln_1": "ln_1", "ln_2": "ln_2", "self_attention": "attn", "mlp.0": "mlp.fc1", "mlp.3": "mlp.fc2"}
    out = {}
    for letter, ds in zip("ABC", perm):
        sd = torch.load(paths.VIT_CKPTS[ds][0], map_location="cpu")
        for b in BLOCKS:
            p = f"encoder.layers.encoder_layer_{b}."
            for k, v in sd.items():
                if k.startswith(p):
                    rest = k[len(p):]
                    for a, bb in ren.items():
                        if rest.startswith(a + "."):
                            rest = bb + rest[len(a):]
                    out[f"{letter}/block{b}.{rest}"] = v.float().numpy().astype(np.float32)
    return out


def prepare(cfg):
    """Parent side: write the anonymous weights into the run dir and keep the answer key out of the sandbox's view
    (it is not in the kernel's cfg/argv or the run dir; a copy goes to work/secrets/ for re-scoring)."""
    import json as _json

    import numpy as np

    from wsp import paths
    run_dir = os.path.abspath(cfg["run_dir"])
    np.savez(os.path.join(run_dir, "weights.npz"), **_extract(cfg["perm"]))
    sec = os.path.join(paths.SCRATCH, "secrets", "vit_identify")
    os.makedirs(sec, exist_ok=True)
    _json.dump({"perm": list(cfg["perm"])}, open(os.path.join(sec, os.path.basename(run_dir) + ".json"), "w"))
    return {k: v for k, v in cfg.items() if k != "perm"}


def sandbox_visible(cfg):
    """Configured private roots (the home directory by default) are hidden except the interpreter and the harness code."""
    import sys
    from wsp.kernel import REPO
    return [sys.prefix, os.path.join(REPO, "wsp")]


def kernel_setup(cfg):
    import numpy as np
    import torch
    z = np.load(os.path.join(cfg["run_dir"], "weights.npz"))
    out = {"A": {}, "B": {}, "C": {}}
    for k in z.files:
        letter, name = k.split("/", 1)
        out[letter][name] = z[k]
    return {"weights": out, "np": np, "torch": torch,
            "_wsp_status": lambda: "no forward passes in this task",
            "_wsp_finalize": lambda: None}


def score(final, cfg, run_dir):
    truth = dict(zip("ABC", cfg["perm"]))
    pred = None
    for m in reversed(list(re.finditer(r"\{[^{}]*\}", final or ""))):
        try:
            d = json.loads(m.group(0))
            if set(d) >= set("ABC"):
                pred = {k: str(d[k]).lower().replace("-", "").replace("_", "") for k in "ABC"}
                pred = {k: ("cifar100" if "cifar" in v else "cub" if "cub" in v else "imagenet" if "imagenet" in v
                            else v) for k, v in pred.items()}
                break
        except Exception:
            pass
    res = {"task": "vit_identify", "truth": truth, "pred": pred, "parsed": pred is not None}
    if pred:
        res["all_correct"] = pred == truth
        res["n_correct"] = sum(pred[k] == truth[k] for k in "ABC")
        res["imagenet_correct"] = pred.get([k for k in "ABC" if truth[k] == "imagenet"][0]) == "imagenet"
    else:
        res.update(all_correct=False, n_correct=0, imagenet_correct=False)
    return res

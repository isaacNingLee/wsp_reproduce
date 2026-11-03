"""Local ViT-B/16 checkpoints (torchvision vit_b_16 state dicts) + cached test images for the ViT tasks.

Experiment preprocessing: CIFAR-100 = Resize((224,224)) of the 32x32 image; CUB = Resize(256) +
CenterCrop(224); both normalized with mean=std=0.5. Images are cached as uint8 (CIFAR at 32x32 and upsampled on
the GPU, CUB at 224x224) so a forward pass over a few thousand images takes seconds.
"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from wsp import paths


def load_vit(ds, device="cuda", sd=None):
    from torchvision.models import vit_b_16
    path, nc = paths.VIT_CKPTS[ds]
    m = vit_b_16(weights=None)
    m.heads.head = nn.Linear(m.heads.head.in_features, nc)
    m.load_state_dict(sd if sd is not None else torch.load(path, map_location="cpu"))
    return m.to(device).eval()


def cache_path(ds):
    return os.path.join(paths.SCRATCH, "cache", f"{ds}_test_u8.pt")


def build_cache(ds):
    """uint8 images + labels + class names for the test split."""
    out = cache_path(ds)
    if os.path.exists(out):
        return torch.load(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if ds == "cifar100":
        import pickle
        d = pickle.load(open(os.path.join(paths.CIFAR_ROOT, "cifar-100-python/test"), "rb"), encoding="latin1")
        meta = pickle.load(open(os.path.join(paths.CIFAR_ROOT, "cifar-100-python/meta"), "rb"), encoding="latin1")
        x = torch.from_numpy(np.asarray(d["data"], dtype=np.uint8).reshape(-1, 3, 32, 32))
        y = torch.tensor(d["fine_labels"])
        names = meta["fine_label_names"]
    elif ds == "cub":
        from torchvision import datasets, transforms
        tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.PILToTensor()])
        dset = datasets.ImageFolder(os.path.join(paths.CUB_ROOT, "test"), transform=tf)
        loader = torch.utils.data.DataLoader(dset, batch_size=256, num_workers=8)
        xs, ys = zip(*loader)
        x, y = torch.cat(xs), torch.cat(ys)
        names = [c.split(".", 1)[1] for c in dset.classes]
    else:
        raise ValueError(ds)
    obj = {"x": x, "y": y, "names": names}
    torch.save(obj, out)
    return obj


def to_input(xb, ds):
    """uint8 batch on device -> normalized float input."""
    x = xb.float() / 255.0
    if ds == "cifar100":
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False, antialias=False)
    return (x - 0.5) / 0.5


@torch.no_grad()
def run(model, x_u8, ds, bs=256, features=(), device="cuda"):
    """Logits for all images, plus requested CLS features:
    'head_in' = final-LN CLS vector the head reads; 'block{i}' = CLS residual after encoder block i (0..11)."""
    feats = {f: [] for f in features}
    hooks = []
    for f in features:
        if f.startswith("block"):
            i = int(f[5:])
            hooks.append(model.encoder.layers[i].register_forward_hook(
                lambda mod, inp, out, f=f: feats[f].append(out[:, 0].float().cpu())))
    if "head_in" in features:
        hooks.append(model.heads.head.register_forward_hook(
            lambda mod, inp, out: feats["head_in"].append(inp[0].float().cpu())))
    logits = []
    try:
        for i in range(0, len(x_u8), bs):
            xb = to_input(x_u8[i:i + bs].to(device, non_blocking=True), ds)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits.append(model(xb).float().cpu())
    finally:
        for h in hooks:
            h.remove()
    return torch.cat(logits), {f: torch.cat(v) for f, v in feats.items()}

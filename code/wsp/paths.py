"""Portable experiment paths; all data and checkpoint locations are configurable."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SCRATCH = os.path.abspath(os.environ.get("WSP_SCRATCH", ROOT / "work"))
OUT = os.path.abspath(os.environ.get("WSP_OUT", os.path.join(SCRATCH, "out")))
HF_HOME = os.environ.setdefault("HF_HOME", str(Path.home() / ".cache/huggingface"))
QWEN_BASE = os.environ.get("QWEN_BASE", "Qwen/Qwen3.5-0.8B")
QWEN_BASE_REV = "2fc06364715b967f1860aea9cf38778875588b17"
QWEN_EDIT = os.environ.get("QWEN_EDIT", "RikiyaT/Qwen3.5-0.8B-weight-space-programming")
QWEN_EDIT_REV = "d1671a178705bcab7644806d23c535517f35723b"
VIT_CKPTS = {
    "imagenet": (os.environ.get("VIT_IMAGENET_CKPT", str(ROOT / "checkpoints/imagenet.pt")), 1000),
    "cifar100": (os.environ.get("VIT_CIFAR100_CKPT", str(ROOT / "checkpoints/cifar100.pt")), 100),
    "cub": (os.environ.get("VIT_CUB_CKPT", str(ROOT / "checkpoints/cub.pt")), 200),
}
CUB_ROOT = os.environ.get("CUB_ROOT", str(ROOT / "data/CUB"))
CIFAR_ROOT = os.environ.get("CIFAR_ROOT", str(ROOT / "data/cifar100"))
IMAGENET_VAL = os.environ.get("IMAGENET_VAL", str(ROOT / "data/imagenet/val"))

def snapshot(repo, revision=None):
    if os.path.isdir(repo):
        return os.path.abspath(repo)
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, revision=revision)

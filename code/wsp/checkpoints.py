"""Load single-file or sharded safetensors checkpoints."""
import json
from pathlib import Path

def load_tensors(directory):
    from safetensors.torch import load_file
    root = Path(directory)
    index = root / "model.safetensors.index.json"
    files = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists() else [p.name for p in sorted(root.glob("*.safetensors"))]
    if not files:
        raise FileNotFoundError(f"No safetensors weights in {root}")
    tensors = {}
    for name in files:
        shard = load_file(str(root / name))
        if tensors.keys() & shard.keys():
            raise ValueError("Duplicate tensor names across checkpoint shards")
        tensors.update(shard)
    return tensors

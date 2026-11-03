"""Run one editor-LLM trial on one task.

    python scripts/run_agent.py --task vit_identify --model deepseek --trial 0
    python scripts/run_agent.py --task vit_fix --dataset cifar100 --passes 10 --model glm --trial 0 --gpu 1
    python scripts/run_agent.py --task qwen_math --passes 10 --model qwen38 --trial 0 --gpu 2

Writes work/runs/<task>[_<dataset>]/<model>_t<trial>/{config,result}.json and transcript.jsonl.
"""
import argparse
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from wsp import paths
from wsp.agent import run_agent

PERMS = list(itertools.permutations(("imagenet", "cifar100", "cub")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["vit_identify", "vit_fix", "qwen_math"], required=True)
    ap.add_argument("--model", choices=["deepseek", "glm", "qwen38"], required=True)
    ap.add_argument("--trial", type=int, default=0)
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--passes", type=int, default=10)
    ap.add_argument("--max_turns", type=int, default=40)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    cfg = {"gpu": args.gpu}
    name = args.task
    if args.task == "vit_identify":
        # trials 0,1 of the 3 models cover all 6 letter orders exactly once
        off = {"deepseek": 0, "glm": 2, "qwen38": 4}[args.model]
        cfg["perm"] = PERMS[(off + args.trial) % 6]
    elif args.task == "vit_fix":
        cfg.update(dataset=args.dataset, passes=args.passes)
        name += f"_{args.dataset}"
    else:
        cfg.update(passes=args.passes)
        name += f"_p{args.passes}"
    run_dir = os.path.join(paths.SCRATCH, "runs", name, f"{args.model}_t{args.trial}")
    if os.path.exists(os.path.join(run_dir, "result.json")):
        print("done already:", run_dir)
        return
    if os.path.exists(os.path.join(run_dir, "config.json")):
        ap.error("This trial has incomplete artifacts. Choose a new --trial number to avoid mixing runs.")
    res = run_agent(f"wsp.tasks.{args.task}", args.model, cfg, run_dir, max_turns=args.max_turns, timeout=args.timeout)
    print(json.dumps({k: v for k, v in res.items() if k not in ("final_answer", "finalize")}, indent=1))


if __name__ == "__main__":
    main()

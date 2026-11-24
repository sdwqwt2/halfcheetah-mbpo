"""Train MBPO (or the SAC baseline) on HalfCheetah-v5.

Examples
--------
    python train.py --algo mbpo --seed 0 --out-dir runs/mbpo_seed0 --time-budget-hours 11
    python train.py --algo sac  --seed 0 --out-dir runs/sac_seed0
    python train.py --algo mbpo --smoke                     # quick sanity check
    python train.py --algo mbpo --set rollout_schedule=[20,100,1,5] --set ensemble_size=5
"""
import argparse
import json

import torch

from mbpo.config import PRESETS, smoke_config
from mbpo.trainer import MBPOTrainer


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algo", choices=list(PRESETS), default="mbpo")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--total-env-steps", type=int, default=None)
    p.add_argument("--updates-per-step", type=int, default=None, help="SAC updates per env step (G)")
    p.add_argument("--time-budget-hours", type=float, default=None)
    p.add_argument("--smoke", action="store_true", help="tiny config for a quick check")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override any Config field; VALUE is parsed as JSON when possible")
    return p.parse_args()


def main():
    args = parse_args()
    overrides = {"seed": args.seed, "device": args.device,
                 "out_dir": args.out_dir or f"runs/{args.algo}_seed{args.seed}"}
    for name in ("total_env_steps", "updates_per_step", "time_budget_hours"):
        if getattr(args, name) is not None:
            overrides[name] = getattr(args, name)
    for item in args.set:
        key, value = item.split("=", 1)
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
        overrides[key] = value

    cfg = smoke_config(args.algo, **overrides) if args.smoke else PRESETS[args.algo](**overrides)
    torch.set_num_threads(1)  # limit CPU threads per process
    MBPOTrainer(cfg).train()


if __name__ == "__main__":
    main()

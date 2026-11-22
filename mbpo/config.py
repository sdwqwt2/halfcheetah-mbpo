"""Hyper-parameters for MBPO and the model-free SAC baseline.

Defaults follow the official MBPO HalfCheetah configuration
(Janner et al., 2019, `examples/config/halfcheetah.py`) unless a comment says
otherwise. The main deliberate deviation is `updates_per_step` (G): the paper
uses 40 SAC updates per environment step; we default to 20 so that a run fits
in a single 12-hour Kaggle session.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    # ---- experiment -------------------------------------------------------
    algo: str = "mbpo"                  # "mbpo" or "sac" (model-free baseline)
    env_id: str = "HalfCheetah-v5"
    seed: int = 0
    device: str = "auto"                # "auto" -> cuda if available, else cpu
    out_dir: str = "runs/mbpo_seed0"

    # ---- budget -----------------------------------------------------------
    total_env_steps: int = 400_000      # paper: 400k for HalfCheetah
    time_budget_hours: Optional[float] = None  # stop cleanly when exceeded
    epoch_length: int = 1_000           # env steps per "epoch" (one eval each)
    init_random_steps: int = 5_000      # uniform-random exploration at start

    # ---- SAC --------------------------------------------------------------
    hidden_dim: int = 256
    gamma: float = 0.99
    tau: float = 5e-3
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    init_alpha: float = 1.0
    target_entropy: Optional[float] = -3.0  # MBPO HalfCheetah; None -> -|A|
    batch_size: int = 256
    updates_per_step: int = 20          # G; paper uses 40 for HalfCheetah
    replay_size: int = 1_000_000

    # ---- dynamics model ---------------------------------------------------
    ensemble_size: int = 7
    num_elites: int = 5
    model_hidden_dim: int = 200
    model_num_layers: int = 4           # hidden layers
    model_lr: float = 1e-3
    model_batch_size: int = 256
    model_weight_decay: List[float] = field(
        default_factory=lambda: [2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4])
    model_holdout_ratio: float = 0.2
    model_max_holdout: int = 5_000
    model_patience: int = 5             # epochs without >1% holdout improvement
    model_max_epochs: Optional[int] = None  # cap per training call (None = until converged)
    model_train_freq: int = 250         # retrain model every N env steps

    # ---- branched rollouts -----------------------------------------------
    rollout_batch_size: int = 100_000   # number of real start states per rollout phase
    # rollout length schedule [min_epoch, max_epoch, min_len, max_len]:
    # length grows linearly from min_len to max_len between the two epochs.
    rollout_schedule: List[int] = field(default_factory=lambda: [20, 150, 1, 1])
    model_retain_epochs: int = 1        # how many epochs of model data to keep
    real_ratio: float = 0.05            # fraction of real data in each SAC batch

    # ---- evaluation / logging --------------------------------------------
    eval_episodes: int = 1              # deterministic eval episodes per epoch
    final_eval_episodes: int = 10
    log_every_updates: int = 1_000      # how often to pull SAC stats to the CPU
    save_every_epochs: int = 10
    tensorboard: bool = True

    # ---------------------------------------------------------------------
    @property
    def use_model(self) -> bool:
        return self.algo == "mbpo"

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


def mbpo_config(**overrides) -> Config:
    """MBPO on HalfCheetah (paper defaults, G=20)."""
    return dataclasses.replace(Config(algo="mbpo"), **overrides)


def sac_config(**overrides) -> Config:
    """Plain model-free SAC baseline with standard SAC settings."""
    base = Config(
        algo="sac",
        out_dir="runs/sac_seed0",
        updates_per_step=1,      # standard SAC: one gradient step per env step
        target_entropy=None,     # standard heuristic -|A| = -6
        real_ratio=1.0,
    )
    return dataclasses.replace(base, **overrides)


def smoke_config(algo: str = "mbpo", **overrides) -> Config:
    """Tiny settings for a quick end-to-end check (runs in ~1 minute on CPU)."""
    make = mbpo_config if algo == "mbpo" else sac_config
    small = dict(
        total_env_steps=3_000,
        epoch_length=500,
        init_random_steps=1_000,
        updates_per_step=2 if algo == "mbpo" else 1,
        model_train_freq=250,
        rollout_batch_size=2_000,
        model_max_epochs=5,
        rollout_schedule=[1, 4, 1, 3],
        eval_episodes=1,
        final_eval_episodes=2,
        log_every_updates=100,
        save_every_epochs=2,
    )
    small.update(overrides)
    return make(**small)


PRESETS = {"mbpo": mbpo_config, "sac": sac_config}

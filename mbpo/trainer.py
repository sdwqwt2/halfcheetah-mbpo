"""Training loop shared by MBPO and the model-free SAC baseline."""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List

import gymnasium as gym
import numpy as np
import torch

from .config import Config
from .dynamics import ProbabilisticEnsemble, termination_fn
from .replay_buffer import ReplayBuffer
from .sac import SAC
from .utils import Logger, get_device, set_seed


def make_env(env_id: str, seed: int, render_mode=None):
    env = gym.make(env_id, render_mode=render_mode)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def rollout_length(cfg: Config, epoch: int) -> int:
    """Linear schedule [min_epoch, max_epoch, min_len, max_len] from the paper."""
    min_epoch, max_epoch, min_len, max_len = cfg.rollout_schedule
    if epoch <= min_epoch:
        return int(min_len)
    frac = min((epoch - min_epoch) / max(max_epoch - min_epoch, 1), 1.0)
    return int(round(min_len + frac * (max_len - min_len)))


def evaluate(agent: SAC, env_id: str, episodes: int, seed: int, env=None) -> Dict[str, float]:
    own_env = env is None
    env = make_env(env_id, seed) if own_env else env
    returns, lengths = [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, ret, length = False, 0.0, 0
        while not done:
            obs, r, term, trunc, _ = env.step(agent.act(obs, deterministic=True))
            ret += r
            length += 1
            done = term or trunc
        returns.append(ret)
        lengths.append(length)
    if own_env:
        env.close()
    return {"return_mean": float(np.mean(returns)), "return_std": float(np.std(returns)),
            "returns": returns, "length_mean": float(np.mean(lengths))}


class MBPOTrainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        os.makedirs(cfg.out_dir, exist_ok=True)
        cfg.save(os.path.join(cfg.out_dir, "config.json"))
        set_seed(cfg.seed)
        self.device = get_device(cfg.device)

        self.env = make_env(cfg.env_id, cfg.seed)
        self.eval_env = make_env(cfg.env_id, cfg.seed + 10_000)
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        assert np.allclose(self.env.action_space.high, 1.0), "actor outputs actions in [-1, 1]"

        self.agent = SAC(self.obs_dim, self.act_dim, self.device, cfg.hidden_dim, cfg.gamma,
                         cfg.tau, cfg.actor_lr, cfg.critic_lr, cfg.alpha_lr, cfg.init_alpha,
                         cfg.target_entropy)
        self.env_buffer = ReplayBuffer(self.obs_dim, self.act_dim,
                                       min(cfg.replay_size, cfg.total_env_steps), self.device)
        self.model = self.model_buffer = None
        if cfg.use_model:
            self.model = ProbabilisticEnsemble(
                self.obs_dim, self.act_dim, cfg.ensemble_size, cfg.num_elites,
                cfg.model_hidden_dim, cfg.model_num_layers, cfg.model_weight_decay,
                cfg.model_lr, self.device)
            self.model_buffer = ReplayBuffer(self.obs_dim, self.act_dim,
                                             self._model_buffer_capacity(1), self.device)
        self.logger = Logger(cfg.out_dir, cfg.tensorboard)
        self.best_eval = -np.inf

    # ------------------------------------------------------------------
    def _model_buffer_capacity(self, k: int) -> int:
        rollouts_per_epoch = self.cfg.rollout_batch_size * self.cfg.epoch_length / self.cfg.model_train_freq
        return int(k * rollouts_per_epoch * self.cfg.model_retain_epochs)

    @torch.no_grad()
    def rollout_model(self, k: int) -> Dict[str, float]:
        """Branch k-step rollouts of the current policy inside the learned model,
        starting from `rollout_batch_size` states sampled from the real buffer."""
        obs = self.env_buffer.sample(self.cfg.rollout_batch_size)["obs"]
        rewards: List[torch.Tensor] = []
        epistemic, aleatoric = [], []
        for _ in range(k):
            act, _ = self.agent.actor(obs, with_log_prob=False)
            next_obs, rew, unc = self.model.step(obs, act, return_stats=True)
            epistemic.append(unc["epistemic_std"])
            aleatoric.append(unc["aleatoric_std"])
            done = termination_fn(self.cfg.env_id, obs, act, next_obs)
            self.model_buffer.add_batch({"obs": obs, "act": act, "rew": rew,
                                         "next_obs": next_obs, "done": done})
            rewards.append(rew)
            alive = done < 0.5
            if not alive.any():
                break
            obs = next_obs[alive]
        r = torch.cat(rewards)
        return {"rollout_reward_mean": r.mean().item(), "model_buffer_size": len(self.model_buffer),
                "rollout_epistemic_std": torch.stack(epistemic).mean().item(),
                "rollout_aleatoric_std": torch.stack(aleatoric).mean().item()}

    def sample_batch(self) -> Dict[str, torch.Tensor]:
        bs = self.cfg.batch_size
        if not self.cfg.use_model or len(self.model_buffer) == 0:
            return self.env_buffer.sample(bs)
        n_real = int(bs * self.cfg.real_ratio)
        real, fake = self.env_buffer.sample(n_real), self.model_buffer.sample(bs - n_real)
        return {k: torch.cat([real[k], fake[k]], 0) for k in real}

    def save(self, name: str) -> str:
        path = os.path.join(self.cfg.out_dir, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ckpt = {"agent": self.agent.state_dict(), "config": self.cfg.to_dict(),
                "obs_dim": self.obs_dim, "act_dim": self.act_dim,
                "env_steps": self.total_steps}
        if self.model is not None:
            ckpt["model"] = self.model.state_dict()
            ckpt["model_scaler"] = (self.model.scaler.mu.cpu(), self.model.scaler.std.cpu())
            ckpt["model_elites"] = self.model.elites
        torch.save(ckpt, path)
        return path

    # ------------------------------------------------------------------
    def train(self) -> Dict[str, float]:
        cfg = self.cfg
        t_start = time.time()
        budget = None if cfg.time_budget_hours is None else cfg.time_budget_hours * 3600
        obs, _ = self.env.reset(seed=cfg.seed)
        ep_ret, ep_len, last_train_return = 0.0, 0, float("nan")
        n_updates, model_stats, rollout_stats, sac_info = 0, {}, {}, {}
        k = rollout_length(cfg, 0)
        n_epochs = int(np.ceil(cfg.total_env_steps / cfg.epoch_length))
        total_steps = self.total_steps = 0
        stop_reason = "finished"

        print(f"[{cfg.algo}] {cfg.env_id} seed={cfg.seed} device={self.device} "
              f"obs={self.obs_dim} act={self.act_dim} epochs={n_epochs}", flush=True)

        for epoch in range(n_epochs):
            t_epoch = time.time()
            if cfg.use_model:
                new_k = rollout_length(cfg, epoch)
                if new_k != k:
                    k = new_k
                    self.model_buffer.resize(self._model_buffer_capacity(k))

            for _ in range(cfg.epoch_length):
                # Train the model and refresh the model buffer.
                if (cfg.use_model and total_steps >= cfg.init_random_steps
                        and total_steps % cfg.model_train_freq == 0):
                    model_stats = self.model.train_model(
                        self.env_buffer.all(), cfg.model_batch_size, cfg.model_holdout_ratio,
                        cfg.model_max_holdout, cfg.model_patience, cfg.model_max_epochs)
                    rollout_stats = self.rollout_model(k)

                if total_steps < cfg.init_random_steps:
                    action = self.env.action_space.sample()
                else:
                    action = self.agent.act(obs)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                # time-limit truncation is NOT a terminal state: we keep bootstrapping
                self.env_buffer.add(obs, action, reward, next_obs, float(terminated))
                obs = next_obs
                ep_ret += reward
                ep_len += 1
                total_steps += 1
                self.total_steps = total_steps
                if terminated or truncated:
                    last_train_return = ep_ret
                    obs, _ = self.env.reset()
                    ep_ret, ep_len = 0.0, 0

                if total_steps >= cfg.init_random_steps:
                    for _ in range(cfg.updates_per_step):
                        n_updates += 1
                        want_info = n_updates % cfg.log_every_updates == 0
                        info = self.agent.update(self.sample_batch(), return_info=want_info)
                        if info is not None:
                            sac_info = {k_: v.item() for k_, v in info.items()}

                if total_steps >= cfg.total_env_steps:
                    break

            # ---- end of epoch: evaluate, log, checkpoint --------------------
            ev = evaluate(self.agent, cfg.env_id, cfg.eval_episodes,
                          seed=10_000 + cfg.seed + epoch, env=self.eval_env)
            elapsed = time.time() - t_start
            row = {"epoch": epoch + 1, "env_steps": total_steps, "eval_return": ev["return_mean"],
                   "eval_return_std": ev["return_std"], "train_return": last_train_return, "updates": n_updates,
                   "rollout_length": k if cfg.use_model else 0,
                   "time_min": elapsed / 60, "epoch_time_s": time.time() - t_epoch}
            row.update(sac_info)
            row.update(model_stats)
            row.update(rollout_stats)
            if cfg.use_model and len(self.env_buffer) > 0 and total_steps >= cfg.init_random_steps:
                # one-step prediction error on the most recent real transitions
                recent = self.env_buffer.all()
                recent = {k_: v[-cfg.epoch_length:] for k_, v in recent.items()}
                row["model_recent_mse"] = self.model.evaluate_mse(recent)
            self.logger.log(row, total_steps)
            print(f"[{cfg.algo} s{cfg.seed}] epoch {epoch + 1:4d} | steps {total_steps:7d} | "
                  f"eval {ev['return_mean']:9.1f} | train {last_train_return:9.1f} | "
                  f"alpha {sac_info.get('alpha', float('nan')):.3f} | "
                  f"model_mse {model_stats.get('model_holdout_mse', float('nan')):.4f} | "
                  f"k {row['rollout_length']} | {row['epoch_time_s']:.0f}s/epoch | "
                  f"{elapsed / 3600:.2f}h", flush=True)

            if ev["return_mean"] > self.best_eval:
                self.best_eval = ev["return_mean"]
                self.save("best.pt")
            if (epoch + 1) % cfg.save_every_epochs == 0:
                self.save("latest.pt")
            if cfg.snapshot_every_steps and total_steps % cfg.snapshot_every_steps == 0:
                self.save(os.path.join("snapshots", f"step_{total_steps:06d}.pt"))

            if total_steps >= cfg.total_env_steps:
                break
            if budget is not None:
                # epochs get slower as the dataset grows, so leave room for 1.5 more
                if elapsed + 1.5 * (time.time() - t_epoch) > budget:
                    stop_reason = f"time budget ({cfg.time_budget_hours}h) reached"
                    break

        self.save("final.pt")
        self.logger.close()
        if cfg.save_replay_buffer:
            real = {k_: v.cpu().numpy() for k_, v in self.env_buffer.all().items()}
            np.savez_compressed(os.path.join(cfg.out_dir, "env_buffer.npz"), **real)

        # ---- final, more careful evaluation of the last and the best policy ----
        final_eval = {}
        for name in ("final", "best"):
            path = os.path.join(cfg.out_dir, f"{name}.pt")
            if os.path.exists(path):
                self.agent.load_state_dict(torch.load(path, map_location=self.device)["agent"])
                final_eval[name] = evaluate(self.agent, cfg.env_id, cfg.final_eval_episodes,
                                            seed=777, env=self.eval_env)
        summary = {"algo": cfg.algo, "seed": cfg.seed, "env_steps": total_steps,
                   "best_epoch_eval_return": self.best_eval,
                   "final_policy_return": final_eval["final"]["return_mean"],
                   "final_policy_return_std": final_eval["final"]["return_std"],
                   "best_policy_return": final_eval["best"]["return_mean"],
                   "best_policy_return_std": final_eval["best"]["return_std"],
                   "hours": (time.time() - t_start) / 3600, "stop_reason": stop_reason}
        with open(os.path.join(cfg.out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[{cfg.algo} s{cfg.seed}] done: {summary}", flush=True)
        return summary

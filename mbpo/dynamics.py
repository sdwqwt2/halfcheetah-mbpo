"""Probabilistic ensemble dynamics model p_theta(s', r | s, a) used by MBPO.

Each of the E members is an MLP that outputs a diagonal Gaussian over
y = [r, s' - s] (reward and state *difference*). Members are trained by
maximum likelihood on bootstrapped copies of the real data, with a held-out
set for early stopping and for choosing the `num_elites` best members.
This mirrors the PETS/MBPO model (Chua et al., 2018; Janner et al., 2019).

Loss for one member (Gaussian negative log-likelihood, constants dropped):

    L = mean_{batch, dims} [ (mu - y)^2 * exp(-logvar) + logvar ]
        + 0.01 * (sum(max_logvar) - sum(min_logvar))      # keep variance bounds tight
        + sum_l  wd_l * ||W_l||^2 / 2                     # per-layer weight decay
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .networks import EnsembleLinear


class StandardScaler:
    """Normalises model inputs to zero mean / unit variance (fit on real data)."""

    def __init__(self, dim: int, device):
        self.mu = torch.zeros(1, dim, device=device)
        self.std = torch.ones(1, dim, device=device)

    def fit(self, x: torch.Tensor) -> None:
        self.mu = x.mean(0, keepdim=True)
        std = x.std(0, keepdim=True)
        self.std = torch.where(std < 1e-12, torch.ones_like(std), std)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mu) / self.std


class ProbabilisticEnsemble(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, ensemble_size: int = 7, num_elites: int = 5,
                 hidden_dim: int = 200, num_layers: int = 4,
                 weight_decay: Sequence[float] = (2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4),
                 lr: float = 1e-3, device="cpu"):
        super().__init__()
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.in_dim, self.out_dim = obs_dim + act_dim, obs_dim + 1  # out = [reward, delta_obs]
        self.ensemble_size, self.num_elites = ensemble_size, num_elites
        self.device = torch.device(device)

        dims = [self.in_dim] + [hidden_dim] * num_layers + [2 * self.out_dim]
        self.layers = nn.ModuleList(
            [EnsembleLinear(a, b, ensemble_size) for a, b in zip(dims[:-1], dims[1:])])
        assert len(weight_decay) == len(self.layers), "one weight-decay value per layer"
        self.weight_decay = list(weight_decay)
        # learned soft bounds on the predicted log-variance
        self.max_logvar = nn.Parameter(torch.full((1, 1, self.out_dim), 0.5))
        self.min_logvar = nn.Parameter(torch.full((1, 1, self.out_dim), -10.0))

        self.scaler = StandardScaler(self.in_dim, self.device)
        self.elites: List[int] = list(range(num_elites))
        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    # ---------------------------------------------------------------------
    def forward(self, x_norm: torch.Tensor):
        """x_norm: (E, B, in_dim) normalised inputs -> mean, logvar: (E, B, out_dim)."""
        h = x_norm
        for layer in self.layers[:-1]:
            h = F.silu(layer(h))  # "swish", as in PETS / MBPO
        out = self.layers[-1](h)
        mean, logvar = out[..., : self.out_dim], out[..., self.out_dim:]
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mean, logvar

    def _predict_all(self, obs, act):
        x = self.scaler.transform(torch.cat([obs, act], -1))
        return self(x.unsqueeze(0).expand(self.ensemble_size, *x.shape))

    def _weight_decay_loss(self):
        return sum(wd * 0.5 * (layer.weight ** 2).sum()
                   for wd, layer in zip(self.weight_decay, self.layers))

    # ---------------------------------------------------------------------
    @staticmethod
    def make_targets(obs, rew, next_obs):
        return torch.cat([rew.unsqueeze(-1), next_obs - obs], -1)

    def train_model(self, data: Dict[str, torch.Tensor], batch_size: int = 256,
                    holdout_ratio: float = 0.2, max_holdout: int = 5000,
                    patience: int = 5, max_epochs: Optional[int] = None,
                    max_seconds: Optional[float] = None) -> Dict[str, float]:
        """Fit the ensemble on all real data; returns training statistics."""
        t0 = time.time()
        inputs = torch.cat([data["obs"], data["act"]], -1)
        targets = self.make_targets(data["obs"], data["rew"], data["next_obs"])
        n = inputs.shape[0]

        perm = torch.randperm(n, device=self.device)
        n_hold = min(int(n * holdout_ratio), max_holdout)
        hold_idx, train_idx = perm[:n_hold], perm[n_hold:]
        self.scaler.fit(inputs[train_idx])
        x_train, y_train = self.scaler.transform(inputs[train_idx]), targets[train_idx]
        x_hold = self.scaler.transform(inputs[hold_idx]).unsqueeze(0).expand(self.ensemble_size, -1, -1)
        y_hold = targets[hold_idx]
        n_train = x_train.shape[0]

        # bootstrap: every member trains on its own resample (with replacement)
        boot = torch.randint(0, n_train, (self.ensemble_size, n_train), device=self.device)

        best = self._holdout_mse(x_hold, y_hold)
        best_state = {k: v.detach().clone() for k, v in self.state_dict().items()}
        epochs_since_update, epoch, train_loss = 0, 0, float("nan")
        while True:
            shuffled = torch.gather(boot, 1, torch.argsort(torch.rand_like(boot, dtype=torch.float32), 1))
            losses = []
            for start in range(0, n_train, batch_size):
                idx = shuffled[:, start:start + batch_size]          # (E, b)
                mean, logvar = self(x_train[idx])                     # (E, b, out)
                inv_var = torch.exp(-logvar)
                nll = (((mean - y_train[idx]) ** 2) * inv_var + logvar).mean((1, 2)).sum()
                loss = nll + 0.01 * (self.max_logvar.sum() - self.min_logvar.sum()) \
                    + self._weight_decay_loss()
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                losses.append(nll.detach())
            epoch += 1
            train_loss = (torch.stack(losses).mean() / self.ensemble_size).item()

            hold = self._holdout_mse(x_hold, y_hold)
            improved = (best - hold) / best.clamp_min(1e-12) > 0.01
            if improved.any():
                best = torch.where(improved, hold, best)
                state = self.state_dict()
                for k, v in state.items():          # keep the best weights of each member
                    if k.startswith("layers.") and v.shape[0] == self.ensemble_size:
                        best_state[k][improved] = v.detach()[improved]
                epochs_since_update = 0
            else:
                epochs_since_update += 1

            if epochs_since_update > patience:
                break
            if max_epochs is not None and epoch >= max_epochs:
                break
            if max_seconds is not None and time.time() - t0 > max_seconds:
                break

        # restore each member's best weights (variance bounds are shared, keep current)
        current = self.state_dict()
        for k in current:
            if k.startswith("layers."):
                current[k].copy_(best_state[k])
        final = self._holdout_mse(x_hold, y_hold)
        self.elites = torch.argsort(final)[: self.num_elites].tolist()
        return {"model_train_loss": train_loss,
                "model_holdout_mse": final[self.elites].mean().item(),
                "model_epochs": epoch,
                "model_train_time": time.time() - t0,
                "model_data_size": n}

    @torch.no_grad()
    def _holdout_mse(self, x_hold, y_hold) -> torch.Tensor:
        mean, _ = self(x_hold)
        return ((mean - y_hold.unsqueeze(0)) ** 2).mean((1, 2))  # (E,)

    # ---------------------------------------------------------------------
    @torch.no_grad()
    def step(self, obs: torch.Tensor, act: torch.Tensor, deterministic: bool = False):
        """Sample (next_obs, reward) for a batch, each row from a random elite member."""
        mean, logvar = self._predict_all(obs, act)                        # (E, B, out)
        sample = mean if deterministic else mean + torch.randn_like(mean) * (0.5 * logvar).exp()
        elites = torch.as_tensor(self.elites, device=obs.device)
        member = elites[torch.randint(len(self.elites), (obs.shape[0],), device=obs.device)]
        chosen = sample[member, torch.arange(obs.shape[0], device=obs.device)]  # (B, out)
        reward, next_obs = chosen[:, 0], obs + chosen[:, 1:]
        return next_obs, reward

    @torch.no_grad()
    def evaluate_mse(self, data: Dict[str, torch.Tensor]) -> float:
        """MSE of the elite-mean prediction on a batch of real transitions."""
        mean, _ = self._predict_all(data["obs"], data["act"])
        pred = mean[self.elites].mean(0)
        y = self.make_targets(data["obs"], data["rew"], data["next_obs"])
        return ((pred - y) ** 2).mean().item()


def termination_fn(env_id: str, obs: torch.Tensor, act: torch.Tensor, next_obs: torch.Tensor):
    """Known termination rule used inside model rollouts (MBPO's static_fns).
    HalfCheetah never terminates (episodes only end by the 1000-step time limit)."""
    if "HalfCheetah" in env_id or "Swimmer" in env_id:
        return torch.zeros(obs.shape[0], device=obs.device)
    raise NotImplementedError(f"no termination function for {env_id}")

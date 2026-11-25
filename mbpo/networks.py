"""Neural network building blocks shared by SAC and the dynamics model.

`EnsembleLinear` holds the weights of E independent linear layers in a single
tensor, so E networks are evaluated with one batched matmul instead of a Python
loop. We use it for the probabilistic dynamics ensemble (E = 7) and for SAC's
twin Q-functions (E = 2).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnsembleLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, ensemble_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(ensemble_size, in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(ensemble_size, 1, out_dim))
        # same init as nn.Linear (Kaiming-uniform), applied per member
        bound = 1.0 / math.sqrt(in_dim)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (E, B, in_dim) -> (E, B, out_dim)
        return torch.baddbmm(self.bias, x, self.weight)


class EnsembleMLP(nn.Module):
    """E independent MLPs evaluated in parallel."""

    def __init__(self, in_dim, out_dim, hidden_dim, num_hidden, ensemble_size, act=nn.ReLU):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * num_hidden
        self.layers = nn.ModuleList(
            [EnsembleLinear(a, b, ensemble_size) for a, b in zip(dims[:-1], dims[1:])])
        self.out = EnsembleLinear(dims[-1], out_dim, ensemble_size)
        self.act = act()
        self.ensemble_size = ensemble_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:  # same input for every member
            x = x.unsqueeze(0).expand(self.ensemble_size, *x.shape)
        for layer in self.layers:
            x = self.act(layer(x))
        return self.out(x)


class TwinQ(nn.Module):
    """Two Q-networks Q1, Q2 (clipped double-Q) evaluated as one ensemble."""

    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.net = EnsembleMLP(obs_dim + act_dim, 1, hidden_dim, 2, ensemble_size=2)

    def forward(self, obs, act):
        q = self.net(torch.cat([obs, act], dim=-1))  # (2, B, 1)
        return q.squeeze(-1)                          # (2, B)


class SquashedGaussianActor(nn.Module):
    """Gaussian policy with tanh-squashed actions."""

    LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0

    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.mu = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Linear(hidden_dim, act_dim)

    def forward(self, obs, deterministic=False, with_log_prob=True):
        h = self.trunk(obs)
        mu = self.mu(h)
        log_std = self.log_std(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp()
        u = mu if deterministic else mu + std * torch.randn_like(mu)  # reparameterisation
        a = torch.tanh(u)
        if not with_log_prob:
            return a, None
        # Compute the log probability with a stable tanh Jacobian correction.
        log_prob = (-0.5 * ((u - mu) / std) ** 2 - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)
        log_prob = log_prob - (2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))).sum(-1)
        return a, log_prob

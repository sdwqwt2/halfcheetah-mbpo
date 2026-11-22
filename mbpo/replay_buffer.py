"""FIFO replay buffer that lives on the training device.

Keeping the data on the GPU means sampling a mini-batch is a single indexing
op, with no host-to-device copy inside the (very hot) SAC update loop.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch

FIELDS = ("obs", "act", "rew", "next_obs", "done")


class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, capacity: int, device: torch.device):
        self.capacity = int(capacity)
        self.device = device
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self._alloc(self.capacity)
        self.ptr = 0
        self.size = 0

    def _alloc(self, capacity):
        z = lambda *s: torch.zeros(capacity, *s, dtype=torch.float32, device=self.device)
        self.data = {"obs": z(self.obs_dim), "act": z(self.act_dim), "rew": z(),
                     "next_obs": z(self.obs_dim), "done": z()}

    def __len__(self):
        return self.size

    # -- writing -------------------------------------------------------------
    def add(self, obs, act, rew, next_obs, done) -> None:
        """Add one transition (numpy / python scalars)."""
        self.add_batch({
            "obs": torch.as_tensor(np.asarray(obs)[None], dtype=torch.float32),
            "act": torch.as_tensor(np.asarray(act)[None], dtype=torch.float32),
            "rew": torch.tensor([float(rew)]),
            "next_obs": torch.as_tensor(np.asarray(next_obs)[None], dtype=torch.float32),
            "done": torch.tensor([float(done)]),
        })

    def add_batch(self, batch: Dict[str, torch.Tensor]) -> None:
        n = batch["obs"].shape[0]
        if n >= self.capacity:  # keep only the newest `capacity` items
            batch = {k: v[-self.capacity:] for k, v in batch.items()}
            n = self.capacity
        idx = (torch.arange(n, device=self.device) + self.ptr) % self.capacity
        for k in FIELDS:
            self.data[k][idx] = batch[k].to(self.device, torch.float32).reshape(n, *self.data[k].shape[1:])
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    # -- reading -------------------------------------------------------------
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return {k: v[idx] for k, v in self.data.items()}

    def all(self) -> Dict[str, torch.Tensor]:
        """All stored transitions in chronological order."""
        if self.size < self.capacity:
            return {k: v[: self.size] for k, v in self.data.items()}
        order = (torch.arange(self.capacity, device=self.device) + self.ptr) % self.capacity
        return {k: v[order] for k, v in self.data.items()}

    def resize(self, new_capacity: int) -> None:
        """Change capacity, keeping the most recent transitions (used when the
        rollout length, and therefore the amount of model data, changes)."""
        new_capacity = int(new_capacity)
        if new_capacity == self.capacity:
            return
        old = self.all()
        self.capacity = new_capacity
        self._alloc(new_capacity)
        self.ptr = self.size = 0
        if old["obs"].shape[0] > 0:
            self.add_batch(old)

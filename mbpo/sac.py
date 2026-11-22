"""Soft Actor-Critic (Haarnoja et al., 2018), the policy optimiser inside MBPO.

One call to `update(batch)` performs one gradient step on each of

  critic:  L_Q  = E[(Q_i(s,a) - y)^2],   y = r + gamma (1-d) [min_j Q'_j(s',a') - alpha log pi(a'|s')],  a' ~ pi(.|s')
  actor:   L_pi = E[alpha log pi(a~|s) - min_j Q_j(s,a~)],                                                a~ ~ pi(.|s)
  alpha:   L_a  = E[-log_alpha (log pi(a~|s) + H_target)]

followed by a Polyak update of the target critic: Q' <- tau Q + (1-tau) Q'.
"""
from __future__ import annotations

import copy
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .networks import SquashedGaussianActor, TwinQ


def _adam(params, lr, device):
    # the fused CUDA kernel is noticeably faster for these small networks
    kwargs = {"fused": True} if device.type == "cuda" else {}
    try:
        return torch.optim.Adam(params, lr=lr, **kwargs)
    except (TypeError, RuntimeError):
        return torch.optim.Adam(params, lr=lr)


class SAC:
    def __init__(self, obs_dim, act_dim, device, hidden_dim=256, gamma=0.99, tau=5e-3,
                 actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4, init_alpha=1.0,
                 target_entropy: Optional[float] = None):
        self.device = device
        self.gamma, self.tau = gamma, tau
        self.actor = SquashedGaussianActor(obs_dim, act_dim, hidden_dim).to(device)
        self.critic = TwinQ(obs_dim, act_dim, hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False)
        self.log_alpha = torch.tensor(np.log(init_alpha), dtype=torch.float32,
                                      device=device, requires_grad=True)
        self.target_entropy = float(-act_dim if target_entropy is None else target_entropy)

        self.actor_opt = _adam(self.actor.parameters(), actor_lr, device)
        self.critic_opt = _adam(self.critic.parameters(), critic_lr, device)
        self.alpha_opt = _adam([self.log_alpha], alpha_lr, device)
        self._critic_params = list(self.critic.parameters())
        self._target_params = list(self.critic_target.parameters())

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    # ---------------------------------------------------------------------
    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        a, _ = self.actor(o.unsqueeze(0) if o.dim() == 1 else o,
                          deterministic=deterministic, with_log_prob=False)
        a = a.cpu().numpy()
        return a[0] if o.dim() == 1 else a

    def update(self, batch: Dict[str, torch.Tensor], return_info: bool = False):
        obs, act, rew = batch["obs"], batch["act"], batch["rew"]
        next_obs, done = batch["next_obs"], batch["done"]
        alpha = self.alpha.detach()

        # ---- critic ------------------------------------------------------
        with torch.no_grad():
            next_a, next_logp = self.actor(next_obs)
            q_next = self.critic_target(next_obs, next_a).min(0).values
            target = rew + self.gamma * (1.0 - done) * (q_next - alpha * next_logp)
        q = self.critic(obs, act)                                 # (2, B)
        critic_loss = F.mse_loss(q, target.unsqueeze(0).expand_as(q), reduction="none").mean(1).sum()
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        # ---- actor -------------------------------------------------------
        for p in self._critic_params:
            p.requires_grad_(False)
        new_a, logp = self.actor(obs)
        q_pi = self.critic(obs, new_a).min(0).values
        actor_loss = (alpha * logp - q_pi).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        for p in self._critic_params:
            p.requires_grad_(True)

        # ---- temperature -------------------------------------------------
        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        # ---- target network ----------------------------------------------
        with torch.no_grad():
            torch._foreach_mul_(self._target_params, 1.0 - self.tau)
            torch._foreach_add_(self._target_params, self._critic_params, alpha=self.tau)

        if return_info:  # tensors; caller converts to floats rarely (avoids GPU syncs)
            return {"critic_loss": critic_loss.detach(), "actor_loss": actor_loss.detach(),
                    "alpha_loss": alpha_loss.detach(), "alpha": alpha,
                    "entropy": -logp.detach().mean(), "q_mean": q.detach().mean()}
        return None

    # ---------------------------------------------------------------------
    def state_dict(self):
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu()}

    def load_state_dict(self, sd):
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])
        self.critic_target.load_state_dict(sd["critic_target"])
        with torch.no_grad():
            self.log_alpha.copy_(sd["log_alpha"].to(self.device))

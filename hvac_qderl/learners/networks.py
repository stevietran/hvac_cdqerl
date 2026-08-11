"""Torch networks for SAC: device-aware

    GaussianActor  u ~ tanh(N(mu(x), sigma(x))) 
    TwinCritic  Q1,Q2(x,u)                               

`x` is the Markov feature vector from `common.augment_obs` (60-D for the
reference topology)

"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


def get_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class GaussianActor(nn.Module):
    def __init__(self, feat_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feat_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)

    def forward(self, x):
        h = self.net(x)
        mu = self.mu(h)
        log_std = torch.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, x):
        """Return (action in [-1,1], log_prob, deterministic tanh(mu))."""
        mu, log_std = self.forward(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mu, std)
        x = normal.rsample()
        u = torch.tanh(x)
        # tanh-corrected log prob
        logp = normal.log_prob(x) - torch.log(1 - u.pow(2) + 1e-6)
        logp = logp.sum(dim=-1, keepdim=True)
        return u, logp, torch.tanh(mu)


class Critic(nn.Module):
    def __init__(self, feat_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))

    def forward(self, x, u):
        return self.net(torch.cat([x, u], dim=-1))


class TwinCritic(nn.Module):
    def __init__(self, feat_dim, act_dim, hidden=64):
        super().__init__()
        self.q1 = Critic(feat_dim, act_dim, hidden)
        self.q2 = Critic(feat_dim, act_dim, hidden)

    def forward(self, x, u):
        return self.q1(x, u), self.q2(x, u)


def flat_params(module: nn.Module) -> np.ndarray:
    return torch.cat([p.detach().reshape(-1) for p in module.parameters()]).cpu().numpy()


def set_flat_params(module: nn.Module, theta: np.ndarray):
    i = 0
    t = torch.as_tensor(theta, dtype=torch.float32)
    for p in module.parameters():
        n = p.numel()
        p.data.copy_(t[i:i + n].view_as(p).to(p.device))
        i += n
    return module

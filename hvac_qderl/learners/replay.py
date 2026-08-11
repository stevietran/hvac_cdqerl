"""Shared replay buffer for SAC / CQD-ERL

Plain transition replay over the Markov feature vector from
`common.augment_obs`. No hidden states are stored.
"""
from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int, feat_dim: int, act_dim: int):
        self.cap = int(capacity)
        self.obs = np.zeros((self.cap, feat_dim), np.float32)
        self.next_obs = np.zeros((self.cap, feat_dim), np.float32)
        self.act = np.zeros((self.cap, act_dim), np.float32)
        self.rew = np.zeros((self.cap, 1), np.float32)
        self.done = np.zeros((self.cap, 1), np.float32)
        self.idx = 0
        self.full = False

    def add(self, obs, act, rew, next_obs, done):
        """`done` must be the BOOTSTRAP flag, not the reset flag.
        """
        i = self.idx
        self.obs[i] = obs; self.next_obs[i] = next_obs
        self.act[i] = act; self.rew[i] = rew; self.done[i] = done
        self.idx = (i + 1) % self.cap
        self.full = self.full or self.idx == 0

    def __len__(self):
        return self.cap if self.full else self.idx

    def sample(self, batch_size: int, device):
        n = len(self)
        j = np.random.randint(0, n, size=batch_size)
        t = lambda a: torch.as_tensor(a[j], device=device)
        return dict(obs=t(self.obs), next_obs=t(self.next_obs), act=t(self.act),
                    rew=t(self.rew), done=t(self.done))

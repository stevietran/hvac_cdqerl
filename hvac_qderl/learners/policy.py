"""Lightweight numpy MLP policy — the evolvable genome

Genome = flattened (W1,b1,W2,b2) of a 2-layer tanh MLP  obs -> action in [-1,1].
"""
from __future__ import annotations

import numpy as np


class NumpyMLPPolicy:
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 32, seed: int = 0):
        self.obs_dim, self.act_dim, self.hidden = obs_dim, act_dim, hidden
        rng = np.random.default_rng(seed)
        # small init keeps early actions mid-range
        self.W1 = rng.normal(0, 0.3, (hidden, obs_dim))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, 0.3, (act_dim, hidden))
        self.b2 = np.zeros(act_dim)

    # ---- flat-parameter interface for pyribs / evolution ---- #
    @property
    def n_params(self) -> int:
        return self.W1.size + self.b1.size + self.W2.size + self.b2.size

    def get_params(self) -> np.ndarray:
        return np.concatenate([self.W1.ravel(), self.b1,
                               self.W2.ravel(), self.b2])

    def set_params(self, theta: np.ndarray) -> "NumpyMLPPolicy":
        i = 0
        n = self.hidden * self.obs_dim
        self.W1 = theta[i:i + n].reshape(self.hidden, self.obs_dim); i += n
        self.b1 = theta[i:i + self.hidden]; i += self.hidden
        n = self.act_dim * self.hidden
        self.W2 = theta[i:i + n].reshape(self.act_dim, self.hidden); i += n
        self.b2 = theta[i:i + self.act_dim]
        return self

    def act(self, x: np.ndarray) -> np.ndarray:
        h = np.tanh(self.W1 @ x + self.b1)
        return np.tanh(self.W2 @ h + self.b2)     # in [-1, 1]

    def act_batch(self, X: np.ndarray) -> np.ndarray:
        """Batched `.act`: `X` is (N, obs_dim) -> (N, act_dim), one vectorised
        pass instead of N single-row calls (notes.md §8.6.4d) -- algebraically
        identical to calling `.act` on every row (`(W1 @ x)_k == (X @ W1.T)_k`
        summed the same way), used by `probe_bd.probe_actions` for the whole
        probe grid at once."""
        h = np.tanh(X @ self.W1.T + self.b1)          # (N, hidden)
        return np.tanh(h @ self.W2.T + self.b2)        # (N, act_dim)


def make_policy(obs_dim, act_dim=2, hidden=32, seed=0, theta=None):
    p = NumpyMLPPolicy(obs_dim, act_dim, hidden, seed)
    if theta is not None:
        p.set_params(np.asarray(theta))
    return p

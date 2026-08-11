"""QD-ERL-Context configuration — deliberately torch-free.

Kept separate from `qd_erl_contextual.py` so the settings can be inspected,
validated and hyperparameter-searched on any machine (and by the test suite)
without importing torch.

"""
from __future__ import annotations

from dataclasses import dataclass


def actor_genome_size(feat_dim: int, hidden: int, act_dim: int = 2) -> int:
    """Parameter count of the torch GaussianActor used as the QD-ERL-Context genome.
    """
    return (feat_dim * hidden + hidden) + (hidden * hidden + hidden) \
        + 2 * (hidden * act_dim + act_dim)


@dataclass
class ContextualQDERLConfig:
    """§8.6 training-loop hyperparameters (notes.md §8.6.1/§8.5).

    The outer loop is Iso+LineDD / SAC-critic-gradient / shared-critic-update /
    periodic-actor-injection, driven in `qd_erl_contextual.ContextualQDERL`.
    The INNER loop (one context cell + F_G36 normalisation + probe-BD
    """
    hidden: int = 64
    n_beh: int = 12
    n_context_cells: int = 18
    n_init: int = 96
    g_ga: int = 8
    g_pg: int = 8
    n_grad_pg: int = 10
    n_grad_core: int = 20
    t_inj: int = 10
    iso_sigma: float = 0.0464
    line_sigma: float = 0.2
    init_sigma: float = 0.5
    gamma: float = 0.98
    tau: float = 0.005
    n_probe: int = 4_500
    bd_calibration_genomes: int = 40
    prefer_gpu: bool = True
    n_workers: int = 1

    def genome_size(self, feat_dim: int) -> int:
        from .common import ACT_DIM
        return actor_genome_size(feat_dim, self.hidden, act_dim=ACT_DIM)

    def effective_iso_sigma(self, n_params: int | None = None) -> float:
        """The mutation scale actually used. `n_params` is accepted (and
        ignored) only for call-site symmetry with the old arm's
        `QDERLConfig.effective_iso_sigma` -- `iso_sigma` here is always the
        plain literal above, never genome-size-derived (see field comment)."""
        return float(self.iso_sigma)

    def total_cells(self) -> int:
        return self.n_context_cells * self.n_beh

    def evals_per_niche(self, total_env_steps: int, steps_per_episode: int = 72) -> float:
        """`steps_per_episode=72` is the §8.6.2 context-episode length (24 h at
        the 20-min control cadence) -- NOT the `n_rep_days+2` episode length
        `QDERLConfig.evals_per_niche` assumes."""
        return (total_env_steps / steps_per_episode) / max(self.total_cells(), 1)

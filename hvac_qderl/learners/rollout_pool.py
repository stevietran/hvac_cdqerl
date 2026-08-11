"""CPU-multiprocessing population-rollout pool for the contextual arm.

Population-based QD algorithms are embarrassingly parallel WITHIN a
generation: every offspring is scored by an independent rollout. 
The dependency is only ACROSS generations (Iso+LineDD and the PG operator sample)
The archive insertion, shared-critic update and periodic actor injection between 
generations still run serially in the main process

"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np

# Populated once per WORKER process by `_worker_init`
_W: dict = {}


def _worker_init(mos_path, price_profile, seed, hidden, n_probe,
                 n_context_cells, kind, feat_dim, act_dim):
    """Runs once when each `ProcessPoolExecutor` worker process starts.
    """
    from ..scenarios import get_annual_context, get_context_archive, get_warmstart_bank
    from ..probe_bd import build_probe_grid

    ctx = get_annual_context(mos_path=mos_path)
    arc = get_context_archive()
    wb = get_warmstart_bank(mos_path=mos_path, price_profile=price_profile)
    n_zones = len(ctx.cfg.zones)
    grid = build_probe_grid(ctx.cfg, n_zones, n_probe=n_probe, seed=seed)

    _W.update(cfg=ctx.cfg, annual=ctx.annual, load=ctx.load,
             context_archive=arc, warmstart_bank=wb, n_zones=n_zones,
             grid=grid, price_profile=price_profile, hidden=hidden,
             n_context_cells=n_context_cells, kind=kind)

    if kind == "torch":
        import torch
        from .networks import GaussianActor
        # One worker == one core
        torch.set_num_threads(1)
        actor = GaussianActor(feat_dim, act_dim, hidden)
        actor.eval()
        _W.update(actor=actor,
                  to_tensor=lambda x: torch.as_tensor(
                      np.asarray(x, dtype=np.float32)).unsqueeze(0))


def _eval_numpy(theta, generation, seed):
    """Worker body for the numpy arm (`contextual_ga_demo.py`, testable)."""
    from .common import evaluate_genome_contextual
    rng = np.random.default_rng(seed)
    F_tilde, bd, cell, day, F_raw = evaluate_genome_contextual(
        theta, _W["cfg"], _W["annual"], _W["load"], _W["context_archive"],
        _W["warmstart_bank"], rng, hidden=_W["hidden"],
        price_profile=_W["price_profile"], probe_grid=_W["grid"],
        n_context_cells=_W["n_context_cells"])
    return dict(generation=generation, F_tilde=F_tilde, bd=bd, cell=cell,
               day=day, F_raw=F_raw, f_g36=_W["warmstart_bank"].f_g36(day))


def _eval_torch(theta, generation, seed):
    """Worker body for the torch arm"""
    from .qd_erl_contextual import rollout_contextual_torch
    rng = np.random.default_rng(seed)
    F_tilde, bd, cell, day, F_raw, f_g36, transitions = rollout_contextual_torch(
        _W["actor"], _W["to_tensor"], theta, _W["cfg"], _W["n_zones"],
        _W["context_archive"], _W["warmstart_bank"], _W["annual"], _W["load"],
        rng, _W["n_context_cells"], _W["grid"], _W["price_profile"])
    return dict(generation=generation, F_tilde=F_tilde, bd=bd, cell=cell,
               day=day, F_raw=F_raw, f_g36=f_g36, transitions=transitions)


class RolloutPool:
    """Owns a `ProcessPoolExecutor` for one training run's lifetime.
    """

    def __init__(self, n_workers: int, kind: str, mos_path=None,
                price_profile=None, seed: int = 0, hidden: int = 32,
                n_probe: int = 4_500, n_context_cells: int = 18,
                feat_dim: int | None = None, act_dim: int | None = None):
        if kind not in ("numpy", "torch"):
            raise ValueError(f"kind must be 'numpy' or 'torch', got {kind!r}")
        if kind == "torch" and (feat_dim is None or act_dim is None):
            raise ValueError("kind='torch' needs feat_dim and act_dim to "
                             "build each worker's CPU actor template")
        self.kind = kind
        self.n_workers = max(1, int(n_workers))
        # Distinct call-seed stream per pool instance
        self._seed_base = int(seed) * 1_000_003
        self._call_id = 0
        self._pool = ProcessPoolExecutor(
            max_workers=self.n_workers, initializer=_worker_init,
            initargs=(mos_path, price_profile, seed, hidden, n_probe,
                      n_context_cells, kind, feat_dim, act_dim))

    def submit_batch(self, thetas, generation: int = 0):
        """Evaluate every genome in `thetas` in parallel.
        """
        thetas = list(thetas)
        if not thetas:
            return []
        fn = _eval_numpy if self.kind == "numpy" else _eval_torch
        seeds = [self._seed_base + self._call_id + i for i in range(len(thetas))]
        self._call_id += len(thetas)
        return list(self._pool.map(fn, thetas, [generation] * len(thetas), seeds))

    def shutdown(self):
        self._pool.shutdown(wait=True, cancel_futures=False)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.shutdown()

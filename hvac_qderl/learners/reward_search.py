"""secondary EA loop: CMA-ES reward/architecture search.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter
from ribs.schedulers import Scheduler

from ..config import default_singapore_config
from ..environment import HVACPlantEnv
from ..weather import SingaporeWeather
from .common import (feature_dim_for, PolicyController, rollout, ACT_DIM)
from .policy import make_policy

N_META = 4
HIDDEN_CHOICES = [32, 64]


def _softplus(z):
    return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0)


def decode(x: np.ndarray):
    w = _softplus(x[:3]) * np.array([1.0, 1.0, 5.0]) + 1e-3
    hidden = HIDDEN_CHOICES[int(np.clip(x[3], 0, len(HIDDEN_CHOICES) - 1e-3)) % len(HIDDEN_CHOICES)]
    return w, hidden


def _cfg_with_weights(w):
    cfg = default_singapore_config()
    cfg.w_energy, cfg.w_comfort, cfg.w_shield = map(float, w)
    return cfg


def meta_objective(trace, cfg) -> float:
    """True operational score (higher better): -energy - hard violation penalties."""
    energy = sum(t["p_total_kw"] for t in trace) * cfg.control_step_h
    rh_viol = np.mean([1.0 if t["rh_violation"] > 0 else 0.0 for t in trace])
    comfort = np.mean([1.0 if t["pmv_disc"] > 0 else 0.0 for t in trace])
    return -(energy + 5000.0 * rh_viol + 2000.0 * comfort)


def inner_search(w, hidden, budget=24, day_type="typical_cool",
                 price_profile=None, seed=0):
    """Light serial policy search under the candidate shaped reward; return the
    meta-objective of the best-shaped policy."""
    cfg = _cfg_with_weights(w)
    env = HVACPlantEnv(config=cfg, weather=SingaporeWeather(day_type, price_profile))
    obs_dim = feature_dim_for(cfg)
    rng = np.random.default_rng(seed)
    n_params = make_policy(obs_dim, ACT_DIM, hidden).n_params
    best_shaped, best_trace = -np.inf, None
    for _ in range(budget):
        theta = rng.normal(0, 0.6, n_params)
        pol = make_policy(obs_dim, ACT_DIM, hidden, theta=theta)
        F, trace = rollout(env, PolicyController(pol, cfg))    # F = shaped return
        if F > best_shaped:
            best_shaped, best_trace = F, trace
    return meta_objective(best_trace, cfg)


@dataclass
class RewardSearchResult:
    best_x: np.ndarray
    best_weights: np.ndarray
    best_hidden: int
    best_meta: float
    history: list


def run_reward_search(iterations=20, batch_size=8, sigma0=0.5,
                      inner_budget=24, seed=0, verbose=True) -> RewardSearchResult:
    # single-cell archive => EvolutionStrategyEmitter behaves as plain CMA-ES
    archive = GridArchive(solution_dim=N_META, dims=(1, 1, 1),
                          ranges=[(0, 1)] * 3, qd_score_offset=-1e6)
    x0 = np.zeros(N_META); x0[3] = 0.5
    emitter = EvolutionStrategyEmitter(archive, x0=x0, sigma0=sigma0,
                                       ranker="obj", batch_size=batch_size, seed=seed)
    scheduler = Scheduler(archive, [emitter])

    best_meta, best_x = -np.inf, x0.copy()
    history = []
    const_measures = np.tile([0.5, 0.5, 0.5], (batch_size, 1))
    for it in range(1, iterations + 1):
        sols = scheduler.ask()
        objs = np.array([inner_search(*decode(s), budget=inner_budget, seed=seed + it)
                         for s in sols])
        scheduler.tell(objs, const_measures)
        k = int(np.argmax(objs))
        if objs[k] > best_meta:
            best_meta, best_x = float(objs[k]), sols[k].copy()
        if verbose:
            w, hid = decode(best_x)
            print(f"  iter {it:3d} | best meta {best_meta:11.1f} | "
                  f"w={np.round(w,2)} hidden={hid}")
        history.append((it, best_meta))
    w, hid = decode(best_x)
    return RewardSearchResult(best_x, w, hid, best_meta, history)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=20)
    ap.add_argument("--inner-budget", type=int, default=24)
    args = ap.parse_args()
    res = run_reward_search(iterations=args.iterations, inner_budget=args.inner_budget)
    print("best weights (w_energy,w_comfort,w_shield):", np.round(res.best_weights, 3),
          "| hidden:", res.best_hidden, "| meta:", round(res.best_meta, 1))

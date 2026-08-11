"""Shared plumbing for the learner stages: obs normalisation, the policy->
controller adapter, and rollout + behaviour-descriptor extraction for
contextual QD-ERL arm (`qd_erl_contextual.py` / `contextual_ga_demo.py`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import SingaporeConfig
from ..environment import HVACPlantEnv
from .policy import make_policy

# Action layout, all in [-1, 1]:
#     a[0] chwst_norm         chilled-water setpoint
#     a[1] cw_fan_norm        cooling-tower fan speed
#     a[2] plant_enable_norm  >= PLANT_ENABLE_THRESHOLD -> plant on

ACT_DIM = 3
FLOW_CAP_MIN = 0.3
PLANT_ENABLE_THRESHOLD = 0.0


def gamma_for_horizon(control_step_min: float, horizon_hours: float = 8.0) -> float:
    """Discount that gives a target *wall-clock* planning horizon.
    """
    steps = max(horizon_hours * 60.0 / max(control_step_min, 1e-6), 2.0)
    return float(min(0.999, 1.0 - 1.0 / steps))

N_EXOG = 21
N_DERIVED_PER_ZONE = 1


def obs_dim_for(cfg: SingaporeConfig) -> int:
    """Width of the RAW observation the environment emits."""
    return 2 * len(cfg.zones) + N_EXOG


def feature_dim_for(cfg: SingaporeConfig) -> int:
    """Width of the vector the actor/critic actually consume.

    = obs_dim_for(cfg) + n_zones      (the dT_z block appended by `augment_obs`)
    For the 15-zone reference topology: 43 + 15 = 58.
    """
    return obs_dim_for(cfg) + N_DERIVED_PER_ZONE * len(cfg.zones)


def normalize_obs(obs: np.ndarray, n_zones: int) -> np.ndarray:
    """Roughly standardise the raw observation so MLP inputs are ~O(1)."""
    x = np.asarray(obs, dtype=float).copy()
    x[:n_zones] = (x[:n_zones] - 24.0) / 5.0                 # zone T
    x[n_zones:2 * n_zones] = (x[n_zones:2 * n_zones] - 0.55) * 4.0   # zone RH
    tail = 2 * n_zones
    x[tail] = (x[tail] - 30.0) / 5.0                         # t_oa
    x[tail + 1] = (x[tail + 1] - 26.0) / 3.0                 # t_wb
    x[tail + 4] = (x[tail + 4] - 8.0) / 2.0                  # prev chwst
    # forecast t_oa (+1/+2/+3 h) and t_wb (+1/+2/+3 h)
    x[tail + 14:tail + 17] = (x[tail + 14:tail + 17] - 30.0) / 5.0
    x[tail + 17:tail + 20] = (x[tail + 17:tail + 20] - 26.0) / 3.0
    return np.clip(x, -5.0, 5.0)


DT_Z_SCALE = 0.5


def augment_obs(obs: np.ndarray, prev_obs: np.ndarray | None,
                n_zones: int) -> np.ndarray:
    """Raw observation -> the Markov feature vector the policy consumes.
    """
    x = normalize_obs(obs, n_zones)
    cur = np.asarray(obs, dtype=float)[:n_zones]
    if prev_obs is None:
        d = np.zeros(n_zones)
    else:
        d = (cur - np.asarray(prev_obs, dtype=float)[:n_zones]) / DT_Z_SCALE
    return np.concatenate([x, np.clip(d, -5.0, 5.0)]).astype(np.float32)


def normalize_obs_batch(obs: np.ndarray, n_zones: int) -> np.ndarray:
    """Batched `normalize_obs`: `obs` is (N, obs_dim) -> (N, obs_dim). Same
    formula per row, vectorised -- see `normalize_obs` for the per-channel
    rationale (unchanged here, just applied column-wise instead of scalar)."""
    x = np.asarray(obs, dtype=float).copy()
    x[:, :n_zones] = (x[:, :n_zones] - 24.0) / 5.0
    x[:, n_zones:2 * n_zones] = (x[:, n_zones:2 * n_zones] - 0.55) * 4.0
    tail = 2 * n_zones
    x[:, tail] = (x[:, tail] - 30.0) / 5.0
    x[:, tail + 1] = (x[:, tail + 1] - 26.0) / 3.0
    x[:, tail + 4] = (x[:, tail + 4] - 8.0) / 2.0
    x[:, tail + 14:tail + 17] = (x[:, tail + 14:tail + 17] - 30.0) / 5.0
    x[:, tail + 17:tail + 20] = (x[:, tail + 17:tail + 20] - 26.0) / 3.0
    return np.clip(x, -5.0, 5.0)


def augment_obs_batch(obs: np.ndarray, prev_obs: np.ndarray | None,
                      n_zones: int) -> np.ndarray:
    """Batched `augment_obs`: `obs`/`prev_obs` are (N, obs_dim) -> (N, feat_dim)
    """
    x = normalize_obs_batch(obs, n_zones)
    cur = np.asarray(obs, dtype=float)[:, :n_zones]
    if prev_obs is None:
        d = np.zeros_like(cur)
    else:
        d = (cur - np.asarray(prev_obs, dtype=float)[:, :n_zones]) / DT_Z_SCALE
    return np.concatenate([x, np.clip(d, -5.0, 5.0)], axis=1).astype(np.float32)


def _lin(a01, lo, hi):
    """Map one action component from [-1, 1] onto [lo, hi]."""
    return lo + 0.5 * (a01 + 1.0) * (hi - lo)


def action_from_vector(a, cfg: SingaporeConfig) -> dict:
    """THE single raw-action -> env-action-dict conversion
    """
    a = np.asarray(a, dtype=float).reshape(-1)
    out = {"chwst": float(_lin(a[0], cfg.chwst_min, cfg.chwst_max)),
           "cw_fan": float(_lin(a[1], 0.2, 1.0))}
    if a.size > 2:
        out["plant_enable"] = bool(a[2] >= PLANT_ENABLE_THRESHOLD)
    if a.size > 3:
        out["flow_cap"] = float(_lin(a[3], FLOW_CAP_MIN, 1.0))
    return out


class PolicyController:
    """Adapt a flat-parameter policy to the env's controller interface."""

    def __init__(self, policy, cfg: SingaporeConfig):
        self.policy = policy
        self.cfg = cfg
        self.n_zones = len(cfg.zones)
        self._prev_obs = None

    def reset(self):
        self._prev_obs = None

    def act(self, obs, info) -> dict:
        x = augment_obs(obs, self._prev_obs, self.n_zones)
        self._prev_obs = np.asarray(obs, dtype=float).copy()
        a = self.policy.act(x)
        return action_from_vector(a, self.cfg)

    _lin = staticmethod(_lin)


def probe_behaviour_descriptor(theta_or_policy, cfg: SingaporeConfig,
                               hidden: int = 64, n_probe: int = None,
                               seed: int = 0, grid=None):
    """`bd(theta) = probe_bd(theta)`. See `hvac_qderl.probe_bd`."""
    from ..probe_bd import probe_bd as _probe_bd, N_PROBE_DEFAULT
    return _probe_bd(theta_or_policy, cfg, hidden=hidden,
                     n_probe=n_probe or N_PROBE_DEFAULT, seed=seed, grid=grid)


N_CONTEXT_CELLS = 18   # 16 data-driven cells + S_peak + S_minpv


def sample_training_context(context_archive, warmstart_bank,
                            rng: np.random.Generator,
                            n_context_cells: int = N_CONTEXT_CELLS):
    """draw a context cell UNIFORMLY, resolve it to the nearest real day, 
    then draw one day-of-week warm-start state. 
    Returns `(cell, day, T_z0, T_m0, W_z0)`.
    """
    from ..context_archive import nearest_day_to_centroid
    cell = int(rng.integers(n_context_cells))
    day = nearest_day_to_centroid(context_archive, cell)
    dow = (warmstart_bank.first_weekday + day) % 7
    T_z0, T_m0, W_z0 = warmstart_bank.sample(dow, rng)
    return cell, day, T_z0, T_m0, W_z0


def context_episode_env(annual, load, cfg: SingaporeConfig, context_archive,
                        cell_id: int, price_profile: str | None = None):
    """build the one-day env for `cell_id`"""
    from ..episodes import build_context_episode, make_env
    spec, day = build_context_episode(annual, load, context_archive, cell_id,
                                      price_profile=price_profile)
    env = make_env(spec, cfg=cfg, for_training=True)
    return env, day


def rollout_with_warmstart(env: HVACPlantEnv, controller, T_z0, T_m0, W_z0):
    """run one context episode from an EXPLICIT warm-started reset state 
    not the default isothermal one. Returns `(F_raw, trace)`"""
    obs = env.reset(T_z0=T_z0, T_m0=T_m0, W_z0=W_z0)
    if hasattr(controller, "reset"):
        controller.reset()
    info = None
    F = 0.0
    trace = []
    done = False
    while not done:
        obs, r, done, info = env.step(controller.act(obs, info))
        F += r
        trace.append(info)
    return F, trace


def normalized_fitness(F_raw: float, F_g36_day: float) -> float:
    """the per-context normalisation"""
    return (F_raw - F_g36_day) / max(abs(F_g36_day), 1e-9)


def evaluate_genome_contextual(theta: np.ndarray, cfg: SingaporeConfig, annual, load,
                               context_archive, warmstart_bank,
                               rng: np.random.Generator, hidden: int = 32,
                               act_dim: int = ACT_DIM,
                               price_profile: str | None = None,
                               probe_grid=None, n_probe: int | None = None,
                               n_context_cells: int = N_CONTEXT_CELLS):
    """inner loop, complete, for a NUMPY genome (`policy.NumpyMLPPolicy`
    via `make_policy`) -- torch-free so it is directly testable without
    ribs/torch. The torch GaussianActor genome used by the actual 
    outer loop (QD-ERL's actor architecture) reuses every function above but
    is implemented in `qd_erl_contextual.py`, which needs torch for the PG
    operator and shared critics.

    Returns `(F_tilde, bd, cell, day, F_raw)`:
    """
    pol = make_policy(feature_dim_for(cfg), act_dim, hidden, theta=theta)
    cell, day, T_z0, T_m0, W_z0 = sample_training_context(
        context_archive, warmstart_bank, rng, n_context_cells)
    env, day_check = context_episode_env(annual, load, cfg, context_archive,
                                         cell, price_profile)
    assert day_check == day, "build_context_episode resolved a different day " \
        "than sample_training_context -- context_archive must have changed " \
        "between the two calls"
    F_raw, trace = rollout_with_warmstart(env, PolicyController(pol, cfg),
                                          T_z0, T_m0, W_z0)
    F_g36 = warmstart_bank.f_g36(day)
    F_tilde = normalized_fitness(F_raw, F_g36)
    from ..probe_bd import probe_bd as _probe_bd, N_PROBE_DEFAULT
    bd = _probe_bd(pol, cfg, n_probe=n_probe or N_PROBE_DEFAULT, grid=probe_grid)
    return float(F_tilde), bd, cell, day, float(F_raw)


def evaluate_genome_on_context_cell(theta: np.ndarray, cfg: SingaporeConfig, annual, load,
                                    context_archive, cell_id: int, day: int,
                                    T_z0, T_m0, W_z0, f_g36: float, hidden: int = 32,
                                    act_dim: int = ACT_DIM,
                                    price_profile: str | None = None) -> float:
    """inner loop with the context cell FORCED

    `day`/`T_z0`/`T_m0`/`W_z0`/`f_g36` are also caller-supplied (typically
    drawn/looked-up ONCE per `cell_id` and reused across every genome

    Returns `F_tilde` only (no behaviour descriptor -- BD is a property of
    theta alone
    """
    pol = make_policy(feature_dim_for(cfg), act_dim, hidden, theta=theta)
    env, day_check = context_episode_env(annual, load, cfg, context_archive,
                                         cell_id, price_profile)
    assert day_check == day, "build_context_episode resolved a different day " \
        "for this cell_id than the caller expected -- context_archive must " \
        "have changed between calls"
    F_raw, _ = rollout_with_warmstart(env, PolicyController(pol, cfg),
                                      T_z0, T_m0, W_z0)
    return float(normalized_fitness(F_raw, f_g36))


def rollout(env: HVACPlantEnv, controller):
    """Run one episode; return (fitness=sum reward, trace)."""
    obs = env.reset()
    if hasattr(controller, "reset"):
        controller.reset()
    info = None
    F = 0.0
    trace = []
    done = False
    while not done:
        obs, r, done, info = env.step(controller.act(obs, info))
        F += r
        trace.append(info)
    return F, trace


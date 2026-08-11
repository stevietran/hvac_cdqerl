"""Policy-probe behaviour descriptor — context-invariant

MECHANISM. Feed a fixed, synthetic battery of observations directly to the
actor (`policy.act(x)`), never through `env.step()`, and summarise the emitted
actions. Because the probe inputs are constants, the descriptor is a pure
function of the genome `theta` and cannot depend on which context (weather
day, representative-day draw, warm-start bank sample, ...)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import psychrometrics as psy

# --------------------------------------------------------------------------- #
# Probe grid axes
# --------------------------------------------------------------------------- #
PROBE_AXES: dict[str, tuple[float, ...]] = {
    "T_z": (23.0, 24.0, 25.0, 27.0, 29.0, 31.0, 32.5),
    "RH": (0.45, 0.55, 0.65),
    "t_oa": (26.0, 29.0, 32.0),
    "w_oa": (0.016, 0.019, 0.022),
    "occ": (0.05, 0.15, 0.55, 1.0),
    "hour": (3.0, 8.0, 13.0, 20.0)
}
PROBE_AXIS_ORDER = ("T_z", "RH", "t_oa", "w_oa", "occ", "hour", "prev_chwst")

N_PROBE_DEFAULT = 4_500
T_Z_DRIFT_THRESHOLD_C = 29.0

PROBE_BD_AXES = ("chwst_toa_gain", "fan_woa_gain",
                 "plant_on_unocc_normal_frac", "plant_on_unocc_drift_frac")
PROBE_BD_AXIS_DOC = {
    "chwst_toa_gain": "OLS slope of a[:,0] (CHWST) ~ t_oa over the full probe "
                      "set — pooled, not regime-split (notes.md §8.4.2: "
                      "occ/unocc regime slopes correlate 0.984 on random MLPs)",
    "fan_woa_gain": "OLS slope of a[:,1] (tower fan) ~ w_oa over the full "
                    "probe set — same pooling rationale as chwst_toa_gain",
    "plant_on_unocc_normal_frac": "mean plant-enable output over unoccupied "
                                  "probe points with T_z < 29 degC",
    "plant_on_unocc_drift_frac": "mean plant-enable output over unoccupied "
                                 "probe points with T_z >= 29 degC — split "
                                 "from *_normal_frac because pooling erases "
                                 "tail behaviour (notes.md §8.4.2: one genome "
                                 "measured 0.049 in this corner vs. 0.421 "
                                 "pooled, an 8.6x difference)",
}

_FIXED_Q_SOLAR_NORM = 0.0        # [ARB] normalised incident solar, held at 0
_FIXED_PREV_CW_FAN = 0.6         # [ARB] midpoint of the [0.2, 1.0] actuator span
_FIXED_P_TOTAL_FRAC = 0.0        # [ARB] p_total / peak_power
_FIXED_PREV_N_ON_FRAC = 0.5      # [ARB] prev_n_on / n_chillers
_FIXED_PLANT_ON = 1.0            # [ARB] "already running" — see module docstring
_FIXED_LOCKOUT_LEFT = 0.0        # [ARB] free to switch (no active anti-cycle hold)
_FIXED_Q_EVAP_FRAC = 0.30
_HOURS_TO_OCC_CAP = 12.0


def _prev_chwst_values(cfg) -> tuple[float, float, float]:
    lo, hi = float(cfg.chwst_min), float(cfg.chwst_max)
    return (lo, 0.5 * (lo + hi), hi)


def grid_size(cfg) -> int:
    """Size of the full (pre-subsample) Cartesian product for this cfg."""
    n = 1
    for name in PROBE_AXIS_ORDER:
        vals = PROBE_AXES[name] if name != "prev_chwst" else _prev_chwst_values(cfg)
        n *= len(vals)
    return n


@dataclass
class ProbeGrid:
    """`N_probe` synthetic raw observations plus the axis values that produced
    each row (needed downstream for the OLS/regime-split axis computation)."""
    raw_obs: np.ndarray      # (N, obs_dim)
    T_z: np.ndarray          # (N,)
    RH: np.ndarray           # (N,)
    t_oa: np.ndarray         # (N,)
    w_oa: np.ndarray         # (N,)
    occ: np.ndarray          # (N,)
    hour: np.ndarray         # (N,)
    prev_chwst: np.ndarray   # (N,)
    n_zones: int
    n_probe: int
    full_grid_size: int
    seed: int

    def unoccupied_mask(self, occ_unoccupied_max: float) -> np.ndarray:
        return self.occ <= occ_unoccupied_max


def _raw_obs_batch(cfg, n_zones: int, T_z, RH, t_oa, w_oa, occ, hour,
                   prev_chwst) -> np.ndarray:
    """Vectorised raw-observation builder, matching `environment.py::_obs`'s
    layout (`T_z[n_zones]`, `RH[n_zones]`, then the 21 exogenous channels in
    exactly the order `_obs` emits them). T_z/RH are broadcast identically to
    every zone — the probe axes are scalars, not per-zone.
    """
    n = len(T_z)
    T_z = np.asarray(T_z, dtype=float)
    RH = np.asarray(RH, dtype=float)
    t_oa = np.asarray(t_oa, dtype=float)
    w_oa = np.asarray(w_oa, dtype=float)
    occ = np.asarray(occ, dtype=float)
    hour = np.asarray(hour, dtype=float)
    prev_chwst = np.asarray(prev_chwst, dtype=float)
    t_wb_cache: dict[tuple[float, float], float] = {}
    t_wb = np.empty(n)
    for i in range(n):
        key = (float(t_oa[i]), float(w_oa[i]))
        if key not in t_wb_cache:
            t_wb_cache[key] = psy.wet_bulb(key[0], key[1])
        t_wb[i] = t_wb_cache[key]

    h = hour % 24.0

    # ---- derived values for the load / lookahead channels ----------------- #
    q_evap = _FIXED_Q_EVAP_FRAC * float(cfg.design_cooling_kw)
    n_ch = max(len(getattr(cfg.plant, "chillers", [])) or 1, 1)
    n_on = max(_FIXED_PREV_N_ON_FRAC * n_ch, 1.0)
    t_cw = np.array([cfg.tower.leaving_cw_temp(float(w), _FIXED_PREV_CW_FAN)
                     for w in t_wb])
    cap_each = np.array([cfg.plant.chillers[0].available_capacity(float(c), float(y))
                         for c, y in zip(prev_chwst, t_cw)])
    util = np.clip(q_evap / np.maximum(n_on * cap_each, 1e-6), 0.0, 2.0)
    occupied_now = occ > float(cfg.occ_setback_max)
    lead = (float(cfg.occupied_start_h) - h) % 24.0
    hours_to_occ = np.where(occupied_now, 0.0, np.minimum(lead, _HOURS_TO_OCC_CAP))

    zone_tz = np.tile(T_z.reshape(-1, 1), (1, n_zones))
    zone_rh = np.tile(RH.reshape(-1, 1), (1, n_zones))
    exog = np.stack([
        # --- present conditions (note: NO w_oa; it is derivable from t_oa+t_wb)
        t_oa, t_wb,
        np.full(n, _FIXED_Q_SOLAR_NORM),
        occ, prev_chwst,
        np.full(n, _FIXED_PREV_CW_FAN),
        np.sin(2 * math.pi * h / 24.0), np.cos(2 * math.pi * h / 24.0),
        np.full(n, _FIXED_P_TOTAL_FRAC),
        np.full(n, _FIXED_PREV_N_ON_FRAC),
        np.full(n, _FIXED_PLANT_ON),
        np.full(n, _FIXED_LOCKOUT_LEFT),
        # --- load
        np.full(n, _FIXED_Q_EVAP_FRAC),
        util,
        # --- lookahead (persistence)
        t_oa, t_oa, t_oa,
        t_wb, t_wb, t_wb,
        hours_to_occ / _HOURS_TO_OCC_CAP,
    ], axis=1)
    return np.concatenate([zone_tz, zone_rh, exog], axis=1)


def build_probe_grid(cfg, n_zones: int, n_probe: int = N_PROBE_DEFAULT,
                     seed: int = 0) -> ProbeGrid:
    """Enumerate the full Cartesian product, then draw `n_probe` rows without
    replacement (notes.md §8.4: "subsample to N_probe = 4,000-5,000 by
    uniform draw without replacement"). Deterministic in `seed`, so a probe_bd
    call is reproducible and — because it never touches env/weather state —
    identical regardless of what context happened to run before it.
    """
    axis_values = {n: (PROBE_AXES[n] if n != "prev_chwst" else _prev_chwst_values(cfg))
                   for n in PROBE_AXIS_ORDER}
    full = grid_size(cfg)
    m = min(int(n_probe), full)
    rng = np.random.default_rng(seed)
    idx = rng.choice(full, size=m, replace=False)
    idx.sort()                    # cheap, stable ordering; result is unaffected

    sizes = [len(axis_values[n]) for n in PROBE_AXIS_ORDER]
    coords = np.array(np.unravel_index(idx, sizes)).T     # (m, 7)
    cols = {n: np.array(axis_values[n])[coords[:, j]]
           for j, n in enumerate(PROBE_AXIS_ORDER)}

    raw = _raw_obs_batch(cfg, n_zones, cols["T_z"], cols["RH"], cols["t_oa"],
                         cols["w_oa"], cols["occ"], cols["hour"],
                         cols["prev_chwst"])
    from .learners.common import obs_dim_for
    want = obs_dim_for(cfg)
    if raw.shape[1] != want:
        raise ValueError(
            f"probe grid width {raw.shape[1]} != environment obs width {want}. "
            f"`probe_bd._raw_obs_batch` must mirror `environment._obs` channel "
            f"for channel -- update it (and `learners.common.N_EXOG`) together.")
    return ProbeGrid(raw_obs=raw, T_z=cols["T_z"], RH=cols["RH"],
                     t_oa=cols["t_oa"], w_oa=cols["w_oa"], occ=cols["occ"],
                     hour=cols["hour"], prev_chwst=cols["prev_chwst"],
                     n_zones=n_zones, n_probe=m, full_grid_size=full, seed=seed)


def probe_actions(policy, grid: ProbeGrid) -> np.ndarray:
    """Run every probe point through the policy, `x = augment_obs(raw,
    prev_obs=None, n_zones)`. Never touches `env.step()`.
    """
    from .learners.common import augment_obs_batch
    X = augment_obs_batch(grid.raw_obs, None, grid.n_zones)   # (n_probe, feat_dim)

    if hasattr(policy, "act_batch"):
        return np.asarray(policy.act_batch(X), dtype=float)

    acts = None
    for i in range(X.shape[0]):
        a = np.asarray(policy.act(X[i]), dtype=float).reshape(-1)
        if acts is None:
            acts = np.empty((X.shape[0], a.size))
        acts[i] = a
    return acts


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Slope of y ~ x by ordinary least squares (no scipy dependency)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    vx = np.var(x)
    if vx < 1e-12:
        return 0.0
    return float(np.cov(x, y, ddof=0)[0, 1] / vx)


@dataclass
class ProbeBD:
    """Raw (un-normalised) descriptor plus enough of the probe to audit"""
    chwst_toa_gain: float
    fan_woa_gain: float
    plant_on_unocc_normal_frac: float
    plant_on_unocc_drift_frac: float
    n_unocc_normal: int
    n_unocc_drift: int

    def as_array(self) -> np.ndarray:
        return np.array([self.chwst_toa_gain, self.fan_woa_gain,
                         self.plant_on_unocc_normal_frac,
                         self.plant_on_unocc_drift_frac], dtype=float)


def compute_probe_bd(actions: np.ndarray, grid: ProbeGrid, cfg,
                     enable_threshold: float | None = None) -> ProbeBD:
    """The four axes, computed from `probe_actions`' output.
    """
    if enable_threshold is None:
        from .learners.common import PLANT_ENABLE_THRESHOLD as enable_threshold

    chwst_gain = _ols_slope(grid.t_oa, actions[:, 0])
    fan_gain = _ols_slope(grid.w_oa, actions[:, 1])

    unocc = grid.unoccupied_mask(cfg.occ_unoccupied_max)
    normal = unocc & (grid.T_z < T_Z_DRIFT_THRESHOLD_C)
    drift = unocc & (grid.T_z >= T_Z_DRIFT_THRESHOLD_C)

    def _frac(mask):
        if not np.any(mask):
            return float("nan")
        return float(np.mean(actions[mask, 2] >= enable_threshold))

    return ProbeBD(chwst_toa_gain=chwst_gain, fan_woa_gain=fan_gain,
                   plant_on_unocc_normal_frac=_frac(normal),
                   plant_on_unocc_drift_frac=_frac(drift),
                   n_unocc_normal=int(normal.sum()), n_unocc_drift=int(drift.sum()))


# --------------------------------------------------------------------------- #
# Top-level convenience: theta -> the 4-vector.
# --------------------------------------------------------------------------- #
def probe_bd(theta_or_policy, cfg, hidden: int = 64,
            n_probe: int = N_PROBE_DEFAULT, seed: int = 0,
            grid: ProbeGrid | None = None) -> np.ndarray:
    """`bd(theta) = probe_bd(theta)`: a pure function of the genome and `cfg` only. 
    Accepts either a raw flat parameter vector or an object that already implements `.act(x)`.

    Passing a pre-built `grid` (`build_probe_grid`) amortises grid
    construction across many genomes, e.g. a calibration sweep.
    """
    from .learners.common import feature_dim_for, ACT_DIM
    from .learners.policy import make_policy

    n_zones = len(cfg.zones)
    if grid is None:
        grid = build_probe_grid(cfg, n_zones, n_probe=n_probe, seed=seed)

    if hasattr(theta_or_policy, "act"):
        policy = theta_or_policy
    else:
        policy = make_policy(feature_dim_for(cfg), ACT_DIM, hidden,
                             theta=np.asarray(theta_or_policy))

    acts = probe_actions(policy, grid)
    return compute_probe_bd(acts, grid, cfg).as_array()


# --------------------------------------------------------------------------- #
# Measurement / calibration
# --------------------------------------------------------------------------- #
@dataclass
class ProbeCalibration:
    """Discrimination of the four axes over a population of random genomes"""
    names: tuple
    values: np.ndarray          # (n_genomes, 4) raw axis values
    scales: np.ndarray          # (n_genomes,) init scale each genome was drawn at
    n_unocc_normal: np.ndarray  # (n_genomes,)
    n_unocc_drift: np.ndarray   # (n_genomes,)
    grid: ProbeGrid

    def spread(self) -> dict:
        """min/max/sd per axis, the exact columns of the §8.4.2 table."""
        out = {}
        for j, n in enumerate(self.names):
            v = self.values[:, j]
            out[n] = dict(min=float(np.min(v)), max=float(np.max(v)),
                         sd=float(np.std(v, ddof=1)))
        return out


def calibrate(cfg=None, n_genomes: int = 18, hidden: int = 64,
             scales=(0.2, 0.6, 1.5, 2.5), n_probe: int = N_PROBE_DEFAULT,
             seed: int = 0) -> ProbeCalibration:
    """Random-genome discrimination sweep
    """
    from .config import default_singapore_config
    from .learners.common import feature_dim_for, ACT_DIM
    from .learners.policy import make_policy

    cfg = cfg or default_singapore_config()
    n_zones = len(cfg.zones)
    grid = build_probe_grid(cfg, n_zones, n_probe=n_probe, seed=seed)
    obs_dim = feature_dim_for(cfg)

    rng = np.random.default_rng(seed)
    scales = tuple(scales)
    values = np.empty((n_genomes, 4))
    used_scales = np.empty(n_genomes)
    n_normal = np.empty(n_genomes, dtype=int)
    n_drift = np.empty(n_genomes, dtype=int)
    for i in range(n_genomes):
        s = scales[i % len(scales)]
        used_scales[i] = s
        n_par = make_policy(obs_dim, ACT_DIM, hidden).n_params
        theta = rng.normal(0, s, n_par)
        pol = make_policy(obs_dim, ACT_DIM, hidden, theta=theta)
        acts = probe_actions(pol, grid)
        res = compute_probe_bd(acts, grid, cfg)
        values[i] = res.as_array()
        n_normal[i], n_drift[i] = res.n_unocc_normal, res.n_unocc_drift

    return ProbeCalibration(names=PROBE_BD_AXES, values=values, scales=used_scales,
                            n_unocc_normal=n_normal, n_unocc_drift=n_drift, grid=grid)


def verify_context_invariance(cfg=None, n_genomes: int = 3, hidden: int = 64,
                              n_probe: int = N_PROBE_DEFAULT, seed: int = 0
                              ) -> dict:
    """Probe the SAME genome before and after a real episode rollout runs 
    in a DIFFERENT training context (`scenarios.make_training_env` with its own seed),
    which mutates env/weather/day-index state that a trace-based descriptor
    would inherit. 
    Returns per-genome max|delta| — every entry should be exactly 0.0.
    """
    from .config import default_singapore_config
    from .learners.common import feature_dim_for, ACT_DIM, PolicyController, rollout
    from .learners.policy import make_policy
    from .scenarios import make_training_env

    cfg = cfg or default_singapore_config()
    obs_dim = feature_dim_for(cfg)
    rng = np.random.default_rng(seed)
    grid = build_probe_grid(cfg, len(cfg.zones), n_probe=n_probe, seed=seed)

    deltas = []
    for i in range(n_genomes):
        n_par = make_policy(obs_dim, ACT_DIM, hidden).n_params
        theta = rng.normal(0, 0.6, n_par)
        bd_before = probe_bd(theta, cfg, hidden=hidden, grid=grid)

        env, ctx, _ = make_training_env(n_rep_days=1, seed=100 + i)
        pol = make_policy(feature_dim_for(ctx.cfg), ACT_DIM, hidden, theta=theta)
        rollout(env, PolicyController(pol, ctx.cfg))     # mutate context state

        bd_after = probe_bd(theta, cfg, hidden=hidden, grid=grid)
        deltas.append(float(np.max(np.abs(bd_after - bd_before))))

    return dict(deltas=deltas, max_delta=max(deltas) if deltas else float("nan"))

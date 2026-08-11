"""Annual dispatch — serving the trained product archive. Implements the "proposal, spec-level detail" section verbatim:

    DAILY (on forecast update):
        1. forecast tomorrow's (q_mean, latent_frac, t_wb_max, ghi_daily)
           -> project into the §8.2 PCA basis
        2. c_candidate = nearest of 18 centroids
        3. commit c_current = c_candidate only after 2 consecutive
           confirming days (else stay put)                -- context hysteresis

    EVERY CONTROL STEP, inside archive[c_current]:
        4. x_t = augment_obs(o_t, o_{t-1})
        5. d_t = windowed BD estimate, in the SAME 4-D space as §8.4's
           probe-BD axes
        6. FILLED = filled niches in archive[c_current]; if empty, fall back
           to the filled niches of the nearest OTHER context cell
        7. j* = argmin_j || d_t - centroid_j || over FILLED, switch_hysteresis=3
        8. u = shield(pi_theta_{c_current,j*}(x_t))  -- the env's own hard
           shield (shield.py) applies inside HVACPlantEnv.step(), same as
           every other controller; this module only returns the raw action
        9. IF no feasible (c,j) anywhere: G36 (final fallback)
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .learners.common import (ACT_DIM, action_from_vector, augment_obs,
                              feature_dim_for)
from .learners.policy import make_policy
from .probe_bd import T_Z_DRIFT_THRESHOLD_C

DEFAULT_WINDOW_STEPS = 36          # 12 h at a 20-min control step -- [HEUR]


# --------------------------------------------------------------------------- #
# Policy reconstruction
# --------------------------------------------------------------------------- #
class _NumpyGaussianActorPolicy:
    """`tanh(mu(x))` for `networks.GaussianActor`, built from a flat `theta`.
    """

    def __init__(self, feat_dim: int, act_dim: int, hidden: int,
                theta: np.ndarray):
        theta = np.asarray(theta, dtype=np.float64).reshape(-1)
        expected = (hidden * feat_dim + hidden + hidden * hidden + hidden
                   + 2 * (act_dim * hidden + act_dim))
        if theta.size != expected:
            raise ValueError(
                f"_NumpyGaussianActorPolicy: theta has {theta.size} params, "
                f"expected {expected} for feat_dim={feat_dim}, hidden={hidden}, "
                f"act_dim={act_dim} -- wrong architecture/shape guess.")
        i = 0

        def take(shape):
            nonlocal i
            n = int(np.prod(shape))
            arr = theta[i:i + n].reshape(shape)
            i += n
            return arr

        self.W1 = take((hidden, feat_dim)); self.b1 = take((hidden,))
        self.W2 = take((hidden, hidden));   self.b2 = take((hidden,))
        self.Wmu = take((act_dim, hidden)); self.bmu = take((act_dim,))
        self.Wls = take((act_dim, hidden)); self.bls = take((act_dim,))  # unused

    def act(self, x: np.ndarray) -> np.ndarray:
        h1 = np.maximum(self.W1 @ x + self.b1, 0.0)
        h2 = np.maximum(self.W2 @ h1 + self.b2, 0.0)
        mu = self.Wmu @ h2 + self.bmu
        return np.tanh(mu)

    def act_batch(self, X: np.ndarray) -> np.ndarray:
        h1 = np.maximum(X @ self.W1.T + self.b1, 0.0)
        h2 = np.maximum(h1 @ self.W2.T + self.b2, 0.0)
        mu = h2 @ self.Wmu.T + self.bmu
        return np.tanh(mu)


def build_dispatch_policy(theta: np.ndarray, feat_dim: int, act_dim: int,
                          hidden: int, policy_kind: str = "numpy_mlp"):
    if policy_kind == "numpy_mlp":
        return make_policy(feat_dim, act_dim, hidden, theta=theta)
    if policy_kind == "torch_actor":
        return _NumpyGaussianActorPolicy(feat_dim, act_dim, hidden, theta)
    raise ValueError(f"unknown policy_kind {policy_kind!r} "
                     "(expected 'numpy_mlp' or 'torch_actor')")
DEFAULT_CONTEXT_CONFIRM_DAYS = 2
DEFAULT_NICHE_HYSTERESIS = 3      
DEFAULT_FITNESS_TIEBREAK_EPS = 0.10


# --------------------------------------------------------------------------- #
# small numpy-only helpers (no scipy dependency, matches probe_bd.py's style)
# --------------------------------------------------------------------------- #
def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    vx = np.var(x)
    if vx < 1e-12:
        return 0.0
    return float(np.cov(x, y, ddof=0)[0, 1] / vx)


def _norm_chwst(chwst: float, cfg) -> float:
    lo, hi = float(cfg.chwst_min), float(cfg.chwst_max)
    return float(np.clip(2.0 * (chwst - lo) / max(hi - lo, 1e-9) - 1.0, -1.0, 1.0))


def _norm_fan(cw_fan: float) -> float:
    lo, hi = 0.2, 1.0
    return float(np.clip(2.0 * (cw_fan - lo) / (hi - lo) - 1.0, -1.0, 1.0))


# --------------------------------------------------------------------------- #
# context-layer geometry (mirrors context_archive.fit_pca_basis' projection)
# --------------------------------------------------------------------------- #
def project_context_features(archive, F_row: np.ndarray) -> np.ndarray:
    """Same standardise -> centre -> project pipeline `build_context_archive`
    fit on the 363 non-extreme days, applied to one new feature row."""
    z = (np.asarray(F_row, dtype=float) - archive.mu) / archive.sd
    return (z - archive.z_mean) @ archive.basis.T


def nearest_context_cell(archive, point2d: np.ndarray) -> int:
    d2 = ((archive.centroids - point2d) ** 2).sum(axis=1)
    return int(np.argmin(d2))


def _other_cell_order(archive) -> dict:
    """cell -> every OTHER cell, ordered nearest-first by centroid distance
    (§8.7 step 6's "nearest OTHER context cell" fallback tier)."""
    C = archive.centroids
    n = len(C)
    out = {}
    for c in range(n):
        d2 = ((C - C[c]) ** 2).sum(axis=1)
        order = [int(j) for j in np.argsort(d2) if j != c]
        out[c] = order
    return out


# --------------------------------------------------------------------------- #
@dataclass
class DispatchStats:
    """Everything §8.7.1 point 2 says to record."""
    context_switches: int = 0
    sentinel_days: dict = field(default_factory=lambda: {"S_peak": 0, "S_minpv": 0})
    niche_switches: int = 0
    context_empty_fallback_steps: int = 0
    final_fallback_steps: int = 0
    n_steps: int = 0
    tiebreak_wins: int = 0
    tiebreak_band_candidates: int = 0
    deployed_fitness_sum: float = 0.0
    deployed_fitness_steps: int = 0
    context_cell_log: list = field(default_factory=list)   # (day, cell) once/day
    # (day, hour, cell, niche, tier, fitness)
    niche_log: list = field(default_factory=list)

    def mean_deployed_fitness(self) -> float | None:
        """Step-weighted mean fitness of the elites actually dispatched."""
        if self.deployed_fitness_steps == 0:
            return None
        return self.deployed_fitness_sum / self.deployed_fitness_steps

    def mean_band_candidates(self) -> float | None:
        """Mean number of niches inside the tiebreak band, over steps that
        reached the band test at all."""
        if self.deployed_fitness_steps == 0:
            return None
        return self.tiebreak_band_candidates / self.deployed_fitness_steps

    def dwell_times(self) -> list:
        """Consecutive-day run-lengths of the committed context cell —
        notes.md §8.7's "mean dwell 1.21 days" / "one switch every 8-9 days"
        statistic, reproduced from the log rather than hand-tallied."""
        if not self.context_cell_log:
            return []
        runs = []
        cur_cell, run_len = self.context_cell_log[0][1], 1
        for _day, cell in self.context_cell_log[1:]:
            if cell == cur_cell:
                run_len += 1
            else:
                runs.append(run_len)
                cur_cell, run_len = cell, 1
        runs.append(run_len)
        return runs


# --------------------------------------------------------------------------- #
class ContextDispatchController:
    """§8.7's two-clock dispatch policy. Implements the standard controller
    interface (`reset()`, `act(obs, info) -> action dict`) used everywhere
    else in this project (G36Controller, the learners' controllers), so it
    plugs directly into `runner.run_episode` / `experiments.evaluate
    .evaluate_controller`.
    """

    def __init__(self, cfg, context_archive, product_archive, *,
                hidden: int = 24, act_dim: int = ACT_DIM,
                policy_kind: str = "numpy_mlp",
                window_steps: int = DEFAULT_WINDOW_STEPS,
                context_confirm_days: int = DEFAULT_CONTEXT_CONFIRM_DAYS,
                niche_switch_hysteresis: int = DEFAULT_NICHE_HYSTERESIS,
                fitness_tiebreak_eps: float = DEFAULT_FITNESS_TIEBREAK_EPS,
                final_fallback=None):
        self.cfg = cfg
        self.arc = context_archive
        self.pa = product_archive
        self.hidden = hidden
        self.act_dim = act_dim
        self.policy_kind = policy_kind
        self.window_steps = window_steps
        self.confirm_days = context_confirm_days
        self.niche_hyst = niche_switch_hysteresis
        self.tiebreak_eps = float(fitness_tiebreak_eps)
        self.n_zones = len(cfg.zones)

        self._other_order = _other_cell_order(context_archive)
        # the N_beh behaviour centroids are ONE shared tessellation across all
        # 18 context sub-archives by construction (product_archive.py's own
        # module docstring); grab it once from any of them.
        self._centroids = np.asarray(product_archive.archive_for(0).centroids)
        lo = np.array([r[0] for r in product_archive.bd_ranges], dtype=float)
        hi = np.array([r[1] for r in product_archive.bd_ranges], dtype=float)
        self._bd_span = np.maximum(hi - lo, 1e-9)
        self._bd_lo = lo
        self._centroids_norm = (self._centroids - lo) / self._bd_span

        self.final_fallback = final_fallback or self._default_final_fallback()
        self.stats = DispatchStats()
        self._policy_cache: dict = {}
        self.reset()

    # ------------------------------------------------------------------ #
    def _default_final_fallback(self):
        """G36 when no (context, niche) is feasible anywhere.

        This used to try `GurobiEconomicMPC` first and fall through to G36 on any
        exception. Both the MPC tier and the try/except are gone: the MPC needed a
        gurobipy licence, so on most machines the fallback silently WAS G36 while
        the code read as though it were an optimiser -- meaning the arm that
        actually ran depended on the licence state of the host. One unconditional
        fallback is honest and reproducible. `final_fallback=` is still accepted by
        `__init__` for callers that want to inject something else.
        """
        from .baselines import G36Controller
        return G36Controller(self.cfg)

    def _normalize_bd(self, v: np.ndarray) -> np.ndarray:
        return (np.asarray(v, dtype=float) - self._bd_lo) / self._bd_span

    # ------------------------------------------------------------------ #
    def reset(self):
        self._prev_obs = None
        self._window = deque(maxlen=self.window_steps)
        self._last_day = None
        self._last_d = self._archive_mean_measures_raw()
        self.c_current = None
        self._pending_c_candidate = None
        self._pending_c_count = 0
        self._j_current = None
        self._pending_j_candidate = None
        self._pending_j_count = 0
        if hasattr(self.final_fallback, "reset"):
            self.final_fallback.reset()

    # ------------------------------------------------------------------ #
    # context layer (daily)
    # ------------------------------------------------------------------ #
    def _update_context(self, day_index: int):
        """§8.7 steps 1-3. `day_index` is TODAY's real calendar day; the
        "forecast" is a perfect look-ahead at TODAY's own already-computed
        features (module docstring's forecast note) -- for a backtest this is
        equivalent to forecasting yesterday for today, just evaluated at the
        boundary instead of the day before it."""
        F = self.arc.features[day_index]
        point = project_context_features(self.arc, F)
        candidate = nearest_context_cell(self.arc, point)

        if self.c_current is None:
            self.c_current = candidate
        elif candidate == self.c_current:
            self._pending_c_candidate, self._pending_c_count = None, 0
        else:
            if candidate == self._pending_c_candidate:
                self._pending_c_count += 1
            else:
                self._pending_c_candidate, self._pending_c_count = candidate, 1
            if self._pending_c_count >= self.confirm_days:
                self.c_current = candidate
                self._pending_c_candidate, self._pending_c_count = None, 0
                self.stats.context_switches += 1

        self.stats.context_cell_log.append((day_index, self.c_current))
        name = self.arc.cell_name(self.c_current)
        if name in self.stats.sentinel_days:
            self.stats.sentinel_days[name] += 1

    # ------------------------------------------------------------------ #
    # behaviour layer (every control step)
    # ------------------------------------------------------------------ #
    def _archive_mean_measures_raw(self) -> np.ndarray:
        arrs = self.pa.to_arrays()
        if len(arrs["measures"]) == 0:
            return np.zeros(len(self.pa.bd_ranges))
        return arrs["measures"].mean(axis=0)

    def _windowed_bd_estimate(self) -> np.ndarray:
        """§8.4's four axes, computed over the trailing realised window
        instead of a synthetic probe battery (module docstring)."""
        if len(self._window) < 4:
            return self._last_d
        arr = np.array(self._window, dtype=float)
        t_oa, w_oa, a0, a1, Tz, occ, enable = arr.T
        d0 = _ols_slope(t_oa, a0)
        d1 = _ols_slope(w_oa, a1)
        unocc = occ <= self.cfg.occ_unoccupied_max
        normal = unocc & (Tz < T_Z_DRIFT_THRESHOLD_C)
        drift = unocc & (Tz >= T_Z_DRIFT_THRESHOLD_C)
        d2 = float(enable[normal].mean()) if normal.any() else float(self._last_d[2])
        d3 = float(enable[drift].mean()) if drift.any() else float(self._last_d[3])
        d = np.array([d0, d1, d2, d3])
        self._last_d = d
        return d

    def _filled_niches(self, context_cell: int):
        """`(index, solution, objective)` for every filled niche in a context
        cell. `objective` is the elite's training fitness"""
        arc = self.pa.archive_for(context_cell)
        if arc.stats.num_elites == 0:
            return (np.empty(0, dtype=int),
                    np.empty((0, self.pa.solution_dim)),
                    np.empty(0, dtype=float))
        d = arc.data(["solution", "index", "objective"])
        return (np.asarray(d["index"]), np.asarray(d["solution"]),
                np.asarray(d["objective"], dtype=float))

    def _select_niche(self, d_t: np.ndarray):
        d_norm = self._normalize_bd(d_t)
        c = self.c_current
        idx, sols, objs = self._filled_niches(c)
        tier = "primary"
        source_cell = c
        if len(idx) == 0:
            tier = "context_fallback"
            self.stats.context_empty_fallback_steps += 1
            for other in self._other_order[c]:
                idx2, sols2, objs2 = self._filled_niches(other)
                if len(idx2) > 0:
                    idx, sols, objs, source_cell = idx2, sols2, objs2, other
                    break
        if len(idx) == 0:
            return None, "final_fallback"

        dists = ((self._centroids_norm[idx] - d_norm) ** 2).sum(axis=1)
        nearest_pos = int(np.argmin(dists))
        if self.tiebreak_eps > 0.0:
            # FITNESS TIEBREAK. Everything within `(1 + eps) * d_min` is
            # behaviourally indistinguishable as far as the probe-BD can tell, 
            # so pick the highest-fitness one among them.
            band = dists <= dists[nearest_pos] * (1.0 + self.tiebreak_eps) + 1e-12
            band_pos = np.flatnonzero(band)
            # Highest fitness in the band
            order = np.lexsort((dists[band_pos], -objs[band_pos]))
            best_pos = int(band_pos[int(order[0])])
            self.stats.tiebreak_band_candidates += int(band_pos.size)
            if best_pos != nearest_pos:
                self.stats.tiebreak_wins += 1
            chosen_pos = best_pos
        else:
            self.stats.tiebreak_band_candidates += 1
            chosen_pos = nearest_pos
        candidate_j = int(idx[chosen_pos])

        # switch_hysteresis=3. Niche identity `j` indexes the ONE
        # shared behaviour tessellation (product_archive.py)
        if self._j_current is None:
            self._j_current = candidate_j
        elif candidate_j == self._j_current:
            self._pending_j_candidate, self._pending_j_count = None, 0
        else:
            if candidate_j == self._pending_j_candidate:
                self._pending_j_count += 1
            else:
                self._pending_j_candidate, self._pending_j_count = candidate_j, 1
            if self._pending_j_count >= self.niche_hyst:
                self._j_current = candidate_j
                self._pending_j_candidate, self._pending_j_count = None, 0
                self.stats.niche_switches += 1

        cur_j = self._j_current
        if cur_j not in idx:
            # the held niche has no elite in the resolved source cell (e.g.
            # just fell back to a different cell) -- cannot persist it
            cur_j = candidate_j
            self._j_current = candidate_j

        pos = int(np.where(idx == cur_j)[0][0])
        theta = sols[pos]
        # `objs[pos]` is the fitness of the elite ACTUALLY dispatched -- after
        # hysteresis, not the candidate the tiebreak proposed
        fitness = float(objs[pos])
        self.stats.deployed_fitness_sum += fitness
        self.stats.deployed_fitness_steps += 1
        return (source_cell, cur_j, theta, fitness), tier

    def _policy_for(self, cell: int, j: int, theta: np.ndarray):
        key = (cell, j)
        pol = self._policy_cache.get(key)
        if pol is None:
            pol = build_dispatch_policy(theta, feature_dim_for(self.cfg),
                                        self.act_dim, self.hidden,
                                        policy_kind=self.policy_kind)
            self._policy_cache[key] = pol
        return pol

    # ------------------------------------------------------------------ #
    def act(self, obs, info) -> dict:
        x = augment_obs(obs, self._prev_obs, self.n_zones)
        self._prev_obs = np.asarray(obs, dtype=float).copy()

        # `info` is None on the very first call of an episode before any
        # step() has run. Day 0's context must still be resolved before the
        # first action, so default to day_index=0 rather than skip the update.
        day_index = int(info.get("source_day", 0)) if info is not None else 0
        if day_index != self._last_day:
            self._update_context(day_index)
            self._last_day = day_index

        d_t = self._windowed_bd_estimate()
        result, tier = self._select_niche(d_t)

        if result is None:
            self.stats.final_fallback_steps += 1
            action = self.final_fallback.act(obs, info)
            log_cell, log_j, log_fit = -1, -1, float("nan")
        else:
            source_cell, j, theta, fitness = result
            pol = self._policy_for(source_cell, j, theta)
            a = pol.act(x)
            action = action_from_vector(a, self.cfg)
            log_cell, log_j, log_fit = source_cell, j, fitness

        self.stats.n_steps += 1
        if info is not None:
            # `fitness` joins the niche log so a run's deployed-quality can be
            # read straight off the CSV, instead of joining it back against an
            # archive snapshot after the fact
            self.stats.niche_log.append(
                (info.get("source_day", -1), info.get("hour", -1.0),
                 log_cell, log_j, tier, log_fit))
            a0n = _norm_chwst(action["chwst"], self.cfg)
            a1n = _norm_fan(action["cw_fan"])
            enable = 1.0 if action.get("plant_enable", True) else 0.0
            # `info` (environment.py::_obs / _aggregate) does not carry w_oa --
            # only t_oa/t_wb are exported, so compute it here from the env's weather table
            w_oa = self._weather_w_oa(info)
            self._window.append((info["t_oa"], w_oa, a0n, a1n,
                                 info["mean_Tz"], info["occ"], enable))
        return action

    # ------------------------------------------------------------------ #
    def bind_env(self, env):
        """Attach the episode's env so `act()` can read `w_oa` (not exported
        in `info`, see `_weather_w_oa`). The evaluation harness
        (`experiments.evaluate.evaluate_controller`) calls `bind_env` on any
        controller that defines it, before the episode starts."""
        self._env = env

    def _weather_w_oa(self, info) -> float:
        env = getattr(self, "_env", None)
        if env is None:
            return 0.0
        hour = float(info.get("hour", env.hour)) - env.physics_dt_h
        return float(env.weather.at(hour).w_oa)

    # `experiments.evaluate.evaluate_controller` reads `niche_switches` as a
    # flat attribute (`getattr(controller, "niche_switches", 0)`)
    @property
    def niche_switches(self) -> int:
        return self.stats.niche_switches

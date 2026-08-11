"""Warm-start bank — day-of-week stratified thermal state

Builds ``BANK[dow] = [(T_z, T_m, W_z), ...]`` by running ONE full-year (8,760 h)
sequential rollout under bare `G36Controller`, and recording the mean-over-zones 
state every time the walk crosses `occupied_start_h` on a new calendar day. 
Each recorded state is tagged with its day-of-week
"""
from __future__ import annotations

import hashlib
import os
import pickle
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

DOW_LABELS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")   # index = dow
WEEKEND_DOW = (5, 6)                                             # Sat, Sun
DEFAULT_FIRST_WEEKDAY = 6      # the .mos year's day 0 is a Sunday (weather_mos.py)

# v2: adds `daily_g36_return`, F_G36[day] baseline cache, recorded
# from the SAME full-year always-on-G36 rollout the bank already runs
WARMSTART_SCHEMA_VERSION = 2


@dataclass
class WarmstartBank:
    """`BANK[dow]` plus the provenance needed to tell a stale cache from a fresh one."""
    bank: dict            # dow(int) -> list[(T_z, T_m, W_z)]
    mos_path: str
    price_profile: str
    first_weekday: int
    reference_schedule: str = "always_on_g36"
    n_zones: int = 0
    build_seconds: float = 0.0
    daily_g36_return: dict = field(default_factory=dict)

    def n(self, dow: int) -> int:
        return len(self.bank.get(dow, []))

    def sample(self, dow: int, rng: np.random.Generator | None = None):
        """Draw one (T_z, T_m, W_z) uniformly from `BANK[dow]` (notes.md §8.1
        "Sampling procedure at training/evaluation time")."""
        states = self.bank[dow]
        rng = rng or np.random.default_rng()
        return states[int(rng.integers(len(states)))]

    def f_g36(self, day: int) -> float:
        """F_G36[day] (notes.md §8.6.3) -- raises rather than silently
        defaulting to 0.0, which would look like a genuine break-even
        baseline instead of a missing entry."""
        try:
            return float(self.daily_g36_return[int(day)])
        except KeyError as e:
            raise KeyError(
                f"no F_G36 baseline recorded for day {day} -- this bank "
                f"predates WARMSTART_SCHEMA_VERSION={WARMSTART_SCHEMA_VERSION} "
                f"(v2 added daily_g36_return); rebuild with "
                f"build_warmstart_bank(rebuild=True)") from e


# --------------------------------------------------------------------------- #
def _cache_path(mos_path: str | None, price_profile: str, first_weekday: int) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    from .scenarios import _config_signature, SIZING_SCHEMA_VERSION
    tag = hashlib.sha1(
        f"{mos_path}|{price_profile}|{first_weekday}"
        f"|sizing_v{SIZING_SCHEMA_VERSION}|cfg{_config_signature()}"
        .encode()).hexdigest()[:12]
    d = os.path.join(here, "_cache")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"warmstart_bank_v{WARMSTART_SCHEMA_VERSION}_{tag}.pkl")


def build_warmstart_bank(mos_path: str | None = None, price_profile: str | None = None,
                         first_weekday: int = DEFAULT_FIRST_WEEKDAY,
                         use_disk_cache: bool = True,
                         rebuild: bool = False) -> WarmstartBank:
    """Run the full-year always-on-G36 rollout and return the day-of-week bank.

    Cached on disk (~57.6 s [MEAS], notes.md §8.1), invalidated the same way
    `scenarios.get_annual_context` is: any change to `_CFG_SIGNATURE_FIELDS`
    or `SIZING_SCHEMA_VERSION` rebuilds rather than silently serving a stale
    pickle (`scenarios.py`'s own rationale, reused verbatim here).
    """
    from .config import resolve_price_profile
    price = resolve_price_profile(price_profile)
    cache = _cache_path(mos_path, price, first_weekday) if use_disk_cache else None
    if cache and not rebuild and os.path.exists(cache):
        try:
            with open(cache, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass                                    # stale/corrupt -> rebuild

    import time
    t0 = time.time()

    from .baselines import G36Controller
    from .scenarios import make_dispatch_env

    env, ctx, spec = make_dispatch_env(full_year=True, price_profile=price,
                                       mos_path=mos_path)
    controller = G36Controller(ctx.cfg)

    obs = env.reset()
    if hasattr(controller, "reset"):
        controller.reset()

    bank: dict[int, list] = defaultdict(list)
    daily_return: dict[int, float] = defaultdict(float)
    occ_h = float(ctx.cfg.occupied_start_h)
    next_target = occ_h                 # absolute `env.hour` of day 0's crossing
    info = None
    done = False
    while not done:
        action = controller.act(obs, info)
        obs, reward, done, info = env.step(action)
        daily_return[info["day_slot"]] += reward
        if env.hour >= next_target:
            day_index = int(round((next_target - occ_h) / 24.0))
            dow = (first_weekday + day_index) % 7
            bank[dow].append((float(env.T_z.mean()), float(env.T_m.mean()),
                              float(env.W_z.mean())))
            next_target += 24.0

    result = WarmstartBank(bank=dict(bank), mos_path=mos_path or "default",
                           price_profile=price, first_weekday=first_weekday,
                           n_zones=env.n_zones, build_seconds=time.time() - t0,
                           daily_g36_return=dict(daily_return))
    if cache:
        try:
            with open(cache, "wb") as f:
                pickle.dump(result, f)
        except Exception:
            pass
    return result


# --------------------------------------------------------------------------- #
def _welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch's t (unequal-variance two-sample), no scipy dependency."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    na, nb = len(a), len(b)
    se = np.sqrt(va / na + vb / nb)
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def bank_stats(wb: WarmstartBank) -> dict:
    """Per-dow mean/sd of T_z, T_m, W_z, plus the between/within-dow """
    per_dow = {}
    tm_arrays = {}
    for dow in range(7):
        states = np.asarray(wb.bank.get(dow, []))
        if states.size == 0:
            continue
        tz, tm, wz = states[:, 0], states[:, 1], states[:, 2]
        tm_arrays[dow] = tm
        per_dow[dow] = dict(
            label=DOW_LABELS[dow], n=len(states),
            Tz_mean=float(tz.mean()), Tz_sd=float(tz.std(ddof=1)),
            Tm_mean=float(tm.mean()), Tm_sd=float(tm.std(ddof=1)),
            Wz_mean=float(wz.mean()), Wz_sd=float(wz.std(ddof=1)),
        )

    stratum_means = np.array([per_dow[d]["Tm_mean"] for d in sorted(per_dow)])
    between_dow_sd = float(stratum_means.std(ddof=1))
    within_sds = np.array([per_dow[d]["Tm_sd"] for d in sorted(per_dow)])
    # plain mean of the seven within-dow sds
    pooled_within_sd = float(within_sds.mean())

    # Welch's t of each dow's T_m against the Tue-Fri pool
    weekday_pool_dows = [d for d in (1, 2, 3, 4) if d in tm_arrays]   # Tue..Fri
    pool = np.concatenate([tm_arrays[d] for d in weekday_pool_dows]) \
        if weekday_pool_dows else np.array([])
    welch_t = {}
    for dow in per_dow:
        if dow in weekday_pool_dows or pool.size == 0:
            continue
        welch_t[dow] = _welch_t(tm_arrays[dow], pool)

    return dict(per_dow=per_dow, between_dow_sd=between_dow_sd,
               pooled_within_dow_sd=pooled_within_sd,
               ratio=between_dow_sd / pooled_within_sd if pooled_within_sd else float("nan"),
               welch_t=welch_t, weekday_pool_dows=weekday_pool_dows)

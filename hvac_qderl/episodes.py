"""Episode construction — representative-day training episodes and full-year
test-only dispatch.

Three builders, and one hard rule:
    build_training_episode(...)     n_rep_days + min-PV + peak-load days.  TRAINING.
    build_context_episode(...)      ONE context-cell day.  TRAINING (QD arm).
    build_context_set_episode(...)  ALL 18 context-cell days, one episode.  TRAINING.
    build_dispatch_episode(...)     the same n_rep_days + 2 set, annual-weighted.  TEST.
    build_full_year_episode(...)    all 365 days / 8,760 h.  **TEST ONLY.**

`assert_not_training` enforces this at the call site, so a full-year weather driver
cannot be handed to a learner by accident.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cooling_load import AnnualLoad, compute_annual_load
from .representative_periods import RepresentativeSet, build_from_weather
from .config import resolve_price_profile
from .weather_mos import AnnualWeather, MosWeather, parse_mos, DAYS_PER_YEAR


class TrainingOnFullYearError(RuntimeError):
    """Raised when a full-year episode is about to be used for training."""


@dataclass
class EpisodeSpec:
    """Everything needed to instantiate an environment for one episode design."""
    weather: MosWeather
    kind: str                       # "representative" | "full_year"
    n_days: int
    horizon_hours: float
    day_weights: np.ndarray
    rep_set: RepresentativeSet | None = None
    train_allowed: bool = True

    def describe(self) -> str:
        w = self.day_weights
        return (f"{self.kind}: {self.n_days} days, {self.horizon_hours:,.0f} h, "
                f"Σweights={w.sum():.0f} "
                f"({'training allowed' if self.train_allowed else 'TEST ONLY'})")


# --------------------------------------------------------------------------- #
def prepare_annual(mos_path: str | None = None, cfg=None,
                   rh_target: float = 0.55):
    """Parse weather + compute the annual cooling load once (shared by builders)."""
    annual = parse_mos(mos_path)
    load = compute_annual_load(annual, cfg=cfg, rh_target=rh_target)
    return annual, load


def build_representative_set(annual: AnnualWeather, load: AnnualLoad,
                             n_rep_days: int, seed: int = 0) -> RepresentativeSet:
    return build_from_weather(annual, load, n_rep_days=n_rep_days, seed=seed)


# --------------------------------------------------------------------------- #
def build_training_episode(annual: AnnualWeather, load: AnnualLoad,
                           n_rep_days: int = 4, price_profile: str | None = None,
                           seed: int = 0,
                           rep_set: RepresentativeSet | None = None) -> EpisodeSpec:
    """The training episode: `n_rep_days` cluster medoids + min-PV + peak-load day.
    """
    rs = rep_set or build_representative_set(annual, load, n_rep_days, seed)
    w = MosWeather(annual, rs.day_indices, price_profile=price_profile,
                   day_weights=rs.weights)
    return EpisodeSpec(weather=w, kind="representative", n_days=rs.n_days,
                       horizon_hours=w.horizon_hours, day_weights=rs.weights,
                       rep_set=rs, train_allowed=True)


def build_dispatch_episode(annual: AnnualWeather, load: AnnualLoad,
                           n_rep_days: int = 4, price_profile: str | None = None,
                           seed: int = 0,
                           rep_set: RepresentativeSet | None = None) -> EpisodeSpec:
    """**Dispatch option A** — the n_rep_days + 2 set, annual-weighted.

    Cheap (a few hundred hours) and unbiased for annual totals because each day
    carries its cluster weight; use this for routine controller comparison.
    """
    spec = build_training_episode(annual, load, n_rep_days, price_profile, seed,
                                  rep_set)
    spec.kind = "representative"
    return spec


def build_context_episode(annual: AnnualWeather, load: AnnualLoad,
                          context_archive, cell_id: int,
                          price_profile: str | None = None) -> tuple[EpisodeSpec, int]:
    """**Context episode**: the single real calendar day
    nearest a context cell's centroid, 24 h / 72 control steps at the
    20-min cadence. This is the training unit of the contextual QD proposal

    `load` is accepted (not read) for signature symmetry with the other
    builders here; the day is resolved purely from `context_archive`, which
    was itself built from `data/annual_load.csv`.

    Returns `(spec, day_index)`
    """
    from .context_archive import nearest_day_to_centroid
    day = nearest_day_to_centroid(context_archive, cell_id)
    from .config import resolve_price_profile
    price = resolve_price_profile(price_profile)
    w = MosWeather(annual, [day], price_profile=price, day_weights=[1.0])
    spec = EpisodeSpec(weather=w, kind="context", n_days=1, horizon_hours=w.horizon_hours,
                       day_weights=np.array([1.0]), rep_set=None, train_allowed=True)
    return spec, day


def build_context_set_episode(annual: AnnualWeather, load: AnnualLoad,
                              context_archive, price_profile: str | None = None,
                              extremes_last: bool = True) -> EpisodeSpec:
    """**Context-set episode**: ALL 18 context cells' days in ONE episode,
    18 x 24 h = 432 h = 1,296 control steps at the 20-min cadence.

    This is the episode design that puts a NON-CONTEXTUAL learner (`sac.py`) on
    the same training distribution as the contextual QD arm (`qd_erl_contextual.py`)

    DAY WEIGHTS are the context archive's own cell populations
    (`ContextArchive.population()`, Sigma == 365 by construction

    `load` is accepted and not read
    the days are resolved purely from `context_archive`, as in
    `build_context_episode`).
    """
    from .context_archive import nearest_day_to_centroid, N_CLUSTERS, N_CELLS

    days_by_cell = [nearest_day_to_centroid(context_archive, c)
                    for c in range(N_CELLS)]
    weights_by_cell = context_archive.population().astype(float)

    if extremes_last:
        # 16 k-means cells chronologically, then S_minpv (17), then S_peak (16).
        order = sorted(range(N_CLUSTERS), key=lambda c: days_by_cell[c])
        order += [N_CLUSTERS + 1, N_CLUSTERS]
    else:
        order = sorted(range(N_CELLS), key=lambda c: days_by_cell[c])

    days = [days_by_cell[c] for c in order]
    weights = np.array([weights_by_cell[c] for c in order], dtype=float)

    price = resolve_price_profile(price_profile)
    w = MosWeather(annual, days, price_profile=price, day_weights=weights)
    return EpisodeSpec(weather=w, kind="context_set", n_days=len(days),
                       horizon_hours=w.horizon_hours, day_weights=weights,
                       rep_set=None, train_allowed=True)


def build_full_year_episode(annual: AnnualWeather, price_profile: str | None = None
                            ) -> EpisodeSpec:
    """**Dispatch option B** — the complete 8,760 h year. TEST ONLY.

    Every day carries weight 1.0, so KPIs are true annual totals with no clustering
    approximation. `train_allowed=False` makes misuse detectable.
    """
    days = list(range(DAYS_PER_YEAR))
    w = MosWeather(annual, days, price_profile=price_profile,
                   day_weights=np.ones(DAYS_PER_YEAR))
    return EpisodeSpec(weather=w, kind="full_year", n_days=DAYS_PER_YEAR,
                       horizon_hours=w.horizon_hours,
                       day_weights=np.ones(DAYS_PER_YEAR), rep_set=None,
                       train_allowed=False)


# --------------------------------------------------------------------------- #
def assert_not_training(spec: EpisodeSpec, context: str = "training"):
    """Guard: refuse to use a test-only episode for training."""
    if not spec.train_allowed:
        raise TrainingOnFullYearError(
            f"Episode kind '{spec.kind}' is TEST ONLY and must not be used for "
            f"{context}. Full-year episodes leave no held-out data (any annual "
            f"saving would be in-sample) and a 105,120-step episode destroys "
            f"credit assignment and population throughput. Use "
            f"build_training_episode(n_rep_days=...) instead. "
            f"See hvac_qderl/episodes.py for the full rationale.")
    return spec


def make_env(spec: EpisodeSpec, cfg=None, for_training: bool = False):
    """Instantiate an environment for an episode spec, enforcing the training rule."""
    from .config import default_singapore_config
    from .environment import HVACPlantEnv
    if for_training:
        assert_not_training(spec, "training")
    return HVACPlantEnv(config=cfg or default_singapore_config(),
                        weather=spec.weather)


# --------------------------------------------------------------------------- #
def annualise(trace, dt_h: float) -> dict:
    """Scale a (possibly weighted, possibly short) episode trace to annual totals.
    For a representative episode each day's contribution is multiplied by its
    cluster weight; for a full-year episode the weights are 1.0 so this returns the
    directly simulated annual total.
    """
    if not trace:
        return {}
    energy_w = cost_w = 0.0
    energy_raw = 0.0
    # ENERGY SPLIT + DELIVERED LOAD
    cool_w = chiller_w = 0.0
    n_days_seen = {}
    for t in trace:
        wgt = t.get("day_weight", 1.0)
        e = t["p_total_kw"] * dt_h
        energy_raw += e
        energy_w += e * wgt
        cool_w += t.get("q_evap_kw", 0.0) * dt_h * wgt
        chiller_w += t.get("p_chiller_kw", 0.0) * dt_h * wgt
        cost_w += t.get("cost_sgd", 0.0) * wgt
        n_days_seen[t.get("day_slot", 0)] = wgt
    rh_viol = float(np.mean([1.0 if t["rh_violation"] > 0 else 0.0 for t in trace]))
    pmv = float(np.mean([1.0 if t["pmv_disc"] > 0 else 0.0 for t in trace]))
    peak = max(t["p_total_kw"] for t in trace)
    return {
        "annual_energy_kwh": energy_w,
        "annual_cost_sgd": cost_w,
        "annual_cool_kwh": cool_w,
        "annual_chiller_kwh": chiller_w,
        "annual_aux_kwh": energy_w - chiller_w,
        "episode_energy_kwh": energy_raw,
        "episode_days": len(n_days_seen),
        "weight_sum": float(sum(n_days_seen.values())),
        "peak_kw": peak,
        "rh_violation_rate": rh_viol,
        "pmv_disc_rate": pmv,
    }

"""Single place that decides which environment an episode runs in.

Before this module the environment was hard-wired to a synthetic 24 h day. Now
every consumer (learners, evaluation, baselines) asks here for an env, and gets
one of three designs:

    training      n_rep_days + min-PV + peak-load days of real .mos weather
    dispatch_rep  the same set, annual-weighted            (dispatch option A)
    dispatch_year the full 8,760 h year                    (dispatch option B, TEST ONLY)

The annual weather + cooling load + representative set are computed **once** and
cached process-wide, because clustering the year on every worker process would
dominate runtime for a 20-core population rollout.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from .cooling_load import compute_annual_load, apply_load_derived_airflow
from .config import SingaporeConfig, default_singapore_config
from .episodes import (EpisodeSpec, build_dispatch_episode, build_full_year_episode,
                       build_training_episode, assert_not_training, make_env)
from .representative_periods import RepresentativeSet
from .weather_mos import parse_mos

# --------------------------------------------------------------------------- #
# defaults (overridable by env vars so worker processes inherit them)
# --------------------------------------------------------------------------- #
DEFAULT_N_REP_DAYS = int(os.environ.get("HVAC_N_REP_DAYS", "4"))
from .config import DEFAULT_PRICE_PROFILE as DEFAULT_PRICE  # noqa: E402
DEFAULT_MOS = os.environ.get("HVAC_MOS_PATH") or None

_CACHE: dict = {}


@dataclass
class AnnualContext:
    """Weather + load + representative set + a plant sized to the calculated load."""
    annual: object
    load: object
    rep_set: RepresentativeSet
    cfg: SingaporeConfig
    n_rep_days: int

    def summary(self) -> str:
        return (self.annual.summary() + "\n" + self.load.summary() + "\n"
                + self.rep_set.summary())

# v4: DESIGN_LOAD_KW (4,797 kW)
# CHILLER_CAPACITY_RT (4 x 360 RT = 1,440 RT)
SIZING_SCHEMA_VERSION = 4


# Config fields that change what an episode DOES, and therefore 
# must invalidate the cached AnnualContext
_CFG_SIGNATURE_FIELDS = (
    "w_energy", "w_comfort", "w_shield",
    "humidity_shield", "comfort_shield", "overtemp_guard",
    "comfort_shield_deadband", "allow_plant_off",
    "plant_min_on_min", "plant_min_off_min",
    "control_step_min", "physics_step_min",
    "T_set", "T_band", "pmv_band", "RH_ceiling", "RH_mould_limit",
    "T_max_unocc", "dewpoint_max_unocc", "dewpoint_margin_K",
    "vav_min_frac_unocc", "optimum_start_max_lead_h",
    "energy_tariff", "demand_charge",
    "design_cooling_kw",
)


def _config_signature() -> str:
    """Hash of the behaviour-affecting config defaults, for the cache key."""
    import hashlib
    from .config import SingaporeConfig
    base = SingaporeConfig()
    blob = "|".join(f"{f}={getattr(base, f, None)!r}"
                    for f in _CFG_SIGNATURE_FIELDS)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def _disk_cache_path(n_rep: int, path, seed: int, rescale: bool) -> str:
    import hashlib
    here = os.path.dirname(os.path.abspath(__file__))
    tag = hashlib.sha1(
        f"{n_rep}|{path}|{seed}|{rescale}|v{SIZING_SCHEMA_VERSION}"
        f"|cfg{_config_signature()}"
        .encode()).hexdigest()[:12]
    d = os.path.join(here, "_cache")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"annual_ctx_v{SIZING_SCHEMA_VERSION}_{tag}.pkl")


def get_annual_context(n_rep_days: int | None = None, mos_path: str | None = None,
                       seed: int = 0, use_load_shr: bool = True,
                       use_disk_cache: bool = True) -> AnnualContext:
    """Cached annual weather / load / clustering, and the plant config.
    """
    n_rep = DEFAULT_N_REP_DAYS if n_rep_days is None else int(n_rep_days)
    path = mos_path or DEFAULT_MOS
    key = (n_rep, path, seed, use_load_shr)
    if key in _CACHE:
        return _CACHE[key]

    disk = _disk_cache_path(n_rep, path, seed, use_load_shr) if use_disk_cache else None
    if disk and os.path.exists(disk):
        try:
            import pickle
            with open(disk, "rb") as f:
                ctx = pickle.load(f)
            _CACHE[key] = ctx
            return ctx
        except Exception:
            pass                                   # stale/corrupt -> rebuild

    annual = parse_mos(path)
    base = default_singapore_config()
    load = compute_annual_load(annual, cfg=base)
    cfg = apply_load_derived_airflow(base, load) if use_load_shr else base
    from .representative_periods import build_from_weather
    rep = build_from_weather(annual, load, n_rep_days=n_rep, seed=seed)
    ctx = AnnualContext(annual=annual, load=load, rep_set=rep, cfg=cfg,
                        n_rep_days=n_rep)
    _CACHE[key] = ctx
    if disk:
        try:
            import pickle
            with open(disk, "wb") as f:
                pickle.dump(ctx, f)
        except Exception:
            pass
    return ctx


# --------------------------------------------------------------------------- #
def make_episode_spec(mode: str = "training", n_rep_days: int | None = None,
                      price_profile: str | None = None,
                      mos_path: str | None = None, seed: int = 0) -> EpisodeSpec:
    """mode: training | dispatch_rep | dispatch_year."""
    ctx = get_annual_context(n_rep_days, mos_path, seed)
    from .config import resolve_price_profile
    price = resolve_price_profile(price_profile)
    if mode == "dispatch_year":
        return build_full_year_episode(ctx.annual, price_profile=price)
    if mode == "dispatch_rep":
        return build_dispatch_episode(ctx.annual, ctx.load,
                                      n_rep_days=ctx.n_rep_days,
                                      price_profile=price, seed=seed,
                                      rep_set=ctx.rep_set)
    if mode == "training":
        return build_training_episode(ctx.annual, ctx.load,
                                      n_rep_days=ctx.n_rep_days,
                                      price_profile=price, seed=seed,
                                      rep_set=ctx.rep_set)
    raise ValueError(f"unknown episode mode {mode!r}")


def make_training_env(n_rep_days: int | None = None,
                      price_profile: str | None = None,
                      mos_path: str | None = None, seed: int = 0):
    """The env every learner should use: representative days, real weather.

    Raises if handed a test-only design, so a full-year env can never be trained on.
    """
    ctx = get_annual_context(n_rep_days, mos_path, seed)
    spec = make_episode_spec("training", n_rep_days, price_profile, mos_path, seed)
    assert_not_training(spec, "training")
    return make_env(spec, cfg=ctx.cfg, for_training=True), ctx, spec


# --------------------------------------------------------------------------- #
# Cache the bank and centroids alongside AnnualContext,
# --------------------------------------------------------------------------- #
_CTX_ARCHIVE_CACHE: dict = {}
_WARMSTART_CACHE: dict = {}


def get_context_archive(csv_path: str | None = None, rep_days_csv: str | None = None):
    """Cached §8.2 context archive (16 data-driven cells + 2 sentinels)."""
    key = (csv_path, rep_days_csv)
    if key in _CTX_ARCHIVE_CACHE:
        return _CTX_ARCHIVE_CACHE[key]
    from .context_archive import build_context_archive
    arc = build_context_archive(csv_path=csv_path, rep_days_csv=rep_days_csv)
    _CTX_ARCHIVE_CACHE[key] = arc
    return arc


def make_context_set_env(price_profile: str | None = None,
                         mos_path: str | None = None, seed: int = 0,
                         extremes_last: bool = True):
    """Training env over ALL 18 context-cell days in ONE 432 h episode.
    """
    from .episodes import build_context_set_episode
    ctx = get_annual_context(None, mos_path, seed)
    spec = build_context_set_episode(ctx.annual, ctx.load, get_context_archive(),
                                     price_profile=price_profile,
                                     extremes_last=extremes_last)
    assert_not_training(spec, "training")
    return make_env(spec, cfg=ctx.cfg, for_training=True), ctx, spec


def get_warmstart_bank(mos_path: str | None = None, price_profile: str | None = None,
                       use_disk_cache: bool = True):
    """Cached §8.1 warm-start bank (also carries the §8.6.3 F_G36 baseline)."""
    path = mos_path or DEFAULT_MOS
    from .config import resolve_price_profile
    price = resolve_price_profile(price_profile)
    key = (path, price)
    if key in _WARMSTART_CACHE:
        return _WARMSTART_CACHE[key]
    from .warmstart import build_warmstart_bank
    wb = build_warmstart_bank(mos_path=mos_path, price_profile=price,
                              use_disk_cache=use_disk_cache)
    _WARMSTART_CACHE[key] = wb
    return wb


def make_dispatch_env(full_year: bool = False, n_rep_days: int | None = None,
                      price_profile: str | None = None,
                      mos_path: str | None = None, seed: int = 0):
    """Evaluation env. `full_year=True` selects dispatch option B (test only)."""
    ctx = get_annual_context(n_rep_days, mos_path, seed)
    mode = "dispatch_year" if full_year else "dispatch_rep"
    spec = make_episode_spec(mode, n_rep_days, price_profile, mos_path, seed)
    return make_env(spec, cfg=ctx.cfg, for_training=False), ctx, spec

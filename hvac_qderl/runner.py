"""Episode runner + KPI aggregation shared by the example run and tests."""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .environment import HVACPlantEnv
from .equipment import KW_PER_RT


@dataclass
class EpisodeKPIs:
    controller: str
    energy_kwh: float
    cost_sgd: float
    peak_kw: float
    mean_kw_per_rt: float
    mean_Tz: float
    max_Tz: float
    mean_RH: float
    max_RH: float
    rh_violation_rate: float          # fraction of steps any zone > RH ceiling
    pmv_disc_rate: float              # fraction of ALL steps outside PMV band
    mean_shr: float
    chiller_starts: int
    # --- shutdown-aware variants ------------------------------------------ #
    pmv_disc_rate_occ: float = 0.0     # discomfort rate among OCCUPIED steps
    kw_per_rt_on: float = 0.0          # kW/RT while the plant is actually on
    plant_on_frac: float = 1.0         # fraction of steps the plant ran
    # --- load-weighted efficiency ------------------------------------------ #
    kw_per_rt_lw: float = 0.0          # sum(p_total) / sum(q_evap/KW_PER_RT)

    def as_row(self):
        return asdict(self)


def _shutdown_aware(pmv_v, kwrt, occ_v, on_v, occ_thresh: float = 0.30) -> dict:
    """Comfort among occupied steps, and efficiency while the plant is running.
    """
    import numpy as _np
    pmv_a, kw_a = _np.asarray(pmv_v), _np.asarray(kwrt)
    occ_a = _np.asarray(occ_v) if occ_v else _np.ones_like(pmv_a)
    on_a = _np.asarray(on_v) if on_v else _np.ones_like(pmv_a)
    occ_m, on_m = occ_a > occ_thresh, on_a > 0.5
    return dict(
        pmv_disc_rate_occ=round(float(pmv_a[occ_m].mean()), 3) if occ_m.any() else 0.0,
        kw_per_rt_on=round(float(kw_a[on_m].mean()), 3) if on_m.any() else 0.0,
        plant_on_frac=round(float(on_a.mean()), 3))


def run_episode(env: HVACPlantEnv, controller, start_hour: float = 0.0,
                T_z0=None, T_m0=None, W_z0=None):
    """Run one full episode; return (EpisodeKPIs, per-step trace list).

    `T_z0`/`T_m0`/`W_z0` pass straight through to `env.reset()` for a 
    warm-started cold start instead of the default isothermal one
    """
    obs = env.reset(start_hour=start_hour, T_z0=T_z0, T_m0=T_m0, W_z0=W_z0)
    if hasattr(controller, "reset"):
        controller.reset()
    info = None
    trace = []
    energy = cost = 0.0
    # load-weighted kW/RT accumulators
    p_sum = rt_sum = 0.0
    kwrt, tz, rh, shr, pmv_v, rhv = [], [], [], [], [], []
    occ_v, on_v = [], []
    tz_max = 0.0
    rh_max = 0.0
    prev_n = None
    starts = 0
    done = False
    while not done:
        action = controller.act(obs, info)
        obs, reward, done, info = env.step(action)
        energy += info["p_total_kw"] * env.dt_h
        cost += info["cost_sgd"]
        p_sum += info["p_total_kw"]
        rt_sum += info["q_evap_kw"] / KW_PER_RT
        kwrt.append(info["kw_per_rt"])
        tz.append(info["mean_Tz"])
        tz_max = max(tz_max, info["max_Tz"])
        rh.append(info["mean_RH"])
        rh_max = max(rh_max, info["max_RH"])
        shr.append(info["shr"])
        pmv_v.append(1.0 if info["pmv_disc"] > 0 else 0.0)
        occ_v.append(float(info.get("occ", 1.0)))
        on_v.append(float(info.get("plant_enable", 1)))
        rhv.append(1.0 if info["rh_violation"] > 0 else 0.0)
        # staging changes between control steps, plus any that happened *within*
        # the step's physics sub-steps (otherwise sub-stepping would hide cycling)
        if prev_n is not None and info["n_chillers_on"] > prev_n:
            starts += info["n_chillers_on"] - prev_n
        starts += int(info.get("starts_in_step", 0))
        prev_n = info["n_chillers_on"]
        trace.append(info)

    kpis = EpisodeKPIs(
        controller=type(controller).__name__,
        energy_kwh=round(energy, 1),
        cost_sgd=round(cost, 1),
        peak_kw=round(env.peak_kw, 1),
        mean_kw_per_rt=round(float(np.mean(kwrt)), 3),
        mean_Tz=round(float(np.mean(tz)), 2),
        max_Tz=round(tz_max, 2),
        mean_RH=round(float(np.mean(rh)), 3),
        max_RH=round(rh_max, 3),
        rh_violation_rate=round(float(np.mean(rhv)), 3),
        pmv_disc_rate=round(float(np.mean(pmv_v)), 3),
        mean_shr=round(float(np.mean(shr)), 3),
        chiller_starts=int(starts),
        kw_per_rt_lw=round(p_sum / rt_sum, 3) if rt_sum > 1e-9 else 0.0,
        **_shutdown_aware(pmv_v, kwrt, occ_v, on_v))
    return kpis, trace

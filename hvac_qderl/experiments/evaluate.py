"""Evaluation harness: dispatch the trained model in the env and
compare achieved KPIs against the G36 baseline.

Evaluation is **paired**: every controller is run on the identical scenario set
(`SCENARIOS`), so per-scenario deltas are attributable to the controller.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from ..config import default_singapore_config
from ..environment import HVACPlantEnv
from ..runner import run_episode
from ..weather import SingaporeWeather
from ..baselines import G36Controller
from .logging_utils import RunPaths, CsvAppender, EVAL_FIELDS

# held-out evaluation grid 
from ..config import DEFAULT_PRICE_PROFILE, PRICE_PROFILES

# Default grids carry ONLY the project tariff
SCENARIOS = [(d, DEFAULT_PRICE_PROFILE) for d in ("typical_cool", "peak_cool")]
SCENARIOS_ALL = [(d, p) for d in ("typical_cool", "peak_cool")
                 for p in PRICE_PROFILES]

# real-weather dispatch grid: the price profile is the only free axis now, because
# the day set comes from the clustered .mos year rather than a synthetic day type
MOS_SCENARIOS = [("rep", DEFAULT_PRICE_PROFILE)]
MOS_SCENARIOS_ALL = [("rep", p) for p in PRICE_PROFILES]
FULL_YEAR_SCENARIOS = [("full_year", DEFAULT_PRICE_PROFILE)]

# ---- the baseline set -------------------------------------------------- #
DEFAULT_BASELINES = ("G36",)


def scenario_name(day: str, price: str) -> str:
    return f"{day}__{price}"


def evaluate_controller(controller, day: str, price: str, cfg=None,
                        save_trace_to: str | None = None,
                        use_mos: bool = False, n_rep_days=None):
    """Run one paired episode; return (kpis, extras, trace).

    `day` is either a synthetic day type ("typical_cool"/"peak_cool") or, when
    `use_mos=True`, one of "rep" (dispatch option A: n_rep_days + 2, annual
    weighted) / "full_year" (option B: 8,760 h, TEST ONLY).
    """
    if use_mos or day in ("rep", "full_year"):
        from ..scenarios import make_dispatch_env
        env, ctx, spec = make_dispatch_env(full_year=(day == "full_year"),
                                           n_rep_days=n_rep_days,
                                           price_profile=price)
        if cfg is not None and abs(cfg.design_cooling_kw
                                   - ctx.cfg.design_cooling_kw) > 1.0:
            raise ValueError(
                f"controller config is sized for {cfg.design_cooling_kw:,.0f} kW "
                f"but the evaluation environment is {ctx.cfg.design_cooling_kw:,.0f} kW. "
                "Build controllers from `scenarios.get_annual_context().cfg`, "
                "not from `default_singapore_config()`.")
        cfg = ctx.cfg
    else:
        cfg = cfg or default_singapore_config()
        env = HVACPlantEnv(config=cfg, weather=SingaporeWeather(day, price))
    # schedules need the episode's occupancy forecast; controllers are built
    # before the env exists, so bind it here rather than fall back to the clock
    if hasattr(controller, "bind_env"):
        controller.bind_env(env)
    kpis, trace = run_episode(env, controller)
    extras = {
        "shield_correction_mean": float(np.mean([t["correction"] for t in trace])),
        "dispatch_niche_switches": int(getattr(controller, "niche_switches", 0)),
        "episode_return": float(sum(t["reward"] for t in trace)),
    }
    # annualise (weights are 1.0 for a full-year walk, cluster weights otherwise)
    from ..episodes import annualise
    extras.update(annualise(trace, env.dt_h))
    if save_trace_to:
        _write_trace(save_trace_to, trace)
    return kpis, extras, trace


def _write_trace(path: str, trace):
    import csv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = list(trace[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in trace:
            w.writerow({k: round(v, 5) if isinstance(v, float) else v
                        for k, v in r.items()})


def build_controllers(cfg, *, sac_ckpt: str | None = None,
                      prefer_gpu: bool = True) -> dict:
    """Assemble the comparison set. Learner entries are skipped (with a printed
    note) when their checkpoint is missing or torch is unavailable.

    The non-learning set is `DEFAULT_BASELINES` = (G36,) and there is no switch
    to widen it. The `mpc_horizon`, `include_scheduled` and `extra_baselines`
    keywords went out with the four arms they controlled (module docstring); a
    caller still passing them will now fail loudly rather than silently evaluate
    a grid that no longer means what it used to.
    """
    ctrls = {"G36": G36Controller(cfg)}
    # G36 emits no `plant_enable`, so environment.py defaults it True and the
    # bare baseline runs the plant 24/7

    # ---- SAC (torch) ---- #
    if sac_ckpt:
        try:
            from ..experiments.checkpoint import load_agent, resolve_checkpoint
            from ..learners.sac import SACController
            ptr = resolve_checkpoint(sac_ckpt)
            agent, _ = load_agent(ptr["checkpoint"], prefer_gpu=prefer_gpu)
            ctrls["SAC"] = SACController(agent, cfg)
        except Exception as e:
            print(f"  [skip SAC] {type(e).__name__}: {e}")

    return ctrls


def run_evaluation(controllers: dict, paths: RunPaths, seed: int = 0,
                   scenarios=None, checkpoint_env_steps: int = 0,
                   save_traces_for=DEFAULT_BASELINES,
                   verbose: bool = True, use_mos: bool = False,
                   n_rep_days=None) -> list[dict]:
    """Evaluate every controller on every scenario; append rows to eval_log.csv.

    With `use_mos=True` the scenarios are real-weather dispatch designs:
    `("rep", price)` = option A (n_rep_days + 2, annual weighted) and
    `("full_year", price)` = option B (8,760 h, test only).
    """
    scenarios = scenarios or (MOS_SCENARIOS if use_mos else SCENARIOS)
    appender = CsvAppender(paths.eval_log, EVAL_FIELDS)
    rows = []
    _mos_cfg = None
    if use_mos or any(d in ("rep", "full_year") for d, _p in scenarios):
        from ..scenarios import get_annual_context
        _mos_cfg = get_annual_context(n_rep_days).cfg
    for name, ctrl in controllers.items():
        for day, price in scenarios:
            # fresh plant state per episode
            cfg = _mos_cfg if (use_mos or day in ("rep", "full_year")) \
                else default_singapore_config()
            trace_path = None
            if name in tuple(save_traces_for) \
                    and price == DEFAULT_PRICE_PROFILE \
                    and day in ("typical_cool", "rep"):
                trace_path = paths.trace(name, scenario_name(day, price))
            kpis, extras, _ = evaluate_controller(
                ctrl, day, price, cfg, save_trace_to=trace_path,
                use_mos=use_mos or day in ("rep", "full_year"),
                n_rep_days=n_rep_days)
            row = dict(controller=name, seed=seed, scenario_day=day,
                       scenario_price=price,
                       checkpoint_env_steps=checkpoint_env_steps,
                       episode_kind=day if day in ("rep", "full_year") else "synthetic",
                       episode_days=extras.get("episode_days", 1),
                       episode_hours=round(24.0 * extras.get("episode_days", 1), 1),
                       weight_sum=round(extras.get("weight_sum", 1.0), 1),
                       annual_energy_kwh=round(extras.get("annual_energy_kwh", 0.0), 1),
                       annual_cost_sgd=round(extras.get("annual_cost_sgd", 0.0), 1),
                       episode_energy_kwh=round(extras.get("episode_energy_kwh", 0.0), 1),
                       energy_kwh=kpis.energy_kwh, cost_sgd=kpis.cost_sgd,
                       peak_kw=kpis.peak_kw,
                       kw_per_rt=kpis.kw_per_rt_lw,
                       kw_per_rt_on=kpis.kw_per_rt_on,
                       plant_on_frac=kpis.plant_on_frac,
                       rh_violation_rate=kpis.rh_violation_rate,
                       pmv_disc_rate=kpis.pmv_disc_rate,
                       mean_Tz=kpis.mean_Tz, max_Tz=kpis.max_Tz,
                       mean_RH=kpis.mean_RH, max_RH=kpis.max_RH,
                       mean_shr=kpis.mean_shr,
                       chiller_starts=kpis.chiller_starts,
                       shield_correction_mean=round(extras["shield_correction_mean"], 5),
                       dispatch_niche_switches=extras["dispatch_niche_switches"])
            appender.append(row)
            rows.append(row)
            if verbose:
                ann = row.get("annual_energy_kwh") or 0.0
                print(f"  {name:24s} {day:10s} {price:15s} "
                      f"ep={row['energy_kwh']:>9,.0f} kWh  "
                      f"annual={ann/1e6:>6.3f} GWh  "
                      f"kW/RT={row['kw_per_rt']:.3f}  "
                      f"RHviol={row['rh_violation_rate']:.3f}", flush=True)
    return rows


# --------------------------------------------------------------------------- #
def comparison_table(rows: list[dict], reference: str = "G36") -> "list[dict]":
    """Aggregate across scenarios and compute paired deltas vs a reference."""
    import pandas as pd
    df = pd.DataFrame(rows)
    if df.empty:
        return []
    num = ["energy_kwh", "cost_sgd", "peak_kw", "kw_per_rt", "kw_per_rt_on",
           "plant_on_frac",
           "rh_violation_rate", "pmv_disc_rate", "max_RH", "max_Tz",
           "chiller_starts", "shield_correction_mean", "dispatch_niche_switches"]
    agg = df.groupby("controller")[num].mean().reset_index()

    # paired per-scenario energy delta vs reference (more honest than mean-of-means)
    piv = df.pivot_table(index=["scenario_day", "scenario_price"],
                         columns="controller", values="energy_kwh")
    out = []
    for _, r in agg.iterrows():
        c = r["controller"]
        d = dict(r)
        if reference in piv.columns and c in piv.columns:
            rel = 100.0 * (piv[c] - piv[reference]) / piv[reference]
            d[f"energy_pct_vs_{reference}"] = round(float(rel.mean()), 2)
            d[f"energy_pct_vs_{reference}_worst"] = round(float(rel.max()), 2)
        out.append(d)
    return out

#!/usr/bin/env python3
"""Annual dispatch backtest -- G36 vs SAC vs QD-ERL-Context (ContextDispatchController)

Baseline set for this run: G36 (bare, 24/7, no schedule wrapper). 
G36 is seed-independent, so it is only (re-)run when no `G36` row 
is already present in the output CSV; 
use `--arms` to choose which learned arm(s) run.

    python run_annual_dispatch.py                       # G36 (if missing) + QD + SAC, seed 0
    python run_annual_dispatch.py --arms qd              # QD-ERL-Context only (+ G36 if missing)
    python run_annual_dispatch.py --arms sac             # SAC only (+ G36 if missing)
    python run_annual_dispatch.py --seed 1 --arms qd sac
    python run_annual_dispatch.py --run runs/QD-ERL-Context_s0_20260808-010930
    python run_annual_dispatch.py --sac-run runs/SAC_s0_20260808-015441
    python run_annual_dispatch.py --window-steps 18 --arms qd
"""


from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import paths
paths.ensure_dirs()

import numpy as np
import pandas as pd

from hvac_qderl.scenarios import get_annual_context, get_context_archive, \
    get_warmstart_bank, make_dispatch_env
from hvac_qderl.baselines import G36Controller
from hvac_qderl.runner import run_episode
from hvac_qderl.episodes import annualise
from hvac_qderl.experiments.checkpoint import resolve_checkpoint, \
    rebuild_product_archive, load_agent
from hvac_qderl.learners.sac import SACController
from hvac_qderl.dispatch import (ContextDispatchController,
                                 DEFAULT_FITNESS_TIEBREAK_EPS)

RUN_LABEL = {"qd": "QD-ERL-Context", "sac": "SAC"}

TRACE_COLUMNS = ["day_slot", "hour", "p_total_kw", "p_chiller_kw",
                 "q_evap_kw", "q_sen_kw", "q_lat_kw", "kw_per_rt",
                 "mean_Tz", "max_Tz",
                 "mean_RH", "max_RH", "rh_violation", "pmv_disc",
                 "plant_enable", "chwst", "cw_fan", "occ", "t_oa", "t_wb",
                 "t_cw", "n_chillers_on", "cost_sgd", "corr_safety","corr_cycle",
                 "chwst_cmd", "q_demand_kw", "capacity_limited",
                 "chwst_float_K", "flow_cap", "airflow_kgs"]


def find_run(learner: str, seed: int) -> str | None:
    """Newest completed run dir for a learner+seed.

    Adopted from `run_evaluation.py`'s `find_run`: prefers the run with the
    most env_steps, because a resumed run that was misfiled (the `runs/` root
    bug) can leave a stale pointer in the original directory -- exactly what
    happened to SAC seed 0.
    """
    pref = f"{RUN_LABEL[learner]}_s{seed}_"
    best, best_steps = None, -1
    if not os.path.isdir(paths.RUNS):
        return None
    for d in os.listdir(paths.RUNS):
        if not d.startswith(pref):
            continue
        man = os.path.join(paths.RUNS, d, "manifest.json")
        if not os.path.isfile(man):
            continue
        try:
            steps = int(json.load(open(man)).get("env_steps", 0))
        except Exception:
            continue
        if steps > best_steps:
            best, best_steps = os.path.join(paths.RUNS, d), steps
    # the misfiled SAC seed-0 resume landed in runs/ itself
    root_man = os.path.join(paths.RUNS, "manifest.json")
    if os.path.isfile(root_man):
        try:
            m = json.load(open(root_man))
            if m.get("learner") == RUN_LABEL[learner] and int(m.get("seed", -1)) == seed:
                if int(m.get("env_steps", 0)) > best_steps:
                    best, best_steps = paths.RUNS, int(m.get("env_steps", 0))
        except Exception:
            pass
    return best if best_steps > 0 else None


def kpi_row(name, run_id, seed, k, e, secs) -> dict:
    return dict(
        controller=name, run_id=run_id, seed=seed, episode_kwh=k.energy_kwh,
        annual_kwh=e["annual_energy_kwh"], annual_cost=e["annual_cost_sgd"],
        annual_cool_kwh=e.get("annual_cool_kwh"),
        annual_chiller_kwh=e.get("annual_chiller_kwh"),
        annual_aux_kwh=e.get("annual_aux_kwh"),
        kw_per_rt=k.kw_per_rt_lw, kw_per_rt_on=k.kw_per_rt_on, peak_kw=k.peak_kw,
        rh_violation_rate=k.rh_violation_rate, pmv_disc_rate=k.pmv_disc_rate,
        pmv_disc_rate_occ=k.pmv_disc_rate_occ, plant_on_frac=k.plant_on_frac,
        max_Tz=k.max_Tz, max_RH=k.max_RH, chiller_starts=k.chiller_starts,
        seconds=round(secs, 1), episode_kind="full_year", episode_days=365)


def upsert_kpis(summary_path: str, new_rows: list[dict]) -> pd.DataFrame:
    """Merge `new_rows` into the on-disk KPI CSV.
    """
    touched = {r["controller"] for r in new_rows}
    existing = pd.read_csv(summary_path) if os.path.exists(summary_path) else pd.DataFrame()
    keyed: dict[tuple[str, str], dict] = {}
    if not existing.empty:
        if "run_id" not in existing.columns:
            existing["run_id"] = ""
        existing["run_id"] = existing["run_id"].fillna("")
        for _, r in existing.iterrows():
            row = r.to_dict()
            rid = row["run_id"]
            if row["controller"] in touched and rid == "":
                continue
            keyed[(row["controller"], rid)] = row
    for row in new_rows:
        keyed[(row["controller"], row.get("run_id") or "")] = row
    out = pd.DataFrame(list(keyed.values()))
    out.to_csv(summary_path, index=False)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arms", nargs="+", choices=["qd", "sac"],
                   default=["qd", "sac"],
                   help="which learned arm(s) to dispatch this run (default: "
                        "both). G36 runs too, UNLESS a G36 row already sits "
                        "in the output CSV.")
    p.add_argument("--seed", type=int, default=0,
                   help="seed used to locate each arm's newest run dir via "
                        "find_run() (default 0)")
    p.add_argument("--run", default=None,
                   help="QD-ERL-Context run dir override (default: newest "
                        "runs/QD-ERL-Context_s<seed>_* by env_steps)")
    p.add_argument("--sac-run", default=None,
                   help="SAC run dir override (default: newest "
                        "runs/SAC_s<seed>_* by env_steps)")
    p.add_argument("--hidden", type=int, default=None,
                   help="override the QD actor hidden width (default: read "
                        "from the run's manifest.json)")
    p.add_argument("--policy-kind", default=None,
                   choices=["numpy_mlp", "torch_actor"],
                   help="override QD policy architecture (default: inferred "
                        "from manifest['learner'])")
    p.add_argument("--window-steps", type=int, default=36,
                   help="QD windowed-BD-estimate length, control steps "
                        "(default 36 = 12h at a 20-min step) [HEUR]")
    p.add_argument("--context-confirm-days", type=int, default=2)
    p.add_argument("--niche-hysteresis", type=int, default=3)
    p.add_argument("--fitness-tiebreak-eps", type=float,
                   default=DEFAULT_FITNESS_TIEBREAK_EPS,
                   help="QD: among niches within this RELATIVE band of the "
                        "minimum squared BD distance, dispatch the "
                        "highest-fitness elite (default "
                        f"{DEFAULT_FITNESS_TIEBREAK_EPS}; 0 = pure "
                        "BD-nearest, the pre-tiebreak behaviour)")
    p.add_argument("--out-dir", default=paths.OUT_BASE)
    p.add_argument("--no-traces", action="store_true")
    args = p.parse_args(argv)

    t_all = time.time()
    run_qd = "qd" in args.arms
    run_sac = "sac" in args.arms

    ctx = get_annual_context()
    cfg = ctx.cfg
    print(f"plant     : {cfg.design_cooling_kw:,.0f} kW design "
          f"({cfg.design_rt:,.0f} RT)")

    summary_path = os.path.join(args.out_dir, "dispatch_context_full_year_kpis.csv")
    existing = pd.read_csv(summary_path) if os.path.exists(summary_path) else pd.DataFrame()
    have_g36 = (not existing.empty) and (existing["controller"] == "G36").any()

    new_rows: list[dict] = []
    traces: list[tuple[str, list]] = []

    # ------------------------------------------------------------------ #
    # G36 baseline -- full year, standard isothermal reset. Seed-independent,
    # so it is skipped once a result already sits in the KPI CSV.
    # ------------------------------------------------------------------ #
    if have_g36:
        g36_row = existing.loc[existing["controller"] == "G36"].iloc[-1]
        g36_annual_kwh = float(g36_row["annual_kwh"])
        g36_rh_violation_rate = float(g36_row["rh_violation_rate"])
        g36_pmv_disc_rate = float(g36_row["pmv_disc_rate"])
        g36_chiller_starts = float(g36_row["chiller_starts"])
        print(f"\nG36        : already in {summary_path} "
              f"({g36_annual_kwh / 1e6:.4f} GWh/yr) -- skipping re-run")
    else:
        env_g36, _, _ = make_dispatch_env(full_year=True)
        t0 = time.time()
        kpis_g36, trace_g36 = run_episode(env_g36, G36Controller(cfg))
        g36_seconds = time.time() - t0
        extras_g36 = annualise(trace_g36, env_g36.dt_h)
        g36_annual_kwh = extras_g36["annual_energy_kwh"]
        g36_rh_violation_rate = kpis_g36.rh_violation_rate
        g36_pmv_disc_rate = kpis_g36.pmv_disc_rate
        g36_chiller_starts = kpis_g36.chiller_starts
        print(f"\nG36        : {g36_annual_kwh / 1e6:.4f} GWh/yr  "
              f"RHviol={kpis_g36.rh_violation_rate:.3f}  "
              f"maxTz={kpis_g36.max_Tz:.2f}  starts={kpis_g36.chiller_starts}  "
              f"({g36_seconds:.1f}s)")
        new_rows.append(kpi_row("G36", "", None, kpis_g36, extras_g36, g36_seconds))
        traces.append(("G36", trace_g36))

    # ------------------------------------------------------------------ #
    # QD-ERL-Context -- full year, banked cold-start at hour 0
    # ------------------------------------------------------------------ #
    if run_qd:
        arc = get_context_archive()
        wb = get_warmstart_bank()

        qd_run = args.run or find_run("qd", args.seed)
        if not qd_run:
            raise FileNotFoundError(
                f"No QD-ERL-Context run found for seed {args.seed} under "
                f"{paths.RUNS} (expected '{RUN_LABEL['qd']}_s{args.seed}_*'). "
                "Pass --run to point at one explicitly.")
        ptr = resolve_checkpoint(qd_run)
        man_path = os.path.join(os.path.dirname(os.path.dirname(ptr["checkpoint"])),
                                "manifest.json")
        manifest = json.load(open(man_path)) if os.path.exists(man_path) else {}
        learner = manifest.get("learner", "")
        policy_kind = args.policy_kind or (
            "torch_actor" if learner == "QD-ERL-Context" else "numpy_mlp")
        hidden = args.hidden or int(manifest.get("config", {}).get("hidden", 24))

        archive_path = ptr.get("archive_npz")
        if not archive_path:
            raise FileNotFoundError(
                f"No product archive found for run '{qd_run}'. "
                "Check that the checkpoint has an associated archive_npz.")
        pa = rebuild_product_archive(archive_path)
        run_id_qd = os.path.basename(os.path.normpath(qd_run))
        print(f"\narchive   : {qd_run}")
        print(f"            {pa.stats.num_elites}/{pa.stats.total_cells} elites "
              f"({100 * pa.stats.coverage:.1f}% coverage), n_beh={pa.n_beh}, "
              f"policy_kind={policy_kind}, hidden={hidden}")

        dow0 = wb.first_weekday % 7
        states0 = np.asarray(wb.bank[dow0])
        T_z0, T_m0, W_z0 = states0.mean(axis=0)
        print(f"warm start: dow={dow0} ({['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][dow0]}) "
              f"T_z0={T_z0:.3f} T_m0={T_m0:.3f} W_z0={W_z0:.5f} "
              f"(mean of {len(states0)} bank samples, §8.1)")

        env_disp, _, _ = make_dispatch_env(full_year=True)
        ctrl = ContextDispatchController(
            cfg, arc, pa, hidden=hidden, policy_kind=policy_kind,
            window_steps=args.window_steps,
            context_confirm_days=args.context_confirm_days,
            niche_switch_hysteresis=args.niche_hysteresis,
            fitness_tiebreak_eps=args.fitness_tiebreak_eps)
        ctrl.bind_env(env_disp)
        t0 = time.time()
        kpis_disp, trace_disp = run_episode(env_disp, ctrl, T_z0=T_z0, T_m0=T_m0,
                                            W_z0=W_z0)
        disp_seconds = time.time() - t0
        extras_disp = annualise(trace_disp, env_disp.dt_h)
        pct_qd = 100.0 * (extras_disp["annual_energy_kwh"] - g36_annual_kwh) / g36_annual_kwh
        print(f"\nQD-ERL-Context: {extras_disp['annual_energy_kwh'] / 1e6:.4f} GWh/yr  "
              f"RHviol={kpis_disp.rh_violation_rate:.3f}  "
              f"maxTz={kpis_disp.max_Tz:.2f}  starts={kpis_disp.chiller_starts}  "
              f"({disp_seconds:.1f}s)  [{pct_qd:+.2f}% vs G36]")
        new_rows.append(kpi_row("ContextDispatch", run_id_qd, args.seed,
                                kpis_disp, extras_disp, disp_seconds))
        traces.append((f"ContextDispatch_s{args.seed}", trace_disp))

        # ------------------------------------------------------------------ #
        # QD-specific (archive dwell/switches)
        # ------------------------------------------------------------------ #
        dwell = ctrl.stats.dwell_times()
        diag = dict(
            run=qd_run, run_id=run_id_qd, policy_kind=policy_kind, hidden=hidden,
            window_steps=args.window_steps,
            context_confirm_days=args.context_confirm_days,
            niche_hysteresis=args.niche_hysteresis,
            archive_elites=pa.stats.num_elites, archive_total_cells=pa.stats.total_cells,
            archive_coverage=pa.stats.coverage, n_beh=pa.n_beh,
            context_switches=ctrl.stats.context_switches,
            n_dwell_runs=len(dwell),
            dwell_mean_days=float(np.mean(dwell)) if dwell else None,
            dwell_median_days=float(np.median(dwell)) if dwell else None,
            dwell_min_days=int(np.min(dwell)) if dwell else None,
            dwell_max_days=int(np.max(dwell)) if dwell else None,
            niche_switches=ctrl.stats.niche_switches,
            # fitness-tiebreak accounting (dispatch.DEFAULT_FITNESS_TIEBREAK_EPS)
            fitness_tiebreak_eps=args.fitness_tiebreak_eps,
            tiebreak_wins=ctrl.stats.tiebreak_wins,
            tiebreak_win_rate=(ctrl.stats.tiebreak_wins
                               / max(ctrl.stats.deployed_fitness_steps, 1)),
            mean_band_candidates=ctrl.stats.mean_band_candidates(),
            mean_deployed_fitness=ctrl.stats.mean_deployed_fitness(),
            context_empty_fallback_steps=ctrl.stats.context_empty_fallback_steps,
            final_fallback_steps=ctrl.stats.final_fallback_steps,
            n_steps=ctrl.stats.n_steps,
            sentinel_days=ctrl.stats.sentinel_days,
            annual_kwh_g36=g36_annual_kwh,
            annual_kwh_dispatch=extras_disp["annual_energy_kwh"],
            pct_vs_g36_energy=pct_qd,
            rh_violation_rate_g36=g36_rh_violation_rate,
            rh_violation_rate_dispatch=kpis_disp.rh_violation_rate,
            pmv_disc_rate_g36=g36_pmv_disc_rate,
            pmv_disc_rate_dispatch=kpis_disp.pmv_disc_rate,
            chiller_starts_g36=g36_chiller_starts,
            chiller_starts_dispatch=kpis_disp.chiller_starts,
            dispatch_seconds=disp_seconds,
        )
        diag_path = os.path.join(args.out_dir,
                                 f"dispatch_context_diagnostics_s{args.seed}.json")
        with open(diag_path, "w", encoding="utf-8") as f:
            json.dump(diag, f, indent=2, default=str)
        print(f"wrote {diag_path}")
        print(f"\ncontext switches/yr : {diag['context_switches']}  "
              f"(dwell mean {diag['dwell_mean_days']}, "
              f"median {diag['dwell_median_days']}, "
              f"n runs {diag['n_dwell_runs']})")
        print(f"niche switches      : {diag['niche_switches']}")
        # both are None only if EVERY step took the final fallback (no archive)
        # in which case there is no deployed elite to report on.
        _mdf, _mbc = diag["mean_deployed_fitness"], diag["mean_band_candidates"]
        print(f"fitness tiebreak    : eps={diag['fitness_tiebreak_eps']}  "
              f"changed the pick on {diag['tiebreak_wins']} steps "
              f"({100 * diag['tiebreak_win_rate']:.1f}%)  "
              f"mean band {'n/a' if _mbc is None else f'{_mbc:.2f}'}"
              f"/{pa.n_beh} niches")
        print(f"deployed fitness    : "
              f"{'n/a' if _mdf is None else f'{_mdf:.4f}'} (step-weighted mean)")
        print(f"sentinel active days: {diag['sentinel_days']}")
        print(f"fallback triggers   : context_empty={diag['context_empty_fallback_steps']}  "
              f"final={diag['final_fallback_steps']}")

        # per-day committed context cell, and per-step niche log
        ctx_log_path = os.path.join(args.out_dir,
                                    f"dispatch_context_cell_log_s{args.seed}.csv")
        pd.DataFrame(ctrl.stats.context_cell_log,
                    columns=["day", "context_cell"]).to_csv(ctx_log_path, index=False)
        print(f"wrote {ctx_log_path}")

        niche_log_path = os.path.join(args.out_dir,
                                      f"dispatch_niche_log_s{args.seed}.csv")
        pd.DataFrame(ctrl.stats.niche_log,
                    columns=["day", "hour", "cell", "niche", "tier",
                             "fitness"]).to_csv(
            niche_log_path, index=False)
        print(f"wrote {niche_log_path}")

    # ------------------------------------------------------------------ #
    # SAC -- full year, plain isothermal reset (same convention as G36)
    # ------------------------------------------------------------------ #
    if run_sac:
        sac_run = args.sac_run or find_run("sac", args.seed)
        if not sac_run:
            raise FileNotFoundError(
                f"No SAC run found for seed {args.seed} under {paths.RUNS} "
                f"(expected '{RUN_LABEL['sac']}_s{args.seed}_*'). Pass "
                "--sac-run to point at one explicitly.")
        run_id_sac = os.path.basename(os.path.normpath(sac_run))
        sac_ptr = resolve_checkpoint(sac_run)
        agent, _ = load_agent(sac_ptr["checkpoint"], prefer_gpu=True)
        sac_ctrl = SACController(agent, cfg)
        print(f"\nSAC       : {sac_run}")

        env_sac, _, _ = make_dispatch_env(full_year=True)
        t0 = time.time()
        kpis_sac, trace_sac = run_episode(env_sac, sac_ctrl)
        sac_seconds = time.time() - t0
        extras_sac = annualise(trace_sac, env_sac.dt_h)
        pct_sac = 100.0 * (extras_sac["annual_energy_kwh"] - g36_annual_kwh) / g36_annual_kwh
        print(f"\nSAC        : {extras_sac['annual_energy_kwh'] / 1e6:.4f} GWh/yr  "
              f"RHviol={kpis_sac.rh_violation_rate:.3f}  "
              f"maxTz={kpis_sac.max_Tz:.2f}  starts={kpis_sac.chiller_starts}  "
              f"({sac_seconds:.1f}s)  [{pct_sac:+.2f}% vs G36]")
        new_rows.append(kpi_row("SAC", run_id_sac, args.seed,
                                kpis_sac, extras_sac, sac_seconds))
        traces.append((f"SAC_s{args.seed}", trace_sac))

    # ------------------------------------------------------------------ #
    # KPI summary CSV
    # ------------------------------------------------------------------ #
    if new_rows:
        upsert_kpis(summary_path, new_rows)
        print(f"\nwrote {summary_path}")
    else:
        print(f"\nnothing new to dispatch -- {summary_path} unchanged "
              "(G36 already present, --arms selected nothing new)")

    if not args.no_traces and traces:
        os.makedirs(paths.TRACES, exist_ok=True)
        for name, trace in traces:
            tp = os.path.join(paths.TRACES, f"full_year_{name}.csv")
            df = pd.DataFrame(trace)
            cols = [c for c in TRACE_COLUMNS if c in df.columns]
            df[cols].to_csv(tp, index=False)
            print(f"wrote {tp}")

    print(f"\ntotal wall time: {time.time() - t_all:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

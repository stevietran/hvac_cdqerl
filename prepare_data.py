#!/usr/bin/env python3
"""Data preparation: weather -> annual cooling load -> representative days.

    python prepare_data.py load    # annual load only
    python prepare_data.py rep-days  --n-rep-days 4    # representative days only
    python prepare_data.py validate  --n-rep-days 4    # clustering error vs the year

Writes into `data/`:
    annual_load.csv            8,760 h ideal-loads heat/moisture balance
    representative_days.csv    N medoids + the forced min-PV and peak-load days
    clustering_validation.csv  clustering error against the true full year
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import paths
paths.ensure_dirs()

import numpy as np

from hvac_qderl.baselines import G36Controller
from hvac_qderl.episodes import annualise
from hvac_qderl.runner import run_episode
from hvac_qderl.scenarios import get_annual_context, make_dispatch_env


# --------------------------------------------------------------------------- #
def write_annual_load(ctx):
    L = ctx.load
    out = paths.ANNUAL_LOAD
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["hour", "day", "t_db_C", "t_wb_C", "w_oa_kgkg", "rh",
                    "ghi_Wm2", "occ", "q_total_kW", "q_sensible_kW",
                    "q_latent_kW", "q_oa_kW", "shr"])
        for h in range(len(L.q_total)):
            w.writerow([h, h // 24, round(ctx.annual.t_db[h], 2),
                        round(ctx.annual.t_wb[h], 2), round(ctx.annual.w_oa[h], 5),
                        round(ctx.annual.rh[h], 3), round(ctx.annual.ghi[h], 1),
                        round(L.occ[h], 3), round(L.q_total[h], 2),
                        round(L.q_sensible[h], 2), round(L.q_latent[h], 2),
                        round(L.q_oa[h], 2), round(L.shr[h], 3)])
    return out


def write_representative_days(ctx):
    rep_out = paths.REPRESENTATIVE_DAYS
    with open(rep_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["day_index", "label", "kind", "weight_days", "weight_fraction",
                    "q_load_mean_kW", "q_load_max_kW", "t_db_mean_C", "t_wb_max_C",
                    "w_oa_mean", "ghi_max_Wm2"])
        for d in ctx.rep_set.days:
            ft = d.features
            w.writerow([d.day_index, d.label, d.kind, round(d.weight_days, 2),
                        round(d.weight_fraction, 5),
                        round(ft.get("q_load_mean", 0), 1),
                        round(ft.get("q_load_max", 0), 1),
                        round(ft.get("t_db_mean", 0), 2),
                        round(ft.get("t_wb_max", 0), 2),
                        round(ft.get("w_oa_mean", 0), 5),
                        round(ft.get("ghi_max", 0), 1)])
    return rep_out


def cmd_load(a):
    ctx = get_annual_context(a.n_rep_days, a.mos, seed=a.seed)
    print("=== (1) Weather ===")
    print(" ", ctx.annual.summary())
    print("\n=== (1) Cooling load — ideal-loads method ===")
    print(" ", ctx.load.summary())
    L = ctx.load
    occ = L.occ > 0.5
    print(f"  latent share of annual load : {100*L.q_latent.sum()/L.q_total.sum():.1f} %")
    print(f"  ventilation (OA) share      : {100*L.q_oa.sum()/L.q_total.sum():.1f} %")
    print(f"  SHR, occupied hours         : {np.nanmean(L.shr[occ]):.3f}")
    print(f"  SHR at the peak hour        : {L.shr[L.q_total.argmax()]:.3f}")
    print(f"  plant sized to design load  : {ctx.cfg.design_cooling_kw:,.0f} kW "
          f"({ctx.cfg.design_rt:,.0f} RT)")

    out = paths.ANNUAL_LOAD
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["hour", "day", "t_db_C", "t_wb_C", "w_oa_kgkg", "rh",
                    "ghi_Wm2", "occ", "q_total_kW", "q_sensible_kW",
                    "q_latent_kW", "q_oa_kW", "shr"])
        for h in range(len(L.q_total)):
            w.writerow([h, h // 24, round(ctx.annual.t_db[h], 2),
                        round(ctx.annual.t_wb[h], 2), round(ctx.annual.w_oa[h], 5),
                        round(ctx.annual.rh[h], 3), round(ctx.annual.ghi[h], 1),
                        round(L.occ[h], 3), round(L.q_total[h], 2),
                        round(L.q_sensible[h], 2), round(L.q_latent[h], 2),
                        round(L.q_oa[h], 2), round(L.shr[h], 3)])
    rep_out = paths.REPRESENTATIVE_DAYS
    with open(rep_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["day_index", "label", "kind", "weight_days", "weight_fraction",
                    "q_load_mean_kW", "q_load_max_kW", "t_db_mean_C", "t_wb_max_C",
                    "w_oa_mean", "ghi_max_Wm2"])
        for d in ctx.rep_set.days:
            ft = d.features
            w.writerow([d.day_index, d.label, d.kind, round(d.weight_days, 2),
                        round(d.weight_fraction, 5),
                        round(ft.get("q_load_mean", 0), 1),
                        round(ft.get("q_load_max", 0), 1),
                        round(ft.get("t_db_mean", 0), 2),
                        round(ft.get("t_wb_max", 0), 2),
                        round(ft.get("w_oa_mean", 0), 5),
                        round(ft.get("ghi_max", 0), 1)])
    print(f"\nwrote {out}\nwrote {rep_out}")
    return ctx


def cmd_rep_days(a):
    ctx = get_annual_context(a.n_rep_days, a.mos, seed=a.seed)
    print("=== Representative days ===")
    rep = get_representative_set(ctx.n_rep_days, ctx.annual, ctx.load, a.seed)
    print(rep.summary())

    rep_out = write_representative_days(ctx, seed=a.seed)
    print(f"\nwrote {rep_out}")
    return ctx


# --------------------------------------------------------------------------- #
def _controllers(cfg, horizon=None, only=None):
    """Controllers available to the clustering-validation command.

    G36 is the only one left. `G36+smart_off`, `FixedSetpoint`, `NearOptimal`
    and `GurobiMPC` were removed from the project (see
    `hvac_qderl/baselines/__init__.py`), so `--controller` can no longer summon
    them. `horizon` is kept only so the existing CLI wiring for `--mpc-horizon`
    does not break; it is unused.
    """
    all_c = {"G36": G36Controller(cfg)}
    if not only:
        from hvac_qderl.experiments.evaluate import DEFAULT_BASELINES
        return {k: all_c[k] for k in DEFAULT_BASELINES}
    want = [c.strip() for c in only.split(",") if c.strip()]
    unknown = [c for c in want if c not in all_c]
    if unknown:
        raise SystemExit(f"unknown controller(s) {unknown}; "
                         f"available: {sorted(all_c)}")
    return {k: v for k, v in all_c.items() if k in want}



def cmd_validate(a):
    """How much error does the representative-day reduction introduce?"""
    ctrl_name = a.controller
    print("=== Clustering fidelity: weighted representative episode vs the true year ===")
    print(f"controller: {ctrl_name}\n")
    print(f"{'design':>12s} {'days':>5s} {'steps':>8s} {'sec':>7s} "
          f"{'annual GWh':>11s} {'err vs year':>12s} {'speedup':>8s}")

    def run(full_year, n_rep):
        env, ctx, spec = make_dispatch_env(full_year=full_year, n_rep_days=n_rep,
                                           price_profile=a.price, mos_path=a.mos,
                                           seed=a.seed)
        c = _controllers(ctx.cfg, a.horizon)[ctrl_name]
        t0 = time.time()
        k, tr = run_episode(env, c)
        el = time.time() - t0
        return annualise(tr, env.dt_h), el, len(tr), spec

    truth, t_year, n_year, _ = run(True, a.n_rep_days)
    results = []
    for n in [int(x) for x in a.sweep.split(",") if x]:
        A, el, n_steps, spec = run(False, n)
        err = 100.0 * (A["annual_energy_kwh"] - truth["annual_energy_kwh"]) \
            / truth["annual_energy_kwh"]
        results.append((n, spec.n_days, n_steps, el, A["annual_energy_kwh"], err,
                        t_year / max(el, 1e-9)))
        print(f"{'rep '+str(n)+'+2':>12s} {spec.n_days:>5d} {n_steps:>8,} {el:>7.1f} "
              f"{A['annual_energy_kwh']/1e6:>11.3f} {err:>11.2f}% "
              f"{t_year/max(el,1e-9):>7.0f}x")
    print(f"{'full year':>12s} {365:>5d} {n_year:>8,} {t_year:>7.1f} "
          f"{truth['annual_energy_kwh']/1e6:>11.3f} {'(reference)':>12s}")

    out = paths.CLUSTERING_VALIDATION
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n_rep_days", "episode_days", "steps", "seconds",
                    "annual_energy_kwh", "error_pct_vs_full_year", "speedup_x"])
        for r in results:
            w.writerow([r[0], r[1], r[2], round(r[3], 2), round(r[4], 1),
                        round(r[5], 3), round(r[6], 1)])
        w.writerow([365, 365, n_year, round(t_year, 2),
                    round(truth["annual_energy_kwh"], 1), 0.0, 1.0])
    print(f"\nwrote {out}")
    print("\nUse this table to choose n_rep_days: the training episode should be the "
          "smallest set whose annual error is acceptable (~2 % at 4+2 here).")
    return results


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--n-rep-days", type=int, default=4)
        p.add_argument("--mos", default=None, help="path to the .mos weather file")
        p.add_argument("--price", default=None,   # None -> project default
                       choices=["constant", "dynamic", "highly_dynamic"])
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--horizon", type=int, default=6, help="MPC horizon steps")

    p = sub.add_parser("load", help="cooling load + representative-day report")
    common(p); p.set_defaults(fn=cmd_load)


    p = sub.add_parser("validate", help="clustering error vs the true full year")
    common(p)
    p.add_argument("--sweep", default="2,4,8,12",
                   help="comma-separated n_rep_days values to compare")
    p.add_argument("--controller", default="G36",
                   choices=["G36", "G36+smart_off", "GurobiMPC",
                            "NearOptimal", "FixedSetpoint"],
                   help="baselines of record are G36 and G36+smart_off; the "
                        "rest are diagnostics")
    p.set_defaults(fn=cmd_validate)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())

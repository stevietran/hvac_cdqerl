#!/usr/bin/env python3
"""Canonical locations for every input and output in this project.

WHY THIS EXISTS
---------------
Paths used to be built ad hoc as `os.path.join(HERE, "something.csv")` in each
`run_*.py`, so the repository root accumulated ~20 loose CSV/JSON artefacts with
no way to tell an *input* from a *result*, or one arm's results from another's.
Anything that reorganises the tree then has to chase those literals through every
script.

LAYOUT
------
    data/                  inputs — weather, derived load, representative days
    outputs/
        base/              baselines: G36
        sac/               SAC arm
        tuning/            Screening / optimisation / confirmation
        figures/           all generated figures (PNG + PDF)
        reports/           generated markdown reports
        traces/            per-step dispatch traces
        calibration/       iso_sigma sweep CSVs (calibrate_contextual_mutation_sigma.py)
    runs/                  training run dirs (checkpoints, manifests, logs)
    logs/                  launcher logs
    docs/                  hand-written analysis
    _archive/              superseded files (this mount forbids delete)

`ensure_dirs()` is idempotent and safe to call at import time in a runner.
"""
from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.abspath(__file__))

# --- inputs ---------------------------------------------------------------- #
DATA = os.path.join(ROOT, "data")
WEATHER_MOS = os.path.join(DATA, "SGP_Singapore.486980_IWEC.mos")
ANNUAL_LOAD = os.path.join(DATA, "annual_load.csv")
REPRESENTATIVE_DAYS = os.path.join(DATA, "representative_days.csv")
CLUSTERING_VALIDATION = os.path.join(DATA, "clustering_validation.csv")

# --- outputs --------------------------------------------------------------- #
OUTPUTS = os.path.join(ROOT, "outputs")
OUT_BASE = os.path.join(OUTPUTS, "base")
OUT_SAC = os.path.join(OUTPUTS, "sac")
OUT_TUNING = os.path.join(OUTPUTS, "tuning")
FIGURES = os.path.join(OUTPUTS, "figures")
REPORTS = os.path.join(OUTPUTS, "reports")
TRACES = os.path.join(OUTPUTS, "traces")
CALIBRATION = os.path.join(OUTPUTS, "calibration")   # §8.6 iso_sigma sweeps

# per-arm output directory, keyed by the CLI learner name. 
# contextual arm QD-ERL-Context has no dedicated `outputs/` subdir
# its artefacts live under `runs/`, and `arm_dir()`
ARM_DIR = {"sac": OUT_SAC}
# ...and by the display name used in run dirs / manifests
ARM_DIR_BY_LABEL = {"SAC": OUT_SAC}

# --- working dirs ---------------------------------------------------------- #
RUNS = os.path.join(ROOT, "runs")
LOGS = os.path.join(ROOT, "logs")
DOCS = os.path.join(ROOT, "docs")
ARCHIVE = os.path.join(ROOT, "_archive")

ALL_DIRS = (DATA, OUTPUTS, OUT_BASE, OUT_SAC,
            OUT_TUNING, FIGURES, REPORTS, TRACES, CALIBRATION, RUNS, LOGS, ARCHIVE)

# baselines / example runs
EXAMPLE_KPIS = os.path.join(OUT_BASE, "singapore_example_kpis.csv")
EXAMPLE_TRACE = os.path.join(OUT_BASE, "singapore_example_trace.csv")
EXAMPLE_REPORT = os.path.join(REPORTS, "singapore_example_report.md")
SCHEDULE_COMPARISON = os.path.join(OUT_BASE, "schedule_comparison.csv")
SCHEDULE_TRACE = os.path.join(OUT_BASE, "schedule_trace_smart_off.csv")
DISPATCH_REP_KPIS = os.path.join(OUT_BASE, "dispatch_rep4_kpis.csv")
DISPATCH_FULL_YEAR_KPIS = os.path.join(OUT_BASE, "dispatch_full_year_kpis.csv")

# tuning
TUNING_TRIALS = os.path.join(OUT_TUNING, "tuning_trials.csv")
TUNING_SCREENING = os.path.join(OUT_TUNING, "tuning_screening.json")
TUNING_WINNER = os.path.join(OUT_TUNING, "tuning_winner.json")
TUNING_CONFIRMATION = os.path.join(OUT_TUNING, "tuning_confirmation.json")
TUNING_STUDY_DB = os.path.join(OUT_TUNING, "tuning_study.db")

# reports
TRAINING_REPORT = os.path.join(REPORTS, "training_report.md")
FINAL_KPI_REPORT = os.path.join(REPORTS, "final_kpi_comparison.md")


def ensure_dirs(*extra: str) -> None:
    """Create the standard tree (idempotent)."""
    for d in ALL_DIRS + tuple(extra):
        os.makedirs(d, exist_ok=True)


def arm_dir(learner: str) -> str:
    """Output directory for a learner, accepting CLI name or display label."""
    key = str(learner)
    if key in ARM_DIR:
        return ARM_DIR[key]
    if key in ARM_DIR_BY_LABEL:
        return ARM_DIR_BY_LABEL[key]
    return os.path.join(OUTPUTS, key.lower().replace("-", "_"))


if __name__ == "__main__":
    ensure_dirs()
    print(f"root: {ROOT}\n")
    for name, path in sorted(globals().items()):
        if name.isupper() and isinstance(path, str) and path.startswith(ROOT):
            rel = os.path.relpath(path, ROOT)
            mark = "     " if os.path.exists(path) else "  (-)"
            print(f"{mark} {name:<24} {rel}")

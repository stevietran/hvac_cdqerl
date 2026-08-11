"""Run layout + metric logging

Everything a figure needs is appended to flat CSVs so plotting is fully decoupled
from training: a run can be trained on a GPU box, the `runs/` folder copied, and
the figures regenerated anywhere (no torch required to plot).

Layout:

    runs/<run_id>/
        manifest.json           # latest run metadata (checkpoint.py writes it)
        train_log.csv           # per logging interval
        eval_log.csv            # per (controller, scenario, seed)
        archive_snapshot.csv    # per elite at selected generations
        traces/trace_<ctrl>_<scenario>.csv
        checkpoints/ckpt_<env_steps>.pt, archive_<env_steps>.npz
        latest.json, best.json
"""
from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Iterable

# --- CSV schemas ------------------------ #
TRAIN_FIELDS = [
    "run_id", "learner", "seed", "generation", "env_steps", "grad_steps",
    "wall_clock_s", "best_return", "mean_return", "eval_return",
    "coverage", "qd_score", "archive_elites", "buffer_size",
    "alpha", "critic_loss", "actor_loss", "resumed_from",
]

EVAL_FIELDS = [
    "controller", "seed", "scenario_day", "scenario_price",
    "checkpoint_env_steps", "energy_kwh", "cost_sgd", "peak_kw",
    "kw_per_rt", "kw_per_rt_on", "plant_on_frac",
    "rh_violation_rate", "pmv_disc_rate",
    "mean_Tz", "max_Tz", "mean_RH", "max_RH", "mean_shr", "chiller_starts",
    "shield_correction_mean", "dispatch_niche_switches",
    "episode_kind", "episode_days", "episode_hours", "weight_sum",
    "annual_energy_kwh", "annual_cost_sgd", "episode_energy_kwh",
]

ARCHIVE_FIELDS = [
    "run_id", "generation", "env_steps", "niche_index",
    "bd_peak_power", "bd_rh_violation", "bd_comfort_dev", "fitness",
]


CONTEXT_ARCHIVE_FIELDS = [
    "run_id", "generation", "env_steps", "context_cell", "behaviour_cell",
    "bd_chwst_toa_gain", "bd_fan_woa_gain", "bd_plant_on_unocc_normal_frac",
    "bd_plant_on_unocc_drift_frac", "fitness",
]


ROLLOUT_FIELDS = [
    "run_id", "generation", "env_steps", "context_cell", "day",
    "fitness_raw", "f_g36", "fitness_tilde", "source",
]

ROLLOUT_SOURCES = ("bootstrap", "iso_line", "random_fallback", "pg", "actor_inject")


def default_runs_root() -> str:
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(here, "runs")


def make_run_id(learner: str, seed: int) -> str:
    return f"{learner}_s{seed}_{time.strftime('%Y%m%d-%H%M%S')}"


@dataclass
class RunPaths:
    """Filesystem layout for one training run."""
    run_id: str
    root: str = ""

    def __post_init__(self):
        if not self.root:
            self.root = os.path.join(default_runs_root(), self.run_id)
        os.makedirs(self.checkpoints, exist_ok=True)
        os.makedirs(self.traces, exist_ok=True)

    @property
    def checkpoints(self):
        return os.path.join(self.root, "checkpoints")

    @property
    def traces(self):
        return os.path.join(self.root, "traces")

    @property
    def train_log(self):
        return os.path.join(self.root, "train_log.csv")

    @property
    def eval_log(self):
        return os.path.join(self.root, "eval_log.csv")

    @property
    def archive_snapshot(self):
        return os.path.join(self.root, "archive_snapshot.csv")

    @property
    def context_archive_snapshot(self):
        """§8.6 contextual training loop's own archive snapshot CSV --
        parallel to `archive_snapshot` above, not a replacement (see
        `CONTEXT_ARCHIVE_FIELDS`)."""
        return os.path.join(self.root, "context_archive_snapshot.csv")

    @property
    def rollout_log(self):
        """Every §8.6.2 rollout's (context, raw fitness, F_G36, normalised
        fitness) -- see `ROLLOUT_FIELDS`."""
        return os.path.join(self.root, "rollout_log.csv")

    @property
    def manifest(self):
        return os.path.join(self.root, "manifest.json")

    @property
    def latest_ptr(self):
        return os.path.join(self.root, "latest.json")

    @property
    def best_ptr(self):
        return os.path.join(self.root, "best.json")

    def ckpt(self, env_steps: int):
        return os.path.join(self.checkpoints, f"ckpt_{env_steps:09d}.pt")

    def archive_npz(self, env_steps: int):
        return os.path.join(self.checkpoints, f"archive_{env_steps:09d}.npz")

    def trace(self, controller: str, scenario: str):
        return os.path.join(self.traces, f"trace_{controller}_{scenario}.csv")


class CsvAppender:
    """Append rows to a CSV, writing the header once. Safe across resumes."""

    def __init__(self, path: str, fields: Iterable[str]):
        self.path = path
        self.fields = list(fields)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self.fields).writeheader()

    def append(self, row: dict):
        clean = {k: row.get(k, "") for k in self.fields}
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.fields).writerow(clean)

    def append_many(self, rows):
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.fields)
            for r in rows:
                w.writerow({k: r.get(k, "") for k in self.fields})


class MetricsLogger:
    """Writes the three training-side CSVs and echoes to stdout."""

    def __init__(self, paths: RunPaths, learner: str, seed: int,
                 resumed_from: str = "", verbose: bool = True):
        self.paths = paths
        self.learner = learner
        self.seed = seed
        self.resumed_from = resumed_from
        self.verbose = verbose
        self.t0 = time.time()
        self._wall_offset = 0.0
        self.train = CsvAppender(paths.train_log, TRAIN_FIELDS)
        self.archive_csv = CsvAppender(paths.archive_snapshot, ARCHIVE_FIELDS)
        self.context_archive_csv = CsvAppender(paths.context_archive_snapshot,
                                               CONTEXT_ARCHIVE_FIELDS)
        self.rollout_csv = CsvAppender(paths.rollout_log, ROLLOUT_FIELDS)

    def set_wall_offset(self, seconds: float):
        """After a resume, keep wall_clock_s monotone across the lineage."""
        self._wall_offset = float(seconds)

    @property
    def wall_clock(self) -> float:
        return self._wall_offset + (time.time() - self.t0)

    def log_train(self, *, env_steps: int, generation: int = 0, grad_steps: int = 0,
                  best_return=None, mean_return=None, eval_return=None,
                  coverage=None, qd_score=None, archive_elites=None,
                  buffer_size=None, alpha=None, critic_loss=None,
                  actor_loss=None):
        row = dict(run_id=self.paths.run_id, learner=self.learner, seed=self.seed,
                   generation=generation, env_steps=env_steps, grad_steps=grad_steps,
                   wall_clock_s=round(self.wall_clock, 2),
                   best_return=_r(best_return), mean_return=_r(mean_return),
                   eval_return=_r(eval_return), coverage=_r(coverage, 5),
                   qd_score=_r(qd_score), archive_elites=archive_elites,
                   buffer_size=buffer_size, alpha=_r(alpha, 4),
                   critic_loss=_r(critic_loss), actor_loss=_r(actor_loss),
                   resumed_from=self.resumed_from)
        self.train.append(row)
        if self.verbose:
            bits = [f"steps {env_steps:>9,}"]
            if generation:
                bits.append(f"gen {generation:>5d}")
            if eval_return is not None:
                bits.append(f"eval {eval_return:>10.2f}")
            elif best_return is not None:
                bits.append(f"best {best_return:>10.2f}")
            if coverage is not None:
                bits.append(f"cov {coverage:6.3f}")
            if qd_score is not None:
                bits.append(f"QD {qd_score:>12.1f}")
            bits.append(f"{self.wall_clock:7.1f}s")
            print("  " + " | ".join(bits), flush=True)

    def log_archive(self, archive, generation: int, env_steps: int):
        """Snapshot every elite's (BD, fitness) for the illumination plots."""
        try:
            data = archive.data(["measures", "objective", "index"])
            meas, objs, idxs = data["measures"], data["objective"], data["index"]
        except Exception:
            return 0
        rows = []
        for m, o, i in zip(meas, objs, idxs):
            rows.append(dict(run_id=self.paths.run_id, generation=generation,
                             env_steps=env_steps, niche_index=int(i),
                             bd_peak_power=round(float(m[0]), 5),
                             bd_rh_violation=round(float(m[1]), 5),
                             bd_comfort_dev=round(float(m[2]), 5),
                             fitness=round(float(o), 4)))
        self.archive_csv.append_many(rows)
        return len(rows)

    def log_context_archive(self, archive, generation: int, env_steps: int):
        """Snapshot every elite of a §8.6 `ProductArchive` -- its own schema
        (`CONTEXT_ARCHIVE_FIELDS`), not `log_archive`'s (see the module-level
        comment on `CONTEXT_ARCHIVE_FIELDS`). Torch-free: `archive.to_arrays()`
        only needs pyribs, and the CSV this writes needs neither to read back
        (figures.py's F19/F20 read this file directly)."""
        a = archive.to_arrays()
        rows = []
        for ctx, idx, m, o in zip(a["context_cell"], a["index"], a["measures"],
                                  a["objective"]):
            rows.append(dict(run_id=self.paths.run_id, generation=generation,
                             env_steps=env_steps, context_cell=int(ctx),
                             behaviour_cell=int(idx),
                             bd_chwst_toa_gain=round(float(m[0]), 6),
                             bd_fan_woa_gain=round(float(m[1]), 6),
                             bd_plant_on_unocc_normal_frac=round(float(m[2]), 6),
                             bd_plant_on_unocc_drift_frac=round(float(m[3]), 6),
                             fitness=round(float(o), 6)))
        self.context_archive_csv.append_many(rows)
        return len(rows)

    def log_rollout(self, *, generation: int, env_steps: int, context_cell: int,
                    day: int, fitness_raw: float, f_g36: float,
                    fitness_tilde: float, source: str = ""):
        """One row per §8.6.2 rollout (every candidate, not just archived
        elites) -- the raw-vs-normalised pair figures.py's F19 reads.
        `source` (optional) is the operator that produced this candidate's
        genome, one of `ROLLOUT_SOURCES` -- figures.py's F25 reads it."""
        self.rollout_csv.append(dict(
            run_id=self.paths.run_id, generation=generation, env_steps=env_steps,
            context_cell=int(context_cell), day=int(day),
            fitness_raw=round(float(fitness_raw), 6),
            f_g36=round(float(f_g36), 6),
            fitness_tilde=round(float(fitness_tilde), 6),
            source=source or ""))


def _r(x, nd=4):
    return "" if x is None else round(float(x), nd)


def write_json(path: str, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

"""Product archive, archive[context_cell][behaviour_cell]

    archive[context_cell][behaviour_cell],  context_cell in 1..18,  behaviour_cell in 1..N_beh
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ribs.archives import CVTArchive

N_CONTEXT_CELLS = 18   # §8.2: 16 data-driven cells + S_peak + S_minpv
N_BEH_DEFAULT = 12     # §8.5: recommended N_beh (128.6 evals/niche at a 2M-step budget)


# --------------------------------------------------------------------------- #
def calibrate_probe_bd_ranges(cfg, n_genomes: int = 40, hidden: int = 64,
                              seed: int = 0, pad_frac: float = 0.05):
    """1-99% quantile ranges for the 4 probe-BD axes. BD_RANGES_RAW calibration corpus 
    is 87 random genomes + 16 archive elites
    """
    from ..probe_bd import calibrate
    cal = calibrate(cfg=cfg, n_genomes=n_genomes, hidden=hidden, seed=seed)
    lo = np.percentile(cal.values, 1, axis=0)
    hi = np.percentile(cal.values, 99, axis=0)
    span = np.maximum(hi - lo, 1e-9)
    lo = lo - pad_frac * span
    hi = hi + pad_frac * span
    return [(float(a), float(b)) for a, b in zip(lo, hi)]


# --------------------------------------------------------------------------- #
@dataclass
class ProductArchiveStats:
    num_elites: int
    total_cells: int
    obj_max: float
    obj_mean: float
    qd_score: float
    coverage: float
    per_context_elites: np.ndarray   # (n_context_cells,)


class ProductArchive:
    """archive[context_cell][behaviour_cell]"""

    def __init__(self, solution_dim: int, bd_ranges=None, cfg=None,
                n_context_cells: int = N_CONTEXT_CELLS, n_beh: int = N_BEH_DEFAULT,
                seed: int = 0, qd_score_offset: float = -10.0,
                samples: int = 20_000):
        if bd_ranges is None:
            if cfg is None:
                raise ValueError("ProductArchive needs bd_ranges or cfg "
                                 "(to calibrate them via calibrate_probe_bd_ranges)")
            bd_ranges = calibrate_probe_bd_ranges(cfg, seed=seed)
        self.solution_dim = solution_dim
        self.bd_ranges = bd_ranges
        self.n_context_cells = n_context_cells
        self.n_beh = n_beh
        self.qd_score_offset = qd_score_offset
        # One CVTArchive per context cell, all built from the SAME `bd_ranges`
        # and `seed` so their k-means-fit centroids coincide
        self._archives = [
            CVTArchive(solution_dim=solution_dim, centroids=n_beh, ranges=bd_ranges,
                      samples=samples, seed=seed, qd_score_offset=qd_score_offset)
            for _ in range(n_context_cells)]

    @property
    def total_cells(self) -> int:
        return self.n_context_cells * self.n_beh

    def archive_for(self, context_cell: int) -> CVTArchive:
        return self._archives[context_cell]

    # ------------------------------------------------------------------ #
    def add(self, context_cell: int, solution, objective: float, measures):
        """`j*` (nearest of the N_beh behaviour centroids) and elitist insertion """
        return self._archives[int(context_cell)].add_single(
            np.asarray(solution), float(objective), np.asarray(measures))

    def sample_elites(self, n: int, rng: np.random.Generator | None = None):
        """Pool elites across every context cell and draw `n` uniformly.
        Used by the Iso+LineDD operator, which is context-agnostic """
        rng = rng or np.random.default_rng()
        sols, objs, meas = [], [], []
        for arc in self._archives:
            if arc.stats.num_elites == 0:
                continue
            d = arc.data()
            sols.append(d["solution"]); objs.append(d["objective"]); meas.append(d["measures"])
        if not sols:
            return None
        sols = np.concatenate(sols); objs = np.concatenate(objs); meas = np.concatenate(meas)
        idx = rng.integers(0, len(sols), size=n)
        return dict(solution=sols[idx], objective=objs[idx], measures=meas[idx])

    def all_solutions(self) -> np.ndarray:
        """Every elite genome pooled across every context cell -- for
        Iso+LineDD's two-parent draw once >=2 elites exist anywhere."""
        sols = [arc.data("solution") for arc in self._archives if arc.stats.num_elites > 0]
        return np.concatenate(sols) if sols else np.empty((0, self.solution_dim))

    # ------------------------------------------------------------------ #
    @property
    def stats(self) -> ProductArchiveStats:
        per_context = np.array([a.stats.num_elites for a in self._archives])
        num_elites = int(per_context.sum())
        filled = [a.stats for a in self._archives if a.stats.num_elites > 0]
        obj_max = max((float(s.obj_max) for s in filled), default=float("-inf"))
        obj_mean = (sum(float(s.obj_mean) * s.num_elites for s in filled) / num_elites
                   if num_elites else float("nan"))
        qd_score = float(sum(float(a.stats.qd_score) for a in self._archives))
        return ProductArchiveStats(
            num_elites=num_elites, total_cells=self.total_cells, obj_max=obj_max,
            obj_mean=obj_mean, qd_score=qd_score,
            coverage=num_elites / self.total_cells, per_context_elites=per_context)

    # ------------------------------------------------------------------ #
    def to_arrays(self) -> dict:
        """Flatten every context cell's elites into one set of parallel
        arrays, for disk serialisation """
        sols, objs, meas, idxs, ctxs = [], [], [], [], []
        for c, arc in enumerate(self._archives):
            if arc.stats.num_elites == 0:
                continue
            d = arc.data(["solution", "objective", "measures", "index"])
            sols.append(d["solution"]); objs.append(d["objective"])
            meas.append(d["measures"]); idxs.append(d["index"])
            ctxs.append(np.full(len(d["solution"]), c, dtype=np.int64))
        if not sols:
            return dict(solution=np.empty((0, self.solution_dim), dtype=np.float32),
                       objective=np.empty(0, dtype=np.float64),
                       measures=np.empty((0, len(self.bd_ranges)), dtype=np.float32),
                       index=np.empty(0, dtype=np.int64),
                       context_cell=np.empty(0, dtype=np.int64))
        return dict(solution=np.concatenate(sols).astype(np.float32),
                   objective=np.concatenate(objs).astype(np.float64),
                   measures=np.concatenate(meas).astype(np.float32),
                   index=np.concatenate(idxs).astype(np.int64),
                   context_cell=np.concatenate(ctxs))

    @classmethod
    def from_arrays(cls, arrays: dict, solution_dim: int, bd_ranges,
                    n_context_cells: int = N_CONTEXT_CELLS,
                    n_beh: int = N_BEH_DEFAULT, seed: int = 0,
                    qd_score_offset: float = -10.0, samples: int = 20_000
                    ) -> "ProductArchive":
        """Rebuild a live `ProductArchive` from `to_arrays()`'s output and
        re-insert every stored elite (for resume / offline analysis)."""
        arc = cls(solution_dim=solution_dim, bd_ranges=bd_ranges,
                  n_context_cells=n_context_cells, n_beh=n_beh, seed=seed,
                  qd_score_offset=qd_score_offset, samples=samples)
        for c, sol, obj, meas in zip(arrays["context_cell"], arrays["solution"],
                                     arrays["objective"], arrays["measures"]):
            arc.add(int(c), sol, float(obj), meas)
        return arc

    def context_behaviour_grid(self) -> np.ndarray:
        """(n_context_cells, n_beh) grid of the best (elite) objective per
        (context, behaviour) cell, NaN where empty"""
        grid = np.full((self.n_context_cells, self.n_beh), np.nan)
        for c, arc in enumerate(self._archives):
            if arc.stats.num_elites == 0:
                continue
            d = arc.data()
            grid[c, d["index"]] = d["objective"]
        return grid

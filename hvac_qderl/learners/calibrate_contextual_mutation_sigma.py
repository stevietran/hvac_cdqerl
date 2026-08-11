"""iso_sigma calibration 

For each candidate `sigma`, offspring `theta' are drawn from a corpus of base genomes 
(several random-init scales x seeds). `probe_bd` is evaluated before and after, 
normalised per axis by `calibrate_probe_bd_ranges`, and the displacement is expressed 
in "cells moved" = normalised L2 displacement / empirical cell width. The recommended
sigma is the one whose median cells-moved is closest to 1.0


USAGE
    python -m hvac_qderl.learners.calibrate_contextual_mutation_sigma \\
        --genome numpy/torch --hidden 64 --n-beh 12
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------- #
# shared measurement primitives -- torch-free
# --------------------------------------------------------------------------- #
def _normalize(bd: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Per-axis min-max into [0, 1] -- the same normalisation the archive's
    tessellation implicitly uses (`bd_ranges` = the CVTArchive's `ranges`),
    so a displacement of 1.0 in this space is comparable to the empirical
    cell width measured in the same space."""
    return (np.asarray(bd, dtype=float) - lo) / np.maximum(hi - lo, 1e-9)


def empirical_cell_width(bd_ranges, n_beh: int, seed: int = 0,
                         samples: int = 20_000) -> float:
    """Median nearest-neighbour spacing between a CVTArchive's own centroids,
    in the SAME normalised [0,1]^4 space as `_normalize` -- a measured
    replacement for the old grid archive's `n_cells ** (-1/D)` cube-width
    proxy, appropriate to a Voronoi (CVT) tessellation instead of a grid.
    """
    from ribs.archives import CVTArchive
    lo = np.array([r[0] for r in bd_ranges], dtype=float)
    hi = np.array([r[1] for r in bd_ranges], dtype=float)
    # solution_dim is irrelevant here -- only the tessellation is used.
    arc = CVTArchive(solution_dim=1, centroids=n_beh, ranges=bd_ranges,
                     samples=samples, seed=seed)
    c = _normalize(arc.centroids, lo, hi)          # (n_beh, D)
    if len(c) < 2:
        return float("nan")
    d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return float(np.median(d.min(axis=1)))


def _geometric_sigmas(lo: float = 0.01, hi: float = 1.0, n: int = 10) -> list[float]:
    return [float(v) for v in np.geomspace(lo, hi, n)]


@dataclass
class CalibrationResult:
    genome_kind: str
    hidden: int
    n_par: int
    n_beh: int
    cell_width: float
    sigmas: list = field(default_factory=list)
    n_samples: list = field(default_factory=list)     # (genome, offspring) pairs / sigma
    median_cells: list = field(default_factory=list)
    p25_cells: list = field(default_factory=list)
    p75_cells: list = field(default_factory=list)
    target: float = 1.0
    band: tuple = (0.5, 2.0)
    recommended_sigma: float = float("nan")
    in_band: bool = False

    def recommend(self) -> None:
        """Pick the swept sigma whose median cells-moved is closest to
        `target`, preferring one that actually lands inside `band` (mirrors
        the old calibration's 0.2-4.0 acceptance window, tightened here
        because the cell width is measured rather than assumed)."""
        med = np.array(self.median_cells, dtype=float)
        sig = np.array(self.sigmas, dtype=float)
        in_band = (med >= self.band[0]) & (med <= self.band[1])
        pool = sig[in_band] if in_band.any() else sig
        med_pool = med[in_band] if in_band.any() else med
        i = int(np.argmin(np.abs(med_pool - self.target)))
        self.recommended_sigma = float(pool[i])
        self.in_band = bool(in_band.any())


def _sweep(base_genomes: list[np.ndarray], bd_fn, n_par: int, bd_ranges,
          n_beh: int, sigmas: list[float], n_offspring: int,
          rng: np.random.Generator, cell_width: float,
          genome_kind: str, hidden: int) -> CalibrationResult:
    lo = np.array([r[0] for r in bd_ranges], dtype=float)
    hi = np.array([r[1] for r in bd_ranges], dtype=float)
    res = CalibrationResult(genome_kind=genome_kind, hidden=hidden, n_par=n_par,
                            n_beh=n_beh, cell_width=cell_width)
    for sigma in sigmas:
        cells = []
        for theta in base_genomes:
            bd0 = _normalize(bd_fn(theta), lo, hi)
            for _ in range(n_offspring):
                theta2 = theta + sigma * rng.normal(size=n_par)
                bd1 = _normalize(bd_fn(theta2), lo, hi)
                cells.append(float(np.linalg.norm(bd1 - bd0)) / max(cell_width, 1e-9))
        cells = np.array(cells)
        res.sigmas.append(float(sigma))
        res.n_samples.append(int(len(cells)))
        res.median_cells.append(float(np.median(cells)))
        res.p25_cells.append(float(np.percentile(cells, 25)))
        res.p75_cells.append(float(np.percentile(cells, 75)))
    res.recommend()
    return res


def _base_genome_corpus(n_par: int, scales, n_per_scale: int,
                        rng: np.random.Generator) -> list[np.ndarray]:
    """Random-init genomes at several scales (mirrors the old calibration's
    "several init scales, several seeds" corpus, notes.md §8.6.4b). No
    archive elites yet the first time this runs -- same documented gap
    `product_archive.calibrate_probe_bd_ranges` already carries; re-run
    against real elites once a trained archive exists."""
    out = []
    for s in scales:
        for _ in range(n_per_scale):
            out.append(rng.normal(0, s, n_par))
    return out


# --------------------------------------------------------------------------- #
# genome kit: numpy MLP (contextual_ga_demo.py's arm) -- torch-free
# --------------------------------------------------------------------------- #
def sweep_numpy_genome(cfg=None, hidden: int = 24, n_beh: int = 6,
                       sigmas: list[float] | None = None,
                       base_scales=(0.1, 0.3, 0.5, 1.0), n_per_scale: int = 4,
                       n_offspring: int = 8, seed: int = 0,
                       n_probe: int = 800, bd_calibration_genomes: int = 24,
                       samples: int = 20_000) -> CalibrationResult:
    """Calibrate `iso_sigma` for `learners.policy.NumpyMLPPolicy` (the
    `contextual_ga_demo.py` genome) -- runs on any machine, no torch."""
    from ..config import default_singapore_config
    from .common import feature_dim_for, ACT_DIM
    from .policy import make_policy
    from .product_archive import calibrate_probe_bd_ranges
    from ..probe_bd import build_probe_grid, probe_bd

    cfg = cfg or default_singapore_config()
    obs_dim = feature_dim_for(cfg)
    n_par = make_policy(obs_dim, ACT_DIM, hidden).n_params
    rng = np.random.default_rng(seed)

    grid = build_probe_grid(cfg, len(cfg.zones), n_probe=n_probe, seed=seed)
    bd_ranges = calibrate_probe_bd_ranges(cfg, n_genomes=bd_calibration_genomes,
                                          hidden=hidden, seed=seed)
    cell_width = empirical_cell_width(bd_ranges, n_beh, seed=seed, samples=samples)

    def bd_fn(theta):
        return probe_bd(theta, cfg, hidden=hidden, grid=grid)

    base = _base_genome_corpus(n_par, base_scales, n_per_scale, rng)
    sigmas = sigmas or _geometric_sigmas()
    return _sweep(base, bd_fn, n_par, bd_ranges, n_beh, sigmas, n_offspring,
                 rng, cell_width, genome_kind="numpy", hidden=hidden)


# --------------------------------------------------------------------------- #
# genome kit: torch GaussianActor (qd_erl_contextual.py's arm) -- needs torch
# --------------------------------------------------------------------------- #
def sweep_torch_genome(cfg=None, hidden: int = 64, n_beh: int = 12,
                       sigmas: list[float] | None = None,
                       base_scales=(0.1, 0.3, 0.5, 1.0), n_per_scale: int = 4,
                       n_offspring: int = 8, seed: int = 0,
                       n_probe: int = 4_500, bd_calibration_genomes: int = 40,
                       samples: int = 20_000, prefer_gpu: bool = True
                       ) -> CalibrationResult:
    """Calibrate `ContextualQDERLConfig.iso_sigma` for the real torch
    `GaussianActor` training genome. Needs torch (not installed in this
    sandbox -- run on a GPU box). Uses `_TorchActorProbePolicy`
    (`qd_erl_contextual.py`, notes.md §8.6.4b bugfix) so `probe_bd` scores
    the REAL actor, not a numpy-genome misreading of its flat params.
    """
    import torch
    from ..config import default_singapore_config
    from .common import feature_dim_for, ACT_DIM
    from .networks import GaussianActor, flat_params, get_device
    from .product_archive import calibrate_probe_bd_ranges
    from .qd_erl_contextual import _TorchActorProbePolicy
    from ..probe_bd import build_probe_grid, probe_bd

    cfg = cfg or default_singapore_config()
    feat_dim = feature_dim_for(cfg)
    device = get_device(prefer_gpu)
    actor = GaussianActor(feat_dim, ACT_DIM, hidden).to(device)
    n_par = flat_params(actor).size
    rng = np.random.default_rng(seed)

    def to_tensor(x_np):
        return torch.as_tensor(x_np, dtype=torch.float32,
                               device=device).unsqueeze(0)

    grid = build_probe_grid(cfg, len(cfg.zones), n_probe=n_probe, seed=seed)
    bd_ranges = calibrate_probe_bd_ranges(cfg, n_genomes=bd_calibration_genomes,
                                          hidden=hidden, seed=seed)
    cell_width = empirical_cell_width(bd_ranges, n_beh, seed=seed, samples=samples)

    from .networks import set_flat_params

    def bd_fn(theta):
        set_flat_params(actor, theta)
        return probe_bd(_TorchActorProbePolicy(actor, to_tensor), cfg, grid=grid)

    base = _base_genome_corpus(n_par, base_scales, n_per_scale, rng)
    sigmas = sigmas or _geometric_sigmas()
    return _sweep(base, bd_fn, n_par, bd_ranges, n_beh, sigmas, n_offspring,
                 rng, cell_width, genome_kind="torch", hidden=hidden)


# --------------------------------------------------------------------------- #
# CSV I/O
# --------------------------------------------------------------------------- #
CSV_FIELDS = ["genome_kind", "hidden", "n_par", "n_beh", "cell_width",
             "sigma", "n_samples", "median_cells", "p25_cells", "p75_cells",
             "target", "band_lo", "band_hi", "recommended_sigma", "in_band"]


def save_csv(res: CalibrationResult, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for sigma, n, med, p25, p75 in zip(res.sigmas, res.n_samples,
                                           res.median_cells, res.p25_cells,
                                           res.p75_cells):
            w.writerow(dict(genome_kind=res.genome_kind, hidden=res.hidden,
                            n_par=res.n_par, n_beh=res.n_beh,
                            cell_width=round(res.cell_width, 6),
                            sigma=sigma, n_samples=n,
                            median_cells=round(med, 6), p25_cells=round(p25, 6),
                            p75_cells=round(p75, 6), target=res.target,
                            band_lo=res.band[0], band_hi=res.band[1],
                            recommended_sigma=round(res.recommended_sigma, 6),
                            in_band=res.in_band))
    return path


# --------------------------------------------------------------------------- #
def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--genome", choices=["numpy", "torch"], default="numpy")
    ap.add_argument("--hidden", type=int, default=None,
                    help="default: 24 for numpy (contextual_ga_demo.py's "
                         "default), 64 for torch (ContextualQDERLConfig's)")
    ap.add_argument("--n-beh", type=int, default=None,
                    help="default: 6 for numpy, 12 for torch")
    ap.add_argument("--n-per-scale", type=int, default=4)
    ap.add_argument("--n-offspring", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args(argv)

    import paths
    paths.ensure_dirs()

    if args.genome == "numpy":
        hidden = args.hidden if args.hidden is not None else 24
        n_beh = args.n_beh if args.n_beh is not None else 6
        res = sweep_numpy_genome(hidden=hidden, n_beh=n_beh,
                                 n_per_scale=args.n_per_scale,
                                 n_offspring=args.n_offspring, seed=args.seed)
    else:
        hidden = args.hidden if args.hidden is not None else 64
        n_beh = args.n_beh if args.n_beh is not None else 12
        res = sweep_torch_genome(hidden=hidden, n_beh=n_beh,
                                 n_per_scale=args.n_per_scale,
                                 n_offspring=args.n_offspring, seed=args.seed)

    out = args.out_csv or os.path.join(
        paths.CALIBRATION, f"iso_sigma_calibration_{args.genome}.csv")
    save_csv(res, out)

    print(f"genome={res.genome_kind}  hidden={res.hidden}  n_par={res.n_par}  "
         f"n_beh={res.n_beh}  cell_width={res.cell_width:.4f}")
    for sigma, med, p25, p75, n in zip(res.sigmas, res.median_cells,
                                       res.p25_cells, res.p75_cells, res.n_samples):
        flag = " <-- recommended" if abs(sigma - res.recommended_sigma) < 1e-9 else ""
        print(f"  sigma={sigma:8.4f}  cells moved: median={med:6.3f} "
             f"[{p25:6.3f}, {p75:6.3f}]  (n={n}){flag}")
    band_note = "within" if res.in_band else "OUTSIDE (nearest available)"
    print(f"\nrecommended iso_sigma = {res.recommended_sigma:.4f}  "
         f"(target {res.target} cell, band {res.band}, {band_note})")
    print(f"saved: {out}")
    print(f"figure: python figures.py --only F21")
    return out


if __name__ == "__main__":
    main()

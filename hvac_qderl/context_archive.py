"""Context archive — 16 data-driven cells (extremes excluded) + 2 fixed sentinel
centroids (notes.md §8.2).

Pipeline, exactly as specified:

    F(d)          4-feature daily vector, one row per calendar day, read
                  straight off `data/annual_load.csv` (§8.2 "one row per
                  calendar day, from data/annual_load.csv")
    PCA basis     fit on the 363 NON-extreme days only, so the two known
                  extremes (peak-cooling-load / min-PV) cannot skew the basis
                  used for the other 363
    k=16 k-means  on the 363 non-extreme days' 2-D projection, with a
                  multi-seed restart that REJECTS any fit containing a
                  singleton cluster (§8.2: "run k-means with >=5 seeds and
                  reject a fit with singleton clusters -- implementation
                  note, not yet automated" -- this module is that automation)
    sentinels     the two extremes, projected into the SAME basis, appended
                  as 2 more FIXED centroids (not additional k-means clusters)
    assignment    all 365 days (including the 2 extremes) assigned to the
                  nearest of the 18 centroids by Euclidean distance

No sklearn in this sandbox (`pip list` confirms it is absent), so PCA is a
plain `numpy.linalg.svd` and k-means is a from-scratch Lloyd's-algorithm
implementation, matching the numpy-only-fallback convention already used by
`representative_periods.py`'s PAM k-medoids.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import paths

DAYS_PER_YEAR = 365
N_CLUSTERS = 16
N_CELLS = N_CLUSTERS + 2                     # + S_peak, S_minpv
FEATURE_NAMES = ("q_mean", "latent_frac", "t_wb_max", "ghi_daily")


# --------------------------------------------------------------------------- #
@dataclass
class ContextArchive:
    features: np.ndarray            # (365, 4)  raw F(d)
    proj: np.ndarray                # (365, 2)  PCA projection, ALL days
    mu: np.ndarray                  # (4,)      feature mean, fit on the 363
    sd: np.ndarray                  # (4,)      feature sd,   fit on the 363
    z_mean: np.ndarray              # (4,)      mean of standardised 363 (~0)
    basis: np.ndarray               # (2, 4)    top-2 right singular vectors
    singular_values: np.ndarray     # (4,)
    explained_variance: np.ndarray  # (4,)      fraction of total variance
    participation_ratio: float      # effective dimensionality, of 4
    centroids: np.ndarray           # (18, 2)   16 k-means + 2 sentinel, in order
    cell_labels: np.ndarray         # (365,)    nearest of 18, for every day
    peak_day: int
    minpv_day: int
    kmeans_seed: int
    kmeans_seeds_tried: int
    kmeans_valid_seeds: int
    kmeans_inertia: float

    def population(self) -> np.ndarray:
        """n_days per cell, index 0..15 = C0..C15 (k-means), 16 = S_peak, 17 = S_minpv."""
        return np.bincount(self.cell_labels, minlength=N_CELLS)

    def cell_name(self, i: int) -> str:
        return "S_peak" if i == N_CLUSTERS else "S_minpv" if i == N_CLUSTERS + 1 \
            else f"C{i}"

    def summary(self) -> str:
        pop = self.population()
        lines = [f"context archive: {N_CLUSTERS} data-driven cells + 2 sentinels "
                f"(PC1+PC2 = {100*self.explained_variance[:2].sum():.1f}% var, "
                f"participation ratio {self.participation_ratio:.2f}/4)",
                f"k-means: seed {self.kmeans_seed} "
                f"({self.kmeans_valid_seeds}/{self.kmeans_seeds_tried} seeds tried "
                f"had no singleton cluster), inertia {self.kmeans_inertia:.2f}"]
        for i in range(N_CELLS):
            lines.append(f"  {self.cell_name(i):>8}  n_days={pop[i]:3d}")
        lines.append(f"  {'total':>8}  n_days={pop.sum():3d}  (must be 365)")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
def _extreme_days(rep_days_csv: str | None = None) -> tuple[int, int]:
    """(peak_load_day, min_pv_day) -- read from `data/representative_days.csv`
    """
    path = rep_days_csv or paths.REPRESENTATIVE_DAYS
    try:
        df = pd.read_csv(path)
        peak = int(df.loc[df["kind"] == "peak_load", "day_index"].iloc[0])
        minpv = int(df.loc[df["kind"] == "min_pv", "day_index"].iloc[0])
        return peak, minpv
    except Exception:
        return 163, 174                      # notes.md §8.2 documented values


def daily_context_features(csv_path: str | None = None) -> np.ndarray:
    """F(d), shape (365, 4), columns = FEATURE_NAMES, straight off the hourly CSV.

        q_mean(d)      = mean_h  q_total_kW[d,h]
        latent_frac(d) = sum_h q_latent_kW[d,h]  /  sum_h q_total_kW[d,h]
        t_wb_max(d)    = max_h  t_wb_C[d,h]
        ghi_daily(d)   = sum_h  ghi_Wm2[d,h]
    """
    path = csv_path or paths.ANNUAL_LOAD
    df = pd.read_csv(path)
    g = df.groupby("day", sort=True)
    q_mean = g["q_total_kW"].mean()
    latent_frac = g["q_latent_kW"].sum() / g["q_total_kW"].sum()
    t_wb_max = g["t_wb_C"].max()
    ghi_daily = g["ghi_Wm2"].sum()
    n_days = len(q_mean)
    if n_days != DAYS_PER_YEAR:
        raise ValueError(f"{path} has {n_days} distinct 'day' values, expected "
                         f"{DAYS_PER_YEAR}")
    F = np.column_stack([q_mean.to_numpy(), latent_frac.to_numpy(),
                         t_wb_max.to_numpy(), ghi_daily.to_numpy()])
    return F


# --------------------------------------------------------------------------- #
def fit_pca_basis(F: np.ndarray, exclude: set[int]):
    """Standardise + fit a 2-D PCA basis on the non-excluded rows only; project
    every row (excluded ones included) into that basis. Returns
    (proj_all, mu, sd, z_mean, basis, singular_values, explained_variance,
    participation_ratio).
    """
    mask = np.array([d not in exclude for d in range(len(F))])
    F_fit = F[mask]
    mu, sd = F_fit.mean(axis=0), F_fit.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    Z_fit = (F_fit - mu) / sd
    z_mean = Z_fit.mean(axis=0)
    Zc = Z_fit - z_mean
    # economy SVD: Zc = U @ diag(S) @ Vh : Vh rows are the principal axes
    _, S, Vh = np.linalg.svd(Zc, full_matrices=False)
    basis = Vh[:2]                                    # (2, 4)
    explained_variance = (S ** 2) / (S ** 2).sum()
    participation_ratio = float((S ** 2).sum() ** 2 / (S ** 4).sum())

    Z_all = (F - mu) / sd
    proj_all = (Z_all - z_mean) @ basis.T              # (365, 2)
    return (proj_all, mu, sd, z_mean, basis, S, explained_variance,
           participation_ratio)


# --------------------------------------------------------------------------- #
def _kmeanspp_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = X.shape[0]
    centers = np.empty((k, X.shape[1]))
    centers[0] = X[int(rng.integers(n))]
    d2 = ((X - centers[0]) ** 2).sum(axis=1)
    for j in range(1, k):
        probs = d2 / d2.sum() if d2.sum() > 0 else np.ones(n) / n
        centers[j] = X[int(rng.choice(n, p=probs))]
        d2 = np.minimum(d2, ((X - centers[j]) ** 2).sum(axis=1))
    return centers


def _kmeans_once(X: np.ndarray, k: int, seed: int, n_iter: int = 300):
    """Lloyd's algorithm, k-means++ init, numpy only. Empty clusters (not just
    singletons) are re-seeded to the current farthest point so a run never
    crashes on a NaN centroid; the singleton REJECTION happens one level up.
    """
    rng = np.random.default_rng(seed)
    centers = _kmeanspp_init(X, k, rng)
    labels = np.zeros(len(X), dtype=int)
    for _ in range(n_iter):
        d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = d2.argmin(axis=1)
        if np.array_equal(new_labels, labels) and _ > 0:
            labels = new_labels
            break
        labels = new_labels
        new_centers = centers.copy()
        for j in range(k):
            members = X[labels == j]
            if len(members) > 0:
                new_centers[j] = members.mean(axis=0)
            else:                                       # empty -> farthest point
                worst = d2[np.arange(len(X)), labels].argmax()
                new_centers[j] = X[worst]
        centers = new_centers
    d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    labels = d2.argmin(axis=1)
    inertia = float(d2[np.arange(len(X)), labels].sum())
    return centers, labels, inertia


def fit_kmeans_reject_singletons(X: np.ndarray, k: int = N_CLUSTERS,
                                 min_valid_seeds: int = 5, max_seeds: int = 200,
                                 start_seed: int = 0):
    """Try seeds `start_seed, start_seed+1, ...` until at least `min_valid_seeds`
    produce a k=16 fit with NO singleton (or empty) cluster; return the
    lowest-inertia valid fit.
    """
    valid = []
    tried = 0
    for seed in range(start_seed, start_seed + max_seeds):
        tried += 1
        centers, labels, inertia = _kmeans_once(X, k, seed)
        sizes = np.bincount(labels, minlength=k)
        if sizes.min() >= 2:
            valid.append((inertia, seed, centers, labels))
        if len(valid) >= min_valid_seeds and tried >= min_valid_seeds:
            # keep searching a little past the floor only if cheap; stop once
            # we have the required number of singleton-free candidates AND
            # have tried at least that many seeds (matches ">=5 seeds").
            break
    if not valid:
        raise RuntimeError(
            f"no singleton-free k={k} fit found in {tried} seeds starting at "
            f"{start_seed} -- widen the search (max_seeds) or inspect the data.")
    valid.sort(key=lambda t: t[0])                      # lowest inertia first
    inertia, seed, centers, labels = valid[0]
    return dict(centers=centers, labels=labels, inertia=inertia, seed=seed,
               seeds_tried=tried, valid_seeds=len(valid))


# --------------------------------------------------------------------------- #
def build_context_archive(csv_path: str | None = None, rep_days_csv: str | None = None,
                          min_valid_seeds: int = 5, max_seeds: int = 200,
                          start_seed: int = 0) -> ContextArchive:
    peak_day, minpv_day = _extreme_days(rep_days_csv)
    exclude = {peak_day, minpv_day}

    F = daily_context_features(csv_path)
    (proj_all, mu, sd, z_mean, basis, S, explained_variance,
     participation_ratio) = fit_pca_basis(F, exclude)

    fit_mask = np.array([d not in exclude for d in range(DAYS_PER_YEAR)])
    proj_363 = proj_all[fit_mask]

    km = fit_kmeans_reject_singletons(proj_363, k=N_CLUSTERS,
                                      min_valid_seeds=min_valid_seeds,
                                      max_seeds=max_seeds, start_seed=start_seed)

    sentinel_centroids = np.stack([proj_all[peak_day], proj_all[minpv_day]])
    centroids_18 = np.vstack([km["centers"], sentinel_centroids])

    d2 = ((proj_all[:, None, :] - centroids_18[None, :, :]) ** 2).sum(axis=2)
    cell_labels = d2.argmin(axis=1)

    return ContextArchive(
        features=F, proj=proj_all, mu=mu, sd=sd, z_mean=z_mean, basis=basis,
        singular_values=S, explained_variance=explained_variance,
        participation_ratio=participation_ratio, centroids=centroids_18,
        cell_labels=cell_labels, peak_day=peak_day, minpv_day=minpv_day,
        kmeans_seed=km["seed"], kmeans_seeds_tried=km["seeds_tried"],
        kmeans_valid_seeds=km["valid_seeds"], kmeans_inertia=km["inertia"])


# --------------------------------------------------------------------------- #
def nearest_day_to_centroid(archive: ContextArchive, cell_id: int) -> int:
    """The real calendar day nearest a context cell's centroid, in the SAME
    projected 2-D space `build_context_archive`"""
    d2 = ((archive.proj - archive.centroids[cell_id]) ** 2).sum(axis=1)
    return int(np.argmin(d2))


if __name__ == "__main__":
    arc = build_context_archive()
    print(arc.summary())
    print(f"day {arc.peak_day} (peak-load)  -> {arc.cell_name(arc.cell_labels[arc.peak_day])}")
    print(f"day {arc.minpv_day} (min-PV)    -> {arc.cell_name(arc.cell_labels[arc.minpv_day])}")

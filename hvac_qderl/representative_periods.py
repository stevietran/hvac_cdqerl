"""Reduce the 365-day year to `n_rep_days` representative days + 2 forced extremes.
Cluster the 365 days by a standardised daily feature vector with **k-medoids**
FORCE-INCLUDE two extremes so worst cases are never averaged away:
- peak-cooling-load day   (max daily cooling load)
- minimum-PV day          (min daily PV per kWp = the overcast monsoon day)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

try:
    from sklearn_extra.cluster import KMedoids
    _HAVE_SKEXTRA = True
except Exception:                                    # pragma: no cover
    _HAVE_SKEXTRA = False

DAYS_PER_YEAR = 365


@dataclass
class RepresentativeDay:
    label: str                    # "day_198" / "EXTREME_peak_cooling_day_163"
    day_index: int                # 0..364 source day
    weight_days: float            # annual days represented (Σ == 365)
    hours: np.ndarray             # source hour indices (length 24)
    features: Dict[str, float] = field(default_factory=dict)
    is_extreme: bool = False
    kind: str = "cluster"         # cluster | peak_load | min_pv

    @property
    def weight_fraction(self) -> float:
        return self.weight_days / DAYS_PER_YEAR


@dataclass
class RepresentativeSet:
    days: List[RepresentativeDay]
    n_rep_days: int
    peak_load_day: int
    min_pv_day: int
    method: str = ""

    @property
    def day_indices(self) -> List[int]:
        return [d.day_index for d in self.days]

    @property
    def weights(self) -> np.ndarray:
        return np.array([d.weight_days for d in self.days])

    @property
    def n_days(self) -> int:
        return len(self.days)

    @property
    def horizon_hours(self) -> float:
        return 24.0 * self.n_days

    def summary(self) -> str:
        L = [f"{self.n_days} representative days "
             f"(n_rep_days={self.n_rep_days} + peak-load + min-PV), "
             f"episode = {self.horizon_hours:.0f} h, method={self.method}",
             f"episode order: {' -> '.join(str(d.day_index) for d in self.days)}",
             f"{'day':>5} {'weight':>8} {'frac':>7}  kind"]
        for d in self.days:
            L.append(f"{d.day_index:>5} {d.weight_days:>8.1f} "
                     f"{100*d.weight_fraction:>6.2f}%  {d.kind}")
        L.append(f"{'':>5} {self.weights.sum():>8.1f} "
                 f"{100*self.weights.sum()/DAYS_PER_YEAR:>6.2f}%  (must total 365)")
        return "\n".join(L)


# --------------------------------------------------------------------------- #
def daily_features(vectors: Dict[str, np.ndarray]) -> np.ndarray:
    """Standardised (365, F) daily feature matrix.

    For each control-relevant hourly signal we take the daily mean AND the daily
    max: the mean separates hot/mild days, the max separates days that share a mean
    but differ in peak — which is what the peak-demand term in the reward reacts to.
    """
    feats, names = [], []
    for name in ("q_load", "q_latent", "t_db", "t_wb", "w_oa", "ghi"):
        if name not in vectors:
            continue
        v = np.asarray(vectors[name])[:DAYS_PER_YEAR * 24].reshape(DAYS_PER_YEAR, 24)
        feats.append(v.mean(axis=1)); names.append(f"{name}_mean")
        feats.append(v.max(axis=1)); names.append(f"{name}_max")
    # latent fraction of the day's load: separates monsoon (humid, moderate temp)
    # days from hot-dry-ish days with the same total load — a different control regime
    if "q_load" in vectors and "q_latent" in vectors:
        ql = np.asarray(vectors["q_load"])[:8760].reshape(DAYS_PER_YEAR, 24).sum(1)
        qlat = np.asarray(vectors["q_latent"])[:8760].reshape(DAYS_PER_YEAR, 24).sum(1)
        feats.append(qlat / np.maximum(ql, 1e-9)); names.append("latent_fraction")

    X = np.column_stack(feats).astype(float)
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def _pam_fallback(X: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Minimal PAM/k-medoids returning medoid row indices (no sklearn needed)."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    # k-medoids++ style seeding: first medoid random, rest far from chosen ones
    medoids = [int(rng.integers(n))]
    for _ in range(k - 1):
        dmin = D[:, medoids].min(axis=1)
        probs = dmin ** 2
        s = probs.sum()
        medoids.append(int(rng.choice(n, p=probs / s)) if s > 0
                       else int(rng.integers(n)))
    medoids = np.array(sorted(set(medoids)))
    while len(medoids) < k:                      # top up if duplicates collapsed
        cand = int(rng.integers(n))
        if cand not in medoids:
            medoids = np.append(medoids, cand)
    for _ in range(100):
        labels = D[:, medoids].argmin(axis=1)
        new_med = medoids.copy()
        for j in range(len(medoids)):
            members = np.where(labels == j)[0]
            if len(members) == 0:
                continue
            costs = D[np.ix_(members, members)].sum(axis=1)
            new_med[j] = members[costs.argmin()]
        if np.array_equal(np.sort(new_med), np.sort(medoids)):
            break
        medoids = new_med
    return medoids


def build_representative_days(vectors: Dict[str, np.ndarray], n_rep_days: int,
                              force_extremes: bool = True, seed: int = 0,
                              extremes_last: bool = True
                              ) -> RepresentativeSet:
    """`vectors`: name -> 8760-array. Needs `q_load`; `pv_per_kwp` enables min-PV.

    `extremes_last=True` (default) emits the cluster medoids in chronological order
    followed by min-PV then peak-load, so the episode-wide peak ratchet in the
    reward is only frozen over the final day (see module docstring).
    `extremes_last=False` restores the original fully-chronological order.
    """
    X = daily_features(vectors)
    n_rep_days = int(max(1, min(n_rep_days, DAYS_PER_YEAR - 2)))

    if _HAVE_SKEXTRA:
        km = KMedoids(n_clusters=n_rep_days, metric="euclidean", method="pam",
                      init="k-medoids++", random_state=seed)
        labels = km.fit_predict(X)
        medoids = km.medoid_indices_
        method = "sklearn_extra.KMedoids(pam)"
    else:
        medoids = _pam_fallback(X, n_rep_days, seed)
        D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
        labels = D[:, medoids].argmin(axis=1)
        method = "numpy PAM fallback (k-medoids++ seed)"

    rep = {int(m): float(np.sum(labels == j)) for j, m in enumerate(medoids)}

    q_day = np.asarray(vectors["q_load"])[:8760].reshape(DAYS_PER_YEAR, 24).sum(1)
    peak_day = int(q_day.argmax())
    if "pv_per_kwp" in vectors:
        pv_day = np.asarray(vectors["pv_per_kwp"])[:8760].reshape(DAYS_PER_YEAR, 24).sum(1)
        minpv_day = int(pv_day.argmin())
    else:
        ghi_day = np.asarray(vectors["ghi"])[:8760].reshape(DAYS_PER_YEAR, 24).sum(1)
        minpv_day = int(ghi_day.argmin())

    if force_extremes:
        for d in (peak_day, minpv_day):
            if d not in rep:
                # steal one day of weight from the cluster that owned it, so Σw stays 365
                owner = int(medoids[labels[d]])
                if owner in rep and rep[owner] > 1:
                    rep[owner] -= 1.0
                rep[d] = rep.get(d, 0.0) + 1.0

    # ---- episode ordering ------------------------------------------------- #
    # Default: cluster medoids chronologically, then min-PV, then peak-load
    if extremes_last:
        tail: List[int] = []
        for d in (minpv_day, peak_day):          # min-PV second-last, peak last
            if d in rep and d not in tail:
                tail.append(d)
        order = [d for d in sorted(rep.keys()) if d not in tail] + tail
    else:
        order = sorted(rep.keys())                      # chronological (legacy)

    days: List[RepresentativeDay] = []
    for d in order:
        hrs = np.arange(d * 24, d * 24 + 24)
        if d == peak_day:
            lab, kind = f"EXTREME_peak_cooling_day_{d}", "peak_load"
        elif d == minpv_day:
            lab, kind = f"EXTREME_min_pv_day_{d}", "min_pv"
        else:
            lab, kind = f"day_{d}", "cluster"
        feats = {}
        for nm in ("q_load", "t_db", "t_wb", "w_oa", "ghi", "q_latent"):
            if nm in vectors:
                v = np.asarray(vectors[nm])[hrs]
                feats[f"{nm}_mean"] = float(v.mean())
                feats[f"{nm}_max"] = float(v.max())
        days.append(RepresentativeDay(
            label=lab, day_index=d, weight_days=rep[d], hours=hrs,
            features=feats, is_extreme=kind != "cluster", kind=kind))

    # normalise weights to exactly 365 (numerical safety)
    tot = sum(p.weight_days for p in days)
    for p in days:
        p.weight_days *= DAYS_PER_YEAR / tot

    return RepresentativeSet(days=days, n_rep_days=n_rep_days,
                             peak_load_day=peak_day, min_pv_day=minpv_day,
                             method=method)


# --------------------------------------------------------------------------- #
def build_from_weather(annual, load, n_rep_days: int, seed: int = 0,
                       extremes_last: bool = True) -> RepresentativeSet:
    """Convenience: assemble the feature vectors from AnnualWeather + AnnualLoad."""
    from .cooling_load import pv_per_kwp
    vectors = {
        "q_load": load.q_total,
        "q_latent": load.q_latent,
        "t_db": annual.t_db,
        "t_wb": annual.t_wb,
        "w_oa": annual.w_oa,
        "ghi": annual.ghi,
        "pv_per_kwp": pv_per_kwp(annual),
    }
    return build_representative_days(vectors, n_rep_days, seed=seed,
                                     extremes_last=extremes_last)

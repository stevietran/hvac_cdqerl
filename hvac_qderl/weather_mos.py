"""Annual weather from an EnergyPlus/TMY `.mos` file (SGP_Singapore.486980_IWEC.mos).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from . import psychrometrics as psy
from .weather import Disturbance

# 0-based column positions
COL_TIME, COL_TDB, COL_TDP, COL_RH, COL_PRES = 0, 1, 2, 3, 4
COL_GHI, COL_DNI, COL_DHI, COL_WSPD = 8, 9, 10, 16

HOURS_PER_YEAR = 8760
DAYS_PER_YEAR = 365


def default_mos_path() -> str:
    """Locate the bundled Singapore IWEC file (searched in a few usual places)."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cands = [
        os.path.join(here, "data", "SGP_Singapore.486980_IWEC.mos"),
        os.path.join(here, "..", "..", "j2", "optimizer", "load",
                     "SGP_Singapore.486980_IWEC.mos"),
        os.path.join(here, "..", "optimizer", "load",
                     "SGP_Singapore.486980_IWEC.mos"),
        "/sessions/zealous-upbeat-ptolemy/mnt/optimizer/load/SGP_Singapore.486980_IWEC.mos",
    ]
    for c in cands:
        if os.path.exists(c):
            return os.path.abspath(c)
    raise FileNotFoundError(
        "SGP_Singapore.486980_IWEC.mos not found. Place it in code/data/ or pass "
        "an explicit path.")


@dataclass
class AnnualWeather:
    """8,760-hour weather record plus derived psychrometric quantities."""
    t_db: np.ndarray            # dry-bulb [degC]
    t_dp: np.ndarray            # dew point [degC]
    rh: np.ndarray              # relative humidity [0-1]
    pressure: np.ndarray        # [Pa]
    ghi: np.ndarray             # global horizontal irradiance [W/m2]
    dni: np.ndarray             # direct normal [W/m2]
    dhi: np.ndarray             # diffuse horizontal [W/m2]
    wind: np.ndarray            # [m/s]
    w_oa: np.ndarray = field(default=None)     # humidity ratio [kg/kg]
    t_wb: np.ndarray = field(default=None)     # wet-bulb [degC]
    source: str = ""

    def __post_init__(self):
        if self.w_oa is None:
            self.w_oa = np.array([psy.W_from_TRH(t, r, p) for t, r, p
                                  in zip(self.t_db, self.rh, self.pressure)])
        if self.t_wb is None:
            self.t_wb = np.array([psy.wet_bulb(t, w, p) for t, w, p
                                  in zip(self.t_db, self.w_oa, self.pressure)])

    # ---- convenience reshapes -------------------------------------------- #
    def daily(self, name: str) -> np.ndarray:
        """(365, 24) view of any field."""
        return np.asarray(getattr(self, name))[:HOURS_PER_YEAR].reshape(
            DAYS_PER_YEAR, 24)

    @property
    def n_hours(self) -> int:
        return len(self.t_db)

    def summary(self) -> str:
        return (f"{os.path.basename(self.source)}: {self.n_hours} h | "
                f"DB {self.t_db.min():.1f}-{self.t_db.max():.1f} degC "
                f"(mean {self.t_db.mean():.1f}) | "
                f"WB max {self.t_wb.max():.1f} degC | "
                f"RH {100*self.rh.min():.0f}-{100*self.rh.max():.0f}% | "
                f"GHI peak {self.ghi.max():.0f} W/m2, "
                f"annual {self.ghi.sum()/1000:.0f} kWh/m2")

def parse_mos(path: str | None = None) -> AnnualWeather:
    """Read a `.mos` TMY/IWEC file into an AnnualWeather record."""
    path = path or default_mos_path()
    cols = {k: [] for k in ("tdb", "tdp", "rh", "pres", "ghi", "dni", "dhi", "wspd")}
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("double"):
                continue
            parts = line.split("\t")
            if len(parts) < 30:
                parts = line.split()
            if len(parts) < 30:
                continue
            try:
                cols["tdb"].append(float(parts[COL_TDB]))
                cols["tdp"].append(float(parts[COL_TDP]))
                cols["rh"].append(float(parts[COL_RH]))
                cols["pres"].append(float(parts[COL_PRES]))
                cols["ghi"].append(float(parts[COL_GHI]))
                cols["dni"].append(float(parts[COL_DNI]))
                cols["dhi"].append(float(parts[COL_DHI]))
                cols["wspd"].append(float(parts[COL_WSPD]))
            except (ValueError, IndexError):
                continue

    n = len(cols["tdb"])
    if n < HOURS_PER_YEAR:
        raise ValueError(f"{path}: only {n} data rows parsed, expected >= 8760")
    trim = lambda k: np.asarray(cols[k][:HOURS_PER_YEAR], dtype=float)

    rh = np.clip(trim("rh") / 100.0, 0.01, 1.0)
    pres = trim("pres")
    pres[pres <= 0] = psy.P_ATM
    return AnnualWeather(t_db=trim("tdb"), t_dp=trim("tdp"), rh=rh, pressure=pres,
                         ghi=np.clip(trim("ghi"), 0, None),
                         dni=np.clip(trim("dni"), 0, None),
                         dhi=np.clip(trim("dhi"), 0, None),
                         wind=np.clip(trim("wspd"), 0, None),
                         source=path)

# --------------------------------------------------------------------------- #
# Episode weather driver
# --------------------------------------------------------------------------- #
def _is_weekend(day_index: int, first_weekday: int = 6) -> bool:
    """IWEC #DATA PERIODS for this file starts on a Sunday (weekday 6)."""
    return ((first_weekday + day_index) % 7) in (5, 6)

class MosWeather:
    """Serves `Disturbance` from the annual record over a schedule of days.

    `day_schedule` is the ordered list of source day indices (0..364) the episode
    walks through. Episode hour `h` maps to slot `int(h // 24)` of the schedule
    and hour-of-day `h % 24`, so an episode of `len(day_schedule)` days is
    exactly `24 * len(day_schedule)` hours long. Values are linearly interpolated
    between hourly samples, which matters because the control step is 5 min.

    Occupancy is still schedule-based (the weather file carries no occupancy) but
    is now **weekday-aware** from the real calendar day
    """

    def __init__(self, annual: AnnualWeather, day_schedule,
                 price_profile: str | None = None, occupancy_weekends: bool = True,
                 first_weekday: int = 6, day_weights=None):
        self.annual = annual
        self.days = [int(d) for d in day_schedule]
        if not self.days:
            raise ValueError("day_schedule must contain at least one day")
        from .config import resolve_price_profile
        self.price_profile = resolve_price_profile(price_profile)
        self.occupancy_weekends = occupancy_weekends
        self.first_weekday = first_weekday
        self.day_weights = (np.ones(len(self.days)) if day_weights is None
                            else np.asarray(day_weights, dtype=float))
        if len(self.day_weights) != len(self.days):
            raise ValueError("day_weights must match day_schedule length")

    # ------------------------------------------------------------------ #
    @property
    def n_days(self) -> int:
        return len(self.days)

    @property
    def horizon_hours(self) -> float:
        return 24.0 * self.n_days

    @property
    def solar_ceiling(self) -> float:
        """Peak GHI [W/m2] in the source record
        """
        return float(self.annual.ghi.max())

    def _sample(self, arr, day: int, hod: float) -> float:
        """Linear interpolation within the day (wrapping at midnight)."""
        h0 = int(np.floor(hod)) % 24
        h1 = (h0 + 1) % 24
        f = hod - np.floor(hod)
        i0, i1 = day * 24 + h0, day * 24 + h1
        return float((1.0 - f) * arr[i0] + f * arr[i1])

    def at(self, hour: float) -> Disturbance:
        slot = int(hour // 24)
        slot = min(max(slot, 0), self.n_days - 1)      # clamp at episode end
        day = self.days[slot]
        hod = hour - 24.0 * slot
        a = self.annual

        t_oa = self._sample(a.t_db, day, hod)
        w_oa = self._sample(a.w_oa, day, hod)
        t_wb = self._sample(a.t_wb, day, hod)
        ghi = self._sample(a.ghi, day, hod)

        occ = self._occupancy(day, hod)
        price = self._price(hod)
        return Disturbance(t_oa=t_oa, w_oa=w_oa, t_wb=t_wb,
                           q_solar_wm2=ghi, occ_frac=occ, price=price)

    # ------------------------------------------------------------------ #
    def _occupancy(self, day: int, hod: float) -> float:
        if self.occupancy_weekends and _is_weekend(day, self.first_weekday):
            return 0.15 if 9.0 <= hod <= 15.0 else 0.05     # skeleton weekend crew
        if hod < 7.0 or hod > 20.0:
            return 0.05
        if hod < 9.0:
            return 0.05 + 0.95 * (hod - 7.0) / 2.0
        if hod <= 18.0:
            return 1.0
        return max(0.05, 1.0 - (hod - 18.0) / 2.0)

    def _price(self, hod: float) -> float:
        if self.price_profile == "constant":
            return 1.0
        if self.price_profile == "highly_dynamic":
            return 2.0 if 13.0 <= hod <= 17.0 else (0.6 if hod < 7.0 else 1.0)
        return 1.6 if 13.0 <= hod <= 17.0 else (0.7 if hod < 7.0 else 1.0)

    # ------------------------------------------------------------------ #
    def describe(self) -> str:
        kinds = []
        for d in self.days:
            tag = "weekend" if _is_weekend(d, self.first_weekday) else "weekday"
            kinds.append(f"day{d}({tag})")
        return f"{self.n_days} days x 24 h = {self.horizon_hours:.0f} h: " + \
               ", ".join(kinds)

def load_annual(path: str | None = None) -> AnnualWeather:
    """Cached-free convenience loader."""
    return parse_mos(path)

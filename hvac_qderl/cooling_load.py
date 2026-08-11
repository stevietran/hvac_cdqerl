"""Annual building cooling-load calculation from the .mos weather file.

**The proposed method** (an *ideal-loads* heat/moisture balance, the same idea as
EnergyPlus's `IdealLoadsAirSystem`): hold every zone exactly at its temperature
and humidity setpoint, integrate only the *slow* states, and read off the cooling
power the plant would have to deliver. Because the zone is pinned, the result is
the **load** — a property of the building and weather, independent of any
controller — which is exactly what is needed to (a) size the plant and (b) build
the daily feature vectors the representative-day clustering runs on.

Per hour, per zone:

    Q_sen_zone = (T_oa - T_z)/R_oa + (T_m - T_z)/R_zm + Q_int,s + Q_sol
    Q_lat_zone = h_fg · [ ṁ_inf (W_oa - W_z) + ṁ_gen ]
    Q_sen_OA   = ṁ_oa · c_p · (T_oa - T_z)          (ventilation, SS 553 rate)
    Q_lat_OA   = ṁ_oa · h_fg · (W_oa - W_z)         (dominant term in the tropics)

with the mass node integrated as
    C_m dT_m/dt = (T_z - T_m)/R_zm + (T_oa - T_m)/R_om   (backward Euler)

Coil load = Σ_zones (Q_sen_zone + Q_lat_zone) + Q_sen_OA + Q_lat_OA, floored at 0
(a tropical office never needs heating).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import psychrometrics as psy
from .config import SingaporeConfig, default_singapore_config
from .equipment import KW_PER_RT
from .weather_mos import AnnualWeather, DAYS_PER_YEAR, HOURS_PER_YEAR, _is_weekend


@dataclass
class AnnualLoad:
    """Hourly annual cooling load and the quantities derived from it."""
    q_total: np.ndarray          # coil load [kW], 8760
    q_sensible: np.ndarray       # [kW]
    q_latent: np.ndarray         # [kW]
    shr: np.ndarray              # sensible heat ratio [-]
    q_oa: np.ndarray             # ventilation share of the load [kW]
    occ: np.ndarray              # occupancy fraction used [-]
    config_name: str = ""
    floor_area_m2: float = 0.0
    meta: dict = field(default_factory=dict)

    # ---- aggregates ---------------------------------------------------- #
    def daily_sum(self, arr=None) -> np.ndarray:
        a = self.q_total if arr is None else arr
        return a[:HOURS_PER_YEAR].reshape(DAYS_PER_YEAR, 24).sum(axis=1)

    def daily_peak(self, arr=None) -> np.ndarray:
        a = self.q_total if arr is None else arr
        return a[:HOURS_PER_YEAR].reshape(DAYS_PER_YEAR, 24).max(axis=1)

    @property
    def annual_kwh(self) -> float:
        return float(self.q_total.sum())          # hourly steps -> kWh directly

    @property
    def peak_kw(self) -> float:
        return float(self.q_total.max())

    def design_kw(self, percentile: float = 99.6) -> float:
        """Design capacity at an ASHRAE-style percentile (robust to a single spike)."""
        return float(np.percentile(self.q_total, percentile))

    @property
    def peak_day(self) -> int:
        return int(self.daily_sum().argmax())

    @property
    def eui_cooling_kwh_m2(self) -> float:
        return self.annual_kwh / max(self.floor_area_m2, 1.0)

    def summary(self) -> str:
        return (f"annual cooling {self.annual_kwh/1000:,.0f} MWh_th "
                f"({self.eui_cooling_kwh_m2:.0f} kWh/m2) | "
                f"peak {self.peak_kw:,.0f} kW ({self.peak_kw/KW_PER_RT:,.0f} RT, "
                f"{1000*self.peak_kw/max(self.floor_area_m2,1):.0f} W/m2) | "
                f"design@99.6% {self.design_kw():,.0f} kW | "
                f"mean SHR {np.nanmean(self.shr):.2f} | peak day {self.peak_day}")


def _occupancy(day: int, hod: float, weekends: bool = True,
               first_weekday: int = 6) -> float:
    """Same schedule MosWeather uses, so load and simulation agree."""
    if weekends and _is_weekend(day, first_weekday):
        return 0.15 if 9.0 <= hod <= 15.0 else 0.05
    if hod < 7.0 or hod > 20.0:
        return 0.05
    if hod < 9.0:
        return 0.05 + 0.95 * (hod - 7.0) / 2.0
    if hod <= 18.0:
        return 1.0
    return max(0.05, 1.0 - (hod - 18.0) / 2.0)


def compute_annual_load(annual: AnnualWeather,
                        cfg: SingaporeConfig | None = None,
                        rh_target: float = 0.55,
                        substeps: int = 4,
                        weekends: bool = True) -> AnnualLoad:
    """Ideal-loads annual cooling-load calculation.

    `rh_target` is the zone humidity the plant is assumed to hold (below the 0.65
    ceiling of §16.2); it sets the latent duty. `substeps` sub-divides each hour
    for the mass-node integration.
    """
    cfg = cfg or default_singapore_config()
    zones = cfg.zones
    n_h = min(annual.n_hours, HOURS_PER_YEAR)
    T_z = cfg.T_set
    W_z = psy.W_from_TRH(T_z, rh_target)

    # ventilation rate (SS 553 / ASHRAE 62.1): 0.3 L/s.m2 + 3.8 L/s.person
    total_area = cfg.total_area
    peak_persons = sum(z.occ_density * z.area_m2 for z in zones)

    # per-zone constants as arrays for speed
    R_oa = np.array([z.R_oa for z in zones])
    R_zm = np.array([z.R_zm for z in zones])
    R_om = np.array([z.R_om for z in zones])
    C_m = np.array([z.C_m for z in zones])
    areas = np.array([z.area_m2 for z in zones])
    aperture = np.array([z.solar_aperture_m2 for z in zones])
    light_plug_w = np.array([(z.lighting_wm2 + z.plug_wm2) * z.area_m2 for z in zones])
    occ_dens = np.array([z.occ_density * z.area_m2 for z in zones])
    sens_pp = np.array([z.sens_per_person_w for z in zones])
    lat_pp_gph = np.array([z.lat_per_person_gph for z in zones])
    infil_rate = float(getattr(zones[0], "infiltration_kgs_per_m2", 1.0e-4))

    T_m = np.full(len(zones), T_z + 1.0)          # warm start; settles in days
    dt = 3600.0 / substeps

    q_tot = np.zeros(n_h)
    q_sen_a = np.zeros(n_h)
    q_lat_a = np.zeros(n_h)
    q_oa_a = np.zeros(n_h)
    occ_a = np.zeros(n_h)

    for h in range(n_h):
        day, hod = h // 24, float(h % 24)
        t_oa = annual.t_db[h]
        w_oa = annual.w_oa[h]
        ghi = annual.ghi[h]
        occ = _occupancy(day, hod, weekends)
        occ_a[h] = occ

        # ---- internal gains ---- #
        q_int_s = (light_plug_w * (0.3 + 0.7 * occ) + occ_dens * occ * sens_pp) / 1000.0
        m_gen = occ_dens * occ * lat_pp_gph / 1000.0 / 3600.0        # kg/s
        q_sol = ghi * aperture / 1000.0                              # kW

        # ---- mass node (backward Euler over substeps, T_z pinned) ---- #
        for _ in range(substeps):
            a = C_m / dt + 1.0 / R_zm + 1.0 / R_om
            b = C_m / dt * T_m + T_z / R_zm + t_oa / R_om
            T_m = b / a

        # ---- zone sensible + latent to hold setpoint ---- #
        q_sen_zone = ((t_oa - T_z) / R_oa + (T_m - T_z) / R_zm + q_int_s + q_sol)
        # infiltration from the config's per-area leakage rate (~0.1 ACH)
        m_inf = infil_rate * areas                                   # kg/s
        q_lat_zone = psy.H_FG * (m_inf * max(w_oa - W_z, 0.0) + m_gen)

        # ---- ventilation outdoor-air load ---- #
        v_oa = 0.3e-3 * total_area + 3.8e-3 * peak_persons * occ      # m3/s
        m_oa = psy.RHO_AIR * v_oa                                    # kg/s
        q_oa_sen = m_oa * psy.CP_AIR * max(t_oa - T_z, 0.0)
        q_oa_lat = m_oa * psy.H_FG * max(w_oa - W_z, 0.0)

        sen = float(np.clip(q_sen_zone, 0, None).sum()) + q_oa_sen
        lat = float(q_lat_zone.sum()) + q_oa_lat
        q_sen_a[h] = sen
        q_lat_a[h] = lat
        q_oa_a[h] = q_oa_sen + q_oa_lat
        q_tot[h] = max(sen + lat, 0.0)

    shr = np.divide(q_sen_a, np.maximum(q_tot, 1e-9))
    return AnnualLoad(q_total=q_tot, q_sensible=q_sen_a, q_latent=q_lat_a,
                      shr=shr, q_oa=q_oa_a, occ=occ_a,
                      config_name=cfg.name, floor_area_m2=total_area,
                      meta={"rh_target": rh_target, "T_set": T_z,
                           "weather": annual.source, "substeps": substeps})


# --------------------------------------------------------------------------- #
def pv_per_kwp(annual: AnnualWeather, noct: float = 45.0, alpha_t: float = -0.0035,
               i_noc: float = 800.0, t_a_noc: float = 20.0,
               inverter_eff: float = 0.96) -> np.ndarray:
    """AC PV yield per kWp [kW/kWp] — used to identify the **min-PV day**.
    """
    ghi = annual.ghi
    t_cell = annual.t_db + (ghi / i_noc) * (noct - t_a_noc)
    dc = (ghi / 1000.0) * (1.0 + alpha_t * (t_cell - 25.0))
    return np.clip(dc, 0.0, None) * inverter_eff


def apply_load_derived_airflow(cfg: SingaporeConfig,
                               load: AnnualLoad) -> SingaporeConfig:
    """Set the AIR-SIDE design point from the calculated load. No capacity.
    """
    import copy

    new = copy.deepcopy(cfg)
    design = float(new.design_cooling_kw)        # installed capacity, hard-coded

    # coincident SHR at the hour whose total load is closest to the design point
    i_design = int(np.argmin(np.abs(load.q_total - design)))
    shr_design = float(load.shr[i_design])
    if not (0.3 <= shr_design <= 1.0):           # degenerate hour; fall back
        shr_design = float(np.nanpercentile(load.shr, 95))

    new.design_shr = shr_design
    new.m_air_design = (shr_design * design) / (1.006 * new.supply_deltaT_K)
    new.coil.m_air_design = new.m_air_design
    new.coil.ua_design = 1.6 * new.m_air_design * 1.006

    new.sizing_basis = dict(
        design_cooling_kw=design,
        n_chillers=len(new.plant.chillers),
        q_each_kw=new.plant.chillers[0].q_ref_kw,
        design_shr=shr_design,
        simulated_peak_ref_kw=float(load.peak_kw))
    return new

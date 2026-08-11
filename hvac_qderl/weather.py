"""Exogenous disturbance w(t) — weather, occupancy, solar, tariff

A synthetic but Singapore-representative cooling day: near-constant high dry-bulb
and wet-bulb with a mild diurnal swing, high RH, a solar bell curve, and an
office occupancy schedule (~08:00-19:00)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from . import psychrometrics as psy


@dataclass
class Disturbance:
    t_oa: float          # outdoor dry-bulb [degC]
    w_oa: float          # outdoor humidity ratio [kg/kg]
    t_wb: float          # outdoor wet-bulb [degC]
    q_solar_wm2: float   # transmitted solar on facade [W/m2]
    occ_frac: float      # occupancy fraction of peak [0-1]
    price: float         # energy price multiplier [-]


class SingaporeWeather:
    """Generates the disturbance signal at an arbitrary time-of-day (hours)."""

    def __init__(self, day_type: str = "typical_cool",
                 price_profile: str | None = None):
        self.day_type = day_type
        from .config import resolve_price_profile
        self.price_profile = resolve_price_profile(price_profile)
        # peak-cool day is hotter/more humid than a typical day (BOPTEST scenarios)
        self.db_mean, self.db_amp = (31.0, 3.5) if day_type == "peak_cool" else (30.0, 3.0)
        self.rh_mean = 0.82 if day_type == "peak_cool" else 0.80
        self.solar_ceiling = 800.0

    def at(self, hour: float) -> Disturbance:
        h = hour % 24.0
        # diurnal dry-bulb: min ~06:00, max ~15:00
        t_oa = self.db_mean + self.db_amp * math.sin(2 * math.pi * (h - 9.0) / 24.0)
        # RH anti-correlated with temperature (higher at night)
        rh = self.rh_mean - 0.10 * math.sin(2 * math.pi * (h - 9.0) / 24.0)
        rh = min(max(rh, 0.55), 0.98)
        w_oa = psy.W_from_TRH(t_oa, rh)
        t_wb = psy.wet_bulb(t_oa, w_oa)
        # solar bell curve, daylight ~07:00-19:00, peak = self.solar_ceiling at facade
        if 7.0 <= h <= 19.0:
            q_solar = self.solar_ceiling * max(0.0, math.sin(math.pi * (h - 7.0) / 12.0))
        else:
            q_solar = 0.0
        # occupancy: ramp 08:00, plateau, ramp down 19:00
        if h < 7.0 or h > 20.0:
            occ = 0.05
        elif h < 9.0:
            occ = 0.05 + 0.95 * (h - 7.0) / 2.0
        elif h <= 18.0:
            occ = 1.0
        else:
            occ = max(0.05, 1.0 - (h - 18.0) / 2.0)
        if self.price_profile == "constant":
            price = 1.0
        elif self.price_profile == "highly_dynamic":
            price = 2.0 if 13.0 <= h <= 17.0 else (0.6 if h < 7.0 else 1.0)
        else:
            price = 1.6 if 13.0 <= h <= 17.0 else (0.7 if h < 7.0 else 1.0)
        return Disturbance(t_oa=t_oa, w_oa=w_oa, t_wb=t_wb,
                           q_solar_wm2=q_solar, occ_frac=occ, price=price)

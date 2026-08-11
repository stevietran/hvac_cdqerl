"""Psychrometrics — the moist-air relations.

SI units throughout:
    T       [degC]        dry-bulb / dew-point temperature
    W       [kg/kg]       humidity ratio (mass water / mass dry air)
    p       [Pa]          total (barometric) pressure, default 101_325
    p_w     [Pa]          water-vapour partial pressure
    RH, phi [-]           relative humidity in [0, 1]
    h       [kJ/kg d.a.]  moist-air specific enthalpy per kg dry air
"""
from __future__ import annotations

import math

P_ATM = 101_325.0            # Pa, standard sea-level pressure (Singapore ~ sea level)
CP_AIR = 1.006              # kJ/kg.K, dry-air specific heat
CP_VAP = 1.86               # kJ/kg.K, water-vapour specific heat
H_FG = 2501.0              # kJ/kg, latent heat of vaporisation at 0 degC
RHO_AIR = 1.164            # kg/m3, moist-air density ~ 30 degC (used for volumetric flows)
MW_RATIO = 0.621945        # ratio of molecular weights water/dry-air


def p_ws(T: float) -> float:
    """Saturation vapour pressure over liquid water [Pa] (Magnus/Arden-Buck).

    Accurate to <0.1 % over 0-60 degC, the range that matters for HVAC air.
    """
    return 610.94 * math.exp(17.625 * T / (T + 243.04))


def W_from_pw(p_w: float, p: float = P_ATM) -> float:
    """Humidity ratio from vapour partial pressure."""
    p_w = min(p_w, 0.999 * p)
    return MW_RATIO * p_w / (p - p_w)


def pw_from_W(W: float, p: float = P_ATM) -> float:
    """Vapour partial pressure from humidity ratio."""
    return W * p / (MW_RATIO + W)


def W_from_TRH(T: float, RH: float, p: float = P_ATM) -> float:
    """Humidity ratio from dry-bulb temperature and relative humidity (0-1)."""
    p_w = max(0.0, min(RH, 1.0)) * p_ws(T)
    return W_from_pw(p_w, p)


def RH_from_TW(T: float, W: float, p: float = P_ATM) -> float:
    """Relative humidity (0-1) from dry-bulb and humidity ratio."""
    return pw_from_W(W, p) / p_ws(T)


def dew_point(W: float, p: float = P_ATM) -> float:
    """Dew-point temperature [degC] from humidity ratio (invert Magnus)."""
    p_w = max(pw_from_W(W, p), 1.0)
    a = math.log(p_w / 610.94)
    return 243.04 * a / (17.625 - a)


def enthalpy(T: float, W: float) -> float:
    """Moist-air specific enthalpy [kJ/kg dry air]."""
    return CP_AIR * T + W * (H_FG + CP_VAP * T)


def wet_bulb(T: float, W: float, p: float = P_ATM, tol: float = 1e-4) -> float:
    """Thermodynamic wet-bulb temperature [degC] by bisection on the
    psychrometric energy balance h(T,W) = h(Twb, Wsat(Twb)) - (W - Wsat) * cp_w * Twb.
    """
    h = enthalpy(T, W)
    lo, hi = -20.0, T
    for _ in range(60):
        twb = 0.5 * (lo + hi)
        w_s = W_from_pw(p_ws(twb), p)
        # enthalpy of saturated air minus liquid-water term carried in at Twb
        h_star = enthalpy(twb, w_s) - (w_s - W) * 4.186 * twb
        if h_star > h:
            hi = twb
        else:
            lo = twb
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def sat_humidity_ratio(T: float, p: float = P_ATM) -> float:
    """Humidity ratio on the saturation curve at temperature T (used for ADP)."""
    return W_from_pw(p_ws(T), p)

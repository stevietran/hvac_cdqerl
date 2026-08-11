"""Deterministic safety shield.

    u_t = Pi_{C_t}(u_raw) = argmin_{u in C_t} 0.5 * ||u - u_raw||^2

Every raw control is projected onto the feasible set before it touches the
environment, identically in training and deployment
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ActionBounds:
    """Actuator limits.
    """
    chwst_min: float
    chwst_max: float
    cw_fan_min: float = 0.2
    cw_fan_max: float = 1.0
    sat_min: float = 11.0
    sat_max: float = 16.0
    chwst_ramp_per_hour: float = 6.0      # K/h  (== 0.5 K per 5-min step)
    cw_fan_ramp_per_hour: float = 3.6     # 1/h  (== 0.3 per 5-min step)
    step_h: float = 5.0 / 60.0            # physics step, set from the config

    @property
    def chwst_ramp(self) -> float:
        return self.chwst_ramp_per_hour * self.step_h

    @property
    def cw_fan_ramp(self) -> float:
        return self.cw_fan_ramp_per_hour * self.step_h


class SafetyShield:
    """Box + ramp projection, and optionally the humidity actuator clamp.
    """

    def __init__(self, bounds: ActionBounds, rh_ceiling: float = 0.65,
                 enforce_humidity: bool = True):
        self.b = bounds
        self.rh_ceiling = rh_ceiling
        self.enforce_humidity = enforce_humidity

    @staticmethod
    def _clip(x, lo, hi):
        return lo if x < lo else (hi if x > hi else x)

    def project(self, action: dict, prev: dict | None, rh_current: float) -> dict:
        """Project a raw action dict onto the feasible set.
        """
        b = self.b
        span_c = max(b.chwst_max - b.chwst_min, 1e-9)
        span_f = max(b.cw_fan_max - b.cw_fan_min, 1e-9)

        def _disp(c1, f1, c0, f0):
            return abs(c1 - c0) / span_c + abs(f1 - f0) / span_f

        raw_c, raw_f = action["chwst"], action["cw_fan"]

        # ---- stage 1: box (actuator range) ----
        chwst = self._clip(raw_c, b.chwst_min, b.chwst_max)
        cw_fan = self._clip(raw_f, b.cw_fan_min, b.cw_fan_max)
        corr_bounds = _disp(chwst, cw_fan, raw_c, raw_f)
        box_c, box_f = chwst, cw_fan

        # ---- stage 2: ramp (slew rate) — enforced, not charged ----
        if prev is not None:
            chwst = self._clip(chwst, prev["chwst"] - b.chwst_ramp,
                               prev["chwst"] + b.chwst_ramp)
            cw_fan = self._clip(cw_fan, prev["cw_fan"] - b.cw_fan_ramp,
                                prev["cw_fan"] + b.cw_fan_ramp)
        corr_ramp = _disp(chwst, cw_fan, box_c, box_f)
        ramp_c, ramp_f = chwst, cw_fan

        # ---- stage 3: humidity safety clamp — the charged one ----
        if self.enforce_humidity and rh_current > self.rh_ceiling:
            over = min((rh_current - self.rh_ceiling) / 0.10, 1.0)
            chwst = self._clip(chwst, b.chwst_min,
                               b.chwst_max - over * (b.chwst_max - b.chwst_min))
            cw_fan = max(cw_fan, 0.6 + 0.4 * over)
        corr_safety = _disp(chwst, cw_fan, ramp_c, ramp_f)

        return {"chwst": chwst, "cw_fan": cw_fan,
                "correction": corr_safety,          # what the reward charges
                "corr_safety": corr_safety,
                "corr_bounds": corr_bounds,         # diagnostics only
                "corr_ramp": corr_ramp,
                "corr_total": corr_safety + corr_bounds + corr_ramp}

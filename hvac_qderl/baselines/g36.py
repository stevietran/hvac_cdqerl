"""ASHRAE Guideline 36 trim-and-respond baseline
"""
from __future__ import annotations

from ..config import SingaporeConfig


class G36Controller:
    """G36 trim-and-respond.
    """

    def __init__(self, cfg: SingaporeConfig,
                 trim_per_hour: float = 1.2, respond_per_hour: float = 3.0,
                 max_requests: int = 4, fan_gain_per_hour: float = 3.6):
        self.cfg = cfg
        step_h = getattr(cfg, "control_step_h", cfg.zone_step_min / 60.0)
        self.step_h = step_h
        self.trim = trim_per_hour * step_h            # K per call
        self.respond = respond_per_hour * step_h      # K per call per request
        # first-order approach to the tower-fan target, also rate-based
        self.fan_gain = min(1.0, fan_gain_per_hour * step_h)
        self.max_requests = max_requests
        self.chwst = cfg.chwst_default
        self.cw_fan = 0.7

    def reset(self):
        self.chwst = self.cfg.chwst_default
        self.cw_fan = 0.7

    def act(self, obs, info) -> dict:
        cfg = self.cfg
        # ---- CHWST trim-and-respond -------------------------------------- #
        # trim up (energy saving); respond down on RH/temperature requests.
        requests = 0
        if info is not None:
            if info["max_RH"] > cfg.RH_ceiling:
                requests += min(self.max_requests,
                                int((info["max_RH"] - cfg.RH_ceiling) / 0.02) + 1)
            if info["max_Tz"] > cfg.T_set + cfg.T_band:
                requests += min(self.max_requests,
                                int((info["max_Tz"] - cfg.T_set - cfg.T_band) / 0.3) + 1)
        if requests == 0:
            self.chwst += self.trim                       # trim up, save energy
        else:
            self.chwst -= self.respond * min(requests, self.max_requests)
        self.chwst = min(max(self.chwst, cfg.chwst_min), cfg.chwst_max)

        # ---- cooling-tower / condenser-water reset ----------------------- #
        # Track a small approach to wet-bulb: push fan when heat-rejection load
        # is high, ease off at low load (energy vs. condenser-temp trade).
        if info is not None:
            load_frac = info["q_evap_kw"] / max(cfg.design_cooling_kw, 1.0)
            target_fan = 0.45 + 0.55 * min(load_frac, 1.0)
            # nudge toward target (trim-and-respond style, avoid hunting)
            self.cw_fan += self.fan_gain * (target_fan - self.cw_fan)
        self.cw_fan = min(max(self.cw_fan, 0.2), 1.0)

        return {"chwst": self.chwst, "cw_fan": self.cw_fan}

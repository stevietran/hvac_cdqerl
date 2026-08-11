"""Equipment performance models — the algebraic closures g(x,u,w,theta).

Each component is an explicit, differentiable evaluation
Units: power [kW], heat [kW], temperature [degC], flow [kg/s].
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from . import psychrometrics as psy

KW_PER_RT = 3.51685          # 1 refrigeration ton = 3.51685 kW cooling
CP_WATER = 4.186             # kJ/kg.K


# --------------------------------------------------------------------------- #
# Chiller  (DOE-2 / EnergyPlus performance-curve model)
# --------------------------------------------------------------------------- #
@dataclass
class Chiller:
    """Water-cooled centrifugal chiller, bi-quadratic CAPFT/EIRFT + EIRFPLR.

    P_ch = (Q_ref / COP_ref) * CAPFT(Tchws,Tcw) * EIRFT(Tchws,Tcw) * EIRFPLR(PLR)
    Q_avail = Q_ref * CAPFT ;  PLR = Q_evap / Q_avail
    Default curve coefficients are representative centrifugal-chiller values.
    """
    q_ref_kw: float                       # reference (design) cooling capacity [kW]
    cop_ref: float = 5.9                  # design COP (~0.60 kW/RT at chiller)
    t_chws_ref: float = 7.0               # deg C, reference leaving CHW temp
    t_cw_ref: float = 30.0                # deg C, reference entering condenser water
    plr_min: float = 0.15                 # minimum stable part-load ratio
    # bi-quadratic coefficients f = a + b*x + c*x^2 + d*y + e*y^2 + f*x*y
    capft: tuple = (0.257896, 0.0389016, -0.00021708,
                    0.0469314, -0.00094464, -0.00052781)
    eirft: tuple = (0.933884, -0.058212, 0.00450036,
                    0.00243, 0.000486, -0.001215)
    eirfplr: tuple = (0.222903, 0.313387, 0.463581)   # a + b*PLR + c*PLR^2

    @staticmethod
    def _biquad(c, x, y):
        return c[0] + c[1] * x + c[2] * x * x + c[3] * y + c[4] * y * y + c[5] * x * y

    def available_capacity(self, t_chws: float, t_cw: float) -> float:
        capft = max(0.3, self._biquad(self.capft, t_chws, t_cw))
        return self.q_ref_kw * capft

    def power(self, q_evap: float, t_chws: float, t_cw: float) -> float:
        """Electrical power [kW] to deliver q_evap of cooling."""
        if q_evap <= 1e-6:
            return 0.0
        capft = max(0.3, self._biquad(self.capft, t_chws, t_cw))
        q_avail = self.q_ref_kw * capft
        plr = min(max(q_evap / q_avail, self.plr_min), 1.0)
        eirft = max(0.3, self._biquad(self.eirft, t_chws, t_cw))
        eirfplr = max(0.1, self.eirfplr[0] + self.eirfplr[1] * plr
                      + self.eirfplr[2] * plr * plr)
        p_ref = self.q_ref_kw / self.cop_ref
        # part-load power scales with actual (not available) load through PLR term
        return p_ref * capft * eirft * eirfplr


# --------------------------------------------------------------------------- #
# Cooling tower  (Merkel / effectiveness, the tropical bottleneck)
# --------------------------------------------------------------------------- #
@dataclass
class CoolingTower:
    """Evaporative tower. Leaving condenser-water temp = wet-bulb + approach,
    where approach shrinks with fan speed (more air => closer approach) but has
    a hard floor set by the tight tropical wet-bulb margin (§12.7)."""
    approach_min: float = 3.0             # K, tightest achievable approach at full fan
    approach_max: float = 8.0             # K, approach at minimum fan speed
    fan_kw_design: float = 60.0           # kW, design tower-fan power (per cell group)

    def leaving_cw_temp(self, t_wb: float, fan_speed: float) -> float:
        """t_cws (supply to condenser) given wet-bulb and fan speed in [0,1]."""
        fan_speed = min(max(fan_speed, 0.0), 1.0)
        approach = self.approach_max - (self.approach_max - self.approach_min) * fan_speed
        return t_wb + approach

    def fan_power(self, fan_speed: float) -> float:
        return self.fan_kw_design * min(max(fan_speed, 0.0), 1.0) ** 3


# --------------------------------------------------------------------------- #
# Cooling coil  (bypass-factor / apparatus-dew-point wet coil)
# --------------------------------------------------------------------------- #
@dataclass
class CoolingCoil:
    """ADP / bypass-factor coil giving coupled sensible + latent duty.

    T_adp sits on the saturation curve just above the leaving CHW temp; the
    bypass factor BF = exp(-NTU) blends entering air toward the ADP state.
    """
    ua_design: float = 0.0                # kW/K, design conductance (sets NTU)
    m_air_design: float = 1.0             # kg/s design air flow (for NTU scaling)
    adp_approach: float = 0.6             # K, ADP above leaving CHW temp

    def outlet(self, t_in, w_in, m_air, t_chws):
        """Return (T_out, W_out, Q_tot, Q_sen, Q_lat) [degC,kg/kg,kW,kW,kW]."""
        m_air = max(m_air, 1e-3)
        ntu = self.ua_design / (m_air * psy.CP_AIR)
        bf = min(max((2.71828 ** (-ntu)), 0.02), 0.98)
        t_adp = t_chws + self.adp_approach
        w_adp = psy.sat_humidity_ratio(t_adp)
        t_out = t_adp + bf * (t_in - t_adp)
        w_out = w_adp + bf * (w_in - w_adp)
        w_out = min(w_out, w_in)          # coil cannot humidify
        h_in = psy.enthalpy(t_in, w_in)
        h_out = psy.enthalpy(t_out, w_out)
        q_tot = m_air * (h_in - h_out)                       # kW
        q_sen = m_air * psy.CP_AIR * (t_in - t_out)          # kW
        q_lat = m_air * psy.H_FG * (w_in - w_out)            # kW
        return t_out, w_out, max(q_tot, 0.0), max(q_sen, 0.0), max(q_lat, 0.0)


# --------------------------------------------------------------------------- #
# Fans & pumps  (affinity / cube law)
# --------------------------------------------------------------------------- #
def affinity_power(design_kw: float, flow_ratio: float) -> float:
    """Fan or pump power under the cube law, clamped to [0,1] flow ratio."""
    return design_kw * min(max(flow_ratio, 0.0), 1.2) ** 3


# --------------------------------------------------------------------------- #
# Chiller-plant staging 
# --------------------------------------------------------------------------- #
@dataclass
class ChillerPlant:
    """A bank of identical chillers with a simple stage-up/down state machine."""
    chillers: List[Chiller]
    stage_up_frac: float = 0.90
    stage_down_frac: float = 0.80
    _n_on: int = field(default=1, init=False)

    def n_on(self) -> int:
        return self._n_on

    def update_staging(self, q_load, t_chws, t_cw, enable: bool = True):
        """Advance the staging state machine given current total load.

        `enable=False` shuts the plant down completely (n_on = 0). Without this
        the minimum was one chiller running forever: measured on the reference
        case, the plant never dropped below **two** machines overnight and burned
        ~40 % of annual energy holding an empty building at 22 degC.
        """
        if not enable:
            self._n_on = 0
            return 0
        n = max(self._n_on, 1)                       # restart from at least one
        cap_each = self.chillers[0].available_capacity(t_chws, t_cw)
        # STAGE UP IN ONE MOVE, NOT ONE MACHINE AT A TIME.
        # Stage-DOWN is deliberately left at one machine per call. 
        n_need = int(np.ceil(q_load / max(self.stage_up_frac * cap_each, 1e-9))) \
            if q_load > 0 else 1
        if n < len(self.chillers) and n_need > n:
            n = min(n_need, len(self.chillers))
        elif n > 1 and q_load < self.stage_down_frac * (n - 1) * cap_each:
            # Stage down when the load *fits* in n-1 machines with margin.
            n -= 1
        self._n_on = max(0, min(n, len(self.chillers)))
        return self._n_on


    def power(self, q_load, t_chws, t_cw):
        """Total chiller electrical power; load split evenly across running units."""
        n = self._n_on
        if n <= 0 or q_load <= 0:
            return 0.0
        q_each = q_load / n
        return sum(self.chillers[i].power(q_each, t_chws, t_cw) for i in range(n))

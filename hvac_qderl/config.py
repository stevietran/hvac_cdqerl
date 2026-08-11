"""Singapore reference case-study configuration
"""
from __future__ import annotations

import os

from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# ELECTRICITY TARIFF
#
# `constant` is the default: cost then tracks energy, so an energy result and a
# cost result cannot disagree. The dynamic profiles are RETAINED for the thermal
# storage work, where a time-varying tariff is the whole point -- this plant has
# no storage yet, so load shifting is limited to pre-cooling building mass.
#
#   constant        1.0 flat
#   dynamic         0.7 / 1.0 / 1.6   peak 13:00-17:00
#   highly_dynamic  0.6 / 1.0 / 2.0   peak 13:00-17:00
# --------------------------------------------------------------------------- #

PRICE_PROFILES = ("constant", "dynamic", "highly_dynamic")
DEFAULT_PRICE_PROFILE = os.environ.get("HVAC_PRICE_PROFILE", "constant")


def resolve_price_profile(price_profile: str | None = None) -> str:
    """None -> the project default. Raises on an unknown name rather than
    silently falling through to flat pricing."""
    p = price_profile or DEFAULT_PRICE_PROFILE
    if p not in PRICE_PROFILES:
        raise ValueError(f"unknown price_profile {p!r}; expected one of "
                         f"{PRICE_PROFILES}")
    return p

from typing import List

from .equipment import Chiller, ChillerPlant, CoolingTower, CoolingCoil, KW_PER_RT

# --------------------------------------------------------------------------- #
# PLANT SIZING — ONE PLACE, HARD-CODED
# --------------------------------------------------------------------------- #
DESIGN_LOAD_KW = 4797.0 # from the always-on G36 baseline
N_CHILLERS = 4
CHILLER_CAPACITY_RT = 360.0          # per unit; 4 x 360 = 1,440 RT installed


@dataclass
class ZoneParams:
    """3R2C sensible + 1 moisture node per zone (§16.2 reduced-model row)."""
    area_m2: float
    # 3R2C thermal (grey-box; site-identified in practice, §12.2)
    C_z: float          # kJ/K  air-node capacitance
    C_m: float          # kJ/K  mass-node capacitance
    R_oa: float         # K/kW  outdoor<->air resistance
    R_zm: float         # K/kW  air<->mass resistance
    R_om: float         # K/kW  outdoor<->mass resistance
    # internal gains (lighting 9-12, plug 10-15 W/m2, ~0.1 person/m2)
    lighting_wm2: float = 10.0
    plug_wm2: float = 12.0
    occ_density: float = 0.10           # persons / m2 (peak)
    # per-person gains
    sens_per_person_w: float = 75.0
    # ASHRAE Fundamentals ch.18: seated office work ~ 55 W latent/person,
    # i.e. 55 W / 2501 kJ/kg = 0.022 g/s = ~79 g/h. (200 g/h would be moderate
    # physical work and pushes the load SHR unrealistically low for an office.)
    lat_per_person_gph: float = 80.0
    infiltration_kgs_per_m2: float = 1.0e-4   # ~0.1 ACH for a sealed office
    v_zone_m3: float = 0.0              # zone air volume (set from area x height)
    solar_aperture_m2: float = 0.0      # effective window*SHGC area (0 for core)


@dataclass
class SingaporeConfig:
    """Full reference-building configuration."""
    name: str = "SG_reference_water_cooled_office"

    # ---- building geometry ---- #
    n_floors: int = 3                   # 3 representative floors (floor multiplier)
    floor_height_m: float = 3.0
    zones: List[ZoneParams] = field(default_factory=list)

    # ---- climate (SG-Changi, ~33 degC DB / 26 degC WB) ------------ #
    design_db: float = 33.0
    design_wb: float = 26.0
    oa_rh_range: tuple = (0.70, 0.90)

    # ---- comfort / humidity limits -------------------------------- #
    T_set: float = 24.0                 # zone setpoint, mid of 23-25 degC band
    T_band: float = 1.0                 # +/- deadband
    RH_ceiling: float = 0.65            # comfort/IAQ limit while occupied (SS 553)
    pmv_band: float = 0.5

    # ---- occupancy modes -------------------------------------------------- #
    # the schedule produces three natural bands 
    # 0.05 baseline (4,905 h/yr, 56 % of hours but only 30 % of annual load)
    # 0.15 weekend/ramp crew (735 h), and 
    # 0.7-1.0 full occupancy (2,600 h, 55 % of load).
    occ_unoccupied_max: float = 0.06    # <= this: nobody there
    occ_setback_max: float = 0.30       # <= this: skeleton crew / ramp

    # Unoccupied setback
    T_set_unocc: float = 29.0
    T_band_unocc: float = 2.0

    RH_ceiling_unocc: float = 0.70      # unoccupied humidity target
    RH_mould_limit: float = 0.75        # HARD cap at all hours (never relax)
    T_max_unocc: float = 32.0           # empty-building temperature limit
    dewpoint_max_unocc: float = 17.0    # degC — absolute-moisture night guard
    dewpoint_margin_K: float = 1.5      # surface-condensation guard
    humidity_shield: bool = True        # RH clamp + mould / dew-point overrides
    comfort_shield: bool = True         # occupied over-band plant-on latch
    overtemp_guard: bool = True         # T_max_unocc plant-on override
    comfort_shield_deadband: float = 0.3   # K of hysteresis before releasing

    # ---- plant scheduling (see docs/plant_scheduling.md) ------------------ #
    occupied_start_h: float = 8.0       # scheduled occupancy start
    occupied_end_h: float = 19.0        # scheduled occupancy end
    optimum_start_max_lead_h: float = 3.0   # cap on pre-start lead time
    allow_plant_off: bool = True        # master switch for shutdown logic
    vav_min_frac_unocc: float = 0.1
    plant_min_on_min: float = 90.0
    plant_min_off_min: float = 60.0

    # ------------------------------------------------------------------ #
    def occupancy_mode(self, occ_frac: float) -> str:
        """occupied | setback | unoccupied — from the load-profile thresholds."""
        if occ_frac <= self.occ_unoccupied_max:
            return "unoccupied"
        if occ_frac <= self.occ_setback_max:
            return "setback"
        return "occupied"

    def setpoints_for(self, occ_frac: float):
        """(T_set, T_band, RH_ceiling) active at this occupancy."""
        mode = self.occupancy_mode(occ_frac)
        if mode == "occupied":
            return self.T_set, self.T_band, self.RH_ceiling
        if mode == "setback":
            # interpolate so the transition is not a step change the plant has
            # to chase; at occ_setback_max it equals the occupied setpoint
            f = (occ_frac - self.occ_unoccupied_max) / \
                max(self.occ_setback_max - self.occ_unoccupied_max, 1e-9)
            f = min(max(f, 0.0), 1.0)
            return (self.T_set_unocc + f * (self.T_set - self.T_set_unocc),
                    self.T_band_unocc + f * (self.T_band - self.T_band_unocc),
                    self.RH_ceiling_unocc + f * (self.RH_ceiling - self.RH_ceiling_unocc))
        return self.T_set_unocc, self.T_band_unocc, self.RH_ceiling_unocc

    # ---- control / RL cadence ------------------------------------- #
    control_step_min: float = 20.0      # agent / supervisory decision cadence
    physics_step_min: float = 5.0       # DAE integration sub-step
    plant_step_min: float = 15.0        # plant cadence
    horizon_hours: float = 24.0

    @property
    def zone_step_min(self) -> float:
        """Backward-compatible alias: the simulation (physics) step."""
        return self.physics_step_min

    @property
    def n_substeps(self) -> int:
        return max(1, int(round(self.control_step_min / self.physics_step_min)))

    @property
    def control_step_h(self) -> float:
        return self.control_step_min / 60.0

    # ---- CHW / setpoints ------------------------------------------ #
    chwst_min: float = 6.7
    chwst_max: float = 9.0
    chwst_default: float = 7.0
    chw_deltaT: float = 6.0             # K, design CHW loop delta-T (5-7 K)

    # ---- condenser / tower (§16.2: CW supply ~29-30, approach 3-4 K) ------ #
    cw_supply_design: float = 29.5
    tower_approach_min: float = 3.0
    tower_approach_max: float = 8.0

    # ---- reward weights ------------------------------------------- #
    # SIMPLIFIED TO THREE TERMS. The reward is now exactly
    #   -(w_energy*e_term + w_comfort*c_term + w_shield*s_term) * dt_h
    w_energy: float = 1.0               # reference term
    w_comfort: float = 1.0              # ~11x the measured parity threshold (0.087)
    w_shield: float = 2               # price of being bailed out by any guard/timer

    # ---- tariff (demand charge + EMA demand-flex) ------------------ #
    energy_tariff: float = 0.28         # SGD/kWh (indicative)
    demand_charge: float = 1.0         # SGD/kW-month (indicative)

    # ---- air-side sizing inputs ------------------------------------------- #
    design_shr: float = 0.65
    supply_deltaT_K: float = 11.0       # 24 -> 13 degC supply air

    # ---- design capacities (populated in __post_init__) ------------------- #
    design_cooling_kw: float = 0.0
    plant: ChillerPlant = None
    coil: CoolingCoil = None
    tower: CoolingTower = None
    ahu_fan_kw_design: float = 0.0
    chw_pump_kw_design: float = 0.0
    cw_pump_kw_design: float = 0.0
    m_air_design: float = 0.0
    sizing_basis: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.zones:
            self.zones = _default_zones(self.n_floors, self.floor_height_m)

        # design basis and plant #
        self.design_cooling_kw = DESIGN_LOAD_KW
        q_each = CHILLER_CAPACITY_RT * KW_PER_RT              
        chillers = [Chiller(q_ref_kw=q_each, cop_ref=5.9,
                            t_chws_ref=self.chwst_default,
                            t_cw_ref=self.cw_supply_design)
                    for _ in range(N_CHILLERS)]
        self.plant = ChillerPlant(chillers=chillers)

        # cooling tower
        self.tower = CoolingTower(approach_min=self.tower_approach_min,
                                  approach_max=self.tower_approach_max,
                                  fan_kw_design=0.010 * self.design_cooling_kw)

        # design air flow: sized on sensible load / (cp*(T_z - T_sa))
        # `design_shr` (0.65, latent-heavy); it is replaced
        # by the load's own coincident SHR whenever the plant is sized from a
        # computed annual load. deltaT_air ~ 24 - 13 = 11 K.
        q_sen_design = self.design_shr * self.design_cooling_kw
        self.m_air_design = q_sen_design / (1.006 * self.supply_deltaT_K)   # kg/s

        # cooling coil: UA sized to give ~ 13 degC SAT off ~7 degC CHW
        self.coil = CoolingCoil(ua_design=1.6 * self.m_air_design * 1.006,
                                m_air_design=self.m_air_design,
                                adp_approach=0.6)

        # fan / pump design powers (fraction-of-plant heuristics)
        self.ahu_fan_kw_design = 0.045 * self.design_cooling_kw
        self.chw_pump_kw_design = 0.030 * self.design_cooling_kw
        self.cw_pump_kw_design = 0.028 * self.design_cooling_kw

        # zone air volumes
        for z in self.zones:
            z.v_zone_m3 = z.area_m2 * self.floor_height_m

    # convenience
    @property
    def total_area(self) -> float:
        return sum(z.area_m2 for z in self.zones)

    @property
    def design_rt(self) -> float:
        """Building DESIGN LOAD in RT. Not the plant -- see `installed_rt`."""
        return self.design_cooling_kw / KW_PER_RT

    @property
    def installed_kw(self) -> float:
        """Total INSTALLED chiller capacity [kW] at reference conditions."""
        return sum(c.q_ref_kw for c in self.plant.chillers)

    @property
    def installed_rt(self) -> float:
        return self.installed_kw / KW_PER_RT

    @property
    def p_ref_kw(self) -> float:
        """Design TOTAL plant electrical draw [kW] — the energy non-dimensionaliser.
        Chiller at reference COP plus every design auxiliary, so `P/p_ref_kw` is
        ~1.0 with the whole plant at design and the energy reward term lands in
        the same "fraction x hours" units as comfort, humidity and the rest.
        """
        cop_ref = getattr(self.plant.chillers[0], "cop_ref", 5.9) if getattr(
            self, "plant", None) and self.plant.chillers else 5.9
        return (self.design_cooling_kw / max(cop_ref, 1e-6)
                + self.chw_pump_kw_design + self.cw_pump_kw_design
                + self.ahu_fan_kw_design + self.tower.fan_kw_design)


def _default_zones(n_floors: int, h: float) -> List[ZoneParams]:
    """5 zones/floor (4 perimeter + 1 core)
    Perimeter zones see envelope + solar (larger R to outdoor, real solar gain);
    the core is internally-dominated (near-adiabatic envelope).  Areas scaled so
    one floor ~ 14,250 m2 (42,757 m2 / 3), matching the reference building.
    """
    zones: List[ZoneParams] = []
    floor_area = 42_757.0 / 3.0
    perim_area = floor_area * 0.15      # each of 4 perimeter zones
    core_area = floor_area - 4 * perim_area

    # RC defined via physically-sane per-floor-area conductances [kW/K per m2]
    # and capacitances [kJ/K per m2]; R = 1/(h*area).  These give an air-to-mass
    # coupling that lets cold supply air actually cool the zone while the slab
    # provides realistic thermal inertia (grey-box).
    Cz_pm, Cm_pm = 20.0, 150.0          # kJ/K.m2
    h_zm = 0.006                        # kW/K.m2  air<->mass (internal surfaces)
    h_oa_perim, h_om_perim = 0.0006, 0.00020   # kW/K.m2 envelope (perimeter)
    h_oa_core, h_om_core = 0.00004, 0.00002    # near-adiabatic core

    for _ in range(n_floors):
        for _p in range(4):             # perimeter
            zones.append(ZoneParams(
                area_m2=perim_area,
                C_z=Cz_pm * perim_area, C_m=Cm_pm * perim_area,
                R_oa=1.0 / (h_oa_perim * perim_area),
                R_zm=1.0 / (h_zm * perim_area),
                R_om=1.0 / (h_om_perim * perim_area),
                lighting_wm2=10.0, plug_wm2=12.0, occ_density=0.10,
                solar_aperture_m2=0.05 * perim_area))
        zones.append(ZoneParams(       # core (internally dominated, no solar)
            area_m2=core_area,
            C_z=Cz_pm * core_area, C_m=Cm_pm * core_area,
            R_oa=1.0 / (h_oa_core * core_area),
            R_zm=1.0 / (h_zm * core_area),
            R_om=1.0 / (h_om_core * core_area),
            lighting_wm2=10.0, plug_wm2=13.0, occ_density=0.11,
            solar_aperture_m2=0.0))
    return zones


def default_singapore_config() -> SingaporeConfig:
    """The reference Singapore case study of §16.2."""
    return SingaporeConfig()

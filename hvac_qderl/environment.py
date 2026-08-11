"""Tier-1 reduced-order HVAC environment

A Gym-style POMDP wrapping the reduced-order DAE. Integrated with a
semi-implicit (backward-Euler) step at the zone control cadence.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import psychrometrics as psy
from .config import SingaporeConfig
from .equipment import affinity_power, CP_WATER, KW_PER_RT
from .shield import ActionBounds, SafetyShield
from .weather import SingaporeWeather


PLANT_TIMER_EPS_H = 1e-9
CHWS_FLOAT_ITERS = 40
FORECAST_HOURS = (1.0, 2.0, 3.0)


class HVACPlantEnv:
    """Reduced-order Singapore plant+zone environment."""

    def __init__(self, config: SingaporeConfig | None = None,
                 weather: SingaporeWeather | None = None,
                 use_shield: bool = True):
        self.cfg = config or SingaporeConfig()
        self.weather = weather or SingaporeWeather()
        self.dt = self.cfg.physics_step_min * 60.0         # s   — physics
        self.physics_dt_h = self.cfg.physics_step_min / 60.0
        self.n_sub = self.cfg.n_substeps
        self.dt_h = self.physics_dt_h * self.n_sub
        self.n_zones = len(self.cfg.zones)
        self.horizon_hours = float(getattr(self.weather, "horizon_hours",
                                           self.cfg.horizon_hours))
        self.day_weights = np.asarray(getattr(self.weather, "day_weights",
                                              None) if hasattr(
            self.weather, "day_weights") else [1.0] * max(1, int(round(
                self.horizon_hours / 24.0))), dtype=float)
        self.solar_ceiling = float(getattr(self.weather, "solar_ceiling", 800.0))
        self.peak_power = max(self.cfg.p_ref_kw, 1e-6)
        self.use_shield = use_shield
        self.shield = SafetyShield(
            ActionBounds(chwst_min=self.cfg.chwst_min, chwst_max=self.cfg.chwst_max,
                         step_h=self.cfg.physics_step_min / 60.0),
            rh_ceiling=self.cfg.RH_ceiling,
            enforce_humidity=getattr(self.cfg, "humidity_shield", True))

        self.solar_aperture = np.array([z.solar_aperture_m2 for z in self.cfg.zones])
        self._peak_persons = sum(z.occ_density * z.area_m2 for z in self.cfg.zones)
        self._total_area = self.cfg.total_area

        design_frac = np.array([z.area_m2 for z in self.cfg.zones])
        design_frac = design_frac / design_frac.sum()
        self.m_sa_max = self.cfg.m_air_design * design_frac * 2.4
        self.m_sa_min = self.m_sa_max * 0.08

        self.reset()

    def reset(self, start_hour: float = 0.0, warmup_steps: int = 12,
             T_z0=None, T_m0=None, W_z0=None):
        """Reset the episode. Two modes:
        Default (`T_z0=None`) -- the original isothermal `T_z = T_m = T_set`

        Warm-started (any of `T_z0`/`T_m0`/`W_z0` given: the state is overwritten
        with an ALREADY-REALISTIC draw from the day-of-week warm-start bank 
        (`warmstart.WarmstartBank.sample`)
        """
        self.hour = start_hour
        self.step_idx = 0
        warm_started = T_z0 is not None or T_m0 is not None or W_z0 is not None
        self.T_z = np.full(self.n_zones,
                           self.cfg.T_set if T_z0 is None else T_z0, dtype=float)
        self.T_m = np.full(self.n_zones,
                           self.cfg.T_set if T_m0 is None else T_m0, dtype=float)
        w0 = psy.W_from_TRH(self.cfg.T_set, 0.58) if W_z0 is None else W_z0
        self.W_z = np.full(self.n_zones, w0, dtype=float)
        self.prev_action = {"chwst": self.cfg.chwst_default, "cw_fan": 0.7}
        self.peak_kw = 0.0
        self.prev_n_on = self.cfg.plant.n_on()
        self._q_evap_prev = 0.0
        self._plant_on, self._plant_since = True, -1e9
        self._lockout_left = 0.0
        self._comfort_hold = False
        if not warm_started:
            for _ in range(warmup_steps):
                self._physics_step({"chwst": self.cfg.chwst_default, "cw_fan": 0.7})
            self.hour = start_hour
            self.step_idx = 0
        self.peak_kw = 0.0
        return self._obs(self.weather.at(self.hour), 0.0)

    def _internal_gains(self, dist):
        """Per-zone sensible [kW] and latent moisture generation [kg/s]."""
        occ = dist.occ_frac
        q_sen = np.zeros(self.n_zones)
        m_gen = np.zeros(self.n_zones)
        for i, z in enumerate(self.cfg.zones):
            light_plug = (z.lighting_wm2 + z.plug_wm2) * z.area_m2 * (0.3 + 0.7 * occ)
            n_people = z.occ_density * z.area_m2 * occ
            q_people = n_people * z.sens_per_person_w
            q_sen[i] = (light_plug + q_people) / 1000.0
            m_gen[i] = n_people * z.lat_per_person_gph / 1000.0 / 3600.0
        return q_sen, m_gen

    def step(self, action: dict):
        """One **control** step: hold `action` across `n_sub` physics sub-steps.
        The agent decides every `control_step_min`; the DAE is integrated every
        `physics_step_min`. Returned `info` aggregates the sub-steps (power and
        states averaged, extrema kept, energy/cost summed) and `reward` is the
        sum, so a control step is a faithful time-integral regardless of how the
        two cadences are set.
        """
        total_r = 0.0
        infos = []
        obs = None
        done = False
        for _ in range(self.n_sub):
            obs, r, done, info = self._physics_step(action)
            total_r += r
            infos.append(info)
            if done:
                break
        return obs, total_r, done, self._aggregate(infos, total_r)

    @staticmethod
    def _aggregate(infos: list, total_r: float) -> dict:
        """Collapse sub-step infos into one control-step record."""
        if len(infos) == 1:
            infos[0]["reward"] = total_r
            infos[0]["substeps"] = 1
            infos[0]["starts_in_step"] = 0
            infos[0]["chwst_end"] = infos[0]["chwst"]
            infos[0]["cw_fan_end"] = infos[0]["cw_fan"]
            return infos[0]
        out = dict(infos[-1])
        mean_of = ("p_total_kw", "p_chiller_kw", "p_chwpump_kw", "p_cwpump_kw",
                   "p_towerfan_kw", "p_ahufan_kw", "q_evap_kw", "q_sen_kw",
                   "q_lat_kw", "shr", "kw_per_rt", "airflow_kgs", "t_sa",
                   "mean_Tz", "mean_RH", "rh_violation", "comfort_dev",
                   "pmv_disc", "correction", "chwst", "cw_fan", "t_cw",
                   "plant_enable", "rh_mould_breach", "pmv_disc_eff",
                   "comfort_dev_eff", "forced_on", "comfort_forced",
                   "rw_energy", "rw_comfort", "rw_shield",
                   "corr_safety", "corr_bounds", "corr_ramp", "corr_total",
                   "corr_cycle",
                   "chwst_cmd", "q_demand_kw", "capacity_limited",
                   "chwst_float_K", "flow_cap")
        for k in mean_of:
            if k in out:
                out[k] = float(np.mean([i[k] for i in infos]))
        out["max_Tz"] = max(i["max_Tz"] for i in infos)
        out["max_RH"] = max(i["max_RH"] for i in infos)
        out["chwst_end"] = infos[-1]["chwst"]
        out["cw_fan_end"] = infos[-1]["cw_fan"]
        out["cost_sgd"] = float(sum(i["cost_sgd"] for i in infos))
        n_on = [i["n_chillers_on"] for i in infos]
        out["starts_in_step"] = int(sum(max(0, b - a) for a, b in zip(n_on, n_on[1:])))
        out["reward"] = total_r
        out["substeps"] = len(infos)
        return out

    def _physics_step(self, action: dict):
        cfg = self.cfg
        dist = self.weather.at(self.hour)

        rh_zone = np.array([psy.RH_from_TW(self.T_z[i], self.W_z[i]) for i in range(self.n_zones)])
        worst_rh = float(rh_zone.max())
        if self.use_shield:
            applied = self.shield.project(action, self.prev_action, worst_rh)
        else:
            applied = {"chwst": min(max(action["chwst"], cfg.chwst_min), cfg.chwst_max),
                       "cw_fan": min(max(action["cw_fan"], 0.2, ), 1.0),
                       "correction": 0.0, "corr_safety": 0.0,
                       "corr_bounds": 0.0, "corr_ramp": 0.0, "corr_total": 0.0}
        chwst = applied["chwst"]
        cw_fan = applied["cw_fan"]

        mode = cfg.occupancy_mode(dist.occ_frac)
        t_set, t_band, rh_ceiling = cfg.setpoints_for(dist.occ_frac)
        enable = bool(action.get("plant_enable", True)) if cfg.allow_plant_off else True
        held = False
        need = (cfg.plant_min_on_min if self._plant_on else cfg.plant_min_off_min) / 60.0
        elapsed = self.hour - self._plant_since
        self._lockout_left = float(np.clip((need - elapsed) / max(need, 1e-9), 0.0, 1.0))
        if enable != self._plant_on and elapsed < need - PLANT_TIMER_EPS_H:
            enable, held = self._plant_on, True
        corr_cycle = 1.0 if held else 0.0
        forced_on = False
        worst_dp = max(psy.dew_point(self.W_z[i]) for i in range(self.n_zones))
        if getattr(cfg, "humidity_shield", True) and (
                worst_rh > cfg.RH_mould_limit or self._dewpoint_risk()
                or worst_dp > cfg.dewpoint_max_unocc):
            enable, forced_on = True, True
        if getattr(cfg, "overtemp_guard", True) and (
                float(self.T_z.max()) > cfg.T_max_unocc):
            enable, forced_on = True, True

        comfort_forced = False
        if getattr(cfg, "comfort_shield", False):
            t_max_zone = float(self.T_z.max())
            if dist.occ_frac > cfg.occ_setback_max:
                if t_max_zone > t_set + t_band:
                    self._comfort_hold = True
                elif t_max_zone < t_set + t_band - cfg.comfort_shield_deadband:
                    self._comfort_hold = False
            else:
                self._comfort_hold = False
            if self._comfort_hold:
                enable, forced_on, comfort_forced = True, True, True

        err = self.T_z - t_set
        frac = np.clip(0.5 + err / (2.0 * t_band), 0.0, 1.0)
        min_frac = (1.0 if mode == "occupied" else (0.5 if mode == "setback" else cfg.vav_min_frac_unocc))
        m_sa = self.m_sa_min * min_frac + frac * (self.m_sa_max - self.m_sa_min * min_frac)
        flow_cap = float(action.get("flow_cap", 1.0))
        if comfort_forced:
            flow_cap = 1.0
        flow_cap = float(np.clip(flow_cap, 0.0, 1.0))
        m_sa = m_sa * flow_cap
        if enable:
            m_vent = self._ventilation_floor_kgs(dist, mode)
            M_now = float(m_sa.sum())
            if m_vent > M_now > 1e-9:
                m_sa = m_sa * (m_vent / M_now)
        if not enable:
            m_sa = self.m_sa_min * 0.25
        M = float(m_sa.sum())

        w_ret = float((m_sa * self.W_z).sum() / max(M, 1e-6))
        t_ret = float((m_sa * self.T_z).sum() / max(M, 1e-6))
        m_oa_req = self._oa_requirement_kgs(dist, mode)             # kg/s
        m_oa = min(m_oa_req, self.OA_MIX_CAP * M)                  # kg/s, capped
        oa_frac = m_oa / max(M, 1e-6)
        t_mix = (1 - oa_frac) * t_ret + oa_frac * dist.t_oa
        w_mix = (1 - oa_frac) * w_ret + oa_frac * dist.w_oa

        chwst_cmd = chwst
        q_demand = 0.0
        capacity_limited = False
        if enable:
            t_cw_pre = cfg.tower.leaving_cw_temp(dist.t_wb, cw_fan)
            q_demand = cfg.coil.outlet(t_mix, w_mix, M, chwst_cmd)[2]
            n_stage = cfg.plant.update_staging(q_demand, chwst_cmd, t_cw_pre, enable=True)
            q_cap = n_stage * cfg.plant.chillers[0].available_capacity(chwst_cmd, t_cw_pre)
            if q_demand > q_cap:
                capacity_limited = True
                lo, hi = chwst_cmd, max(t_mix, chwst_cmd + 1e-6)
                for _ in range(CHWS_FLOAT_ITERS):
                    mid = 0.5 * (lo + hi)
                    f = (cfg.coil.outlet(t_mix, w_mix, M, mid)[2]
                         - n_stage * cfg.plant.chillers[0].available_capacity(mid, t_cw_pre))
                    if f > 0.0:
                        lo = mid
                    else:
                        hi = mid
                chwst = 0.5 * (lo + hi)
            t_sa, w_sa, q_tot, q_sen_coil, q_lat_coil = cfg.coil.outlet(t_mix, w_mix, M, chwst)
        else:
            t_sa, w_sa = t_mix, w_mix
            q_tot = q_sen_coil = q_lat_coil = 0.0

        q_int_s, m_gen = self._internal_gains(dist)
        q_sol = dist.q_solar_wm2 * self.solar_aperture / 1000.0
        cp = psy.CP_AIR
        T_z_new = np.empty(self.n_zones)
        T_m_new = np.empty(self.n_zones)
        W_z_new = np.empty(self.n_zones)
        for i, z in enumerate(cfg.zones):
            gz = cp * m_sa[i]
            a = z.C_z / self.dt + 1.0 / z.R_oa + 1.0 / z.R_zm + gz
            b = (z.C_z / self.dt * self.T_z[i] + dist.t_oa / z.R_oa
                 + self.T_m[i] / z.R_zm + q_int_s[i] + q_sol[i] + gz * t_sa)
            T_z_new[i] = b / a
            am = z.C_m / self.dt + 1.0 / z.R_zm + 1.0 / z.R_om
            bm = z.C_m / self.dt * self.T_m[i] + T_z_new[i] / z.R_zm + dist.t_oa / z.R_om
            T_m_new[i] = bm / am
            m_inf = 0.02 * m_sa[i]
            mass = psy.RHO_AIR * z.v_zone_m3 / self.dt
            W_z_new[i] = ((mass * self.W_z[i] + m_sa[i] * w_sa + m_inf * dist.w_oa
                           + m_gen[i]) / (mass + m_sa[i] + m_inf))
        self.T_z, self.T_m, self.W_z = T_z_new, T_m_new, W_z_new

        q_evap = q_tot
        t_cw = cfg.tower.leaving_cw_temp(dist.t_wb, cw_fan)

        if enable:
            n_on = cfg.plant.n_on()
        else:
            n_on = cfg.plant.update_staging(0.0, chwst, t_cw, enable=False)
        p_ch = cfg.plant.power(q_evap, chwst, t_cw)
        q_rej = q_evap + p_ch

        if enable:
            m_chw = q_evap / (CP_WATER * cfg.chw_deltaT)
            m_chw_design = cfg.design_cooling_kw / (CP_WATER * cfg.chw_deltaT)
            p_chwp = affinity_power(cfg.chw_pump_kw_design, m_chw / max(m_chw_design, 1e-6))
            p_cwp = affinity_power(cfg.cw_pump_kw_design, q_rej / max(cfg.design_cooling_kw * 1.25, 1e-6))
            p_towerfan = cfg.tower.fan_power(cw_fan)
            p_ahufan = affinity_power(cfg.ahu_fan_kw_design, M / max(cfg.m_air_design, 1e-6))
        else:
            p_chwp = p_cwp = p_towerfan = p_ahufan = 0.0
        p_total = p_ch + p_chwp + p_cwp + p_towerfan + p_ahufan
        self.peak_kw = max(self.peak_kw, p_total)

        rh_zone = np.array([psy.RH_from_TW(self.T_z[i], self.W_z[i]) for i in range(self.n_zones)])
        rt = max(q_evap / KW_PER_RT, 1e-6)
        kw_per_rt = p_total / rt if q_evap > 1e-6 else 0.0
        comfort_dev = np.maximum(np.abs(self.T_z - t_set) - t_band, 0.0).mean()
        pmv = self._pmv(self.T_z, rh_zone)
        pmv_disc = float(np.maximum(np.abs(pmv) - cfg.pmv_band, 0.0).mean())
        occ_w = float(np.clip((dist.occ_frac - cfg.occ_unoccupied_max) / max(cfg.occ_setback_max - cfg.occ_unoccupied_max, 1e-9), 0.0, 1.0))
        pmv_disc_eff = pmv_disc * occ_w
        comfort_dev_eff = comfort_dev * occ_w
        rh_viol = float((rh_zone > rh_ceiling).mean())
        rh_mould = float((rh_zone > cfg.RH_mould_limit).mean())
        starts = max(0, n_on - self.prev_n_on)
        cycle = abs(n_on - self.prev_n_on)
        self.prev_n_on = n_on
        self._q_evap_prev = float(q_evap)

        p_ref = max(cfg.p_ref_kw, 1e-6)
        e_term = p_total / p_ref
        c_term = pmv_disc_eff / max(cfg.pmv_band, 1e-9)
        s_term = applied["correction"] + corr_cycle

        dt_h = self.physics_dt_h
        cost = p_total * dt_h * cfg.energy_tariff * dist.price
        reward = -(cfg.w_energy * e_term + cfg.w_comfort * c_term + cfg.w_shield * s_term) * dt_h

        if enable != self._plant_on:
            self._plant_on, self._plant_since = enable, self.hour
        self.prev_action = {"chwst": chwst, "cw_fan": cw_fan}
        self.hour += dt_h
        self.step_idx += 1
        done = self.hour >= self.horizon_hours

        slot = min(int((self.hour - dt_h) // 24), len(self.day_weights) - 1)
        info = dict(
            day_slot=max(slot, 0), day_weight=float(self.day_weights[max(slot, 0)]),
            source_day=int(getattr(self.weather, "days", [0])[max(slot, 0)]) if hasattr(self.weather, "days") else 0,
            hour=self.hour,
            chwst=chwst, chwst_cmd=chwst_cmd,
            q_demand_kw=q_demand, capacity_limited=int(capacity_limited),
            chwst_float_K=float(chwst - chwst_cmd),
            cw_fan=cw_fan, t_cw=t_cw,
            n_chillers_on=n_on, q_evap_kw=q_evap, q_sen_kw=q_sen_coil,
            q_lat_kw=q_lat_coil, shr=q_sen_coil / max(q_tot, 1e-6),
            t_sa=t_sa, w_sa=w_sa, airflow_kgs=M,
            p_total_kw=p_total, p_chiller_kw=p_ch, p_chwpump_kw=p_chwp,
            p_cwpump_kw=p_cwp, p_towerfan_kw=p_towerfan, p_ahufan_kw=p_ahufan,
            kw_per_rt=kw_per_rt, cost_sgd=cost,
            mean_Tz=float(self.T_z.mean()), max_Tz=float(self.T_z.max()),
            mean_RH=float(rh_zone.mean()), max_RH=float(rh_zone.max()),
            rh_violation=rh_viol, comfort_dev=comfort_dev,
            pmv_disc=pmv_disc, t_wb=dist.t_wb, t_oa=dist.t_oa,
            flow_cap=flow_cap,
            occ_mode=mode, plant_enable=int(enable), forced_on=int(forced_on),
            comfort_forced=int(comfort_forced),
            cycle_held=int(held),
            t_set_active=t_set, rh_ceiling_active=rh_ceiling,
            rh_mould_breach=rh_mould, pmv_disc_eff=pmv_disc_eff,
            comfort_dev_eff=comfort_dev_eff,
            min_Tm=float(self.T_m.min()),
            max_dewpoint=float(max(psy.dew_point(self.W_z[i]) for i in range(self.n_zones))),
            occ=dist.occ_frac, price=dist.price, peak_kw=self.peak_kw,
            correction=applied["correction"], reward=reward,
            corr_safety=applied.get("corr_safety", 0.0),
            corr_bounds=applied.get("corr_bounds", 0.0),
            corr_ramp=applied.get("corr_ramp", 0.0),
            corr_total=applied.get("corr_total", 0.0),
            corr_cycle=corr_cycle,
            rw_energy=e_term, rw_comfort=c_term,
            rw_shield=s_term, rw_starts=starts, cycle_events=cycle)
        return self._obs(dist, p_total), reward, done, info

    def _dewpoint_risk(self) -> bool:
        """True if any zone's dew point is within the margin of its mass surface
        """
        cfg = self.cfg
        for i in range(self.n_zones):
            if psy.dew_point(self.W_z[i]) > self.T_m[i] - cfg.dewpoint_margin_K:
                return True
        return False

    @staticmethod
    def _pmv(T_z, rh):
        """Lightweight PMV proxy (still-air office, met~1.1, clo~0.5).
        Not a full Fanger solve -- a linearisation adequate for a Tier-1 reward term
        """
        return 0.30 * (T_z - 24.0) + 0.80 * (rh - 0.55)

    def _obs(self, dist, p_total):
        """Observation vector, length 2*n_zones + 13
        """
        h = self.hour % 24.0
        rh = np.array([psy.RH_from_TW(self.T_z[i], self.W_z[i]) for i in range(self.n_zones)])
        n_ch = max(len(getattr(self.cfg.plant, "chillers", [])) or 1, 1)
        cap_each = self.cfg.plant.chillers[0].available_capacity(self.prev_action["chwst"], self.cfg.tower.leaving_cw_temp(dist.t_wb, self.prev_action["cw_fan"]))
        util = self._q_evap_prev / max(max(self.prev_n_on, 1) * cap_each, 1e-6)
        fc = [self.weather.at(self.hour + k) for k in FORECAST_HOURS]
        return np.concatenate([
            self.T_z, rh,
            [dist.t_oa, dist.t_wb, dist.q_solar_wm2 / self.solar_ceiling,
             dist.occ_frac, self.prev_action["chwst"], self.prev_action["cw_fan"],
             math.sin(2 * math.pi * h / 24.0),
             math.cos(2 * math.pi * h / 24.0), p_total / self.peak_power,
             self.prev_n_on / n_ch,
             float(self._plant_on),
             self._lockout_left,
             self._q_evap_prev / max(self.cfg.design_cooling_kw, 1e-6),
             float(np.clip(util, 0.0, 2.0)),
             *[d.t_oa for d in fc],
             *[d.t_wb for d in fc],
             self._hours_to_occupancy() / 12.0,
             ]]).astype(float)

    OA_MIX_CAP = 0.8            # m_oa <= OA_MIX_CAP * M (economiser/mixing limit)

    def _oa_requirement_kgs(self, dist, mode: str) -> float:
        """Std 62.1 / SS 553 outdoor-air requirement [kg/s] at this occupancy.
        ~0.3 L/s.m2 area-based + 3.8 L/s.person, scaled by occupancy. 
        ZERO when unoccupied
        """
        if mode == "unoccupied":
            return 0.0
        persons = self._peak_persons * dist.occ_frac
        v_oa = 0.3e-3 * self._total_area + 3.8e-3 * persons          # m3/s
        return float(psy.RHO_AIR * v_oa)

    def _ventilation_floor_kgs(self, dist, mode: str) -> float:
        """Minimum TOTAL supply airflow [kg/s] that can carry the OA requirement.
        """
        return self._oa_requirement_kgs(dist, mode) / self.OA_MIX_CAP

    def _hours_to_occupancy(self, max_lead_h: float = 12.0) -> float:
        """Hours until `occ_frac` next exceeds the setback threshold.
        """
        thr = self.cfg.occ_setback_max
        try:
            if self.weather.at(self.hour).occ_frac > thr:
                return 0.0
            k = 0.5
            while k <= max_lead_h:
                if self.weather.at(self.hour + k).occ_frac > thr:
                    return float(k)
                k += 0.5
        except Exception:
            return max_lead_h
        return max_lead_h

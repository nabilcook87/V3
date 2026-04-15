from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd

from utils.refrigerant_properties import RefrigerantProperties
from utils.refrigerant_densities import RefrigerantDensities
from utils.refrigerant_viscosities import RefrigerantViscosities
from utils.supercompliq_co2 import RefrigerantProps
from utils.pressure_temp_converter import PressureTemperatureConverter

from functools import lru_cache

# ---- module-level singletons to avoid re-instantiation overhead ----
_PROPS = RefrigerantProperties()
_DENS = RefrigerantDensities()
_VISC = RefrigerantViscosities()
_CONV = PressureTemperatureConverter()

@lru_cache(maxsize=4096)
def _props_cached(ref: str, T_C: float) -> dict:
    # RefrigerantProperties returns a dict of properties for a given saturation temperature (°C)
    return _PROPS.get_properties(ref, T_C)

@lru_cache(maxsize=4096)
def _density_cached(ref: str, T_K: float, SH: float) -> float:
    return _DENS.get_density(ref, T_K, SH)

@lru_cache(maxsize=4096)
def _visc_cached(ref: str, T_K: float, SH: float) -> float:
    return _VISC.get_viscosity(ref, T_K, SH)

@lru_cache(maxsize=4096)
def _t2p_cached(ref: str, T_C: float) -> float:
    return _CONV.temp_to_pressure(ref, T_C)

@lru_cache(maxsize=4096)
def _p2t_cached(ref: str, P_bar: float) -> float:
    return _CONV.pressure_to_temp(ref, P_bar)

def _friction_factor(Re: float, eps: float, D: float) -> float:
    """Fast explicit friction factor.
    Uses laminar 64/Re, otherwise Swamee–Jain approximation of Colebrook.
    """
    if Re <= 0:
        return 0.0
    if Re < 2000.0:
        return 64.0 / Re
    return 0.25 / (math.log10((eps / (3.7 * D)) + (5.74 / (Re ** 0.9))) ** 2)

@dataclass
class RiserContext:

    refrigerant: str
    T_evap: float
    T_cond: float
    minliq_temp: float
    superheat_K: float
    max_penalty_K: float

    L: float
    SRB: int
    LRB: int
    bends_45: int
    MAC: int
    ptrap: int
    ubend: int
    ball: int
    globe: int
    PLF: float

    selected_material: str

    pipe_row_for_size: Callable[[str, Optional[str]], pd.Series]

    gc_max_pres: Optional[float] = None
    gc_min_pres: Optional[float] = None

    props: RefrigerantProperties = RefrigerantProperties()
    props_sup: RefrigerantProps = RefrigerantProps()
    dens: RefrigerantDensities = RefrigerantDensities()
    visc: RefrigerantViscosities = RefrigerantViscosities()
    conv: PressureTemperatureConverter = PressureTemperatureConverter()

@dataclass
class PipeResult:

    DP_kPa: float
    DT_K: float
    velocity_m_s: float
    mass_flow_kg_s: float
    dp_pipe: float
    dp_fit: float
    dp_valve: float
    dp_plf: float
    ID_m: float
    area_m2: float


@dataclass
class DoubleRiserResult:

    size_small: str
    size_large: str

    M_total: float
    M_small: float
    M_large: float

    DP_kPa: float
    DT_K: float

    small_result: PipeResult
    large_result: PipeResult

    dp_pipe: float
    dp_fit: float
    dp_valve: float
    dp_plf: float

def _velocity1_prop_for_refrigerant(refrigerant: str, superheat_K: float) -> float:

    if refrigerant in ["R744", "R744 TC"]:
        return 1.0

    if refrigerant == "R404A":
        return (0.0328330590542629 * superheat_K) - 1.47748765744183 if superheat_K > 45 else 0.0

    if refrigerant == "R134a":
        if superheat_K > 30:
            return (
                -0.000566085879684639 * (superheat_K ** 2) +
                0.075049554857083 * superheat_K -
                1.74200935399632
            )
        return 0.0

    if refrigerant in ["R407F", "R407A", "R410A", "R22", "R502", "R507A", "R448A", "R449A", "R717"]:
        return 1.0

    if refrigerant == "R407C":
        return 0.0

    if superheat_K > 30:
        return (
            0.0000406422632403154 * (superheat_K ** 2) -
            0.000541007136813307 * superheat_K +
            0.748882946418884
        )
    return 0.769230769230769

def pipe_results_for_massflow(
    size_inch: str,
    branch_mass_flow_kg_s: float,
    ctx: RiserContext,
    gauge: Optional[str] = None,
) -> PipeResult:

    ref = ctx.refrigerant
    T_evap = ctx.T_evap
    T_cond = ctx.T_cond
    T_min = ctx.minliq_temp
    SH = ctx.superheat_K
    penalty = ctx.max_penalty_K

    props = ctx.props
    props_sup = ctx.props_sup
    dens = ctx.dens
    visc = ctx.visc
    conv = ctx.conv

    row = ctx.pipe_row_for_size(size_inch, gauge)
    ID_mm = float(row["ID_mm"])
    ID_m = ID_mm / 1000
    A = math.pi * (ID_m / 2)**2

    if ref == "R744 TC":
        if ctx.gc_max_pres is None or ctx.gc_min_pres is None:
            raise ValueError("R744 TC requires discharge pressures.")

        h_in = props_sup.get_enthalpy_sup(ctx.gc_max_pres, T_cond)

        if ctx.gc_min_pres >= 73.8:
            h_inmin = props_sup.get_enthalpy_sup(ctx.gc_min_pres, T_min)
        elif ctx.gc_min_pres <= 72.13:
            h_inmin = _props_cached("R744", T_min)["enthalpy_liquid2"]
        else:
            raise ValueError("Disallowed CO2 TC region")

        h_evap = _props_cached("R744", T_evap)["enthalpy_vapor"]
        h_10K = _props_cached("R744", T_evap)["enthalpy_super"]

        h_inlet = h_in
        h_inletmin = h_inmin

    else:
        h_in = _props_cached(ref, T_cond)["enthalpy_liquid2"]
        h_inmin = _props_cached(ref, T_min)["enthalpy_liquid2"]

        h_inlet = _props_cached(ref, T_cond)["enthalpy_liquid"]
        h_inletmin = _props_cached(ref, T_min)["enthalpy_liquid"]

        h_evap = _props_cached(ref, T_evap)["enthalpy_vapor"]
        h_10K = _props_cached(ref, T_evap)["enthalpy_super"]

    hdiff_10K = h_10K - h_evap
    h_super = h_evap + hdiff_10K * min(max(SH, 5.0), 30.0) / 10.0

    Δh = h_evap - h_in
    Δh_min = h_evap - h_inmin

    if Δh <= 0:
        Q = 0
        m = max(branch_mass_flow_kg_s, 0.01)
    else:
        m = max(branch_mass_flow_kg_s, 0)
        Q = m * Δh

    m_min = Q / Δh_min if Δh_min > 0 else 0.01

    if ref == "R744 TC":
        density_super = _density_cached("R744", T_evap - penalty + 273.15, SH)
        ds2a = _density_cached("R744", T_evap + 273.15, ((SH + 5) / 2))
        ds2b = _density_cached("R744", T_evap - penalty + 273.15, ((SH + 5) / 2))
        density_super2 = (ds2a + ds2b) / 2
        density_sat = _props_cached("R744", T_evap)["density_vapor"]
        density_5K = _density_cached("R744", T_evap + 273.15, 5)

    else:
        density_super = _density_cached(ref, T_evap - penalty + 273.15, SH)
        ds2a = _density_cached(ref, T_evap + 273.15, ((SH + 5) / 2))
        ds2b = _density_cached(ref, T_evap - penalty + 273.15, ((SH + 5) / 2))
        density_super2 = (ds2a + ds2b) / 2
        density_sat = _props_cached(ref, T_evap)["density_vapor"]
        density_5K = _density_cached(ref, T_evap + 273.15, 5)

    density_mix = (density_super + density_5K) / 2

    v1 = m / (A * density_mix) if density_mix > 0 else 0
    v1min = m_min / (A * density_mix) if density_mix > 0 else 0
    v2 = m / (A * density_super2) if density_super2 > 0 else 0
    v2min = m_min / (A * density_super2) if density_super2 > 0 else 0

    w = _velocity1_prop_for_refrigerant(ref, SH)
    v = max(v1*w + v2*(1-w), v1min*w + v2min*(1-w))

    if ref == "R744 TC":
        vis_sup = _visc_cached("R744", T_evap - penalty + 273.15, SH)
        vs2a = _visc_cached("R744", T_evap + 273.15, ((SH+5)/2))
        vs2b = _visc_cached("R744", T_evap - penalty + 273.15, ((SH+5)/2))
        vis2 = (vs2a + vs2b) / 2
        vis5 = _visc_cached("R744", T_evap + 273.15, 5)
    else:
        vis_sup = _visc_cached(ref, T_evap - penalty + 273.15, SH)
        vs2a = _visc_cached(ref, T_evap + 273.15, ((SH+5)/2))
        vs2b = _visc_cached(ref, T_evap - penalty + 273.15, ((SH+5)/2))
        vis2 = (vs2a + vs2b) / 2
        vis5 = _visc_cached(ref, T_evap + 273.15, 5)

    vis = (vis_sup + vis5)/2
    vis_final = vis*w + vis2*(1-w)

    # Reynolds
    rho_recalc = m / (v*A) if v > 0 else density_mix
    Re = rho_recalc * v * ID_m / (vis_final/1e6) if vis_final > 0 else 0

    # friction factor
    eps = 0.00004572 if ctx.selected_material in ["Steel SCH40","Steel SCH80"] else 0.000001524

    f = _friction_factor(Re, eps, ID_m)

    # K-factors
    K_SRB = float(row["SRB"])
    K_LRB = float(row["LRB"])
    K_BALL = float(row["BALL"])
    K_GLOBE = float(row["GLOBE"])

    B_SRB = ctx.SRB + 0.5*ctx.bends_45 + 2*ctx.ubend + 3*ctx.ptrap
    B_LRB = ctx.LRB + ctx.MAC

    q = 0.5*rho_recalc*(v**2)/1000

    dp_pipe = f*(ctx.L/ID_m)*q
    dp_plf = q*ctx.PLF
    dp_fit = q*(K_SRB*B_SRB + K_LRB*B_LRB)
    dp_valve = q*(K_BALL*ctx.ball + K_GLOBE*ctx.globe)

    DP = dp_pipe + dp_fit + dp_valve + dp_plf

    if ref == "R744 TC":
        Pevap = _t2p_cached("R744", T_evap)
        P2 = Pevap - DP/100
        T2 = _p2t_cached("R744", P2)
    else:
        Pevap = _t2p_cached(ref, T_evap)
        P2 = Pevap - DP/100
        T2 = _p2t_cached(ref, P2)

    DT = T_evap - T2

    return PipeResult(
        DP_kPa=DP,
        DT_K=DT,
        velocity_m_s=v,
        mass_flow_kg_s=m,
        dp_pipe=dp_pipe,
        dp_fit=dp_fit,
        dp_valve=dp_valve,
        dp_plf=dp_plf,
        ID_m=ID_m,
        area_m2=A,
    )

def balance_double_riser(
    size_small: str,
    size_large: str,
    M_total_kg_s: float,
    ctx: RiserContext,
    gauge_small: Optional[str] = None,
    gauge_large: Optional[str] = None,
    tol_kPa: float = 0.001,
    max_iter: int = 100,
) -> DoubleRiserResult:

    if M_total_kg_s <= 0:
        raise ValueError("Total mass flow must be > 0.")

    lo = 0.000001*M_total_kg_s
    hi = 0.999999*M_total_kg_s

    res_s = None
    res_l = None

    for _ in range(max_iter):
        M_small = 0.5*(lo+hi)
        M_large = M_total_kg_s - M_small

        res_s = pipe_results_for_massflow(size_small, M_small, ctx, gauge=gauge_small)
        res_l = pipe_results_for_massflow(size_large, M_large, ctx, gauge=gauge_large)

        diff = res_s.DP_kPa - res_l.DP_kPa

        if abs(diff) <= tol_kPa:
            break

        if diff > 0:
            hi = M_small
        else:
            lo = M_small

    if res_s is None or res_l is None:
        raise RuntimeError("Double riser balance failed.")

    return DoubleRiserResult(
        size_small=size_small,
        size_large=size_large,
        M_total=M_total_kg_s,
        M_small=res_s.mass_flow_kg_s,
        M_large=res_l.mass_flow_kg_s,
        DP_kPa=(res_s.DP_kPa + res_l.DP_kPa)/2,
        DT_K=(res_s.DT_K + res_l.DT_K)/2,
        small_result=res_s,
        large_result=res_l,
        dp_pipe=(res_s.dp_pipe + res_l.dp_pipe)/2,
        dp_fit=(res_s.dp_fit + res_l.dp_fit)/2,
        dp_valve=(res_s.dp_valve + res_l.dp_valve)/2,
        dp_plf=(res_s.dp_plf + res_l.dp_plf)/2,
    )

def compute_double_riser_oil_metrics(
    dr,
    refrigerant: str,
    T_evap: float,
    density_foroil: float,
    oil_density: float,
    jg_half: float,
    mass_flow_foroil: float,
    mass_flow_foroilmin: float,
    MOR_correction: float,
    MOR_correctionmin: float,
    MOR_correction2: float,
):
    rs = dr.small_result
    rl = dr.large_result

    # Geometry
    small_ID_m = rs.ID_m
    small_area = rs.area_m2
    large_ID_m = rl.ID_m
    large_area = rl.area_m2

    # Min mass flows
    MinMassFlux_small = (jg_half ** 2) * (
        (density_foroil * 9.81 * small_ID_m * (oil_density - density_foroil)) ** 0.5
    )
    MinMassFlux_large = (jg_half ** 2) * (
        (density_foroil * 9.81 * large_ID_m * (oil_density - density_foroil)) ** 0.5
    )

    MinMassFlow_small = MinMassFlux_small * small_area
    MinMassFlow_large = MinMassFlux_large * large_area

    # Full-flow MOR (small riser)
    MOR_full_flow_1 = (
        MinMassFlow_small / mass_flow_foroil
    ) * 100.0 * (1 - MOR_correction) * (1 - MOR_correction2)

    MOR_full_flow_2 = (
        MinMassFlow_small / mass_flow_foroilmin
    ) * 100.0 * (1 - MOR_correctionmin) * (1 - MOR_correction2)

    # Large riser MOR
    M_largeprop = dr.M_large / dr.M_total

    M_largeoil_1 = M_largeprop * mass_flow_foroil
    M_largeoil_2 = M_largeprop * mass_flow_foroilmin

    MOR_large_1 = (
        MinMassFlow_large / M_largeoil_1
    ) * 100.0 * (1 - MOR_correction) * (1 - MOR_correction2)

    MOR_large_2 = (
        MinMassFlow_large / M_largeoil_2
    ) * 100.0 * (1 - MOR_correctionmin) * (1 - MOR_correction2)

    # Validity windows
    MOR_full_flow: Optional[float]
    MOR_large: Optional[float]

    if refrigerant in ["R23", "R508B"]:
        if T_evap < -86 or T_evap > -42:
            MOR_full_flow = None
            MOR_large = None
        else:
            MOR_full_flow = max(MOR_full_flow_1, MOR_full_flow_2)
            MOR_large = max(MOR_large_1, MOR_large_2)
    else:
        if T_evap < -40 or T_evap > 4:
            MOR_full_flow = None
            MOR_large = None
        else:
            MOR_full_flow = max(MOR_full_flow_1, MOR_full_flow_2)
            MOR_large = max(MOR_large_1, MOR_large_2)

    # SST
    SST = T_evap - dr.DT_K

    return MOR_full_flow, MOR_large, SST, M_largeprop



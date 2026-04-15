
from dataclasses import dataclass
from typing import Optional
import math

COPPER_WALL_TOL = 0.9
MILD_STEEL_WALL_TOL = 0.875
STAINLESS_WALL_TOL = 0.85
ALUMINIUM_WALL_TOL = 0.9

SEAMLESS_STEEL_STRESS_PSI = 15000.0
ERW_STEEL_STRESS_PSI = 12800.0 * 0.85
CW_STEEL_STRESS_PSI = 6800.0 * 0.60

@dataclass(frozen=True)
class Stress:
    value: float
    unit: str  # "MPa" or "psi"

@dataclass(frozen=True)
class WallThickness:
    mm: float
    inch: float

COPPER_GAUGE_WALL_IN = {
    22: 0.028,
    21: 0.032,
    20: 0.036,
    19: 0.040,
    18: 0.048,
    16: 0.064,
    14: 0.080,
    12: 0.104,
}

def k65_wall_tolerance(od_mm: float, wall_mm_nom: float) -> float:
    # VB logic
    if od_mm < 18 and wall_mm_nom < 1:
        return 0.9
    if od_mm < 18 and wall_mm_nom >= 1:
        return 0.87
    if od_mm >= 18 and wall_mm_nom < 1:
        return 0.9
    return 0.85

def k65_copper_pipe_stress_mpa(temp_c: float) -> float:
    T2 = (temp_c - 32.0) / 1.8

    A0 = 350.000000000019
    A1 = -0.363883782161995
    A2 = -5.60566148650349E-04
    A3 = -2.22112681890077E-06
    A4 = 4.7315606824367E-08
    A5 = -2.00885764117952E-10
    A6 = 2.5573812776535E-13

    return A0 + (A1 * T2) + (A2 * T2**2) + (A3 * T2**3) + (A4 * T2**4) + (A5 * T2**5) + (A6 * T2**6)

def aluminium_pipe_stress_mpa(temp_c: float) -> float:
    T2 = (temp_c * 1.8) + 32

    A0 = 8499.99990167723
    A1 = 4.16666961650599
    A2 = -5.83333682318413E-02
    A3 = 3.33333542883621E-04
    A4 = -6.66667343674366E-07
    A5 = 1.12102053282222E-15
    A6 = -7.46321063659667E-19
    
    return ((A0 + (A1 * T2) + (A2 * T2**2) + (A3 * T2**3) + (A4 * T2**4) + (A5 * T2**5) + (A6 * T2**6)) / 1000) * 6.895

def bsen_mpa(temp_c: float) -> float:
    T2 = temp_c
    
    A0 = 40.9999999999807
    A1 = 0.222619047772079
    A2 = -9.34523810126485E-03
    A3 = 1.33333333413095E-04
    A4 = -7.87619048051395E-07
    A5 = 1.56190476273047E-09
    
    return A0 + (A1 * T2) + (A2 * T2**2) + (A3 * T2**3) + (A4 * T2**4) + (A5 * T2**5)

def bsen_dki_mpa(temp_c: float) -> float:
    T2 = temp_c
    
    A0 = 192
    A1 = 0.0999999999999837
    A2 = -0.00119999999999987
    
    return A0 + (A1 * T2) + (A2 * T2**2)

def allowable_stress(
    *,
    pipe_index: int,
    circuit: str,
    copper_calc: Optional[str],
    temp_c: float,
    mwp_temp_c: float | None = None,
) -> Stress:

    if pipe_index == 1:
        if copper_calc == "DKI":
            return Stress(
                value=int(round(bsen_dki_mpa(mwp_temp_c))),
                unit="MPa",
            )
        else:
            return Stress(
                value=int(round(bsen_mpa(mwp_temp_c))),
                unit="MPa",
            )

    if pipe_index == 6:
        temp_f = max(mwp_temp_c * 9.0 / 5.0 + 32.0, 100.0)
        return Stress(
            value=pipe_stress_psi(temp_f),
            unit="psi",
        )

    if pipe_index == 7:
        return Stress(
            value=int(round(aluminium_pipe_stress_mpa(mwp_temp_c))),
            unit="MPa",
        )

    if pipe_index == 8:
        return Stress(
            value=int(round(k65_copper_pipe_stress_mpa(mwp_temp_c))),
            unit="MPa",
        )

    if pipe_index in (2, 5):  # steel
        return Stress(value=15000.0, unit="psi")

    if pipe_index in (3, 4):  # stainless
        return Stress(value=70000.0, unit="psi")

    raise ValueError(f"Unsupported pipe index: {pipe_index}")

def calc_wall_thickness(
    *,
    pipe_index: int,
    od_mm: float,
    id_mm: Optional[float] = None,
    gauge: Optional[int] = None,
) -> WallThickness:

    if pipe_index == 1:
        if gauge not in COPPER_GAUGE_WALL_IN:
            raise ValueError("Invalid gauge for EN12735 copper")

        t_in = COPPER_GAUGE_WALL_IN[gauge] * COPPER_WALL_TOL
        t_mm = t_in * 25.4

        return WallThickness(mm=t_mm, inch=t_in)

    if id_mm is None:
        raise ValueError("ID must be provided for non-gauge pipes")

    t_mm = (od_mm - id_mm) / 2.0

    # Apply tolerances
    if pipe_index in (2, 5):  # steel
        t_mm *= MILD_STEEL_WALL_TOL
    elif pipe_index in (3, 4):  # stainless
        t_mm *= STAINLESS_WALL_TOL
    elif pipe_index == 7:  # aluminium
        t_mm *= ALUMINIUM_WALL_TOL
    elif pipe_index == 8:  # K65 copper (VB piecewise tolerance)
        wall_mm_nom = (od_mm - id_mm) / 2.0
        t_mm = wall_mm_nom * k65_wall_tolerance(od_mm, wall_mm_nom)

    t_in = t_mm / 25.4

    return WallThickness(mm=t_mm, inch=t_in)

def calc_mwp(
    *,
    pipe_index: int,
    stress: Stress,
    wall: WallThickness,
    od_mm: float,
    id_mm: float | None,
    mwp_temp_c: float,
    copper_calc: Optional[str],
) -> float:
    PSI_TO_BAR = 0.0689476

    # Helpers
    def _od_in() -> float:
        return od_mm / 25.4

    def _nom_wall_in() -> float:
        if id_mm is None:
            raise ValueError("This pipe type requires id_mm.")
        return (_od_in() - (id_mm / 25.4)) / 2.0

    if pipe_index == 6:
        wall_in = _nom_wall_in() * COPPER_WALL_TOL

        temp_f = mwp_temp_c * 9.0 / 5.0 + 32.0
        if temp_f < 100.0:
            temp_f = 100.0

        stress_psi = pipe_stress_psi(temp_f)

        mwp_psi = (2.0 * stress_psi * wall_in) / (_od_in() - 0.8 * wall_in)
        return mwp_psi * PSI_TO_BAR

    if pipe_index in (2, 5):
        deduction = 0.025 if od_mm <= 61 else 0.065
    
        wall_in_nom = _nom_wall_in()
        wall_eff = (wall_in_nom * MILD_STEEL_WALL_TOL) - deduction
    
        mwps = {}
        for weld, stress_psi in steel_weld_stresses_by_size(od_mm).items():
            mwp_psi = (2.0 * stress_psi * wall_eff) / _od_in()
            mwps[weld] = mwp_psi * PSI_TO_BAR
    
        return mwps

    if pipe_index in (3, 4):
        wall_in_nom = _nom_wall_in()
        mwp_psi = ((wall_in_nom * 2.0 * 70000.0 * STAINLESS_WALL_TOL) / _od_in()) / 4.0 * 0.7
        return mwp_psi * PSI_TO_BAR

    if stress.unit == "MPa":
        mwp_bar = (20.0 * stress.value * wall.mm) / (od_mm - wall.mm)
    elif stress.unit == "psi":
        mwp_psi = (20.0 * stress.value * wall.inch) / (_od_in() - wall.inch)
        mwp_bar = mwp_psi * PSI_TO_BAR
    else:
        raise ValueError(f"Unsupported stress unit: {stress.unit}")

    if copper_calc == "DKI" and pipe_index == 1:
        mwp_bar /= 3.5

    if copper_calc == "DKI" and pipe_index == 8:
        if mwp_temp_c == 50:
            mwp_bar = mwp_bar * 1.5
        if mwp_temp_c == 100:
            mwp_bar = mwp_bar * 1.5
        if mwp_temp_c == 150:
            mwp_bar = mwp_bar * 1.357

    if copper_calc == "DKI" and pipe_index == 7:
        if mwp_temp_c == 50:
            mwp_bar = mwp_bar * 1.357
        if mwp_temp_c == 100:
            mwp_bar = mwp_bar * 1.357
        if mwp_temp_c == 150:
            mwp_bar = mwp_bar * 1.357

    return mwp_bar

from utils.refrigerant_properties import RefrigerantProperties

def calc_design_pressure_bar_g(
    *,
    refrigerant: str,
    design_temp_c: float,
    circuit: str,
    r744_tc_pressure_bar_g: float | None = None,
) -> float:

    # CO₂ transcritical override
    if refrigerant.upper() == "R744 TC" and circuit != "Suction":
        if r744_tc_pressure_bar_g is None:
            raise ValueError("R744 transcritical design pressure must be provided")
        return r744_tc_pressure_bar_g

    if refrigerant == "R744 TC" and circuit == "Suction":
        refrigerant = "R744"

    props = RefrigerantProperties()
    data = props.get_properties(refrigerant, design_temp_c)

    # VB logic:
    # Liquid / Pumped → bubble point
    # Suction / Discharge → dew point
    if circuit in ("Liquid", "Pumped"):
        p_abs = data["pressure_bar2"]   # bubble
    else:
        p_abs = data["pressure_bar"]    # dew

    # Convert abs → gauge
    return p_abs - 1.01325

def calc_pressure_limits(
    *,
    design_pressure_bar_g: float,
    dp_standard: str,
) -> dict[str, float]:

    if dp_standard == "BS EN 378":
        return {
            "design": design_pressure_bar_g,
            "leak_test": design_pressure_bar_g,
            "pressure_test": 1.3 * design_pressure_bar_g,
            "hp_cutout": 0.9 * design_pressure_bar_g,
            "relief_setting": design_pressure_bar_g,
            "rated_discharge": 1.1 * design_pressure_bar_g,
        }

    if dp_standard == "ASME B31.5 - 2006":
        return {
            "design": design_pressure_bar_g,
            "leak_test": design_pressure_bar_g,
            "pressure_test": 1.3 * design_pressure_bar_g,
            "hp_cutout": 0.9 * design_pressure_bar_g,
            "relief_setting": design_pressure_bar_g,
            "rated_discharge": 1.1 * design_pressure_bar_g,
        }

    raise ValueError(f"Unknown design pressure standard: {dp_standard}")

def system_pressure_check(
    *,
    refrigerant: str,
    design_temp_c: float,
    mwp_temp_c: float,        # ← NEW
    circuit: str,
    pipe_index: int,
    od_mm: float,
    id_mm: float | None = None,
    gauge: int | None = None,
    copper_calc: str | None = None,
    r744_tc_pressure_bar_g: float | None = None,
    dp_standard: str = "BS EN 378",
) -> dict:

    if refrigerant == "R744 TC" and circuit == "Suction":
        refrigerant = "R744"
    
    if dp_standard == "BS EN 378":
        design_pressure = calc_design_pressure_bar_g(
            refrigerant=refrigerant,
            design_temp_c=design_temp_c,
            circuit=circuit,
            r744_tc_pressure_bar_g=r744_tc_pressure_bar_g,
        )
    
    elif dp_standard == "ASME B31.5 - 2006":
        design_pressure = calc_design_pressure_bar_g(
            refrigerant=refrigerant,
            design_temp_c=design_temp_c,
            circuit=circuit,
            r744_tc_pressure_bar_g=r744_tc_pressure_bar_g,
        )
    
    else:
        raise ValueError(f"Unknown design pressure standard: {dp_standard}")

    pressure_limits = calc_pressure_limits(
        design_pressure_bar_g=design_pressure,
        dp_standard=dp_standard,
    )

    stress = allowable_stress(
        pipe_index=pipe_index,
        circuit=circuit,
        copper_calc=copper_calc,
        temp_c=design_temp_c,
        mwp_temp_c=mwp_temp_c,
    )

    wall = calc_wall_thickness(
        pipe_index=pipe_index,
        od_mm=od_mm,
        id_mm=id_mm,
        gauge=gauge,
    )

    mwp = calc_mwp(
        pipe_index=pipe_index,
        stress=stress,
        wall=wall,
        od_mm=od_mm,
        id_mm=id_mm,
        mwp_temp_c=mwp_temp_c,   # ← pass separately
        copper_calc=copper_calc,
    )

    mwp_multi = calc_mwp_multi_temp(
        pipe_index=pipe_index,
        stress=stress,
        wall=wall,
        od_mm=od_mm,
        id_mm=id_mm,
        copper_calc=copper_calc,
    )

    if isinstance(mwp, dict):
        passes = {k: v >= design_pressure for k, v in mwp.items()}
        margin = {k: v - design_pressure for k, v in mwp.items()}
    else:
        passes = mwp >= design_pressure
        margin = mwp - design_pressure

    return {
        "design_pressure_bar_g": design_pressure,
        "pressure_limits_bar": pressure_limits,
        "allowable_stress": stress,
        "wall_thickness": wall,
        "mwp_bar": mwp,
        "mwp_multi_temp": mwp_multi,
        "pass": passes,
        "margin_bar": margin,
    }

def system_pressure_check_double_riser(
    *,
    refrigerant: str,
    design_temp_c: float,
    mwp_temp_c: float,
    circuit: str,
    dp_standard: str,

    # --- Branch A ---
    pipe_index_a: int,
    od_mm_a: float,
    id_mm_a: float | None,
    gauge_a: int | None,

    # --- Branch B ---
    pipe_index_b: int,
    od_mm_b: float,
    id_mm_b: float | None,
    gauge_b: int | None,

    copper_calc: str | None = None,
    r744_tc_pressure_bar_g: float | None = None,
) -> dict:
    """
    Double riser pressure check.
    Evaluates both branches and carries forward the WORST (lowest) MWP.
    """

    # --- Run normal pressure check on each branch ---
    res_a = system_pressure_check(
        refrigerant=refrigerant,
        design_temp_c=design_temp_c,
        mwp_temp_c=mwp_temp_c,
        circuit=circuit,
        pipe_index=pipe_index_a,
        od_mm=od_mm_a,
        id_mm=id_mm_a,
        gauge=gauge_a,
        copper_calc=copper_calc,
        r744_tc_pressure_bar_g=r744_tc_pressure_bar_g,
        dp_standard=dp_standard,
    )

    res_b = system_pressure_check(
        refrigerant=refrigerant,
        design_temp_c=design_temp_c,
        mwp_temp_c=mwp_temp_c,
        circuit=circuit,
        pipe_index=pipe_index_b,
        od_mm=od_mm_b,
        id_mm=id_mm_b,
        gauge=gauge_b,
        copper_calc=copper_calc,
        r744_tc_pressure_bar_g=r744_tc_pressure_bar_g,
        dp_standard=dp_standard,
    )

    design_p = res_a["design_pressure_bar_g"]  # identical for both

    # --- Helper to extract governing MWP ---
    def min_mwp_val(x):
        return min(x.values()) if isinstance(x, dict) else x

    # Governing MWP at the selected reference temp
    mwp_a = min_mwp_val(res_a["mwp_bar"])
    mwp_b = min_mwp_val(res_b["mwp_bar"])

    # If both are steel dicts, keep dict form by taking min per weld key
    if isinstance(res_a["mwp_bar"], dict) and isinstance(res_b["mwp_bar"], dict):
        mwp_bar = {}
        for k in set(res_a["mwp_bar"]) | set(res_b["mwp_bar"]):
            va = res_a["mwp_bar"].get(k)
            vb = res_b["mwp_bar"].get(k)
            if va is None:
                mwp_bar[k] = vb
            elif vb is None:
                mwp_bar[k] = va
            else:
                mwp_bar[k] = min(va, vb)
    else:
        # otherwise return the governing float (worst case)
        mwp_bar = min(mwp_a, mwp_b)

    # Governing multi-temp table (so UI works unchanged)
    mwp_multi_temp = {}
    for t in (50, 100, 150):
        a = res_a.get("mwp_multi_temp", {}).get(t)
        b = res_b.get("mwp_multi_temp", {}).get(t)
        if a is None and b is None:
            continue
        if a is None:
            mwp_multi_temp[t] = min_mwp_val(b)
        elif b is None:
            mwp_multi_temp[t] = min_mwp_val(a)
        else:
            mwp_multi_temp[t] = min(min_mwp_val(a), min_mwp_val(b))

    design_p = res_a["design_pressure_bar_g"]

    # pass/margin compatible with your UI
    if isinstance(mwp_bar, dict):
        passes = {k: v >= design_p for k, v in mwp_bar.items()}
        margin = {k: v - design_p for k, v in mwp_bar.items()}
    else:
        passes = mwp_bar >= design_p
        margin = mwp_bar - design_p

    return {
        "design_pressure_bar_g": design_p,
        "pressure_limits_bar": res_a["pressure_limits_bar"],
        "mwp_bar": mwp_bar,
        "mwp_multi_temp": mwp_multi_temp,
        "pass": passes,
        "margin_bar": margin,
        # optional, handy for debugging:
        "branch_a": res_a,
        "branch_b": res_b,
    }

def pipe_stress_psi(temp_f: float) -> float:
    T2 = temp_f
    A0 = 37600.0274576506
    A1 = -878.367490116384
    A2 = 9.83034307114902
    A3 = -5.78667251130062E-02
    A4 = 1.87333522081964E-04
    A5 = -3.14666979085113E-07
    A6 = 2.13333541253137E-10
    return A0 + (A1 * T2) + (A2 * T2**2) + (A3 * T2**3) + (A4 * T2**4) + (A5 * T2**5) + (A6 * T2**6)

def steel_weld_stresses_by_size(od_mm: float) -> dict[str, float]:
    """
    Returns applicable steel stresses (psi) keyed by weld type.
    Always includes 'seamless'.
    """
    stresses = {
        "seamless": SEAMLESS_STEEL_STRESS_PSI,
    }

    # OD thresholds correspond to ~2" and ~4"
    # 2" OD ≈ 60.3 mm
    # 4" OD ≈ 114.3 mm

    if od_mm < 60.3:
        stresses["cw"] = CW_STEEL_STRESS_PSI
    elif 60.3 <= od_mm <= 114.3:
        stresses["cw"] = CW_STEEL_STRESS_PSI
        stresses["erw"] = ERW_STEEL_STRESS_PSI
    else:
        stresses["erw"] = ERW_STEEL_STRESS_PSI

    return stresses

def calc_mwp_multi_temp(
    *,
    pipe_index: int,
    stress: Stress,
    wall: WallThickness,
    od_mm: float,
    id_mm: float | None,
    copper_calc: Optional[str],
    temps: tuple[int, ...] = (50, 100, 150),
) -> dict[int, float]:

    results = {}

    if pipe_index not in (6, 7, 8, 1):
        return results

    for t in temps:
        stress_t = allowable_stress(
            pipe_index=pipe_index,
            circuit="Discharge",      # stress tables are temp-based, not circuit-limited
            copper_calc=copper_calc,
            temp_c=t,
            mwp_temp_c=t,
        )

        mwp_t = calc_mwp(
            pipe_index=pipe_index,
            stress=stress_t,
            wall=wall,
            od_mm=od_mm,
            id_mm=id_mm,
            mwp_temp_c=t,
            copper_calc=copper_calc,
        )

        results[t] = mwp_t

    return results

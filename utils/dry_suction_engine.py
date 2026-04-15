"""Dry suction sizing engine (shared by single-pipe and network modes).

Goal: ensure the thermodynamics + sizing logic used for:
  * the existing single-pipe "Dry Suction" mode, and
  * the new multi-pipe network mode

are *identical*.

This module is a lift of the `get_pipe_results()` logic currently embedded in
`app.py` under `if mode == "Dry Suction"`.

Public API:
  - make_pipe_row_for_size(material_df)
  - strongest_gauge_for_size(material_df, size_inch)
  - make_ctx(...)
  - make_get_pipe_results(...)

`make_get_pipe_results(...)` returns a callable:
    get_pipe_results(size_inch: str, gauge_override: str|None) -> (MORfinal, dt)

Where MORfinal is returned as NaN if the legacy code would return a blank.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple
import math


@dataclass(frozen=True)
class DrySuctionInputs:
    refrigerant: str
    T_evap: float
    T_cond: float
    minliq_temp: float
    superheat_K: float
    max_penalty_K: float
    evap_capacity_kw: float

    # segment geometry / fittings / dp factors
    L: float
    SRB: float
    LRB: float
    bends_45: float
    MAC: float
    ptrap: float
    ubend: float
    ball: float
    globe: float
    PLF: float

    selected_material: str

    # Only used for R744 TC branch
    gc_max_pres: Optional[float] = None
    gc_min_pres: Optional[float] = None


def make_pipe_row_for_size(material_df) -> Callable[[str, Optional[str]], Any]:
    """Row lookup matching the single-pipe UI behavior."""

    def _pipe_row_for_size(size_inch: str, gauge: Optional[str] = None):
        rows = material_df[
            material_df["Nominal Size (inch)"].astype(str).str.strip() == str(size_inch).strip()
        ]
        if rows.empty:
            raise KeyError(f"Pipe size not found: {size_inch}")

        if "Gauge" in rows.columns and rows["Gauge"].notna().any():
            if gauge is not None:
                rows_g = rows[rows["Gauge"] == gauge]
                if not rows_g.empty:
                    return rows_g.iloc[0]
            return rows.iloc[0]

        return rows.iloc[0]

    return _pipe_row_for_size


def strongest_gauge_for_size(material_df, size_inch: str) -> Optional[str]:
    rows = material_df[
        material_df["Nominal Size (inch)"].astype(str).str.strip() == str(size_inch).strip()
    ]
    if rows.empty or "Gauge" not in rows.columns or rows["Gauge"].isna().all():
        return None
    return max(rows["Gauge"].dropna().astype(str).unique())


def make_ctx(
    *,
    inputs: DrySuctionInputs,
    pipe_row_for_size: Callable[[str, Optional[str]], Any],
):
    """Create the `RiserContext` used by utils.double_riser.balance_double_riser."""
    from utils.double_riser import RiserContext

    return RiserContext(
        refrigerant=inputs.refrigerant,
        T_evap=inputs.T_evap,
        T_cond=inputs.T_cond,
        minliq_temp=inputs.minliq_temp,
        superheat_K=inputs.superheat_K,
        max_penalty_K=inputs.max_penalty_K,

        L=inputs.L,
        SRB=inputs.SRB,
        LRB=inputs.LRB,
        bends_45=inputs.bends_45,
        MAC=inputs.MAC,
        ptrap=inputs.ptrap,
        ubend=inputs.ubend,
        ball=inputs.ball,
        globe=inputs.globe,
        PLF=inputs.PLF,

        selected_material=inputs.selected_material,
        pipe_row_for_size=pipe_row_for_size,

        gc_max_pres=inputs.gc_max_pres if inputs.refrigerant == "R744 TC" else None,
        gc_min_pres=inputs.gc_min_pres if inputs.refrigerant == "R744 TC" else None,
    )


def make_get_pipe_results(
    *,
    inputs: DrySuctionInputs,
    pipe_row_for_size: Callable[[str, Optional[str]], Any],
) -> Callable[[str, Optional[str], float], Tuple[float, float, float, float, float, float, float, float, float, float, float, float]]:
    """Return get_pipe_results(size_inch, gauge_override=None, extra_dp_kPa=0.0).

    Returns:
        (MORfinal, dt, velocity_m_sfinal, q_kPa, ID_m)
    """

    # Dependencies are imported here so the app can keep its existing import layout.
    from utils.refrigerant_densities import RefrigerantDensities
    from utils.refrigerant_properties import RefrigerantProperties
    from utils.refrigerant_viscosities import RefrigerantViscosities
    from utils.supercompliq_co2 import RefrigerantProps
    from utils.pressure_temp_converter import PressureTemperatureConverter

    refrigerant = inputs.refrigerant
    T_evap = inputs.T_evap
    T_cond = inputs.T_cond
    minliq_temp = inputs.minliq_temp
    superheat_K = inputs.superheat_K
    max_penalty = inputs.max_penalty_K
    evap_capacity_kw = inputs.evap_capacity_kw

    L = inputs.L
    SRB = inputs.SRB
    LRB = inputs.LRB
    _45 = inputs.bends_45
    MAC = inputs.MAC
    ptrap = inputs.ptrap
    ubend = inputs.ubend
    ball = inputs.ball
    globe = inputs.globe
    PLF = inputs.PLF

    selected_material = inputs.selected_material
    gc_max_pres = inputs.gc_max_pres
    gc_min_pres = inputs.gc_min_pres

    # NOTE:
    # Network mode needs to model tee/junction losses that occur *between* pipe segments.
    # The legacy VB tool handled this by adding a junction loss (k-factor * velocity pressure)
    # to the *previous* segment's pressure drop.
    #
    # To support that without rewriting the thermodynamics, we allow callers to add an
    # extra pressure drop (kPa) that is applied after straight/fitting/valve/PLF losses
    # but before converting pressure drop to an equivalent temperature penalty (dt).
    def get_pipe_results(
        size_inch: str,
        gauge_override: Optional[str] = None,
        extra_dp_kPa: float = 0.0,
    ) -> Tuple[float, float, float, float, float, float, float, float, float, float, float, float]:
        pipe_row = pipe_row_for_size(size_inch, gauge_override)

        try:
            ID_mm_local = float(pipe_row["ID_mm"])
        except Exception:
            return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")

        ID_m_local = ID_mm_local / 1000.0
        area_m2_local = math.pi * (ID_m_local / 2) ** 2

        dens = RefrigerantDensities()
        props = RefrigerantProperties()
        props_sup = RefrigerantProps()

        # --- densities ---
        if refrigerant == "R744 TC":
            density_super = dens.get_density("R744", T_evap - max_penalty + 273.15, superheat_K)
            density_super2a = dens.get_density("R744", T_evap + 273.15, ((superheat_K + 5) / 2))
            density_super2b = dens.get_density("R744", T_evap - max_penalty + 273.15, ((superheat_K + 5) / 2))
            density_super2 = (density_super2a + density_super2b) / 2
            density_super_foroil = dens.get_density("R744", T_evap + 273.15, min(max(superheat_K, 5), 30))
            density_sat = props.get_properties("R744", T_evap)["density_vapor"]
            density_5K = dens.get_density("R744", T_evap + 273.15, 5)
        else:
            density_super = dens.get_density(refrigerant, T_evap - max_penalty + 273.15, superheat_K)
            density_super2a = dens.get_density(refrigerant, T_evap + 273.15, ((superheat_K + 5) / 2))
            density_super2b = dens.get_density(refrigerant, T_evap - max_penalty + 273.15, ((superheat_K + 5) / 2))
            density_super2 = (density_super2a + density_super2b) / 2
            density_super_foroil = dens.get_density(refrigerant, T_evap + 273.15, min(max(superheat_K, 5), 30))
            density_sat = props.get_properties(refrigerant, T_evap)["density_vapor"]
            density_5K = dens.get_density(refrigerant, T_evap + 273.15, 5)

        density = (density_super + density_5K) / 2
        density_foroil = (density_super_foroil + density_sat) / 2

        # --- enthalpies / mass flows ---
        if refrigerant == "R744 TC":
            h_in = props_sup.get_enthalpy_sup(gc_max_pres, T_cond)

            if gc_min_pres is None:
                return float("nan"), float("nan")
            if gc_min_pres >= 73.8:
                h_inmin = props_sup.get_enthalpy_sup(gc_min_pres, minliq_temp)
            elif gc_min_pres <= 72.13:
                h_inmin = props.get_properties("R744", minliq_temp)["enthalpy_liquid2"]
            else:
                return float("nan"), float("nan")

            h_inlet = h_in
            h_inletmin = h_inmin
            h_evap = props.get_properties("R744", T_evap)["enthalpy_vapor"]
            h_10K = props.get_properties("R744", T_evap)["enthalpy_super"]
        else:
            h_in = props.get_properties(refrigerant, T_cond)["enthalpy_liquid2"]
            h_inmin = props.get_properties(refrigerant, minliq_temp)["enthalpy_liquid2"]
            h_inlet = props.get_properties(refrigerant, T_cond)["enthalpy_liquid"]
            h_inletmin = props.get_properties(refrigerant, minliq_temp)["enthalpy_liquid"]
            h_evap = props.get_properties(refrigerant, T_evap)["enthalpy_vapor"]
            h_10K = props.get_properties(refrigerant, T_evap)["enthalpy_super"]

        hdiff_10K = h_10K - h_evap
        hdiff_custom = hdiff_10K * min(max(superheat_K, 5), 30) / 10
        h_super = h_evap + hdiff_custom
        h_foroil = (h_evap + h_super) / 2

        delta_h = h_evap - h_in
        delta_hmin = h_evap - h_inmin

        delta_h_foroil = h_foroil - h_inlet
        delta_h_foroilmin = h_foroil - h_inletmin

        mass_flow_kg_s = evap_capacity_kw / delta_h if delta_h > 0 else 0.01
        mass_flow_kg_smin = evap_capacity_kw / delta_hmin if delta_hmin > 0 else 0.01
        mass_flow_foroil = evap_capacity_kw / delta_h_foroil if delta_h_foroil > 0 else 0.01
        mass_flow_foroilmin = evap_capacity_kw / delta_h_foroilmin if delta_h_foroilmin > 0 else 0.01

        # --- velocities ---
        v1 = mass_flow_kg_s / (area_m2_local * density)
        v1min = mass_flow_kg_smin / (area_m2_local * density)
        v2 = mass_flow_kg_s / (area_m2_local * density_super2)
        v2min = mass_flow_kg_smin / (area_m2_local * density_super2)

        if refrigerant in ("R744", "R744 TC"):
            velocity1_prop = 1
        elif refrigerant == "R404A":
            velocity1_prop = (0.0328330590542629 * superheat_K) - 1.47748765744183 if superheat_K > 45 else 0
        elif refrigerant == "R134a":
            velocity1_prop = (
                (-0.000566085879684639 * (superheat_K ** 2))
                + (0.075049554857083 * superheat_K)
                - 1.74200935399632
            ) if superheat_K > 30 else 0
        elif refrigerant in ["R407F", "R407A", "R410A", "R22", "R502", "R507A", "R448A", "R449A", "R717"]:
            velocity1_prop = 1
        elif refrigerant == "R407C":
            velocity1_prop = 0
        else:
            velocity1_prop = (
                (0.0000406422632403154 * (superheat_K ** 2))
                - (0.000541007136813307 * superheat_K)
                + 0.748882946418884
            ) if superheat_K > 30 else 0.769230769230769

        velocity_m_s = (v1 * velocity1_prop) + (v2 * (1 - velocity1_prop))
        velocity_m_smin = (v1min * velocity1_prop) + (v2min * (1 - velocity1_prop))
        velocity_m_sfinal = max(velocity_m_s, velocity_m_smin)

        # --- oil density ---
        if refrigerant in ["R23", "R508B"]:
            oil_density_sat = (-0.853841209044878 * T_evap) + 999.190772536527
            oil_density_super = (-0.853841209044878 * (T_evap + min(max(superheat_K, 5), 30))) + 999.190772536527
        else:
            oil_density_sat = (-0.00356060606060549 * (T_evap ** 2)) - (0.957878787878808 * T_evap) + 963.595454545455
            oil_density_super = (
                (-0.00356060606060549 * ((T_evap + min(max(superheat_K, 5), 30)) ** 2))
                - (0.957878787878808 * (T_evap + min(max(superheat_K, 5), 30)))
                + 963.595454545455
            )
        oil_density = (oil_density_sat + oil_density_super) / 2

        # --- jg_half map ---
        jg_map = {
            "R404A": 0.860772464072673, "R134a": 0.869986729796935, "R407F": 0.869042493641944,
            "R744": 0.877950613678719, "R744 TC": 0.877950613678719, "R407A": 0.867374311574041,
            "R410A": 0.8904423325365, "R407C": 0.858592104849471, "R22": 0.860563058394146,
            "R502": 0.858236706656266, "R507A": 0.887709710291009, "R449A": 0.867980496631757,
            "R448A": 0.86578818145833, "R717": 0.854957410951708, "R290": 0.844975139695726,
            "R1270": 0.849089717732815, "R600a": 0.84339338979887, "R1234ze": 0.867821375349728,
            "R1234yf": 0.860767472602571, "R12": 0.8735441986466, "R11": 0.864493203834913,
            "R454B": 0.869102255850291, "R450A": 0.865387140496035, "R513A": 0.861251244627232,
            "R454A": 0.868161104592492, "R455A": 0.865687329727713, "R454C": 0.866423016875524,
            "R32": 0.875213309852597, "R23": 0.865673418568001, "R508B": 0.864305626845382,
        }
        jg_half = jg_map.get(refrigerant, 0.865)

        # --- MOR ---
        MinMassFlux = (jg_half ** 2) * ((density_foroil * 9.81 * ID_m_local * (oil_density - density_foroil)) ** 0.5)
        MinMassFlow = MinMassFlux * area_m2_local
        MOR_pre = (MinMassFlow / mass_flow_foroil) * 100
        MOR_premin = (MinMassFlow / mass_flow_foroilmin) * 100

        if refrigerant in ["R23", "R508B"]:
            MOR_correctliq = T_cond + 47.03
            MOR_correctliqmin = minliq_temp + 47.03
            evapoil = T_evap + 46.14
        else:
            MOR_correctliq = T_cond
            MOR_correctliqmin = minliq_temp
            evapoil = T_evap

        # MOR_correction
        if refrigerant == "R744":
            MOR_correction = (0.000225755013421421 * MOR_correctliq) - 0.00280879370374927
        elif refrigerant == "R744 TC":
            MOR_correction = (0.0000603336117708171 * h_in) - 0.0142318718120024
        elif refrigerant in ["R407A", "R449A", "R448A", "R502"]:
            MOR_correction = (0.00000414431651323856 * (MOR_correctliq ** 2)) + (0.000381908525139781 * MOR_correctliq) - 0.0163450053041212
        elif refrigerant == "R507A":
            MOR_correction = (0.000302619054048837 * MOR_correctliq) - 0.00930188913363997
        elif refrigerant == "R22":
            MOR_correction = (0.000108153843367715 * MOR_correctliq) - 0.00329248681202757
        elif refrigerant == "R407C":
            xx = max(MOR_correctliq, -32.0716410083429)
            MOR_correction = (0.00000420322918839302 * (xx ** 2)) + (0.000269608915211859 * xx) - 0.0134546663857195
        elif refrigerant == "R410A":
            MOR_correction = 0
        elif refrigerant == "R407F":
            xx = max(MOR_correctliq, -34.4346433150568)
            MOR_correction = (0.00000347332380289385 * (xx ** 2)) + (0.000239205332540693 * xx) - 0.0121545316131988
        elif refrigerant == "R134a":
            MOR_correction = (0.000195224660107459 * MOR_correctliq) - 0.00591757011487048
        elif refrigerant == "R404A":
            xx = max(MOR_correctliq, -22.031637377024)
            MOR_correction = (0.0000156507169104918 * (xx ** 2)) + (0.000689621839324826 * xx) - 0.0392
        else:
            xx = max(MOR_correctliq, -23.6334996273983)
            MOR_correction = (0.00000461020482461793 * (xx ** 2)) + (0.000217910548009675 * xx) - 0.012074621594626

        if refrigerant == "R744":
            MOR_correctionmin = (0.000225755013421421 * MOR_correctliqmin) - 0.00280879370374927
        elif refrigerant == "R744 TC":
            MOR_correctionmin = (0.0000603336117708171 * h_inmin) - 0.0142318718120024
        elif refrigerant in ["R407A", "R449A", "R448A", "R502"]:
            MOR_correctionmin = (0.00000414431651323856 * (MOR_correctliqmin ** 2)) + (0.000381908525139781 * MOR_correctliqmin) - 0.0163450053041212
        elif refrigerant == "R507A":
            MOR_correctionmin = (0.000302619054048837 * MOR_correctliqmin) - 0.00930188913363997
        elif refrigerant == "R22":
            MOR_correctionmin = (0.000108153843367715 * MOR_correctliqmin) - 0.00329248681202757
        elif refrigerant == "R407C":
            xx = max(MOR_correctliqmin, -32.0716410083429)
            MOR_correctionmin = (0.00000420322918839302 * (xx ** 2)) + (0.000269608915211859 * xx) - 0.0134546663857195
        elif refrigerant == "R410A":
            MOR_correctionmin = 0
        elif refrigerant == "R407F":
            xx = max(MOR_correctliqmin, -34.4346433150568)
            MOR_correctionmin = (0.00000347332380289385 * (xx ** 2)) + (0.000239205332540693 * xx) - 0.0121545316131988
        elif refrigerant == "R134a":
            MOR_correctionmin = (0.000195224660107459 * MOR_correctliqmin) - 0.00591757011487048
        elif refrigerant == "R404A":
            xx = max(MOR_correctliqmin, -22.031637377024)
            MOR_correctionmin = (0.0000156507169104918 * (xx ** 2)) + (0.000689621839324826 * xx) - 0.0392
        else:
            xx = max(MOR_correctliqmin, -23.6334996273983)
            MOR_correctionmin = (0.00000461020482461793 * (xx ** 2)) + (0.000217910548009675 * xx) - 0.012074621594626

        if refrigerant in ("R744", "R744 TC"):
            MOR_correction2 = (-0.0000176412848988908 * (evapoil ** 2)) - (0.00164308248808803 * evapoil) - 0.0184308798286039
        elif refrigerant == "R407A":
            MOR_correction2 = (-0.000864076433837511 * evapoil) - 0.0145018190416687
        elif refrigerant == "R449A":
            MOR_correction2 = (-0.000835375233693285 * evapoil) - 0.0138846063856621
        elif refrigerant == "R448A":
            MOR_correction2 = (0.00000171366802431428 * (evapoil ** 2)) - (0.000865528727278154 * evapoil) - 0.0152961902042161
        elif refrigerant == "R502":
            MOR_correction2 = (0.00000484734071020993 * (evapoil ** 2)) - (0.000624822304716683 * evapoil) - 0.0128725684240106
        elif refrigerant == "R507A":
            MOR_correction2 = (-0.000701333343440148 * evapoil) - 0.0114900933623056
        elif refrigerant == "R22":
            MOR_correction2 = (0.00000636798209134899 * (evapoil ** 2)) - (0.000157783204337396 * evapoil) - 0.00575251626397381
        elif refrigerant == "R407C":
            MOR_correction2 = (-0.00000665735727676349 * (evapoil ** 2)) - (0.000894860288947537 * evapoil) - 0.0116054361757929
        elif refrigerant == "R410A":
            MOR_correction2 = (-0.000672268853990701 * evapoil) - 0.0111802230098585
        elif refrigerant == "R407F":
            MOR_correction2 = (0.00000263731418614519 * (evapoil ** 2)) - (0.000683997257738699 * evapoil) - 0.0126005968942147
        elif refrigerant == "R134a":
            MOR_correction2 = (-0.00000823045532174214 * (evapoil ** 2)) - (0.00108063672211041 * evapoil) - 0.0217411206961643
        elif refrigerant == "R404A":
            MOR_correction2 = (0.00000342378568620316 * (evapoil ** 2)) - (0.000329572335134041 * evapoil) - 0.00706087606597149
        else:
            MOR_correction2 = (-0.000711441807827186 * evapoil) - 0.0118194116436425

        if refrigerant in ["R23", "R508B"]:
            if T_evap < -86 or T_evap > -42:
                MORfinal = ""
            else:
                MOR = (1 - MOR_correction) * (1 - MOR_correction2) * MOR_pre
                MORmin = (1 - MOR_correctionmin) * (1 - MOR_correction2) * MOR_premin
                MORfinal = max(MOR, MORmin)
        else:
            if T_evap < -40 or T_evap > 4:
                MORfinal = ""
            else:
                MOR = (1 - MOR_correction) * (1 - MOR_correction2) * MOR_pre
                MORmin = (1 - MOR_correctionmin) * (1 - MOR_correction2) * MOR_premin
                MORfinal = max(MOR, MORmin)

        density_recalc = mass_flow_kg_s / (velocity_m_s * area_m2_local)

        visc = RefrigerantViscosities()
        if refrigerant == "R744 TC":
            viscosity_super = visc.get_viscosity("R744", T_evap - max_penalty + 273.15, superheat_K)
            viscosity_super2a = visc.get_viscosity("R744", T_evap + 273.15, ((superheat_K + 5) / 2))
            viscosity_super2b = visc.get_viscosity("R744", T_evap - max_penalty + 273.15, ((superheat_K + 5) / 2))
            viscosity_super2 = (viscosity_super2a + viscosity_super2b) / 2
            viscosity_5K = visc.get_viscosity("R744", T_evap + 273.15, 5)
        else:
            viscosity_super = visc.get_viscosity(refrigerant, T_evap - max_penalty + 273.15, superheat_K)
            viscosity_super2a = visc.get_viscosity(refrigerant, T_evap + 273.15, ((superheat_K + 5) / 2))
            viscosity_super2b = visc.get_viscosity(refrigerant, T_evap - max_penalty + 273.15, ((superheat_K + 5) / 2))
            viscosity_super2 = (viscosity_super2a + viscosity_super2b) / 2
            viscosity_5K = visc.get_viscosity(refrigerant, T_evap + 273.15, 5)

        viscosity = (viscosity_super + viscosity_5K) / 2
        viscosity_final = (viscosity * velocity1_prop) + (viscosity_super2 * (1 - velocity1_prop))

        reynolds = (density_recalc * velocity_m_sfinal * ID_m_local) / (viscosity_final / 1_000_000)

        eps = 0.00004572 if selected_material in ["Steel SCH40", "Steel SCH80"] else 0.000001524

        if reynolds < 2000:
            f = 64 / reynolds
        else:
            tol = 1e-5
            max_iter = 60
            flo, fhi = 1e-5, 0.1

            def balance(gg):
                s = math.sqrt(gg)
                lhs = 1.0 / s
                rhs = -2.0 * math.log10((eps / (3.7 * ID_m_local)) + 2.51 / (reynolds * s))
                return lhs, rhs

            f = 0.5 * (flo + fhi)
            for _ in range(max_iter):
                f = 0.5 * (flo + fhi)
                lhs, rhs = balance(f)
                if abs(1.0 - lhs / rhs) < tol:
                    break
                if (lhs - rhs) > 0.0:
                    flo = f
                else:
                    fhi = f

        q_kPa = 0.5 * density_recalc * (velocity_m_sfinal ** 2) / 1000.0
        dp_pipe_kPa = f * (L / ID_m_local) * q_kPa
        dp_plf_kPa = q_kPa * PLF

        try:
            K_SRB = float(pipe_row["SRB"])
            K_LRB = float(pipe_row["LRB"])
            K_BALL = float(pipe_row["BALL"])
            K_GLOBE = float(pipe_row["GLOBE"])
        except Exception:
            return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")

        B_SRB = SRB + 0.5 * _45 + 2.0 * ubend + 3.0 * ptrap
        B_LRB = LRB + MAC

        dp_fittings_kPa = q_kPa * (K_SRB * B_SRB + K_LRB * B_LRB)
        dp_valves_kPa = q_kPa * (K_BALL * ball + K_GLOBE * globe)
        dp_total_kPa = dp_pipe_kPa + dp_fittings_kPa + dp_valves_kPa + dp_plf_kPa

        # Optional junction / tee loss applied by network-level logic.
        try:
            dp_total_kPa = dp_total_kPa + float(extra_dp_kPa)
        except Exception:
            # Keep the original dp_total_kPa if extra_dp_kPa is not usable.
            pass

        converter = PressureTemperatureConverter()
        if refrigerant == "R744 TC":
            evappres = converter.temp_to_pressure("R744", T_evap)
            postcirc = evappres - (dp_total_kPa / 100)
            postcirctemp = converter.pressure_to_temp("R744", postcirc)
        else:
            evappres = converter.temp_to_pressure(refrigerant, T_evap)
            postcirc = evappres - (dp_total_kPa / 100)
            postcirctemp = converter.pressure_to_temp(refrigerant, postcirc)

        dt = T_evap - postcirctemp

        mor_num = float("nan") if MORfinal == "" else float(MORfinal)

        dp_collated_kPa = dp_fittings_kPa + dp_plf_kPa + float(extra_dp_kPa)

        maxmass = max(mass_flow_kg_s, mass_flow_kg_smin)

        volflow = maxmass / density_recalc

        # Return extra internals that the network solver needs for junction losses.
        #   - velocity_m_sfinal : final velocity used in the pressure drop calc
        #   - q_kPa             : velocity pressure (kPa)
        #   - ID_m_local        : internal diameter (m)
        return mor_num, float(dt), float(velocity_m_sfinal), float(q_kPa), float(ID_m_local), float(dp_total_kPa), float(dp_pipe_kPa), float(dp_collated_kPa), float(dp_valves_kPa), float(density_recalc), float(maxmass), float(volflow)

    return get_pipe_results

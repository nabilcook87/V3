import streamlit as st
import pandas as pd
from utils.pressure_temp_converter import PressureTemperatureConverter

from utils.network_solver_vb import (
    Circuit,
    solve_network,
    segment_equivalent_length_m,
    worst_case_path_equivalent_length_m,
)
from utils.dry_suction_engine import DrySuctionInputs, make_pipe_row_for_size, make_get_pipe_results
from utils.dry_suction_selectors import select_horizontal_low_dt, select_single_riser  # kept import; not used
from utils.dry_suction_selectors import select_double_riser_exact
from utils.dry_suction_engine import make_ctx
from utils.double_riser import balance_double_riser, compute_double_riser_oil_metrics

def _build_ctx_for_circuit(circ):

    seg_meta = circ.meta or {}

    inputs = DrySuctionInputs(
        refrigerant=refrigerant,
        T_evap=T_evap,
        T_cond=T_cond,
        minliq_temp=minliq_temp,
        superheat_K=superheat_K,
        max_penalty_K=float(max_penalty),
        evap_capacity_kw=circ.duty_kw,

        L=float(seg_meta.get("L",0.0)),
        SRB=float(seg_meta.get("SRB",0)),
        LRB=float(seg_meta.get("LRB",0)),
        bends_45=float(seg_meta.get("_45",0)),
        MAC=float(seg_meta.get("MAC",0)),
        ptrap=float(seg_meta.get("ptrap",0)),
        ubend=float(seg_meta.get("ubend",0)),
        ball=float(seg_meta.get("ball",0)),
        globe=float(seg_meta.get("globe",0)),
        PLF=float(seg_meta.get("PLF",0.0)),

        selected_material=selected_material,
        gc_max_pres=gc_max_pres if refrigerant=="R744 TC" else None,
        gc_min_pres=gc_min_pres if refrigerant=="R744 TC" else None,
    )

    return make_ctx(inputs=inputs,pipe_row_for_size=pipe_row_for_size)

st.set_page_config(page_title="Network — Dry Suction", layout="wide")
st.title("MicroPipe")

col1, col2 = st.columns(2)
with col1:
# -----------------------------
# Global Dry Suction inputs
# -----------------------------
    with st.expander("Dry Suction Inputs", expanded=True):
        c1, c2 = st.columns(2)
    
        with c1:
            refrigerant = st.selectbox(
                "Refrigerant",
                ["R404A", "R134a", "R407F", "R744", "R744 TC", "R410A",
                "R407C", "R507A", "R448A", "R449A", "R22", "R32", "R454A", "R454C", "R455A", "R407A",
                "R290", "R1270", "R600a", "R717", "R1234ze", "R1234yf", "R12", "R11", "R454B", "R450A", "R513A", "R23", "R508B", "R502"],
                index=0,
            )
            T_evap = st.number_input("Evaporating Temperature (°C)", value=-10.0, step=1.0)
            superheat_K = st.number_input("Superheat (K)", value=10.0, step=1.0, min_value=0.0)
            if refrigerant == "R744 TC":
                gc_max_pres = st.number_input("R744 TC: GC max pressure (bar abs)", value=90.0, step=1.0)
            pipe_data = pd.read_csv("data/pipe_pressure_ratings_full.csv")
            if refrigerant == "R717":
                excluded_materials = ["Copper ASTM", " Copper EN12735", "K65 Copper", "Reflok Aluminium"]
                pipe_materials = sorted(m for m in pipe_data["Material"].dropna().unique() if m not in excluded_materials)
            else:
                pipe_materials = sorted(pipe_data["Material"].dropna().unique())
            selected_material = st.selectbox("Pipe Material", pipe_materials)
            dp_standard = st.selectbox(
                "Design Pressure Standard",
                ["BS EN 378", "ASME B31.5 - 2006"],
                index=0,
                key="manual_dp_standard",
            )
    
        with c2:
            T_cond = st.number_input("Max Liquid Temperature (°C)", value=40.0, step=1.0)
            minliq_temp = st.number_input("Min Liquid Temperature (°C)", value=20.0, step=1.0)
            max_penalty = st.number_input("Max Penalty (K)", value=1.0, step=0.5, min_value=0.0)
            if refrigerant == "R744 TC":
                gc_min_pres = st.number_input("R744 TC: GC min pressure (bar abs)", value=75.0, step=1.0)
            circuit = "Suction"
            if refrigerant == "R744 TC" and circuit != "Suction":
                design_temp_c = None
                r744_tc_pressure_bar_g = st.number_input(
                    "R744 Transcritical Design Pressure (bar(g))",
                    min_value=75.0,
                    max_value=150.0,
                    step=5.0,
                    value=120.0,
                    key="manual_r744_tc_pressure",
                )
            else:
                if circuit in ("Suction", "Pumped"):
                    design_temp_c = st.number_input(
                        "Design Temperature (°C)",
                        min_value=0.0,
                        max_value=0.0,
                        value=0.0,
                        step=1.0,
                        key="manual_design_temp_low",
                    )
                else:
                    design_temp_c = st.number_input(
                        "Design Temperature (°C)",
                        min_value=0.0,
                        max_value=0.0,
                        value=0.0,
                        step=1.0,
                        key="manual_design_temp_high",
                    )
                    
                r744_tc_pressure_bar_g = None
            copper_calc = st.selectbox(
                "Copper MWP Calculation Standard",
                ["BS1306", "DKI"],
                index=0,
                key="manual_copper_calc",
            )

# -----------------------------
# Pipe table & material
# -----------------------------

material_df = pipe_data[pipe_data["Material"] == selected_material].copy()
if material_df.empty:
    st.error(f"No rows for material '{selected_material}'.")
    st.stop()

sizes_df = (
    material_df[["Nominal Size (inch)", "Nominal Size (mm)"]]
    .dropna(subset=["Nominal Size (inch)"])
    .assign(**{"Nominal Size (inch)": lambda d: d["Nominal Size (inch)"].astype(str).str.strip()})
    .drop_duplicates(subset=["Nominal Size (inch)"], keep="first")
)

mm_map = {
    str(r["Nominal Size (inch)"]).strip(): float(r["Nominal Size (mm)"]) if pd.notna(r["Nominal Size (mm)"]) else float("inf")
    for _, r in sizes_df.iterrows()
}

pipe_sizes = sorted(mm_map.keys(), key=lambda s: mm_map.get(s, float("inf")))
pipe_row_for_size = make_pipe_row_for_size(material_df)

# -----------------------------
# Network topology editor
# -----------------------------
st.subheader("Network Topology")

DEFAULT_TOPO = pd.DataFrame(
    [
        {"Label": "E1", "Next Label": "C1", "Terminal": True, "Header": False, "mode_override": "Horizontal", "junction_type": 1},
        {"Label": "E2", "Next Label": "C1", "Terminal": True, "Header": False, "mode_override": "Horizontal", "junction_type": 1},
        {"Label": "N1", "Next Label": "C2", "Terminal": False, "Header": False, "mode_override": "Double Riser", "junction_type": 1},
        {"Label": "H1", "Next Label": "", "Terminal": False, "Header": False, "mode_override": "Horizontal", "junction_type": 1},
        {"Label": "E3", "Next Label": "C3", "Terminal": True, "Header": False, "mode_override": "Horizontal", "junction_type": 1},
        {"Label": "E4", "Next Label": "C3", "Terminal": True, "Header": False, "mode_override": "Horizontal", "junction_type": 1},
        {"Label": "N2", "Next Label": "C4", "Terminal": False, "Header": False, "mode_override": "Double Riser", "junction_type": 1},
        {"Label": "C1", "Next Label": "N1", "Terminal": False, "Header": False, "mode_override": "Horizontal", "junction_type": 1},
        {"Label": "C2", "Next Label": "H1", "Terminal": False, "Header": False, "mode_override": "Horizontal", "junction_type": 1},
        {"Label": "C3", "Next Label": "N2", "Terminal": False, "Header": False, "mode_override": "Horizontal", "junction_type": 1},
        {"Label": "C4", "Next Label": "H1", "Terminal": False, "Header": False, "mode_override": "Horizontal", "junction_type": 1},
    ]
)

DEFAULT_SEG = {
    "duty_kw": 0.0,
    "L": 0.1,
    "SRB": 0,
    "LRB": 0,
    "_45": 0,
    "MAC": 0,
    "ptrap": 0,
    "ubend": 0,
    "ball": 0,
    "globe": 0,
    "PLF": 0.0,
    "mode_override": "Horizontal",  # inherit|Horizontal|Single Riser
    "required_oil_override": 100.0,
    # Tee / junction connection type for the *downstream* segment when it joins an upstream main.
    # Mirrors legacy VB: 1 = side (90°), 2 = end (180°).
    "junction_type": 1,
}

DEFAULT_SEG_BY_LABEL = {
    "E1": {**DEFAULT_SEG, "duty_kw": 2.5, "L": 1.0, "SRB": 3, "LRB": 0, "ptrap": 0, "ball": 1},
    "E2": {**DEFAULT_SEG, "duty_kw": 1.8, "L": 1.0, "SRB": 3, "LRB": 0, "ptrap": 0, "ball": 1},
    "N1": {**DEFAULT_SEG, "L": 4.0, "SRB": 0, "LRB": 0, "ptrap": 2, "ball": 0, "required_oil_override": 20.0},
    "H1": {**DEFAULT_SEG, "L": 20.0, "SRB": 3, "LRB": 5, "ptrap": 0, "ball": 1},
    "E3": {**DEFAULT_SEG, "duty_kw": 5.7, "L": 1.0, "SRB": 3, "LRB": 0, "ptrap": 0, "ball": 1},
    "E4": {**DEFAULT_SEG, "duty_kw": 8.4, "L": 1.0, "SRB": 3, "LRB": 0, "ptrap": 0, "ball": 1},
    "N2": {**DEFAULT_SEG, "L": 4.0, "SRB": 0, "LRB": 0, "ptrap": 2, "ball": 0, "required_oil_override": 20.0},
    "C1": {**DEFAULT_SEG},
    "C2": {**DEFAULT_SEG},
    "C3": {**DEFAULT_SEG},
    "C4": {**DEFAULT_SEG},
}

def _strongest_gauge(material_df, size_inch: str):
    rows = material_df[material_df["Nominal Size (inch)"].astype(str).str.strip() == str(size_inch).strip()]
    if rows.empty or "Gauge" not in rows.columns or rows["Gauge"].isna().all():
        return None
    return max(rows["Gauge"].dropna().unique())

def _kfactor_by_size(material_df, size_inch: str, col: str) -> float:
    """
    Return a k-factor (e.g. SRB/LRB) for a nominal size from the pipe CSV.

    IMPORTANT:
    - VB's SRBendKfactors() is size-based.
    - Your CSV may have multiple rows per size (different Gauge / ratings).
    - For junction modelling we must NOT let gauge choice change k-factors.
    """
    try:
        s = str(size_inch).strip()
        rows = material_df[material_df["Nominal Size (inch)"].astype(str).str.strip() == s]
        if rows.empty or col not in rows.columns:
            return 0.0
        # Pick the first non-null value for this size.
        v = rows[col].dropna()
        if v.empty:
            return 0.0
        return float(v.iloc[0])
    except Exception:
        return 0.0

def _junction_f3(*, ratio: float, is_enlargement: bool) -> float:
    """Legacy VB polynomial for enlargement/reduction loss factor (F3)."""
    A0, A1, A2, A3, A4, A5 = (
        1.00428571427386,
        -2.68809523786478,
        9.10714285562815,
        -30.9523809480179,
        46.4285714229368,
        -23.8095238068658,
    )
    aa0, aa1, aa2, aa3, aa4, aa5 = (
        0.572857142861904,
        9.40476189546585e-02,
        -4.55357142795777,
        15.476190744165,
        -23.2142857119884,
        11.904761903676,
    )

    r = float(ratio)
    if (r > 0.8) and (r <= 1.0):
        return 0.05
    if r < 0.1:
        return 0.9 if is_enlargement else 0.6
    if (r <= 0.8) and (r >= 0.1):
        if is_enlargement:
            return A0 + (A1 * r) + (A2 * r**2) + (A3 * r**3) + (A4 * r**4) + (A5 * r**5)
        return aa0 + (aa1 * r) + (aa2 * r**2) + (aa3 * r**3) + (aa4 * r**4) + (aa5 * r**5)
    return 0.0

if "topology_df" not in st.session_state:
    st.session_state.topology_df = DEFAULT_TOPO.copy()

if "segment_meta" not in st.session_state:
    st.session_state.segment_meta = {}

topo = st.data_editor(
    st.session_state.topology_df,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Label": st.column_config.TextColumn(required=True),
        "Next Label": st.column_config.TextColumn(help="Upstream circuit label (blank means root)"),
        "Terminal": st.column_config.CheckboxColumn(),
        "Header": st.column_config.CheckboxColumn(),
        "mode_override": st.column_config.SelectboxColumn(
            "Orientation",
            options=["Horizontal","Single Riser","Double Riser"],
            required=True,
        ),
        "junction_type": st.column_config.SelectboxColumn(
            "Junction Type",
            options=[1, 2],
            required=True,
            help="1=side (90°), 2=end (180°). Used only when the upstream segment has >1 incoming branch.",
        ),
    },
)
topo["Label"] = topo["Label"].astype(str).str.strip()
topo["Next Label"] = topo["Next Label"].astype(str).str.strip()
if "mode_override" not in topo.columns:
    topo["mode_override"] = "Horizontal"
if "junction_type" not in topo.columns:
    topo["junction_type"] = 1
st.session_state.topology_df = topo

# Initialize meta for ALL circuits (loads + pipes + headers)
all_labels_list = [x for x in topo["Label"].astype(str).str.strip().tolist() if x]
for lab in all_labels_list:
    if lab not in st.session_state.segment_meta:
        st.session_state.segment_meta[lab] = DEFAULT_SEG_BY_LABEL.get(lab, DEFAULT_SEG).copy()

# drop removed
all_labels_set = set(all_labels_list)
for lab in list(st.session_state.segment_meta.keys()):
    if lab not in all_labels_set:
        del st.session_state.segment_meta[lab]

with col2:
    # -----------------------------
    # Circuit detail editor (loads + pipes)
    # -----------------------------
    with st.expander("Pipe Details", expanded=True):
        if not all_labels_list:
            st.info("Add at least one circuit.")
            st.stop()

        colu1, colu2 = st.columns(2)
        with colu1:
            selected_label = st.selectbox("Select Circuit", all_labels_list, index=0)
            seg = st.session_state.segment_meta.get(selected_label, DEFAULT_SEG.copy())
        with colu2:
            copy_from = st.selectbox("Copy From", ["(none)"] + [l for l in all_labels_list if l != selected_label], index=0)
        if st.button("Copy Into Selected") and copy_from != "(none)":
            st.session_state.segment_meta[selected_label] = {**st.session_state.segment_meta[copy_from]}
            seg = st.session_state.segment_meta[selected_label]
            st.success(f"Copied {copy_from} → {selected_label}")
        
        with st.form("seg_form"):
            a, b, c, d = st.columns(4)
            selected_topo_row = topo[topo["Label"] == selected_label]
            is_terminal_selected = False if selected_topo_row.empty else bool(selected_topo_row.iloc[0]["Terminal"])
            is_oil_selected = False if selected_topo_row.empty else selected_topo_row.iloc[0]["mode_override"] in ["Single Riser", "Double Riser"]
            with a:
                duty_kw = st.number_input(
                    "Duty (kW)",
                    value=float(seg.get("duty_kw", 0.0)),
                    min_value=0.0,
                    step=0.1,
                    disabled=not is_terminal_selected,
                )
                SRB = st.number_input("Short Radius Bends", value=int(seg.get("SRB", 0)), min_value=0, step=1)
                ptrap = st.number_input("P-Traps", value=int(seg.get("ptrap", 0)), min_value=0, step=1)
            with b:
                L = st.number_input("Length (m)", value=float(seg.get("L", 0.0)), min_value=0.0, step=0.5)
                LRB = st.number_input("Long Radius Bends", value=int(seg.get("LRB", 0)), min_value=0, step=1)
                ubend = st.number_input("U-Bends", value=int(seg.get("ubend", 0)), min_value=0, step=1)
            with c:
                req_val = st.number_input(
                    "Required Oil Duty (%)",
                    value=float(seg.get("required_oil_override") if seg.get("required_oil_override") is not None else 100.0),
                    min_value=0.0,
                    step=1.0,
                    disabled=not is_oil_selected,
                )
                _45 = st.number_input("45° Bends", value=int(seg.get("_45", 0)), min_value=0, step=1)
                ball = st.number_input("Ball Valves", value=int(seg.get("ball", 0)), min_value=0, step=1)
            with d:
                PLF = st.number_input("Pressure Loss Factors", value=float(seg.get("PLF", 0.0)), min_value=0.0, step=0.1)
                MAC = st.number_input("Machine Bends", value=int(seg.get("MAC", 0)), min_value=0, step=1)
                globe = st.number_input("Globe Valves", value=int(seg.get("globe", 0)), min_value=0, step=1)
            save = st.form_submit_button("Save")
        
        if save:
            st.session_state.segment_meta[selected_label] = {
                "duty_kw": float(duty_kw), "L": float(L), "SRB": int(SRB), "LRB": int(LRB), "_45": int(_45), "MAC": int(MAC),
                "ptrap": int(ptrap), "ubend": int(ubend), "ball": int(ball), "globe": int(globe), "PLF": float(PLF),
                "required_oil_override": float(req_val),
            }
            st.success(f"Saved {selected_label}")

# -----------------------------
# Run
# -----------------------------
if st.button("Calculate"):
    circuits = []
    for _, r in topo.iterrows():
        label = str(r["Label"]).strip()
        if not label:
            continue

        is_load = bool(r.get("Terminal", False))

        # IMPORTANT: meta is now applied to loads too (terminal pipe geometry)
        meta = st.session_state.segment_meta.get(label, DEFAULT_SEG.copy()).copy()

        # Override per-pipe meta with topology table choices
        meta["mode_override"] = str(r.get("mode_override", meta.get("mode_override", "Horizontal"))).strip() or "Horizontal"
        meta["junction_type"] = int(r.get("junction_type", meta.get("junction_type", 1)) or 1)

        circuits.append(
            Circuit(
                label=label,
                next_label=str(r.get("Next Label", "")).strip(),
                is_load=is_load,
                is_header=bool(r.get("Header", False)),
                duty_kw=float(meta.get("duty_kw", 0.0) or 0.0) if is_load else 0.0,
                meta=meta,
            )
        )

    try:
        net = solve_network(circuits)
    except Exception as e:
        st.error(f"Network error: {e}")
        st.stop()

    # ------------------------------------------------------------
    # VB-style path-budget apportionment:
    #   Treat max_penalty (K) as a *path-level* budget.
    #   Convert it to allowable K per equivalent meter using worst path MPL.
    #   NOTE: loads now contribute to MPL because they can have geometry too.
    # ------------------------------------------------------------
    MPL_m, path_len_list = worst_case_path_equivalent_length_m(circuits, net["paths"])
    if MPL_m > 0:
        allowable_penalty_per_m = float(max_penalty) / float(MPL_m)
    else:
        allowable_penalty_per_m = 0.0

    # -----------------------------
    # Helpers for optimisation
    # -----------------------------
    size_to_idx = {s: i for i, s in enumerate(pipe_sizes)}

    def _extract_dt_mor(res):
        """
        Supports engine returning (MORfinal, dt, velocity_m_sfinal, dp_total_kPa, pipe, fittings, valves).
        Returns: (dt, mor, vel, dp_total_kPa, pipe, fittings, valves)
        """
        dt = None
        mor = None
        vel = None
        dp_total_kPa = None
        dp_pipe_kPa = None
        dp_collated_kPa = None
        dp_valves_kPa = None

        if isinstance(res, dict):
            for k in ("dt", "DT", "Penalty", "penalty"):
                if k in res and res[k] is not None:
                    dt = float(res[k])
                    break
            for k in ("MORfinal", "mor", "MOR", "oil_return_pct", "oil"):
                if k in res and res[k] is not None:
                    mor = float(res[k])
                    break
            for k in ("velocity_m_sfinal"):
                if k in res and res[k] is not None:
                    vel = float(res[k])
                    break
            for k in ("dp_total_kPa"):
                if k in res and res[k] is not None:
                    dp_total_kPa = float(res[k])
                    break
            for k in ("dp_pipe_kPa"):
                if k in res and res[k] is not None:
                    dp_pipe_kPa = float(res[k])
                    break
            for k in ("dp_collated_kPa"):
                if k in res and res[k] is not None:
                    dp_collated_kPa = float(res[k])
                    break
            for k in ("dp_valves_kPa"):
                if k in res and res[k] is not None:
                    dp_valves_kPa = float(res[k])
                    break
            return dt, mor, vel, dp_total_kPa, dp_pipe_kPa, dp_collated_kPa, dp_valves_kPa

        if isinstance(res, (tuple, list)):
            if len(res) >= 9:
                try:
                    mor = float(res[0]) if res[0] is not None else None
                    dt = float(res[1]) if res[1] is not None else None
                    vel = float(res[2]) if res[2] is not None else None
                    dp_total_kPa = float(res[5]) if res[5] is not None else None
                    dp_pipe_kPa = float(res[6]) if res[6] is not None else None
                    dp_collated_kPa = float(res[7]) if res[7] is not None else None
                    dp_valves_kPa = float(res[8]) if res[8] is not None else None
                    return dt, mor, vel, dp_total_kPa, dp_pipe_kPa, dp_collated_kPa, dp_valves_kPa
                except Exception:
                    pass
            if len(res) >= 7:
                try:
                    mor = float(res[0]) if res[0] is not None else None
                    dt = float(res[1]) if res[1] is not None else None
                    vel = float(res[2]) if res[2] is not None else None
                    dp_total_kPa = float(res[3]) if res[3] is not None else None
                    dp_pipe_kPa = float(res[4]) if res[4] is not None else None
                    dp_collated_kPa = float(res[5]) if res[5] is not None else None
                    dp_valves_kPa = float(res[6]) if res[6] is not None else None
                    return dt, mor, vel, dp_total_kPa, dp_pipe_kPa, dp_collated_kPa, dp_valves_kPa
                except Exception:
                    pass
            if len(res) >= 2:
                try:
                    mor = float(res[0]) if res[0] is not None else None
                    dt = float(res[1]) if res[1] is not None else None
                    return dt, mor, None
                except Exception:
                    pass

        if hasattr(res, "dt"):
            try:
                dt = float(getattr(res, "dt"))
            except Exception:
                pass
        if hasattr(res, "MORfinal"):
            try:
                mor = float(getattr(res, "MORfinal"))
            except Exception:
                pass
        if hasattr(res, "velocity_m_sfinal"):
            try:
                vel = float(getattr(res, "velocity_m_sfinal"))
            except Exception:
                pass
        if hasattr(res, "dp_total_kPa"):
            try:
                dp_total_kPa = float(getattr(res, "dp_total_kPa"))
            except Exception:
                pass
        if hasattr(res, "dp_pipe_kPa"):
            try:
                dp_pipe_kPa = float(getattr(res, "dp_pipe_kPa"))
            except Exception:
                pass
        if hasattr(res, "dp_collated_kPa"):
            try:
                dp_collated_kPa = float(getattr(res, "dp_collated_kPa"))
            except Exception:
                pass
        if hasattr(res, "dp_valves_kPa"):
            try:
                dp_valves_kPa = float(getattr(res, "dp_valves_kPa"))
            except Exception:
                pass

        return dt, mor, vel, dp_total_kPa, dp_pipe_kPa, dp_collated_kPa, dp_valves_kPa

    def _build_get_pipe_results_for_circuit(circ, max_penalty_for_inputs):
        seg_meta = circ.meta or {}
        ds_inputs_local = DrySuctionInputs(
            refrigerant=refrigerant,
            T_evap=T_evap,
            T_cond=T_cond,
            minliq_temp=minliq_temp,
            superheat_K=superheat_K,
            max_penalty_K=float(max_penalty_for_inputs),
            evap_capacity_kw=circ.duty_kw,

            L=float(seg_meta.get("L", 0.0)),
            SRB=float(seg_meta.get("SRB", 0)),
            LRB=float(seg_meta.get("LRB", 0)),
            bends_45=float(seg_meta.get("_45", 0)),
            MAC=float(seg_meta.get("MAC", 0)),
            ptrap=float(seg_meta.get("ptrap", 0)),
            ubend=float(seg_meta.get("ubend", 0)),
            ball=float(seg_meta.get("ball", 0)),
            globe=float(seg_meta.get("globe", 0)),
            PLF=float(seg_meta.get("PLF", 0.0)),

            selected_material=selected_material,
            gc_max_pres=gc_max_pres if refrigerant == "R744 TC" else None,
            gc_min_pres=gc_min_pres if refrigerant == "R744 TC" else None,
        )
        return make_get_pipe_results(inputs=ds_inputs_local, pipe_row_for_size=pipe_row_for_size)


    def _build_double_riser_callbacks_for_circuit(circ, extra_dp_kPa: float = 0.0):
        seg_meta = circ.meta or {}
        ds_inputs_local = DrySuctionInputs(
            refrigerant=refrigerant,
            T_evap=T_evap,
            T_cond=T_cond,
            minliq_temp=minliq_temp,
            superheat_K=superheat_K,
            max_penalty_K=float(max_penalty),
            evap_capacity_kw=circ.duty_kw,

            L=float(seg_meta.get("L", 0.0)),
            SRB=float(seg_meta.get("SRB", 0)),
            LRB=float(seg_meta.get("LRB", 0)),
            bends_45=float(seg_meta.get("_45", 0)),
            MAC=float(seg_meta.get("MAC", 0)),
            ptrap=float(seg_meta.get("ptrap", 0)),
            ubend=float(seg_meta.get("ubend", 0)),
            ball=float(seg_meta.get("ball", 0)),
            globe=float(seg_meta.get("globe", 0)),
            PLF=float(seg_meta.get("PLF", 0.0)),

            selected_material=selected_material,
            gc_max_pres=gc_max_pres if refrigerant == "R744 TC" else None,
            gc_min_pres=gc_min_pres if refrigerant == "R744 TC" else None,
        )

        ctx_local = make_ctx(inputs=ds_inputs_local, pipe_row_for_size=pipe_row_for_size)
        get_pipe_results_local = make_get_pipe_results(inputs=ds_inputs_local, pipe_row_for_size=pipe_row_for_size)

        from utils.refrigerant_densities import RefrigerantDensities
        from utils.refrigerant_properties import RefrigerantProperties
        from utils.supercompliq_co2 import RefrigerantProps

        props = RefrigerantProperties()
        props_sup = RefrigerantProps()
        dens = RefrigerantDensities()

        if refrigerant == "R744 TC":
            density_super_foroil = dens.get_density("R744", T_evap + 273.15, min(max(superheat_K, 5), 30))
            density_sat = props.get_properties("R744", T_evap)["density_vapor"]
        else:
            density_super_foroil = dens.get_density(refrigerant, T_evap + 273.15, min(max(superheat_K, 5), 30))
            density_sat = props.get_properties(refrigerant, T_evap)["density_vapor"]

        density_foroil = (density_super_foroil + density_sat) / 2

        if refrigerant == "R744 TC":
            h_in = props_sup.get_enthalpy_sup(gc_max_pres, T_cond)

            if gc_min_pres is None:
                raise ValueError("R744 TC requires GC min pressure for double riser.")
            if gc_min_pres >= 73.8:
                h_inmin = props_sup.get_enthalpy_sup(gc_min_pres, minliq_temp)
            elif gc_min_pres <= 72.13:
                h_inmin = props.get_properties("R744", minliq_temp)["enthalpy_liquid2"]
            else:
                raise ValueError("Disallowed CO2 TC region")

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

        mass_flow_kg_s = circ.duty_kw / delta_h if delta_h > 0 else 0.01
        mass_flow_kg_smin = circ.duty_kw / delta_hmin if delta_hmin > 0 else 0.01
        M_total = max(mass_flow_kg_s, mass_flow_kg_smin)

        mass_flow_foroil = circ.duty_kw / delta_h_foroil if delta_h_foroil > 0 else 0.01
        mass_flow_foroilmin = circ.duty_kw / delta_h_foroilmin if delta_h_foroilmin > 0 else 0.01

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

        if refrigerant in ["R23", "R508B"]:
            MOR_correctliq = T_cond + 47.03
            MOR_correctliqmin = minliq_temp + 47.03
            evapoil = T_evap + 46.14
        else:
            MOR_correctliq = T_cond
            MOR_correctliqmin = minliq_temp
            evapoil = T_evap

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

        support = {
            "ctx": ctx_local,
            "M_total": M_total,
            "density_foroil": density_foroil,
            "oil_density": oil_density,
            "jg_half": jg_half,
            "mass_flow_foroil": mass_flow_foroil,
            "mass_flow_foroilmin": mass_flow_foroilmin,
            "MOR_correction": MOR_correction,
            "MOR_correctionmin": MOR_correctionmin,
            "MOR_correction2": MOR_correction2,
        }

        mor_cache = {}
        pair_cache = {}

        def _double_riser_extra_dt_dp(size_large: str, extra_dp_apply_kPa: float):
            """
            Approximate VB-style junction addition for a double riser by applying the
            external junction dp to the large-leg single-pipe engine and using the
            resulting incremental ΔT/ΔP on top of the balanced double-riser result.

            This keeps the sizing loop junction-coupled without changing oil-return
            calculations, which in VB depend on riser hydraulics rather than the tee loss.
            """
            try:
                ex = float(extra_dp_apply_kPa or 0.0)
            except Exception:
                ex = 0.0
            if ex <= 0:
                return 0.0, 0.0

            g_large = _strongest_gauge(material_df, size_large)
            try:
                base_res = get_pipe_results_local(size_large, g_large, 0.0)
                extra_res = get_pipe_results_local(size_large, g_large, ex)
                dt0, _, _, dp0, _, _, _ = _extract_dt_mor(base_res)
                dt1, _, _, dp1, _, _, _ = _extract_dt_mor(extra_res)
                dt_delta = 0.0 if (dt0 is None or dt1 is None) else max(0.0, float(dt1) - float(dt0))
                dp_delta = ex if dp1 is None else max(0.0, float(dp1) - float(dp0 or 0.0))
                return dt_delta, dp_delta
            except Exception:
                return 0.0, ex

        def MOR_full_cached2(size):
            if size not in mor_cache:
                gauge = _strongest_gauge(material_df, size)
                mor_cache[size] = get_pipe_results_local(size, gauge)[0]
            return mor_cache[size]

        def eval_pair_cached(size_small, size_large):
            key = (size_small, size_large, float(extra_dp_kPa or 0.0))
            if key not in pair_cache:
                gauge_small = _strongest_gauge(material_df, size_small)
                gauge_large = _strongest_gauge(material_df, size_large)
                dr = balance_double_riser(
                    size_small=size_small,
                    size_large=size_large,
                    M_total_kg_s=support["M_total"],
                    ctx=support["ctx"],
                    gauge_small=gauge_small,
                    gauge_large=gauge_large,
                )
                dt_delta_k, dp_delta_kpa = _double_riser_extra_dt_dp(size_large=size_large, extra_dp_apply_kPa=extra_dp_kPa)
                try:
                    dr.DT_K = float(dr.DT_K) + float(dt_delta_k)
                except Exception:
                    pass
                try:
                    dr.DP_kPa = float(dr.DP_kPa) + float(dp_delta_kpa)
                except Exception:
                    pass
                try:
                    dr.dp_fit = float(dr.dp_fit) + float(dp_delta_kpa)
                except Exception:
                    pass

                MOR_full, MOR_large, _, _ = compute_double_riser_oil_metrics(
                    dr=dr,
                    refrigerant=refrigerant,
                    T_evap=T_evap,
                    density_foroil=support["density_foroil"],
                    oil_density=support["oil_density"],
                    jg_half=support["jg_half"],
                    mass_flow_foroil=support["mass_flow_foroil"],
                    mass_flow_foroilmin=support["mass_flow_foroilmin"],
                    MOR_correction=support["MOR_correction"],
                    MOR_correctionmin=support["MOR_correctionmin"],
                    MOR_correction2=support["MOR_correction2"],
                )
                pair_cache[key] = (dr, MOR_full, MOR_large)
            return pair_cache[key]

        return MOR_full_cached2, eval_pair_cached

    # -----------------------------
    # Junction / tee loss helpers
    # -----------------------------
    def _compute_junction_extra_dp(*, results_local):
        """
        Return extra_dp_by_label (kPa) to be applied to DOWNSTREAM segments.
    
        VB-equivalent counting:
          - Apply junction (F3 + optional JF + optional SRB) ONCE per physical down->up connection.
          - Do NOT sum the same junction multiple times just because the segment appears on multiple paths.
        """
        extra = {c.label: 0.0 for c in circuits}
    
        # Build quick lookup by label (safer than indexing by path indices)
        circ_by_label = {c.label: c for c in circuits}
    
        for down_c in circuits:
            up_label = (down_c.next_label or "").strip()
            if not up_label:
                continue
            up_c = circ_by_label.get(up_label)
            if up_c is None:
                continue
    
            r_down = results_local.get(down_c.label)
            r_up = results_local.get(up_c.label)
            if not r_down or not r_up:
                continue
    
            d_down = r_down.get("ID_m")
            d_up = r_up.get("ID_m")
            if d_down is None or d_up is None:
                continue
            try:
                d_down = float(d_down)
                d_up = float(d_up)
            except Exception:
                continue
            if d_down <= 0 or d_up <= 0:
                continue
    
            # Always compute F3 and always use VP from the SMALLER diameter side (VB dry suction behaviour).
            try:
                if d_up > d_down:
                    # enlargement along flow (small=downstream, big=upstream)
                    ratio = (d_down / d_up) ** 2
                    vp_kPa = float(r_down.get("q_kPa") or 0.0)  # VP from smaller (downstream) side
                    f3 = _junction_f3(ratio=ratio, is_enlargement=True)
            
                elif d_up < d_down:
                    # reduction along flow (big=downstream, small=upstream)
                    ratio = (d_up / d_down) ** 2
                    vp_kPa = float(r_up.get("q_kPa") or 0.0)    # VP from smaller (upstream) side
                    f3 = _junction_f3(ratio=ratio, is_enlargement=False)
            
                else:
                    # equal diameters: VB sets F3 = 0 (VP side doesn't matter when K=0)
                    vp_kPa = float(r_down.get("q_kPa") or 0.0)
                    f3 = 0.0
            except Exception:
                continue
    
            # Add JF (+SRB) for real merges, and also for the VB suction special case
            # where a single riser feeds directly into a double riser.
            is_merge = int(up_c.incoming_count or 0) > 1
            special_single_to_double = (
                str((down_c.meta or {}).get("mode_override", "")).strip() == "Single Riser"
                and str((up_c.meta or {}).get("mode_override", "")).strip() == "Double Riser"
            )
    
            jf = 0.0
            k_srb = 0.0
            if is_merge or special_single_to_double:
                jt = int((down_c.meta or {}).get("junction_type", 1) or 1)
                if special_single_to_double and not is_merge:
                    jt = 1
                jf = 0.5 if jt == 1 else 0.2
                if jt == 1:
                    size_for_k = r_down.get("pipe_large") if r_down.get("double_riser") and r_down.get("pipe_large") else r_down["selected_size_inch"]
                    k_srb = _kfactor_by_size(material_df, size_for_k, "SRB")
    
            jpd_kPa = (float(f3) + float(jf) + float(k_srb)) * float(vp_kPa)
    
            if jpd_kPa > 0:
                extra[down_c.label] = float(extra.get(down_c.label, 0.0) or 0.0) + float(jpd_kPa)
    
        return extra

    def _recompute_all_with_junction_inplace():
        """Recompute q/ID then junction dp, then dt/MOR/vel/dp_total_kPa/pipes/fittings/valves for all segments in `results`."""
        # Pass 1: compute q_kPa and ID_m with extra_dp = 0 (these are used to calculate tee losses)
        for c in circuits:
            r = results.get(c.label)
            if not r or not r.get("selected_size_inch"):
                continue
            if r.get("double_riser"):
                try:
                    dr0, _, _ = _build_double_riser_callbacks_for_circuit(c, 0.0)[1](r["pipe_small"], r["pipe_large"])
                    r["q_kPa"] = 0.5 * (dr0.large_result.mass_flow_kg_s / (dr0.large_result.velocity_m_s * dr0.large_result.area_m2)) * (dr0.large_result.velocity_m_s ** 2) / 1000.0 if dr0.large_result.velocity_m_s > 0 else None
                    r["ID_m"] = dr0.large_result.ID_m
                except Exception:
                    continue
                continue
            g = _strongest_gauge(material_df, r["selected_size_inch"])
            try:
                mor_i, dt_i, vel_i, q_i, id_i, dp_total_kPa_i, dp_pipe_kPa_i, dp_collated_kPa_i, dp_valves_kPa_i = r["get_pipe_results"](r["selected_size_inch"], g, 0.0)[:9]
                
            except Exception:
                continue
            r["gauge"] = g
            r["q_kPa"] = None if q_i is None else float(q_i)
            r["ID_m"] = None if id_i is None else float(id_i)

        # Pass 2: compute extra junction dp (kPa) per downstream segment
        extra_dp = _compute_junction_extra_dp(results_local=results)

        # Pass 3: recompute dt/MOR/vel/dp_total_kPa/pipes/fittings/valves applying extra dp
        for c in circuits:
            r = results.get(c.label)
            if not r or not r.get("selected_size_inch"):
                continue
            ex = float(extra_dp.get(c.label, 0.0) or 0.0)
            if r.get("double_riser"):
                try:
                    _, eval_pair_cached_ex = _build_double_riser_callbacks_for_circuit(c, ex)
                    dr_ex, MOR_full_ex, MOR_large_ex = eval_pair_cached_ex(r["pipe_small"], r["pipe_large"])
                    r["dt"] = None if dr_ex.DT_K is None else float(dr_ex.DT_K)
                    r["velocity_m_sfinal"] = None if dr_ex.large_result.velocity_m_s is None else float(dr_ex.large_result.velocity_m_s)
                    r["dp_total_kPa"] = None if dr_ex.DP_kPa is None else float(dr_ex.DP_kPa)
                    r["dp_pipe_kPa"] = None if dr_ex.dp_pipe is None else float(dr_ex.dp_pipe)
                    r["dp_collated_kPa"] = None if dr_ex.dp_fit is None else float(dr_ex.dp_fit)
                    r["dp_valves_kPa"] = None if dr_ex.dp_valve is None else float(dr_ex.dp_valve)
                    r["MORfinal"] = None if MOR_full_ex is None else float(MOR_full_ex)
                    r["MOR_large"] = None if MOR_large_ex is None else float(MOR_large_ex)
                    r["ID_m"] = None if dr_ex.large_result.ID_m is None else float(dr_ex.large_result.ID_m)
                    r["q_kPa"] = 0.5 * (dr_ex.large_result.mass_flow_kg_s / (dr_ex.large_result.velocity_m_s * dr_ex.large_result.area_m2)) * (dr_ex.large_result.velocity_m_s ** 2) / 1000.0 if dr_ex.large_result.velocity_m_s > 0 else None
                except Exception:
                    continue
                continue
            g = r.get("gauge")
            if g is None:
                g = _strongest_gauge(material_df, r["selected_size_inch"])
                r["gauge"] = g
            try:
                mor_i, dt_i, vel_i, q_i, id_i, dp_total_kPa_i, dp_pipe_kPa_i, dp_collated_kPa_i, dp_valves_kPa_i, dens_i, maxmass_i, volflow_i = r["get_pipe_results"](r["selected_size_inch"], g, ex)
            except Exception:
                continue
            r["dt"] = None if dt_i is None else float(dt_i)
            r["velocity_m_sfinal"] = None if vel_i is None else float(vel_i)
            r["dp_total_kPa"] = None if dp_total_kPa_i is None else float(dp_total_kPa_i)
            r["dp_pipe_kPa"] = None if dp_pipe_kPa_i is None else float(dp_pipe_kPa_i)
            r["dp_collated_kPa"] = None if dp_collated_kPa_i is None else float(dp_collated_kPa_i)
            r["dp_valves_kPa"] = None if dp_valves_kPa_i is None else float(dp_valves_kPa_i)
            r["MORfinal"] = None if mor_i is None else float(mor_i)
            r["density_recalc"] = None if dens_i is None else float(dens_i)
            r["maxmass"] = None if maxmass_i is None else float(maxmass_i)
            r["volflow"] = None if volflow_i is None else float(volflow_i)

        return extra_dp

    def _snapshot_runtime():
        keys = ("selected_size_inch", "dt", "MORfinal", "velocity_m_sfinal", "dp_total_kPa", "dp_pipe_kPa", "dp_collated_kPa", "dp_valves_kPa", "gauge", "q_kPa", "ID_m", "error", "density_recalc", "maxmass", "volflow")
        return {lab: {k: results[lab].get(k) for k in keys} for lab in results}

    def _restore_runtime(snap):
        for lab, d in snap.items():
            for k, v in d.items():
                results[lab][k] = v

    def _simulate_size_change(lab: str, new_size_inch: str):
        """Temporarily set `lab` to `new_size_inch`, recompute junction-coupled dt/MOR/vel/dp_total_kPa/pipes/fittings/valves, then restore."""
        snap = _snapshot_runtime()
        results[lab]["selected_size_inch"] = new_size_inch
        results[lab]["error"] = None
        _recompute_all_with_junction_inplace()
        dt_v = results[lab].get("dt")
        mor_v = results[lab].get("MORfinal")
        vel_v = results[lab].get("velocity_m_sfinal")
        dp_total_kPa_v = results[lab].get("dp_total_kPa")
        dp_pipe_kPa_v = results[lab].get("dp_pipe_kPa")
        dp_collated_kPa_v = results[lab].get("dp_collated_kPa")
        dp_valves_kPa_v = results[lab].get("dp_valves_kPa")

        dt_by_label_local = {l: (None if results[l]["error"] else results[l]["dt"]) for l in results}
        worst_dt_local, worst_path_labels_local, _ = _path_total_dt_K(net["paths"], dt_by_label_local)

        risers_ok_local = _all_risers_meet_mor()

        _restore_runtime(snap)
        return dt_v, mor_v, vel_v, dp_total_kPa_v, dp_pipe_kPa_v, dp_collated_kPa_v, dp_valves_kPa_v, worst_dt_local, worst_path_labels_local, risers_ok_local

    def _path_total_dt_K(paths, dt_by_label):
        per_path = []
        worst_dt = None
        worst_labels = None
        for p in paths:
            labels = [circuits[i].label for i in p]
            dts = [dt_by_label.get(l) for l in labels]
            if any(v is None for v in dts):
                total = None
            else:
                total = float(sum(dts))
            per_path.append((labels, total))
            if total is not None:
                if worst_dt is None or total > worst_dt:
                    worst_dt = total
                    worst_labels = labels
        return worst_dt, worst_labels, per_path

    def _compute_ratio(lab):
        r = results[lab]
        if r["error"] or r["dt"] is None:
            return None
        denom = r["max_penalty_seg_K"]
        if denom is None or float(denom) <= 0:
            return None
        return float(r["dt"]) / float(denom)

    def _eval_dt_mor_for_size(lab, size_inch):
        r = results[lab]
        best_gauge = _strongest_gauge(material_df, size_inch)
        res = r["get_pipe_results"](size_inch, best_gauge)
        dt, mor, vel, dp_total_kPa, dp_pipe_kPa, dp_collated_kPa, dp_valves_kPa = _extract_dt_mor(res)
        return dt, mor, vel, dp_total_kPa, dp_pipe_kPa, dp_collated_kPa, dp_valves_kPa

    # === CHANGED: MOR is now treated as a MAXIMUM (<= required) ===
    def _all_risers_meet_mor():
        for lab, r in results.items():
            if r["error"]:
                continue
            if r["mode_used"] == "Single Riser":
                req = r["req_oil_used"]
                if req is None:
                    continue
                if r["MORfinal"] is None or float(r["MORfinal"]) > float(req):
                    return False
            elif r["mode_used"] == "Double Riser":
                req = r["req_oil_used"]
                if req is None:
                    continue
                if r["MORfinal"] is None or float(r["MORfinal"]) > float(req):
                    return False
                if r.get("MOR_large") is None or float(r["MOR_large"]) > 100.0:
                    return False
        return True

    # === CHANGED: local riser selector enforces MOR <= required (and dt <= max_penalty_seg) ===
    def _select_single_riser_max_mor(
        pipe_sizes_local,
        get_pipe_results_local,
        max_mor_pct,
        max_penalty_K,
    ):
        best_size = None
        best_mor = None
        best_dt = None
        best_vel = None
        best_dp_total_kPa = None
        best_dp_pipe_kPa = None
        best_dp_collated_kPa = None
        best_dp_valves_kPa = None

        for s in pipe_sizes_local:
            best_gauge = _strongest_gauge(material_df, s)
            res = get_pipe_results_local(s, best_gauge)
            dt, mor, vel, dp_total_kPa, dp_pipe_kPa, dp_collated_kPa, dp_valves_kPa = _extract_dt_mor(res)
            if dt is None or mor is None:
                continue

            if float(mor) <= float(max_mor_pct) and float(dt) <= float(max_penalty_K):
                best_size = s
                best_mor = float(mor)
                best_dt = float(dt)
                best_vel = None if vel is None else float(vel)
                best_dp_total_kPa = None if dp_total_kPa is None else float(dp_total_kPa)
                best_dp_pipe_kPa = None if dp_pipe_kPa is None else float(dp_pipe_kPa)
                best_dp_collated_kPa = None if dp_collated_kPa is None else float(dp_collated_kPa)
                best_dp_valves_kPa = None if dp_valves_kPa is None else float(dp_valves_kPa)
                break

        return best_size, best_mor, best_dt, best_vel, best_dp_total_kPa, best_dp_pipe_kPa, best_dp_collated_kPa, best_dp_valves_kPa

    # === CHANGED: rescue riser selection enforces MOR <= required (ignoring dt) ===
    def _select_single_riser_mor_only(pipe_sizes_local, get_pipe_results_local, required_oil_pct):
        # MOR is a MAXIMUM: choose among MOR-feasible sizes the one with LOWEST dt
        best = None
        best_mor = None
        best_dt = None
        best_vel = None
        best_dp_total_kPa = None
        best_dp_pipe_kPa = None
        best_dp_collated_kPa = None
        best_dp_valves_kPa = None

        for s in pipe_sizes_local:
            best_gauge = _strongest_gauge(material_df, s)
            res = get_pipe_results_local(s, best_gauge)
            dt, mor, vel, dp_total_kPa, dp_pipe_kPa, dp_collated_kPa, dp_valves_kPa = _extract_dt_mor(res)
            if mor is None or dt is None:
                continue

            if float(mor) <= float(required_oil_pct):
                if best_dt is None or float(dt) < float(best_dt):
                    best = s
                    best_mor = float(mor)
                    best_dt = float(dt)
                    best_vel = None if vel is None else float(vel)
                    best_dp_total_kPa = None if dp_total_kPa is None else float(dp_total_kPa)
                    best_dp_pipe_kPa = None if dp_pipe_kPa is None else float(dp_pipe_kPa)
                    best_dp_collated_kPa = None if dp_collated_kPa is None else float(dp_collated_kPa)
                    best_dp_valves_kPa = None if dp_valves_kPa is None else float(dp_valves_kPa)

        return best, best_mor, best_dt, best_vel, best_dp_total_kPa, best_dp_pipe_kPa, best_dp_collated_kPa, best_dp_valves_kPa

    # -----------------------------
    # Initial sizing (equiv-length apportionment)
    # -----------------------------
    results = {}

    for c in circuits:
        seg_meta = c.meta or {}

        mode_used = seg_meta.get("mode_override", "Horizontal")  # now always explicit
        
        req_oil_used = float(seg_meta.get("required_oil_override", 100.0) or 0.0)

        Leq_seg_m = segment_equivalent_length_m(c)
        if MPL_m > 0:
            max_penalty_seg = allowable_penalty_per_m * Leq_seg_m
        else:
            max_penalty_seg = float(max_penalty)

        get_pipe_results_local = _build_get_pipe_results_for_circuit(c, float(max_penalty))

        rec = {
            "Label": c.label,
            "Terminal": bool(c.is_load),
            "next": c.next_label,
            "incoming": c.incoming_count,
            "Duty (kW)": float(c.duty_kw),
            "mode_used": mode_used,
            "req_oil_used": (None if mode_used == "Horizontal" else float(req_oil_used)),
            "Leq_m": float(Leq_seg_m),
            "max_penalty_seg_K": float(max_penalty_seg),
            "get_pipe_results": get_pipe_results_local,
            "selected_size_inch": None,
            "dt": None,
            "MORfinal": None,
            "MOR_large": None,
            "velocity_m_sfinal": None,
            "dp_total_kPa": None,
            "dp_pipe_kPa": None,
            "dp_collated_kPa": None,
            "dp_valves_kPa": None,
            "error": None,
            "riser_relaxed_dt": False,
            # Junction / tee modelling fields
            "junction_type": int(seg_meta.get("junction_type", 1) or 1),
            "gauge": None,
            "q_kPa": None,
            "ID_m": None,
            "pipe_small":None,
            "pipe_large":None,
            "gauge_small":None,
            "gauge_large":None,
            "double_riser":False,
            "density_recalc": None,
            "maxmass": None,
            "volflow": None
        }

        try:
            if mode_used == "Horizontal":
                get_pipe_results_2 = lambda s, g=None: get_pipe_results_local(s, g)[:2]
                best_size, best_dt = select_horizontal_low_dt(
                    pipe_sizes=pipe_sizes,
                    mm_map=mm_map,
                    material_df=material_df,
                    get_pipe_results=get_pipe_results_2,  # <-- wrap to 2-tuple
                    max_penalty_K=float(max_penalty_seg),
                )
                if best_size is None:
                    raise ValueError(f"No size meets ΔT; best ΔT={best_dt:.3f}K")
                rec["selected_size_inch"] = best_size
                rec["dt"] = float(best_dt)

                # grab velocity for chosen size
                dt2, mor2, vel2, dp_total_kPa2, dp_pipe_kPa2, dp_collated_kPa2, dp_valves_kPa2 = _extract_dt_mor(get_pipe_results_local(best_size))
                rec["velocity_m_sfinal"] = None if vel2 is None else float(vel2)
                rec["dp_total_kPa"] = None if dp_total_kPa2 is None else float(dp_total_kPa2)
                rec["dp_pipe_kPa"] = None if dp_pipe_kPa2 is None else float(dp_pipe_kPa2)
                rec["dp_collated_kPa"] = None if dp_collated_kPa2 is None else float(dp_collated_kPa2)
                rec["dp_valves_kPa"] = None if dp_valves_kPa2 is None else float(dp_valves_kPa2)
                rec["MORfinal"] = None if mor2 is None else float(mor2)

            elif mode_used == "Single Riser":
                # CHANGED: use local selector with MOR <= required (not the imported select_single_riser)
                best_size, mor, dt, vel, dp_total_kPa, dp_pipe_kPa, dp_collated_kPa, dp_valves_kPa = _select_single_riser_max_mor(
                    pipe_sizes_local=pipe_sizes,
                    get_pipe_results_local=get_pipe_results_local,
                    max_mor_pct=float(req_oil_used),
                    max_penalty_K=float(max_penalty_seg),
                )

                if best_size is None:
                    # mark as rescue case (MOR-only riser selection, MOR<=required)
                    mor_only_size, mor_only_mor, mor_only_dt, mor_only_vel, mor_only_dp_total_kPa, mor_only_dp_pipe_kPa, mor_only_dp_collated_kPa, mor_only_dp_valves_kPa = _select_single_riser_mor_only(
                        pipe_sizes_local=pipe_sizes,
                        get_pipe_results_local=get_pipe_results_local,
                        required_oil_pct=float(req_oil_used),
                    )
                    if mor_only_size is None:
                        raise ValueError("No size meets MOR (even ignoring ΔT).")

                    rec["selected_size_inch"] = mor_only_size
                    rec["MORfinal"] = float(mor_only_mor)
                    rec["dt"] = float(mor_only_dt)
                    rec["velocity_m_sfinal"] = None if mor_only_vel is None else float(mor_only_vel)
                    rec["dp_total_kPa"] = None if mor_only_dp_total_kPa is None else float(mor_only_dp_total_kPa)
                    rec["dp_pipe_kPa"] = None if mor_only_dp_pipe_kPa is None else float(mor_only_dp_pipe_kPa)
                    rec["dp_collated_kPa"] = None if mor_only_dp_collated_kPa is None else float(mor_only_dp_collated_kPa)
                    rec["dp_valves_kPa"] = None if mor_only_dp_valves_kPa is None else float(mor_only_dp_valves_kPa)
                    rec["riser_relaxed_dt"] = True
                else:
                    rec["selected_size_inch"] = best_size
                    rec["MORfinal"] = float(mor)
                    rec["dt"] = float(dt)
                    rec["velocity_m_sfinal"] = None if vel is None else float(vel)
                    rec["dp_total_kPa"] = None if dp_total_kPa is None else float(dp_total_kPa)
                    rec["dp_pipe_kPa"] = None if dp_pipe_kPa is None else float(dp_pipe_kPa)
                    rec["dp_collated_kPa"] = None if dp_collated_kPa is None else float(dp_collated_kPa)
                    rec["dp_valves_kPa"] = None if dp_valves_kPa is None else float(dp_valves_kPa)
            elif mode_used == "Double Riser":
                MOR_full_cached2, eval_pair_cached = _build_double_riser_callbacks_for_circuit(c)
                best = select_double_riser_exact(
                    pipe_sizes=pipe_sizes,
                    mm_map=mm_map,
                    required_oil_duty_pct=req_oil_used,
                    max_penalty_K=float(max_penalty_seg),
                    MOR_full_cached2=MOR_full_cached2,
                    eval_pair_cached=eval_pair_cached,
                )
                if best is None:
                    raise ValueError("No double riser pair satisfies oil return and ΔT limits.")
                rec["pipe_small"] = best.best_small
                rec["pipe_large"] = best.best_large
                rec["gauge_small"] = _strongest_gauge(material_df, best.best_small)
                rec["gauge_large"] = _strongest_gauge(material_df, best.best_large)
                rec["selected_size_inch"] = f"{best.best_large} + {best.best_small}"
                rec["dt"] = best.best_dr.DT_K
                rec["MORfinal"] = best.best_MOR_full
                rec["MOR_large"] = best.best_MOR_large
                rec["velocity_m_sfinal"] = best.best_dr.large_result.velocity_m_s
                rec["dp_total_kPa"] = best.best_dr.DP_kPa
                rec["dp_pipe_kPa"] = best.best_dr.dp_pipe
                rec["dp_collated_kPa"] = best.best_dr.dp_fit
                rec["dp_valves_kPa"] = best.best_dr.dp_valve
                rec["ID_m"] = best.best_dr.large_result.ID_m
                rec["q_kPa"] = 0.5 * (best.best_dr.large_result.mass_flow_kg_s / (best.best_dr.large_result.velocity_m_s * best.best_dr.large_result.area_m2)) * (best.best_dr.large_result.velocity_m_s ** 2) / 1000.0 if best.best_dr.large_result.velocity_m_s > 0 else None
                rec["double_riser"] = True

        except Exception as e:
            rec["error"] = str(e)

        results[c.label] = rec

    # ------------------------------------------------------------
    # Include junction / tee losses DURING selection by iterating:
    #   1) choose sizes using current per-segment extra junction dp (kPa)
    #   2) recompute junction dp based on chosen sizes
    # until convergence (or max iterations).
    # ------------------------------------------------------------
    extra_dp_by_label = {c.label: 0.0 for c in circuits}
    max_iter_junction = 10
    for _it in range(max_iter_junction):
        prev_sizes = {lab: results[lab]["selected_size_inch"] for lab in results}
        prev_extra = dict(extra_dp_by_label)

        # 1) re-select each segment size with current extra dp
        for c in circuits:
            r = results[c.label]
            if r["error"]:
                continue

            extra_dp_kPa = float(extra_dp_by_label.get(c.label, 0.0) or 0.0)

            def gpr_with_extra(size_inch: str, gauge_override=None):
                return r["get_pipe_results"](size_inch, gauge_override, extra_dp_kPa)

            try:
                if r["mode_used"] == "Horizontal":
                    best_size, best_dt = select_horizontal_low_dt(
                        pipe_sizes=pipe_sizes,
                        mm_map=mm_map,
                        material_df=material_df,
                        get_pipe_results=lambda s, g=None: gpr_with_extra(s, g)[:2],
                        max_penalty_K=float(r["max_penalty_seg_K"]),
                    )
                    if best_size is None:
                        # Retry selection without imposed junction loss for very short pipes
                        # where the local section target is dominated by tee/junction dp.
                        best_size, best_dt = select_horizontal_low_dt(
                            pipe_sizes=pipe_sizes,
                            mm_map=mm_map,
                            material_df=material_df,
                            get_pipe_results=lambda s, g=None: r["get_pipe_results"](s, g, 0.0)[:2],
                            max_penalty_K=float(r["max_penalty_seg_K"]),
                        )
                    if best_size is None:
                        raise ValueError(f"No size meets ΔT; best ΔT={best_dt:.3f}K")

                    best_gauge = _strongest_gauge(material_df, best_size)
                    mor_i, dt_i, vel_i, q_i, id_i, dp_total_kPa_i, dp_pipe_kPa_i, dp_collated_kPa_i, dp_valves_kPa_i = r["get_pipe_results"](best_size, best_gauge, extra_dp_kPa)[:9]

                    r["selected_size_inch"] = best_size
                    r["dt"] = None if dt_i is None else float(dt_i)
                    r["velocity_m_sfinal"] = None if vel_i is None else float(vel_i)
                    r["dp_total_kPa"] = None if dp_total_kPa_i is None else float(dp_total_kPa_i)
                    r["dp_pipe_kPa"] = None if dp_pipe_kPa_i is None else float(dp_pipe_kPa_i)
                    r["dp_collated_kPa"] = None if dp_collated_kPa_i is None else float(dp_collated_kPa_i)
                    r["dp_valves_kPa"] = None if dp_valves_kPa_i is None else float(dp_valves_kPa_i)
                    r["gauge"] = best_gauge
                    r["q_kPa"] = None if q_i is None else float(q_i)
                    r["ID_m"] = None if id_i is None else float(id_i)
                    r["MORfinal"] = None if mor_i is None else float(mor_i)

                elif r["mode_used"] == "Single Riser":
                    # Use local MOR-as-maximum riser selectors (MOR <= required) so the junction
                    # iteration does not revert to legacy selector behaviour.
                    def gpr_size(size_inch: str, gauge_override=None):
                        g_ = gauge_override or _strongest_gauge(material_df, size_inch)
                        return r["get_pipe_results"](size_inch, g_, extra_dp_kPa)

                    best_size, mor, dt, vel, dp_total_kPa, dp_pipe_kPa, dp_collated_kPa, dp_valves_kPa = _select_single_riser_max_mor(
                        pipe_sizes_local=pipe_sizes,
                        get_pipe_results_local=gpr_size,
                        max_mor_pct=float(r["req_oil_used"]),
                        max_penalty_K=float(r["max_penalty_seg_K"]),
                    )

                    if best_size is None:
                        # MOR-only fallback (ignore ΔT) but do not raise if MOR can be met.
                        mor_only_size, mor_only_mor, mor_only_dt, mor_only_vel, mor_only_dp_total_kPa, mor_only_dp_pipe_kPa, mor_only_dp_collated_kPa, mor_only_dp_valves_kPa = _select_single_riser_mor_only(
                            pipe_sizes_local=pipe_sizes,
                            get_pipe_results_local=gpr_size,
                            required_oil_pct=float(r["req_oil_used"]),
                        )
                        if mor_only_size is None:
                            raise ValueError("No size meets MOR (even ignoring ΔT).")
                        best_size = mor_only_size
                        r["riser_relaxed_dt"] = True

                    best_gauge = _strongest_gauge(material_df, best_size)
                    mor_i, dt_i, vel_i, q_i, id_i, dp_total_kPa_i, dp_pipe_kPa_i, dp_collated_kPa_i, dp_valves_kPa_i = r["get_pipe_results"](best_size, best_gauge, extra_dp_kPa)[:9]

                    r["selected_size_inch"] = best_size
                    r["MORfinal"] = None if mor_i is None else float(mor_i)
                    r["dt"] = None if dt_i is None else float(dt_i)
                    r["velocity_m_sfinal"] = None if vel_i is None else float(vel_i)
                    r["dp_total_kPa"] = None if dp_total_kPa_i is None else float(dp_total_kPa_i)
                    r["dp_pipe_kPa"] = None if dp_pipe_kPa_i is None else float(dp_pipe_kPa_i)
                    r["dp_collated_kPa"] = None if dp_collated_kPa_i is None else float(dp_collated_kPa_i)
                    r["dp_valves_kPa"] = None if dp_valves_kPa_i is None else float(dp_valves_kPa_i)
                    r["gauge"] = best_gauge
                    r["q_kPa"] = None if q_i is None else float(q_i)
                    r["ID_m"] = None if id_i is None else float(id_i)

                elif r["mode_used"] == "Double Riser":
                    MOR_full_cached2, eval_pair_cached = _build_double_riser_callbacks_for_circuit(c, extra_dp_kPa)
                    best = select_double_riser_exact(
                        pipe_sizes=pipe_sizes,
                        mm_map=mm_map,
                        required_oil_duty_pct=r["req_oil_used"],
                        max_penalty_K=float(r["max_penalty_seg_K"]),
                        MOR_full_cached2=MOR_full_cached2,
                        eval_pair_cached=eval_pair_cached,
                    )
                    if best is None:
                        raise ValueError("No double riser pair satisfies oil return and ΔT limits.")
                    r["pipe_small"] = best.best_small
                    r["pipe_large"] = best.best_large
                    r["selected_size_inch"] = f"{best.best_large} + {best.best_small}"
                    r["dt"] = None if best.best_dr.DT_K is None else float(best.best_dr.DT_K)
                    r["MORfinal"] = None if best.best_MOR_full is None else float(best.best_MOR_full)
                    r["MOR_large"] = None if best.best_MOR_large is None else float(best.best_MOR_large)
                    r["velocity_m_sfinal"] = None if best.best_dr.large_result.velocity_m_s is None else float(best.best_dr.large_result.velocity_m_s)
                    r["dp_total_kPa"] = None if best.best_dr.DP_kPa is None else float(best.best_dr.DP_kPa)
                    r["dp_pipe_kPa"] = None if best.best_dr.dp_pipe is None else float(best.best_dr.dp_pipe)
                    r["dp_collated_kPa"] = None if best.best_dr.dp_fit is None else float(best.best_dr.dp_fit)
                    r["dp_valves_kPa"] = None if best.best_dr.dp_valve is None else float(best.best_dr.dp_valve)
                    r["ID_m"] = None if best.best_dr.large_result.ID_m is None else float(best.best_dr.large_result.ID_m)
                    r["q_kPa"] = 0.5 * (best.best_dr.large_result.mass_flow_kg_s / (best.best_dr.large_result.velocity_m_s * best.best_dr.large_result.area_m2)) * (best.best_dr.large_result.velocity_m_s ** 2) / 1000.0 if best.best_dr.large_result.velocity_m_s > 0 else None
                    r["double_riser"] = True

            except Exception as e:
                r["selected_size_inch"] = None
                r["gauge"] = None
                r["gauge_small"] = None
                r["gauge_large"] = None
                r["ID_m"] = None
                r["q_kPa"] = None
                r["dp_pipe_kPa"] = None
                r["dp_collated_kPa"] = None
                r["dp_valves_kPa"] = None
                r["dp_total_kPa"] = None
                r["dt"] = None
                r["MOR"] = None
                r["MORfinal"] = None
                r["MOR_large"] = None
                r["error"] = str(e)

        # 2) recompute extra dp from tees, based on the selected sizes
        extra_dp_by_label = _compute_junction_extra_dp(results_local=results)

        # Convergence check: sizes unchanged AND extra dp unchanged
        sizes_same = all(prev_sizes.get(k) == results[k]["selected_size_inch"] for k in results)
        extra_same = all(abs(float(prev_extra.get(k, 0.0)) - float(extra_dp_by_label.get(k, 0.0))) < 1e-9 for k in extra_dp_by_label)
        if sizes_same and extra_same:
            break

    # Ensure dt/MOR/vel/dp_total_kPa/pipes/fittings/valves reflect the final converged extra dp set
    _recompute_all_with_junction_inplace()

    # Current global worst-path dt
    dt_by_label = {lab: (None if results[lab]["error"] else results[lab]["dt"]) for lab in results}
    worst_dt, worst_path_labels, per_path = _path_total_dt_K(net["paths"], dt_by_label)

    # ============================================================
    # MUTUALLY EXCLUSIVE OPTIMISATION STRATEGY CHOICE
    # ============================================================
    needs_rescue = worst_dt is not None and worst_dt > float(max_penalty)

    if needs_rescue:
        # --------------------------------------------------------
        # Strategy 2 ONLY (rescue):
        # Upsize horizontals on offending paths until:
        #   - all risers meet MOR
        #   - worst-case total path dt <= input max_penalty
        # --------------------------------------------------------
        max_steps = 300
        step = 0

        while step < max_steps:
            step += 1

            dt_by_label = {lab: (None if results[lab]["error"] else results[lab]["dt"]) for lab in results}
            worst_dt, worst_path_labels, per_path = _path_total_dt_K(net["paths"], dt_by_label)

            # Stop if we can't compute, or we're feasible
            if worst_dt is None:
                break
            if worst_dt <= float(max_penalty):
                break

            # Paths currently exceeding global budget
            bad_paths = [labels for labels, total in per_path if (total is not None and total > float(max_penalty))]
            if not bad_paths and worst_path_labels:
                bad_paths = [worst_path_labels]

            # Candidate horizontals on those paths: highest (dt / max_penalty_seg_K) first
            cand_set = set()
            for labels in bad_paths:
                for lab in labels:
                    cand_set.add(lab)

            cand_list = []
            for lab in cand_set:
                r = results.get(lab)
                if not r or r["error"] or r["selected_size_inch"] is None or r["dt"] is None:
                    continue
                if r["mode_used"] == "Double Riser":
                    continue
                if r["mode_used"] != "Horizontal":
                    continue
                idx = size_to_idx.get(r["selected_size_inch"])
                if idx is None or idx >= len(pipe_sizes) - 1:
                    continue
                ratio = _compute_ratio(lab)
                if ratio is None:
                    continue
                cand_list.append((ratio, lab))

            if not cand_list:
                # Nothing left to upsize -> cannot rescue further
                break

            cand_list.sort(key=lambda x: x[0], reverse=True)

            move_made = False

            for _, lab_to_up in cand_list:
                r = results[lab_to_up]
                idx = size_to_idx[r["selected_size_inch"]]
                size_up = pipe_sizes[idx + 1]

                (
                    dt_new, mor_new, vel_new,
                    dp_total_kPa_new, dp_pipe_kPa_new, dp_collated_kPa_new, dp_valves_kPa_new,
                    worst_dt_new_sim, _, _
                ) = _simulate_size_change(lab_to_up, size_up)

                if dt_new is None or worst_dt_new_sim is None:
                    continue

                improvement_K = float(worst_dt) - float(worst_dt_new_sim)

                # Require minimum improvement
                if improvement_K <= 0:
                    continue

                # Commit change
                r["selected_size_inch"] = size_up
                r["error"] = None
                _recompute_all_with_junction_inplace()

                move_made = True
                break

            if not move_made:
                # Nothing gave enough improvement -> cannot rescue further
                break

        # Note: we intentionally do NOT run Strategy 1 afterwards in a rescue scenario.

    else:
        # --------------------------------------------------------
        # Strategy 1 ONLY (cost-down):
        # Greedy downsizing in order of lowest (dt / max_penalty_seg_K),
        # accepting only if global worst-path dt stays <= input max_penalty
        # --------------------------------------------------------
        if worst_dt is not None:
            changed = True
            while changed:
                changed = False

                candidates = []
                for lab, r in results.items():
                    if r["error"] or r["selected_size_inch"] is None or r["dt"] is None:
                        continue
                    if r["mode_used"] == "Double Riser":
                        continue
                    idx = size_to_idx.get(r["selected_size_inch"])
                    if idx is None or idx <= 0:
                        continue
                    ratio = _compute_ratio(lab)
                    if ratio is None:
                        continue
                    candidates.append((ratio, lab))

                candidates.sort(key=lambda x: x[0])  # LOWEST ratio first

                for _, lab in candidates:
                    r = results[lab]
                    idx = size_to_idx[r["selected_size_inch"]]
                    size_down = pipe_sizes[idx - 1]

                    dt_new, mor_new, vel_new, dp_total_kPa_new, dp_pipe_kPa_new, dp_collated_kPa_new, dp_valves_kPa_new, worst_dt_new_sim, worst_path_labels_new, risers_ok_sim = _simulate_size_change(lab, size_down)
                    if dt_new is None:
                        continue

                    # CHANGED: If riser, must still meet MOR (as a maximum: <= req)
                    if r["mode_used"] == "Single Riser":
                        req = r["req_oil_used"]
                        if req is not None:
                            if mor_new is None or float(mor_new) > float(req):
                                continue

                                        # Accept only if the junction-coupled network stays feasible
                    if (worst_dt_new_sim is not None) and (worst_dt_new_sim <= float(max_penalty)) and risers_ok_sim:
                        r["selected_size_inch"] = size_down
                        r["error"] = None
                        _recompute_all_with_junction_inplace()
                        worst_dt = worst_dt_new_sim
                        worst_path_labels = worst_path_labels_new
                        changed = True
                        break
                    else:
                        continue

    # -----------------------------
    # Pipe system total volume (m³)
    # -----------------------------
    total_vol_m3 = 0.0
    missing_vol = []
    
    for c in circuits:
        r = results.get(c.label, {})
        meta = c.meta or {}
    
        L_m = float(meta.get("L", 0.0) or 0.0)
        ID_m = r.get("ID_m")
    
        # If sizing failed or no diameter, skip and record
        if (r.get("error")) or (ID_m is None) or (L_m <= 0):
            if L_m > 0 and ID_m is None and not r.get("error"):
                missing_vol.append(c.label)
            continue
    
        ID_m = float(ID_m)
        if ID_m <= 0:
            continue
    
        area_m2 = 3.141592653589793 * (ID_m / 2.0) ** 2
        total_vol_m3 += area_m2 * L_m
    
    total_vol_L = total_vol_m3 * 1000.0  # litres
    
    # -----------------------------
    # Total results (worst-case path)
    # -----------------------------
    
    # Build quick lookup tables from computed results
    dt_by_label = {
        str(lab): (None if results[lab]["error"] else results[lab].get("dt"))
        for lab in results
    }
    
    dp_by_label = {
        str(lab): (None if results[lab]["error"] else results[lab].get("dp_total_kPa"))
        for lab in results
    }
    
    duty_by_label = {c.label: float(c.duty_kw) for c in circuits}
    
    path_summaries = []
    for p in net["paths"]:
        labels = [circuits[i].label for i in p]
    
        # If any circuit on the path failed sizing (dt is None), we can't compute a reliable total dt
        dts = [dt_by_label.get(lab) for lab in labels]
        if any(v is None for v in dts):
            total_dt = None
        else:
            total_dt = float(sum(dts))

        dps = [dp_by_label.get(lab) for lab in labels]
        total_dp = None if any(v is None for v in dps) else float(sum(dps))
    
        # "Total duty" for the path = duty at upstream end (last element), which is aggregated duty
        total_duty_kw = duty_by_label.get(labels[-1], None)
    
        path_summaries.append({
            "Terminal": labels[0],
            "Path": " → ".join(labels),
            "Total Penalty (K)": total_dt,
            "Total Penalty (kPa)": total_dp,
            "Total Duty (kW)": total_duty_kw,
        })
    
    paths_df = pd.DataFrame(path_summaries)

    paths_df = paths_df.round({
        "Total Penalty (K)": 2,
        "Total Penalty (kPa)": 2,
        "Total Duty (kW)": 2
    })

    # Map load label -> total path penalties
    path_dt_by_load = {str(x["Terminal"]): x["Total Penalty (K)"] for _, x in paths_df.iterrows()}
    path_dp_by_load = {str(x["Terminal"]): x["Total Penalty (kPa)"] for _, x in paths_df.iterrows()}

    # -----------------------------
    # Build output dataframe from final results dict
    # -----------------------------
    rows_out = []
    for c in circuits:
        r = results[c.label]
        meta = c.meta or {}

        length_m = float(meta.get("L", 0.0) or 0.0)
    
        bends_count = (
            int(meta.get("SRB", 0) or 0)
            + int(meta.get("LRB", 0) or 0)
            + int(meta.get("_45", 0) or 0)
            + int(meta.get("MAC", 0) or 0)
            + int(meta.get("ptrap", 0) or 0)
            + int(meta.get("ubend", 0) or 0)
        )
    
        valves_count = int(meta.get("ball", 0) or 0) + int(meta.get("globe", 0) or 0)
        
        rows_out.append({
            "Label": r["Label"],
            "Type": "load" if r["Terminal"] else "pipe",
            "Next": r["next"],
            "Duty (kW)": r["Duty (kW)"],
            "Orientation": r["mode_used"],
            "Size(s)": r["selected_size_inch"],
            "Length (m)": length_m,
            "Bends": bends_count,
            "Valves": valves_count,
            "Oil Return %": r["MORfinal"],
            "Velocity (m/s)": r["velocity_m_sfinal"],
            "Penalty (K)": r["dt"],
            "Penalty (kPa)": r["dp_total_kPa"],
            "Pipe (kPa)": r["dp_pipe_kPa"],
            "Fittings (kPa)": r["dp_collated_kPa"],
            "Valves (kPa)": r["dp_valves_kPa"],
            "Flowpath (K)": (path_dt_by_load.get(r["Label"]) if r["Terminal"] else None),
            "Flowpath (kPa)": (path_dp_by_load.get(r["Label"]) if r["Terminal"] else None),
            "Error": r["error"],
        })

    out_df = pd.DataFrame(rows_out)

    out_df = out_df.round({
        "Duty (kW)": 2,
        "Oil Return %": 1,
        "Velocity (m/s)": 2,
        "Penalty (kPa)": 2,
        "Penalty (K)": 2,
        "Pipe (kPa)": 2,
        "Fittings (kPa)": 2,
        "Valves (kPa)": 2,
        "Flowpath (K)": 2,
        "Flowpath (kPa)": 2
    })
    
    st.success("Done")
    st.dataframe(out_df, use_container_width=True)
    
    # Pick worst-case by highest total_dt_K (ignore None)
    paths_valid = paths_df.dropna(subset=["Total Penalty (K)"]).copy()
    paths_valid_dp = paths_df.dropna(subset=["Total Penalty (kPa)"]).copy()
    if paths_valid.empty and paths_valid_dp.empty:
        st.warning("Cannot compute worst-case totals because one or more paths have missing ΔT/P (sizing errors).")
    else:
        worst_row = paths_valid.loc[paths_valid["Total Penalty (K)"].idxmax()]
        worst_row_dp = None if paths_valid_dp.empty else paths_valid_dp.loc[paths_valid_dp["Total Penalty (kPa)"].idxmax()]

        sst = T_evap - worst_row["Total Penalty (K)"]

        converter = PressureTemperatureConverter()
        if refrigerant == "R744 TC":
            evappres = converter.temp_to_pressure("R744", T_evap)
        else:
            evappres = converter.temp_to_pressure(refrigerant, T_evap)
        sucpress = evappres - (worst_row_dp["Total Penalty (kPa)"] / 100)
        # Pick a representative segment to read the values from:
        # Use the upstream end of the worst-ΔT path (last label in "A → B → C").
        labels = [s.strip() for s in str(worst_row["Path"]).split("→")]
        upstream_label = labels[-1] if labels else None
        
        density_recalc = results.get(upstream_label, {}).get("density_recalc") if upstream_label else None
        maxmass = results.get(upstream_label, {}).get("maxmass") if upstream_label else None
        volflow = results.get(upstream_label, {}).get("volflow") if upstream_label else None

        system_mass = None if density_recalc is None else density_recalc * total_vol_L / 1000.0
    
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        with m1:
            if worst_row["Total Duty (kW)"] is not None:
                st.metric("Total Duty (kW)", f'{worst_row["Total Duty (kW)"]:.2f}')
            else:
                st.metric("Total Duty (kW)", "—")
        with m2:
            st.metric("Total Penalty (kPa)", f'{worst_row_dp["Total Penalty (kPa)"]:.2f}')
        with m3:
            st.metric("Total Penalty (K)", f'{worst_row["Total Penalty (K)"]:.2f}')
        with m4:
            st.metric("SST (°C)", f"{sst:.2f}")
        with m5:
            st.metric("Evaporating Pressure (bar(a))", f"{evappres:.2f}")
        with m6:
            st.metric("Suction Pressure (bar(a))", f"{sucpress:.2f}")

        m7, m8, m9, m10, m11, m12 = st.columns(6)
        with m7:
            st.metric("Suction Density (kg/m³)", f"{density_recalc:.2f}" if density_recalc is not None else "—")
        with m8:
            st.metric("Mass Flow Rate (kg/s)", f"{maxmass:.5f}" if maxmass is not None else "—")
        with m9:
            st.metric("Volumetric Flow Rate (m³/s)", f"{volflow:.5f}" if volflow is not None else "—")
        with m10:
            st.metric("Pipe-System Volume (litres)", f"{total_vol_L:.2f}")
        with m11:
            st.metric("Pipe-System Contents (kg)", f"{system_mass:.2f}" if system_mass is not None else "—")
        with m12:
            st.write("")
    
        st.code(str(worst_row["Path"]))
    
    with st.expander("Path Totals", expanded=False):
        st.dataframe(paths_df, use_container_width=True)

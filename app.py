
import streamlit as st
from utils.network_builder import NetworkBuilder
from utils.pressure_temp_converter import PressureTemperatureConverter
from utils.system_pressure_checker import system_pressure_check, system_pressure_check_double_riser
import pandas as pd
import math
import bisect
import numpy as np

def material_to_pipe_index(material: str) -> int:
    m = (material or "").strip().lower()

    # --- Copper ---
    if "en12735" in m or ("copper" in m and "12735" in m):
        return 1
    if "b280" in m or ("copper" in m and "astm" in m):
        return 6
    if "k65" in m:
        return 8

    # --- Aluminium ---
    if "aluminium" in m or "aluminum" in m or "6061" in m:
        return 7

    # --- Stainless MUST come before steel ---
    if "stainless" in m:
        if "sch10" in m or "sch 10" in m or "schedule 10" in m:
            return 3
        if "sch40" in m or "sch 40" in m or "schedule 40" in m:
            return 4
        raise ValueError(f"Unmapped stainless schedule: {material!r}")

    # --- Carbon steel (explicitly excludes stainless now) ---
    if "steel" in m:
        if "sch40" in m or "sch 40" in m or "schedule 40" in m:
            return 2
        if "sch80" in m or "sch 80" in m or "schedule 80" in m:
            return 5
        raise ValueError(f"Unmapped steel schedule: {material!r}")

    raise ValueError(f"Unmapped Material value: {material!r}")

def render_pressure_result(result: dict):
    if not result:
        return

    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

    design_p = result["design_pressure_bar_g"]
    with col1:
        st.metric("Design Pressure (bar(g))", f"{design_p:.2f}")

    pipes = []
    if "branch_a" in result:
        pipes.append(result["branch_a"])
    if "branch_b" in result:
        pipes.append(result["branch_b"])
    if not pipes:
        pipes.append(result)

    # -------- PIPE / BRANCH A --------
    res1 = pipes[0]
    mwp1 = res1["mwp_bar"]
    margin1 = res1["margin_bar"]
    pass1 = res1["pass"]

    with col2:
        if isinstance(mwp1, dict):
            for weld in ("seamless", "erw", "cw"):
                if weld in mwp1:
                    st.metric(
                        f"MWP – {weld.upper()} (bar(g))",
                        f"{mwp1[weld]:.2f}",
                    )
        else:
            st.metric("MWP (bar(g))", f"{mwp1:.2f}")

    with col3:
        if isinstance(margin1, dict):
            for weld in ("seamless", "erw", "cw"):
                if weld in margin1:
                    st.metric(
                        f"Margin – {weld.upper()} (bar)",
                        f"{margin1[weld]:.2f}",
                    )
        else:
            st.metric("Margin (bar)", f"{margin1:.2f}")

    with col4:
        if isinstance(pass1, dict):
            for weld in ("seamless", "erw", "cw"):
                if weld in pass1:
                    if pass1[weld]:
                        st.success(f"{weld.upper()} PASS")
                    else:
                        st.error(f"{weld.upper()} FAIL")
        else:
            if pass1:
                st.success("PASS")
            else:
                st.error("FAIL")

    # -------- PIPE / BRANCH B (if present) --------
    if len(pipes) > 1:
        res2 = pipes[1]
        mwp2 = res2["mwp_bar"]
        margin2 = res2["margin_bar"]
        pass2 = res2["pass"]

        with col5:
            if isinstance(mwp2, dict):
                for weld in ("seamless", "erw", "cw"):
                    if weld in mwp2:
                        st.metric(
                            f"MWP – {weld.upper()} (bar(g))",
                            f"{mwp2[weld]:.2f}",
                        )
            else:
                st.metric("MWP (bar(g))", f"{mwp2:.2f}")

        with col6:
            if isinstance(margin2, dict):
                for weld in ("seamless", "erw", "cw"):
                    if weld in margin2:
                        st.metric(
                            f"Margin – {weld.upper()} (bar)",
                            f"{margin2[weld]:.2f}",
                        )
            else:
                st.metric("Margin (bar)", f"{margin2:.2f}")

        with col7:
            if isinstance(pass2, dict):
                for weld in ("seamless", "erw", "cw"):
                    if weld in pass2:
                        if pass2[weld]:
                            st.success(f"{weld.upper()} PASS")
                        else:
                            st.error(f"{weld.upper()} FAIL")
            else:
                if pass2:
                    st.success("PASS")
                else:
                    st.error("FAIL")

def get_dimensions_for_row(material_df, size_inch: str, gauge: int | None):
    rows = material_df[
        material_df["Nominal Size (inch)"].astype(str).str.strip() == str(size_inch)
    ]

    if rows.empty:
        raise ValueError(f"No pipe data for size {size_inch}")

    if (
        gauge is not None
        and "Gauge" in rows.columns
        and gauge in rows["Gauge"].dropna().unique()
    ):
        row = rows[rows["Gauge"] == gauge].iloc[0]
    else:
        row = rows.iloc[0]

    od_mm = float(row["Nominal Size (mm)"])
    id_mm = float(row["ID_mm"]) if pd.notna(row["ID_mm"]) else None

    return od_mm, id_mm

def governing_mwp(mwp):
    if isinstance(mwp, dict):
        return min(mwp.values()) if mwp else float("nan")
    return mwp

def pressure_checker_inputs(
    *,
    refrigerant: str,
    circuit: str,
    dp_standard: str,
):
    # 1) R744 TC suction override
    if refrigerant == "R744 TC" and circuit == "Suction":
        refrigerant = "R744"

    # 2) MWP reference temp rule
    mwp_temp_c = 150 if circuit == "Discharge" else 50

    # 3) defaults
    if refrigerant == "R744":
        default_high_dt = 25.0
        default_low_dt = 25.0
    elif refrigerant in ("R23", "R508B"):
        default_high_dt = 10.0
        default_low_dt = 10.0
    elif dp_standard == "BS EN 378":
        default_high_dt = 55.0
        default_low_dt = 32.0
    else:
        default_high_dt = 50.0
        default_low_dt = 27.0

    # 4) ranges
    if refrigerant == "R744":
        range_min_low, range_max_low = -20.0, 25.0
        range_min_high, range_max_high = 0.0, 25.0
    elif refrigerant in ("R23", "R508B"):
        range_min_low, range_max_low = -60.0, 10.0
        range_min_high, range_max_high = -30.0, 10.0
    else:
        range_min_low, range_max_low = 20.0, 50.0
        range_min_high, range_max_high = 25.0, 60.0

    return {
        "refrigerant": refrigerant,
        "mwp_temp_c": mwp_temp_c,
        "default_high_dt": default_high_dt,
        "default_low_dt": default_low_dt,
        "range_min_low": range_min_low,
        "range_max_low": range_max_low,
        "range_min_high": range_min_high,
        "range_max_high": range_max_high,
    }

# Make metric numbers & labels smaller
st.markdown("""
<style>
/* number */
div[data-testid="stMetricValue"] > div {
    font-size: 30px;            /* default is ~36px; pick what you like */
    line-height: 1.2;
}
/* label (e.g., "Refrigerant Velocity") */
div[data-testid="stMetricLabel"] > div {
    font-size: 14px;
}
/* optional: delta arrow/number size */
div[data-testid="stMetricDelta"] svg, 
div[data-testid="stMetricDelta"] > div {
    height: 14px; width: 14px; font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Micropipe - Refrigeration Pipe Sizing", layout="wide")
st.title("MicroPipe")

# Sidebar for tools and settings
st.sidebar.title("Tools & Utilities")
tool_selection = st.sidebar.radio("Select Tool", [
    "Manual Calculation",
    "Oil Return Checker",
    "Pressure ↔ Temperature Converter",
    "Pressure Drop ↔ Temperature Penalty",
    "System Pressure Checker"
])
st.sidebar.text("")
st.sidebar.text("")
st.sidebar.text("")
st.sidebar.text("")
st.sidebar.text("")
st.sidebar.text("")
st.sidebar.text("")
st.sidebar.text("")
st.sidebar.image("assets/logo.png", use_container_width=True)

def system_pressure_checker_ui():

    def inch_str_to_float(size: str) -> float:
        """
        Convert nominal pipe size strings like:
        '1', '1/2', '1-1/8', '2-3/8'
        into float inches.
        """
        s = str(size).replace('"', '').strip()
        if not s:
            return float("nan")
    
        total = 0.0
        parts = s.split("-")
        for p in parts:
            if "/" in p:
                num, den = p.split("/")
                total += float(num) / float(den)
            else:
                total += float(p)
    
        return total

    pipe_data = pd.read_csv("data/pipe_pressure_ratings_full.csv")

    required_cols = {"Material", "Nominal Size (inch)", "Nominal Size (mm)", "ID_mm"}
    missing = required_cols - set(pipe_data.columns)
    if missing:
        st.error(f"Pipe CSV missing required columns: {sorted(missing)}")
        return

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### System Pressure Checker")

        refrigerant = st.selectbox(
            "Refrigerant",
            [
                "R404A", "R134a", "R407F", "R744", "R744 TC", "R410A",
                "R407C", "R507A", "R448A", "R449A", "R22", "R32",
                "R454A", "R454C", "R455A", "R407A", "R290", "R1270",
                "R600a", "R717", "R1234ze", "R1234yf", "R12", "R11",
                "R454B", "R450A", "R513A", "R23", "R508B", "R502",
            ],
            disabled=True,
        )

        circuit = st.session_state.get("circuit")
        
        if circuit is None:
            st.error("Circuit has not been set. Please use Manual Calculation first.")
            return

        st.text_input(
            "Circuit Type",
            value=circuit,
            disabled=True,
        )

        if refrigerant == "R744 TC" and circuit == "Suction":
            refrigerant = "R744"

        mwp_options = 150 if circuit == "Discharge" else 50

        cola, colb = st.columns(2)

        with cola:
            copper_calc = st.selectbox(
                "Copper MWP Calculation Standard",
                ["BS1306", "DKI"],
                index=0,
                key="manual_copper_calc",
                disabled=True,
            )

        with colb:
            dp_standard = st.selectbox(
                "Design Pressure Standard",
                ["BS EN 378", "ASME B31.5 - 2006"],
                index=0,
                key="manual_dp_standard",
                disabled=True,
            )

        ctx = pressure_checker_inputs(
            refrigerant=refrigerant,
            circuit=circuit,
            dp_standard=dp_standard,
        )

        r744_tc_pressure_bar_g = None

        if refrigerant == "R744 TC" and circuit != "Suction":
            design_temp_c = None
            r744_tc_pressure_bar_g = st.number_input(
                "R744 Transcritical Design Pressure (bar(g))",
                min_value=75.0,
                max_value=150.0,
                step=5.0,
                value=120.0,
                key="manual_r744_tc_pressure",
                disabled=True,
            )
        else:
            if circuit in ("Suction", "Pumped"):
                design_temp_c = st.number_input(
                    "Design Temperature (°C)",
                    min_value=ctx["range_min_low"],
                    max_value=ctx["range_max_low"],
                    value=ctx["default_low_dt"],
                    step=1.0,
                    key="manual_design_temp_low",
                    disabled=True,
                )
            else:
                design_temp_c = st.number_input(
                    "Design Temperature (°C)",
                    min_value=ctx["range_min_high"],
                    max_value=ctx["range_max_high"],
                    value=ctx["default_high_dt"],
                    step=1.0,
                    key="manual_design_temp_high",
                    disabled=True,
                )

    with col2:

        st.markdown("### Pipe Parameters")

        mwp_temp_c = st.selectbox(
            "MWP Reference Temperature (°C)",
            [50, 100, 150],
            index=[50, 100, 150].index(ctx["mwp_temp_c"]),
            disabled=True,
        )

        pipe_materials = sorted(pipe_data["Material"].dropna().unique())
        selected_material = st.selectbox("Pipe Material", pipe_materials, key="material", disabled=True)

        try:
            pipe_index = material_to_pipe_index(selected_material)
        except ValueError as e:
            st.error(str(e))
            st.info(
                "Fix: either rename the Material entry in the CSV to match a known pattern "
                "(e.g. 'Copper EN12735') or extend material_to_pipe_index()."
            )
            return

        material_df = pipe_data[pipe_data["Material"] == selected_material].copy()
        material_df["Nominal Size (inch)"] = material_df["Nominal Size (inch)"].astype(str)

        pipe_sizes = sorted(
            material_df["Nominal Size (inch)"].dropna().unique(),
            key=inch_str_to_float,
        )

        selected_size = st.selectbox(
            "Nominal Pipe Size (inch)",
            pipe_sizes,
            key="single_size",
        )

        size_df = material_df[
            material_df["Nominal Size (inch)"] == str(selected_size)
        ].copy()

        if size_df.empty:
            st.error("No rows found for the selected material + nominal size.")
            return

        gauge = None
        if "Gauge" in size_df.columns and size_df["Gauge"].notna().any():
            gauges = sorted(size_df["Gauge"].dropna().unique())
            if len(gauges) > 1:
                gauge = st.selectbox("Gauge", gauges, key="single_gauge")
            else:
                gauge = gauges[0]

        od_mm, id_mm = get_dimensions_for_row(material_df, selected_size, gauge)

        if pipe_index != 1 and id_mm is None:
            st.error("This pipe type requires ID_mm in the CSV to calculate wall thickness.")
            return

    result = system_pressure_check(
        refrigerant=refrigerant,
        design_temp_c=design_temp_c,
        mwp_temp_c=mwp_temp_c,
        circuit=circuit,
        pipe_index=pipe_index,
        od_mm=od_mm,
        id_mm=id_mm,
        gauge=gauge,
        copper_calc=copper_calc,
        r744_tc_pressure_bar_g=r744_tc_pressure_bar_g,
        dp_standard=dp_standard,
    )

    limits = result["pressure_limits_bar"]
    mwp_multi = result.get("mwp_multi_temp", {})
    converter = PressureTemperatureConverter()

    if not mwp_multi:
        min_strength = 1.3 * result['design_pressure_bar_g']
        max_strength = None
    else:
        min_strength = 1.3 * result['design_pressure_bar_g']
        max_strength = 1.5 * mwp_multi[50]

    if refrigerant == "R744 TC" and circuit != "Suction":
        design_32 = None
        design_43 = None
        design_55 = None
    else:
        if dp_standard == "BS EN 378":
            if circuit in ("Suction", "Discharge"):
                design_32 = converter.temp_to_pressure(refrigerant, 32) - 1.01325
                design_43 = converter.temp_to_pressure(refrigerant, 43) - 1.01325
                design_55 = converter.temp_to_pressure(refrigerant, 55) - 1.01325
        
            else:
                design_32 = converter.temp_to_pressure2(refrigerant, 32) - 1.01325
                design_43 = converter.temp_to_pressure2(refrigerant, 43) - 1.01325
                design_55 = converter.temp_to_pressure2(refrigerant, 55) - 1.01325
    
        else:
            if circuit in ("Suction", "Discharge"):
                design_32 = converter.temp_to_pressure(refrigerant, 27) - 1.01325
                design_43 = converter.temp_to_pressure(refrigerant, 40) - 1.01325
                design_55 = converter.temp_to_pressure(refrigerant, 50) - 1.01325
        
            else:
                design_32 = converter.temp_to_pressure2(refrigerant, 27) - 1.01325
                design_43 = converter.temp_to_pressure2(refrigerant, 40) - 1.01325
                design_55 = converter.temp_to_pressure2(refrigerant, 50) - 1.01325

    st.markdown("### Results")

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        if refrigerant == "R744 TC" and circuit != "Suction":
            st.metric("", "")
        else:
            if refrigerant == "R744":
                st.metric("30°C", f"{design_55:.2f} bar(g)")
                st.metric("30°C", f"{design_43:.2f} bar(g)")
                st.metric("30°C", f"{design_43:.2f} bar(g)")
            elif refrigerant in ("R23", "R508B"):
                st.metric("10°C", f"{design_55:.2f} bar(g)")
                st.metric("10°C", f"{design_43:.2f} bar(g)")
                st.metric("10°C", f"{design_43:.2f} bar(g)")
            elif dp_standard == "BS EN 378":
                st.metric("55°C", f"{design_55:.2f} bar(g)")
                st.metric("43°C", f"{design_43:.2f} bar(g)")
                st.metric("32°C", f"{design_32:.2f} bar(g)")
            else:
                st.metric("50°C", f"{design_55:.2f} bar(g)")
                st.metric("40°C", f"{design_43:.2f} bar(g)")
                st.metric("27°C", f"{design_32:.2f} bar(g)")

    with col2:
        st.metric("System Design Pressure", f"{result['design_pressure_bar_g']:.2f} bar(g)")
        if refrigerant == "R744 TC" and circuit != "Suction":
            st.metric("", "")
        else:
            st.metric("Leak Test Pressure", f"{limits['leak_test']:.2f} bar(g)")
            st.metric("Pressure Test", f"{limits['pressure_test']:.2f} bar(g)")

    with col3:
        if refrigerant == "R744 TC" and circuit != "Suction":
            st.metric("", "")
        else:
            if circuit in ("Suction", "Pumped"):
                st.metric("High Pressure Cut-out", "N/A")
            else:
                st.metric("High Pressure Cut-out", f"{limits['hp_cutout']:.2f} bar(g)")
            st.metric("Relief Valve Setting", f"{limits['relief_setting']:.2f} bar(g)")
            st.metric("Relief Valve Rated Discharge", f"{limits['rated_discharge']:.2f} bar(g)")

    with col4:
        if mwp_multi:
            st.metric("MWP @ 50°C", f"{mwp_multi[50]:.2f} bar(g)")
            st.metric("MWP @ 100°C", f"{mwp_multi[100]:.2f} bar(g)")
            st.metric("MWP @ 150°C", f"{mwp_multi[150]:.2f} bar(g)")

    with col5:
        mwp = result["mwp_bar"]
        passed = result["pass"]
        margin = result["margin_bar"]
    
        if isinstance(mwp, dict):
            for weld_key in ["seamless", "erw", "cw"]:
                if weld_key not in mwp:
                    continue
    
                st.metric(
                    f"MWP – {weld_key.upper()}",
                    f"{mwp[weld_key]:.2f} bar(g)"
                )
    
                if passed[weld_key]:
                    st.success(
                        f"✅ {weld_key.upper()} PASS "
                        f"(margin {margin[weld_key]:.2f} bar)"
                    )
                else:
                    st.error(
                        f"❌ {weld_key.upper()} FAIL "
                        f"(margin {margin[weld_key]:.2f} bar)"
                    )
    
        else:
            st.metric(
                "Maximum Working Pressure (MWP)",
                f"{mwp:.2f} bar(g)"
            )
    
            if passed:
                st.success(
                    f"✅ Pipe rated for system pressure "
                    f"(margin {margin:.2f} bar)"
                )
            else:
                st.error(
                    f"❌ Pipe NOT rated: MWP {mwp:.2f} bar < "
                    f"Design {result['design_pressure_bar_g']:.2f} bar(g)"
                )

    with col6:
        if max_strength is not None:
            st.metric("Minimum Strength Test", f"{min_strength:.2f} bar(g)")
            st.metric("Maximum Strength Test @ 50°C", f"{max_strength:.2f} bar(g)")

if tool_selection == "Pipe Network Builder":
    builder = NetworkBuilder()
    builder.run()

elif tool_selection == "Pressure ↔ Temperature Converter":
    st.subheader("Saturation Pressure ↔ Temperature Tool")
    converter = PressureTemperatureConverter()

    refrigerant = st.selectbox("Refrigerant", [
        "R404A", "R134a", "R407F", "R744", "R410A",
        "R407C", "R507A", "R448A", "R449A", "R22", "R32", "R454A", "R454C", "R455A", "R407A",
        "R290", "R1270", "R600a", "R717", "R1234ze", "R1234yf", "R12", "R11", "R454B", "R450A", "R513A", "R23", "R508B", "R502"
    ])
    
    col1, col2, col3 = st.columns(3)

    with col1:
        mode = st.radio("Convert:", ["Pressure ➞ Temperature", "Temperature ➞ Pressure"])
    with col2:
        reference = st.radio("Temperature Reference:", ["Dew Point", "Bubble Point"])
    with col3:
        absgauge = st.radio("Pressure Reference:", ["Absolute", "Gauge"])
        
    if reference == "Dew Point":
        if mode == "Pressure ➞ Temperature":
            if absgauge == "Absolute":
                pressure_bar = st.number_input("Saturation Pressure (bar(a))", value=5.0)
                temp_C = converter.pressure_to_temp(refrigerant, pressure_bar)
            else:
                pressure_bar = st.number_input("Saturation Pressure (bar(g))", value=5.0)
                temp_C = converter.pressure_to_temp(refrigerant, pressure_bar + 1.01325)
            st.write(f"**Saturation Temperature:** {temp_C:.2f} °C")
        else:
            temp_C = st.number_input("Saturation Temperature (°C)", value=0.0)
            if absgauge == "Absolute":
                pressure_bar = converter.temp_to_pressure(refrigerant, temp_C)
                st.write(f"**Saturation Pressure:** {pressure_bar:.2f} bar(a)")
            else:
                pressure_bar = converter.temp_to_pressure(refrigerant, temp_C) - 1.01325
                st.write(f"**Saturation Pressure:** {pressure_bar:.2f} bar(g)")
    else:
        if mode == "Pressure ➞ Temperature":
            if absgauge == "Absolute":
                pressure_bar = st.number_input("Saturation Pressure (bar(a))", value=5.0)
                temp_C = converter.pressure2_to_temp(refrigerant, pressure_bar)
            else:
                pressure_bar = st.number_input("Saturation Pressure (bar(g))", value=5.0)
                temp_C = converter.pressure2_to_temp(refrigerant, pressure_bar + 1.01325)
            st.write(f"**Saturation Temperature:** {temp_C:.2f} °C")
        else:
            temp_C = st.number_input("Saturation Temperature (°C)", value=0.0)
            if absgauge == "Absolute":
                pressure_bar = converter.temp_to_pressure2(refrigerant, temp_C)
                st.write(f"**Saturation Pressure:** {pressure_bar:.2f} bar(a)")
            else:
                pressure_bar = converter.temp_to_pressure2(refrigerant, temp_C) - 1.01325
                st.write(f"**Saturation Pressure:** {pressure_bar:.2f} bar(g)")

elif tool_selection == "Pressure Drop ↔ Temperature Penalty":
    st.subheader("Pressure Drop ⇄ Temperature Penalty Tool")
    converter = PressureTemperatureConverter()

    refrigerant = st.selectbox("Refrigerant", [
        "R404A", "R134a", "R407F", "R744", "R410A",
        "R407C", "R507A", "R448A", "R449A", "R22", "R32", "R454A", "R454C", "R455A", "R407A",
        "R290", "R1270", "R600a", "R717", "R1234ze", "R1234yf", "R12", "R11", "R454B", "R450A", "R513A", "R23", "R508B", "R502"
    ])
    T_sat = st.number_input("Saturation Temperature (°C)", value=-10.0)
    
    col1, col2 = st.columns(2)
    with col1:
        direction = st.radio("Convert:", ["ΔP ➞ ΔT", "ΔT ➞ ΔP"])
    with col2:
        reference = st.radio("Temperature Reference:", ["Dew Point", "Bubble Point"])
        
    if reference == "Dew Point":
        if direction == "ΔP ➞ ΔT":
            delta_p_kpa = st.number_input("Pressure Drop (kPa)", value=20.0)
            delta_T = converter.pressure_drop_to_temp_penalty(refrigerant, T_sat, delta_p_kpa)
            st.write(f"**Temperature Penalty:** {delta_T:.3f} K")
        else:
            delta_T = st.number_input("Temperature Penalty (K)", value=0.5)
            delta_p_kpa = converter.temp_penalty_to_pressure_drop(refrigerant, T_sat, delta_T)
            st.write(f"**Equivalent Pressure Drop:** {delta_p_kpa:.2f} kPa")
    else:
        if direction == "ΔP ➞ ΔT":
            delta_p_kpa = st.number_input("Pressure Drop (kPa)", value=20.0)
            delta_T = converter.pressure2_drop_to_temp_penalty(refrigerant, T_sat, delta_p_kpa)
            st.write(f"**Temperature Penalty:** {delta_T:.3f} K")
        else:
            delta_T = st.number_input("Temperature Penalty (K)", value=0.5)
            delta_p_kpa = converter.temp_penalty_to_pressure2_drop(refrigerant, T_sat, delta_T)
            st.write(f"**Equivalent Pressure Drop:** {delta_p_kpa:.2f} kPa")

elif tool_selection == "System Pressure Checker":
    system_pressure_checker_ui()

elif tool_selection == "Oil Return Checker":
    st.subheader("Oil Return Checker")
    
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        refrigerant = st.selectbox("Refrigerant", [
            "R404A", "R134a", "R407F", "R744", "R744 TC", "R410A",
            "R407C", "R507A", "R448A", "R449A", "R22", "R32", "R454A", "R454C", "R455A", "R407A",
            "R290", "R1270", "R600a", "R717", "R1234ze", "R1234yf", "R12", "R11", "R454B", "R450A", "R513A", "R23", "R508B", "R502"
        ])

    # Load pipe data
    pipe_data = pd.read_csv("data/pipe_pressure_ratings_full.csv")

    # --- helpers ---
    def _nps_inch_to_mm(nps_str: str) -> float:
        # e.g. "1-1/8", '1"', "3/8"
        s = str(nps_str).replace('"', '').strip()
        if not s:
            return float('nan')
        parts = s.split('-')
        tot_in = 0.0
        for p in parts:
            p = p.strip()
            if not p:
                continue
            if '/' in p:
                num, den = p.split('/')
                tot_in += float(num) / float(den)
            else:
                tot_in += float(p)
        return tot_in * 25.4  # mm

    ss = st.session_state

    # 1) Pipe material
    with col2:
        if refrigerant == "R717":
            excluded_materials = ["Copper ASTM", " Copper EN12735", "K65 Copper", "Reflok Aluminium"]
            pipe_materials = sorted(m for m in pipe_data["Material"].dropna().unique()
                                    if m not in excluded_materials)
        else:
            pipe_materials = sorted(pipe_data["Material"].dropna().unique())

        selected_material = st.selectbox("Pipe Material", pipe_materials, key="material")
    
    # detect material change
    material_changed = ss.get("last_material") is not None and ss.last_material != selected_material
    ss.last_material = selected_material

    # 2) Sizes for selected material (de-duped)
    material_df = pipe_data[pipe_data["Material"] == selected_material].copy()

    sizes_df = (
        material_df[["Nominal Size (inch)", "Nominal Size (mm)"]]
        .dropna(subset=["Nominal Size (inch)"])
        .assign(**{
            "Nominal Size (inch)": lambda d: d["Nominal Size (inch)"].astype(str).str.strip(),
        })
        .drop_duplicates(subset=["Nominal Size (inch)"], keep="first")
    )

    # make sure we have a numeric mm per nominal (fallback: parse the inch string)
    sizes_df["mm_num"] = pd.to_numeric(sizes_df.get("Nominal Size (mm)"), errors="coerce")
    sizes_df.loc[sizes_df["mm_num"].isna(), "mm_num"] = sizes_df.loc[sizes_df["mm_num"].isna(), "Nominal Size (inch)"].apply(_nps_inch_to_mm)

    pipe_sizes = sizes_df["Nominal Size (inch)"].tolist()
    mm_map = dict(zip(sizes_df["Nominal Size (inch)"], sizes_df["mm_num"]))

    # choose default index
    def _closest_index(target_mm: float) -> int:
        mm_list = [mm_map[s] for s in pipe_sizes]
        return min(range(len(mm_list)), key=lambda i: abs(mm_list[i] - target_mm)) if mm_list else 0

    default_index = 0
    if material_changed and "prev_pipe_mm" in ss:
        default_index = _closest_index(ss.prev_pipe_mm)
    elif selected_material == " Copper EN12735" and ("1-1/8" in pipe_sizes or '1-1/8"' in pipe_sizes):
        # first load or no previous selection → prefer 1-1/8" for  Copper EN12735
        want = "1-1/8" if "1-1/8" in pipe_sizes else '1-1/8"'
        default_index = pipe_sizes.index(want)
    elif "selected_size" in ss and ss.selected_size in pipe_sizes:
        # if Streamlit kept the selection, use it
        default_index = pipe_sizes.index(ss.selected_size)

    disable_valves = st.session_state.get("double_trouble", False)
    
    with col1:
        selected_size = st.selectbox(
            "Nominal Pipe Size (inch)",
            pipe_sizes,
            index=default_index,
            key="selected_size",
            disabled=disable_valves,
        )

    ss.prev_pipe_mm = float(mm_map.get(selected_size, float("nan")))

    # 3) Gauge (if applicable)
    gauge_options = material_df[material_df["Nominal Size (inch)"].astype(str).str.strip() == selected_size]
    if "Gauge" in gauge_options.columns and gauge_options["Gauge"].notna().any():
        gauges = sorted(gauge_options["Gauge"].dropna().unique())
        with col2:
            selected_gauge = st.selectbox("Copper Gauge", gauges, key="gauge", disabled=disable_valves)
        selected_pipe_row = gauge_options[gauge_options["Gauge"] == selected_gauge].iloc[0]
    else:
        selected_pipe_row = gauge_options.iloc[0]

    # Pipe parameters
    pipe_size_inch = selected_pipe_row["Nominal Size (inch)"]
    ID_mm = selected_pipe_row["ID_mm"]

    def gauges_for_size(size_inch: str):
        rows = material_df[material_df["Nominal Size (inch)"].astype(str).str.strip() == str(size_inch)]
        if "Gauge" in rows.columns and rows["Gauge"].notna().any():
            return sorted(rows["Gauge"].dropna().unique())
        return []

    with col1:
        evap_capacity_kw = st.number_input("Evaporator Capacity (kW)", min_value=0.03, max_value=20000.0, value=10.0, step=1.0)

        # --- Base ranges per refrigerant ---
        if refrigerant in ("R23", "R508B"):
            evap_min, evap_max, evap_default = -100.0, -20.0, -80.0
            maxliq_min, maxliq_max, maxliq_default = -100.0, 10.0, -30.0
            minliq_min, minliq_max, minliq_default = -100.0, 10.0, -40.0
        elif refrigerant == "R744":
            evap_min, evap_max, evap_default = -50.0, 20.0, -10.0
            maxliq_min, maxliq_max, maxliq_default = -50.0, 30.0, 15.0
            minliq_min, minliq_max, minliq_default = -50.0, 30.0, 10.0
        else:
            evap_min, evap_max, evap_default = -50.0, 30.0, -10.0
            maxliq_min, maxliq_max, maxliq_default = -50.0, 60.0, 40.0
            minliq_min, minliq_max, minliq_default = -50.0, 60.0, 20.0
    
        # --- Init state (widget-backed) ---
        ss = st.session_state
    
        if "last_refrigerant" not in ss or ss.last_refrigerant != refrigerant:
            ss.maxliq_temp   = maxliq_default
            ss.minliq_temp = minliq_default
            ss.evap_temp   = evap_default
            ss.last_refrigerant = refrigerant
        
        ss.setdefault("maxliq_temp",   maxliq_default)
        ss.setdefault("minliq_temp", minliq_default)
        ss.setdefault("evap_temp",   evap_default)
    
        if "minliq_temp" in ss and "maxliq_temp" in ss:
            ss.minliq_temp = min(ss.maxliq_temp, ss.minliq_temp)
    
        if "minliq_temp" in ss and "maxliq_temp" in ss and "evap_temp" in ss:
            ss.evap_temp = min(ss.maxliq_temp, ss.minliq_temp, ss.evap_temp)
    
        # --- Callbacks implementing your downstream clamping logic ---
        def on_change_maxliq():
            # When cond changes: clamp minliq down to cond, then evap down to minliq
            ss.minliq_temp = min(ss.minliq_temp, ss.maxliq_temp)
            ss.evap_temp   = min(ss.evap_temp,   ss.minliq_temp)
    
        def on_change_minliq():
            # When minliq changes: clamp minliq down to cond, then evap down to minliq
            ss.minliq_temp = min(ss.minliq_temp, ss.maxliq_temp)
            ss.evap_temp   = min(ss.evap_temp,   ss.minliq_temp)
    
        def on_change_evap():
            # When evap changes: clamp evap down to minliq
            ss.evap_temp   = min(ss.evap_temp,   ss.minliq_temp)
    
        # --- Inputs with inclusive caps (≤), same order as your code ---
        if refrigerant == "R744 TC":
            # --- Split Max conditions into half-width boxes ---
            max_col1, max_col2 = st.columns(2)
        
            with max_col1:
                ss.setdefault("gc_max_temp", 38.0)
                gc_max_temp = st.number_input(
                    "Max GC Out Temp (°C)",
                    min_value=-50.0, max_value=50.0,
                    value=ss.gc_max_temp, step=1.0, key="gc_max_temp"
                )
        
            with max_col2:
                ss.setdefault("gc_max_pres", 93.7)
                gc_max_pres = st.number_input(
                    "Max GC Out Pressure (bar(a))",
                    min_value=73.8, max_value=150.0,
                    value=ss.gc_max_pres, step=1.0, key="gc_max_pres"
                )
        
            # --- Split Min conditions into half-width boxes ---
            min_col1, min_col2 = st.columns(2)
        
            with min_col1:
                ss.setdefault("gc_min_temp", 5.0)
                gc_min_temp = st.number_input(
                    "Min GC Out Temp (°C)",
                    min_value=-50.0, max_value=50.0,
                    value=ss.gc_min_temp, step=1.0, key="gc_min_temp"
                )
        
            with min_col2:
                ss.setdefault("gc_min_pres", 75.0)
                gc_min_pres = st.number_input(
                    "Min GC Out Pressure (bar(a))",
                    min_value=40.0, max_value=150.0,
                    value=ss.gc_min_pres, step=1.0, key="gc_min_pres"
                )
        
            # These assignments replace the old "maxliq_temp" and "minliq_temp"
            maxliq_temp = gc_max_temp
            minliq_temp = gc_min_temp

            ss.evap_temp = min(ss.evap_temp, minliq_temp, maxliq_temp)
        
        else:
            # --- Original inputs for normal refrigerants ---
            maxliq_temp = st.number_input(
                "Max Liquid Temperature (°C)",
                min_value=maxliq_min, max_value=maxliq_max,
                value=ss.maxliq_temp, step=1.0, key="maxliq_temp",
                on_change=on_change_maxliq,
            )
        
            minliq_temp = st.number_input(
                "Min Liquid Temperature (°C)",
                min_value=minliq_min, max_value=min(maxliq_temp, minliq_max),
                value=ss.minliq_temp, step=1.0, key="minliq_temp",
                on_change=on_change_minliq,
            )
    
        evaporating_temp = st.number_input(
            "Evaporating Temperature (°C)",
            min_value=evap_min, max_value=min(minliq_temp, evap_max),
            value=ss.evap_temp, step=1.0, key="evap_temp",
            on_change=on_change_evap,
        )

    with col2:
        superheat_K = st.number_input("Superheat (K)", min_value=0.0, max_value=60.0, value=5.0, step=1.0)
        max_penalty = st.number_input("Max Penalty (K)", min_value=0.0, max_value=6.0, value=1.0, step=0.1)
        required_oil_duty_pct = st.number_input("Required Oil Return Duty (%)", min_value=0.0, max_value=100.0, value=100.0, step=5.0)

    with col3:
        L = st.number_input("Pipe Length (m)", min_value=0.1, max_value=300.0, value=10.0, step=1.0, key="L", disabled=True)
        LRB = st.number_input("Long Radius Bends", min_value=0, max_value=50, value=0, step=1, key="LRB", disabled=True)
        SRB = st.number_input("Short Radius Bends", min_value=0, max_value=50, value=0, step=1, key="SRB", disabled=True)
        _45 = st.number_input("45° Bends", min_value=0, max_value=50, value=0, step=1, key="_45", disabled=True)
        MAC = st.number_input("Machine Bends", min_value=0, max_value=50, value=0, step=1, key="MAC", disabled=True)

    if st.session_state.get("double_trouble"):
        st.session_state.ball = 0
        st.session_state.globe = 0

    with col4:
        ptrap = st.number_input("P Traps", min_value=0, max_value=10, value=0, step=1, key="ptrap", disabled=True)
        ubend = st.number_input("U Bends", min_value=0, max_value=10, value=0, step=1, key="ubend", disabled=True)
        ball = st.number_input("Ball Valves", min_value=0, max_value=20, value=0, step=1, key="ball", disabled=True)
        globe = st.number_input("Globe Valves", min_value=0, max_value=20, value=0, step=1, key="globe", disabled=True)
        PLF = st.number_input("Pressure Loss Factors", min_value=0.0, max_value=20.0, value=0.0, step=0.1, key="PLF", disabled=True)

    disable_pipes = not st.session_state.get("double_trouble", False)
    
    def size_mm(size):
        return mm_map.get(size, 0.0)
    
    def on_change_large():
        if size_mm(ss.manual_large) < size_mm(ss.manual_small):
            ss.manual_small = ss.manual_large
    
    def on_change_small():
        if size_mm(ss.manual_small) > size_mm(ss.manual_large):
            ss.manual_large = ss.manual_small

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        manual_large = st.selectbox(
            "Large Riser Size",
            pipe_sizes,
            index=max(pipe_sizes.index(selected_size), 0),
            key="manual_large",
            on_change=on_change_large,
            disabled=disable_pipes
        )
    with col2:
        g_large_opts = gauges_for_size(manual_large)
        gauge_large = None
        if g_large_opts:
            gauge_large = st.selectbox("Large Riser Gauge", g_large_opts, key="gauge_large", disabled=disable_pipes)
    with col3:
        manual_small = st.selectbox(
            "Small Riser Size",
            pipe_sizes,
            index=max(pipe_sizes.index(selected_size) - 2, 0),
            key="manual_small",
            on_change=on_change_small,
            disabled=disable_pipes
        )
    with col4:
        g_small_opts = gauges_for_size(manual_small)
        gauge_small = None
        if g_small_opts:
            gauge_small = st.selectbox("Small Riser Gauge", g_small_opts, key="gauge_small", disabled=disable_pipes)

    from utils.refrigerant_properties import RefrigerantProperties
    from utils.refrigerant_densities import RefrigerantDensities
    from utils.refrigerant_viscosities import RefrigerantViscosities
    from utils.supercompliq_co2 import RefrigerantProps
    from utils.oil_return_checker import check_oil_return

    T_evap = evaporating_temp
    T_cond = maxliq_temp

    props_sup = RefrigerantProps()
    props = RefrigerantProperties()

    if refrigerant == "R744 TC":
        
        h_in = props_sup.get_enthalpy_sup(gc_max_pres, maxliq_temp)
        if gc_min_pres >= 73.8: 
            h_inmin = props_sup.get_enthalpy_sup(gc_min_pres, minliq_temp)
        elif gc_min_pres <= 72.13:
            h_inmin = props.get_properties("R744", minliq_temp)["enthalpy_liquid2"]
        else:
            st.error("This pressure range (72.13–73.8 bar) is not allowed. Please choose another value.")
            st.stop()
        h_inlet = h_in
        h_inletmin = h_inmin
        h_evap = props.get_properties("R744", T_evap)["enthalpy_vapor"]
        h_10K = props.get_properties("R744", T_evap)["enthalpy_super"]

    else:
        h_in = props.get_properties(refrigerant, T_cond)["enthalpy_liquid2"]
        #st.write("h_in:", h_in)
        # for velocity
        h_inmin = props.get_properties(refrigerant, minliq_temp)["enthalpy_liquid2"]
        #st.write("h_inmin:", h_inmin)
        h_inlet = props.get_properties(refrigerant, T_cond)["enthalpy_liquid"]
        #st.write("h_inlet:", h_inlet)
        h_inletmin = props.get_properties(refrigerant, minliq_temp)["enthalpy_liquid"]
        #st.write("h_inletmin:", h_inletmin)
        h_evap = props.get_properties(refrigerant, T_evap)["enthalpy_vapor"]
        #st.write("h_evap:", h_evap)
        h_10K = props.get_properties(refrigerant, T_evap)["enthalpy_super"]
        #st.write("h_10K:", h_10K)
    
    hdiff_10K = h_10K - h_evap
    #st.write("hdiff_10K:", hdiff_10K)
    hdiff_custom = hdiff_10K * min(max(superheat_K, 5), 30) / 10
    #st.write("hdiff_custom:", hdiff_custom)
    h_super = h_evap + hdiff_custom
    #st.write("h_super:", h_super)
    h_foroil = (h_evap + h_super) / 2
    #st.write("h_foroil:", h_foroil)
    
    delta_h = h_evap - h_in
    #st.write("delta_h:", delta_h)
    delta_hmin = h_evap - h_inmin
    #st.write("delta_hmin:", delta_hmin)
    
    delta_h_foroil = h_foroil - h_inlet
    #st.write("delta_h_foroil:", delta_h_foroil)
    delta_h_foroilmin = h_foroil - h_inletmin
    #st.write("delta_h_foroilmin:", delta_h_foroilmin)

    mass_flow_kg_s = evap_capacity_kw / delta_h if delta_h > 0 else 0.01
    #st.write("mass_flow_kg_s:", mass_flow_kg_s)
    mass_flow_kg_smin = evap_capacity_kw / delta_hmin if delta_hmin > 0 else 0.01
    #st.write("mass_flow_kg_smin:", mass_flow_kg_smin)
    M_total = max(mass_flow_kg_s, mass_flow_kg_smin)

    mass_flow_foroil = evap_capacity_kw / delta_h_foroil if delta_h_foroil > 0 else 0.01
    #st.write("mass_flow_foroil:", mass_flow_foroil)
    mass_flow_foroilmin = evap_capacity_kw / delta_h_foroilmin if delta_h_foroilmin > 0 else 0.01
    #st.write("mass_flow_foroilmin:", mass_flow_foroilmin)

    # Calculate velocity for transparency
    if ID_mm is not None:
        ID_m = ID_mm / 1000.0
        #st.write("ID_mm:", ID_mm)
        #st.write("ID_m:", ID_m)
        area_m2 = math.pi * (ID_m / 2) ** 2
        #st.write("area_m2:", area_m2)

        if refrigerant == "R744 TC":
            density_super = RefrigerantDensities().get_density("R744", T_evap - max_penalty + 273.15, superheat_K)
            density_super2a = RefrigerantDensities().get_density("R744", T_evap + 273.15, ((superheat_K + 5) / 2))
            density_super2b = RefrigerantDensities().get_density("R744", T_evap - max_penalty + 273.15, ((superheat_K + 5) / 2))
            density_super2 = (density_super2a + density_super2b) / 2
            density_super_foroil = RefrigerantDensities().get_density("R744", T_evap + 273.15, min(max(superheat_K, 5), 30))
            density_sat = RefrigerantProperties().get_properties("R744", T_evap)["density_vapor"]
            density_5K = RefrigerantDensities().get_density("R744", T_evap + 273.15, 5)    
    
        else:
            density_super = RefrigerantDensities().get_density(refrigerant, T_evap - max_penalty + 273.15, superheat_K)
            #st.write("density_super:", density_super)
            density_super2a = RefrigerantDensities().get_density(refrigerant, T_evap + 273.15, ((superheat_K + 5) / 2))
            #st.write("density_super2a:", density_super2a)
            density_super2b = RefrigerantDensities().get_density(refrigerant, T_evap - max_penalty + 273.15, ((superheat_K + 5) / 2))
            #st.write("density_super2b:", density_super2b)
            density_super2 = (density_super2a + density_super2b) / 2
            #st.write("density_super2:", density_super2)
            density_super_foroil = RefrigerantDensities().get_density(refrigerant, T_evap + 273.15, min(max(superheat_K, 5), 30))
            #st.write("density_super_foroil:", density_super_foroil)
            density_sat = RefrigerantProperties().get_properties(refrigerant, T_evap)["density_vapor"]
            #st.write("density_sat:", density_sat)
            density_5K = RefrigerantDensities().get_density(refrigerant, T_evap + 273.15, 5)
            #st.write("density_5K:", density_5K)
        
        density = (density_super + density_5K) / 2
        #st.write("density:", density)
        density_foroil = (density_super_foroil + density_sat) / 2
        #st.write("density_foroil:", density_foroil)
        velocity_m_s1 = mass_flow_kg_s / (area_m2 * density)
        #st.write("velocity_m_s1:", velocity_m_s1)
        velocity_m_s1min = mass_flow_kg_smin / (area_m2 * density)
        #st.write("velocity_m_s1min:", velocity_m_s1min)
        velocity_m_s2 = mass_flow_kg_s / (area_m2 * density_super2)
        #st.write("velocity_m_s2:", velocity_m_s2)
        velocity_m_s2min = mass_flow_kg_smin / (area_m2 * density_super2)
        #st.write("velocity_m_s2min:", velocity_m_s2min)
        if refrigerant == "R744": velocity1_prop = 1
        elif refrigerant == "R744 TC": velocity1_prop = 1
        elif refrigerant == "R404A":
            if superheat_K > 45: velocity1_prop = (0.0328330590542629 * superheat_K) - 1.47748765744183
            else: velocity1_prop = 0
        elif refrigerant == "R134a":
            if superheat_K > 30: velocity1_prop = (-0.000566085879684639 * (superheat_K ** 2)) + (0.075049554857083 * superheat_K) - 1.74200935399632
            else: velocity1_prop = 0
        elif refrigerant == "R407F": velocity1_prop = 1
        elif refrigerant == "R407A": velocity1_prop = 1
        elif refrigerant == "R410A": velocity1_prop = 1
        elif refrigerant == "R407C": velocity1_prop = 0
        elif refrigerant == "R22": velocity1_prop = 1
        elif refrigerant == "R502": velocity1_prop = 1
        elif refrigerant == "R507A": velocity1_prop = 1
        elif refrigerant == "R448A": velocity1_prop = 1
        elif refrigerant == "R449A": velocity1_prop = 1
        elif refrigerant == "R717": velocity1_prop = 1
        else:
            if superheat_K > 30: velocity1_prop = (0.0000406422632403154 * (superheat_K ** 2)) - (0.000541007136813307 * superheat_K) + 0.748882946418884
            else: velocity1_prop = 0.769230769230769
        # if refrigerant == "R744": velocity1_prop = (-0.0142814388381874 * max(superheat_K, 5)) + 1.07140719419094
        # else: velocity1_prop = (-0.00280805561137312 * max(superheat_K, 5)) + 1.01404027805687
        #st.write("velocity1_prop:", velocity1_prop)
        velocity_m_s = (velocity_m_s1 * velocity1_prop) + (velocity_m_s2 * (1 - velocity1_prop))
        #st.write("velocity_m_s:", velocity_m_s)
        velocity_m_smin = (velocity_m_s1min * velocity1_prop) + (velocity_m_s2min * (1 - velocity1_prop))
        #st.write("velocity_m_smin:", velocity_m_smin)
        if refrigerant in ["R23", "R508B"]:
            oil_density_sat = (-0.853841209044878 * T_evap) + 999.190772536527
            oil_density_super = (-0.853841209044878 * (T_evap + min(max(superheat_K, 5), 30))) + 999.190772536527
        else:
            oil_density_sat = (-0.00356060606060549 * (T_evap ** 2)) - (0.957878787878808 * T_evap) + 963.595454545455
            oil_density_super = (-0.00356060606060549 * ((T_evap + min(max(superheat_K, 5), 30)) ** 2)) - (0.957878787878808 * (T_evap + min(max(superheat_K, 5), 30))) + 963.595454545455
        #st.write("oil_density_sat:", oil_density_sat)
        #st.write("oil_density_super:", oil_density_super)
        oil_density = (oil_density_sat + oil_density_super) / 2
        #st.write("oil_density:", oil_density)
        
        if refrigerant == "R404A": jg_half = 0.860772464072673
        elif refrigerant == "R134a": jg_half = 0.869986729796935
        elif refrigerant == "R407F": jg_half = 0.869042493641944
        elif refrigerant == "R744": jg_half = 0.877950613678719
        elif refrigerant == "R744 TC": jg_half = 0.877950613678719
        elif refrigerant == "R407A": jg_half = 0.867374311574041
        elif refrigerant == "R410A": jg_half = 0.8904423325365
        elif refrigerant == "R407C": jg_half = 0.858592104849471
        elif refrigerant == "R22": jg_half = 0.860563058394146
        elif refrigerant == "R502": jg_half = 0.858236706656266
        elif refrigerant == "R507A": jg_half = 0.887709710291009
        elif refrigerant == "R449A": jg_half = 0.867980496631757
        elif refrigerant == "R448A": jg_half = 0.86578818145833
        elif refrigerant == "R717": jg_half = 0.854957410951708
        elif refrigerant == "R290": jg_half = 0.844975139695726
        elif refrigerant == "R1270": jg_half = 0.849089717732815
        elif refrigerant == "R600a": jg_half = 0.84339338979887
        elif refrigerant == "R1234ze": jg_half = 0.867821375349728
        elif refrigerant == "R1234yf": jg_half = 0.860767472602571
        elif refrigerant == "R12": jg_half = 0.8735441986466
        elif refrigerant == "R11": jg_half = 0.864493203834913
        elif refrigerant == "R454B": jg_half = 0.869102255850291
        elif refrigerant == "R450A": jg_half = 0.865387140496035
        elif refrigerant == "R513A": jg_half = 0.861251244627232
        elif refrigerant == "R454A": jg_half = 0.868161104592492
        elif refrigerant == "R455A": jg_half = 0.865687329727713
        elif refrigerant == "R454C": jg_half = 0.866423016875524
        elif refrigerant == "R32": jg_half = 0.875213309852597
        elif refrigerant == "R23": jg_half = 0.865673418568001
        elif refrigerant == "R508B": jg_half = 0.864305626845382
        #st.write("jg_half:", jg_half)
        
        MinMassFlux = (jg_half ** 2) * ((density_foroil * 9.81 * ID_m * (oil_density - density_foroil)) ** 0.5)
        #st.write("MinMassFluxy:", MinMassFlux)
        MinMassFlow = MinMassFlux * area_m2
        #st.write("MinMassFlow:", MinMassFlow)
        MOR_pre = (MinMassFlow / mass_flow_foroil) * 100
        #st.write("MOR_pre:", MOR_pre)
        MOR_premin = (MinMassFlow / mass_flow_foroilmin) * 100
        #st.write("MOR_premin:", MOR_premin)

        if refrigerant in ["R23", "R508B"]:
            MOR_correctliq = T_cond + 47.03
        else:
            MOR_correctliq = T_cond
        if refrigerant in ["R23", "R508B"]:
            MOR_correctliqmin = minliq_temp + 47.03
        else:
            MOR_correctliqmin = minliq_temp
        if refrigerant in ["R23", "R508B"]:
            evapoil = T_evap + 46.14
        else:
            evapoil = T_evap
        #st.write("MOR_correctliq:", MOR_correctliq)
        #st.write("evapoil:", evapoil)
        if refrigerant == "R744": MOR_correction = (0.000225755013421421 * MOR_correctliq) - 0.00280879370374927
        elif refrigerant == "R744 TC": MOR_correction = (0.0000603336117708171 * h_in) - 0.0142318718120024
        elif refrigerant == "R407A": MOR_correction = (0.00000414431651323856 * (MOR_correctliq ** 2)) + (0.000381908525139781 * MOR_correctliq) - 0.0163450053041212
        elif refrigerant == "R449A": MOR_correction = (0.00000414431651323856 * (MOR_correctliq ** 2)) + (0.000381908525139781 * MOR_correctliq) - 0.0163450053041212
        elif refrigerant == "R448A": MOR_correction = (0.00000414431651323856 * (MOR_correctliq ** 2)) + (0.000381908525139781 * MOR_correctliq) - 0.0163450053041212
        elif refrigerant == "R502": MOR_correction = (0.00000414431651323856 * (MOR_correctliq ** 2)) + (0.000381908525139781 * MOR_correctliq) - 0.0163450053041212
        elif refrigerant == "R507A": MOR_correction = (0.000302619054048837 * MOR_correctliq) - 0.00930188913363997
        elif refrigerant == "R22": MOR_correction = (0.000108153843367715 * MOR_correctliq) - 0.00329248681202757
        elif refrigerant == "R407C": MOR_correction = (0.00000420322918839302 * (max(MOR_correctliq, -32.0716410083429) ** 2)) + (0.000269608915211859 * max(MOR_correctliq, -32.0716410083429)) - 0.0134546663857195
        elif refrigerant == "R410A": MOR_correction = 0
        elif refrigerant == "R407F": MOR_correction = (0.00000347332380289385 * (max(MOR_correctliq, -34.4346433150568) ** 2)) + (0.000239205332540693 * max(MOR_correctliq, -34.4346433150568)) - 0.0121545316131988
        elif refrigerant == "R134a": MOR_correction = (0.000195224660107459 * MOR_correctliq) - 0.00591757011487048
        elif refrigerant == "R404A": MOR_correction = (0.0000156507169104918 * (max(MOR_correctliq, -22.031637377024) ** 2)) + (0.000689621839324826 * max(MOR_correctliq, -22.031637377024)) - 0.0392
        else: MOR_correction = (0.00000461020482461793 * (max(MOR_correctliq, -23.6334996273983) ** 2)) + (0.000217910548009675 * max(MOR_correctliq, -23.6334996273983)) - 0.012074621594626
        #st.write("MOR_correction:", MOR_correction)

        if refrigerant == "R744": MOR_correctionmin = (0.000225755013421421 * MOR_correctliqmin) - 0.00280879370374927
        elif refrigerant == "R744 TC": MOR_correctionmin = (0.0000603336117708171 * h_inmin) - 0.0142318718120024
        elif refrigerant == "R407A": MOR_correctionmin = (0.00000414431651323856 * (MOR_correctliqmin ** 2)) + (0.000381908525139781 * MOR_correctliqmin) - 0.0163450053041212
        elif refrigerant == "R449A": MOR_correctionmin = (0.00000414431651323856 * (MOR_correctliqmin ** 2)) + (0.000381908525139781 * MOR_correctliqmin) - 0.0163450053041212
        elif refrigerant == "R448A": MOR_correctionmin = (0.00000414431651323856 * (MOR_correctliqmin ** 2)) + (0.000381908525139781 * MOR_correctliqmin) - 0.0163450053041212
        elif refrigerant == "R502": MOR_correctionmin = (0.00000414431651323856 * (MOR_correctliqmin ** 2)) + (0.000381908525139781 * MOR_correctliqmin) - 0.0163450053041212
        elif refrigerant == "R507A": MOR_correctionmin = (0.000302619054048837 * MOR_correctliqmin) - 0.00930188913363997
        elif refrigerant == "R22": MOR_correctionmin = (0.000108153843367715 * MOR_correctliqmin) - 0.00329248681202757
        elif refrigerant == "R407C": MOR_correctionmin = (0.00000420322918839302 * (max(MOR_correctliqmin, -32.0716410083429) ** 2)) + (0.000269608915211859 * max(MOR_correctliqmin, -32.0716410083429)) - 0.0134546663857195
        elif refrigerant == "R410A": MOR_correctionmin = 0
        elif refrigerant == "R407F": MOR_correctionmin = (0.00000347332380289385 * (max(MOR_correctliqmin, -34.4346433150568) ** 2)) + (0.000239205332540693 * max(MOR_correctliqmin, -34.4346433150568)) - 0.0121545316131988
        elif refrigerant == "R134a": MOR_correctionmin = (0.000195224660107459 * MOR_correctliqmin) - 0.00591757011487048
        elif refrigerant == "R404A": MOR_correctionmin = (0.0000156507169104918 * (max(MOR_correctliqmin, -22.031637377024) ** 2)) + (0.000689621839324826 * max(MOR_correctliqmin, -22.031637377024)) - 0.0392
        else: MOR_correctionmin = (0.00000461020482461793 * (max(MOR_correctliqmin, -23.6334996273983) ** 2)) + (0.000217910548009675 * max(MOR_correctliqmin, -23.6334996273983)) - 0.012074621594626
        #st.write("MOR_correctionmin:", MOR_correctionmin)

        if refrigerant == "R744": MOR_correction2 = (-0.0000176412848988908 * (evapoil ** 2)) - (0.00164308248808803 * evapoil) - 0.0184308798286039
        elif refrigerant == "R744 TC": MOR_correction2 = (-0.0000176412848988908 * (evapoil ** 2)) - (0.00164308248808803 * evapoil) - 0.0184308798286039
        elif refrigerant == "R407A": MOR_correction2 = (-0.000864076433837511 * evapoil) - 0.0145018190416687
        elif refrigerant == "R449A": MOR_correction2 = (-0.000835375233693285 * evapoil) - 0.0138846063856621
        elif refrigerant == "R448A": MOR_correction2 = (0.00000171366802431428 * (evapoil ** 2)) - (0.000865528727278154 * evapoil) - 0.0152961902042161
        elif refrigerant == "R502": MOR_correction2 = (0.00000484734071020993 * (evapoil ** 2)) - (0.000624822304716683 * evapoil) - 0.0128725684240106
        elif refrigerant == "R507A": MOR_correction2 = (-0.000701333343440148 * evapoil) - 0.0114900933623056
        elif refrigerant == "R22": MOR_correction2 = (0.00000636798209134899 * (evapoil ** 2)) - (0.000157783204337396 * evapoil) - 0.00575251626397381
        elif refrigerant == "R407C": MOR_correction2 = (-0.00000665735727676349 * (evapoil ** 2)) - (0.000894860288947537 * evapoil) - 0.0116054361757929
        elif refrigerant == "R410A": MOR_correction2 = (-0.000672268853990701 * evapoil) - 0.0111802230098585
        elif refrigerant == "R407F": MOR_correction2 = (0.00000263731418614519 * (evapoil ** 2)) - (0.000683997257738699 * evapoil) - 0.0126005968942147
        elif refrigerant == "R134a": MOR_correction2 = (-0.00000823045532174214 * (evapoil ** 2)) - (0.00108063672211041 * evapoil) - 0.0217411206961643
        elif refrigerant == "R404A": MOR_correction2 = (0.00000342378568620316 * (evapoil ** 2)) - (0.000329572335134041 * evapoil) - 0.00706087606597149
        else: MOR_correction2 = (-0.000711441807827186 * evapoil) - 0.0118194116436425
        #st.write("MOR_correction2:", MOR_correction2)
        
        if refrigerant in ["R23", "R508B"]:
            if T_evap < -86:
                MOR = ""
                MORmin = ""
                MORfinal = ""
            elif T_evap > -42:
                MOR = ""
                MORmin = ""
                MORfinal = ""
            else:
                MOR = (1 - MOR_correction) * (1 - MOR_correction2) * MOR_pre
                MORmin = (1 - MOR_correctionmin) * (1 - MOR_correction2) * MOR_premin
                MORfinal = max(MOR, MORmin)
        else:    
            if T_evap < -40:
                MOR = ""
                MORmin = ""
                MORfinal = ""
            elif T_evap > 4:
                MOR = ""
                MORmin = ""
                MORfinal = ""
            else:
                MOR = (1 - MOR_correction) * (1 - MOR_correction2) * MOR_pre
                MORmin = (1 - MOR_correctionmin) * (1 - MOR_correction2) * MOR_premin
                MORfinal = max(MOR, MORmin)
        #st.write("MOR:", MOR)
        #st.write("MORmin:", MORmin)
        #st.write("MORfinal:", MORfinal)
        velocity_m_sfinal = max(velocity_m_s, velocity_m_smin)
        #st.write("velocity_m_sfinal:", velocity_m_sfinal)
    else:
        velocity_m_s = None
        velocity_m_smin = None
        velocity_m_sfinal = None

    # Oil return check
    adjusted_duty_kw = evap_capacity_kw * (required_oil_duty_pct / 100.0)
    #st.write("adjusted_duty_kw:", adjusted_duty_kw)

    density_recalc = mass_flow_kg_s / (velocity_m_s * area_m2)
    #st.write("density_recalc:", density_recalc)
    
    if MORfinal == "":
        MinCap = ""
    else:
        MinCap = MORfinal * evap_capacity_kw / 100

    def _pipe_row_for_size(size_inch: str, gauge: str | None = None):
        rows = material_df[
            material_df["Nominal Size (inch)"].astype(str).str.strip() == str(size_inch)
        ]
    
        if "Gauge" in rows.columns and rows["Gauge"].notna().any():
            if gauge is not None:
                rows_g = rows[rows["Gauge"] == gauge]
                if not rows_g.empty:
                    return rows_g.iloc[0]
            # fallback if gauge is None or invalid
            return rows.iloc[0]
    
        return rows.iloc[0]

    from utils.double_riser import RiserContext, balance_double_riser
    
    # Only meaningful for R744 TC
    gc_max = gc_max_pres if refrigerant == "R744 TC" else None
    gc_min = gc_min_pres if refrigerant == "R744 TC" else None
    
    ctx = RiserContext(
        refrigerant=refrigerant,
        T_evap=T_evap,
        T_cond=T_cond,
        minliq_temp=minliq_temp,
        superheat_K=superheat_K,
        max_penalty_K=max_penalty,
    
        L=L,
        SRB=SRB,
        LRB=LRB,
        bends_45=_45,
        MAC=MAC,
        ptrap=ptrap,
        ubend=ubend,
        ball=0,
        globe=0,
        PLF=PLF,
    
        selected_material=selected_material,
        pipe_row_for_size=_pipe_row_for_size,
    
        gc_max_pres=gc_max,
        gc_min_pres=gc_min,
    )
    
    double_trouble = st.checkbox("Double Riser Mode", key="double_trouble")

    if double_trouble:
        dr = balance_double_riser(
            manual_small,
            manual_large,
            M_total,
            ctx,
            gauge_small=gauge_small,
            gauge_large=gauge_large,
        )

        rs = dr.small_result
        rl = dr.large_result

        from utils.double_riser import compute_double_riser_oil_metrics

        MOR_full_flow, MOR_large, SST, M_largeprop = compute_double_riser_oil_metrics(
            dr=dr,
            refrigerant=refrigerant,
            T_evap=T_evap,
            density_foroil=density_foroil,
            oil_density=oil_density,
            jg_half=jg_half,
            mass_flow_foroil=mass_flow_foroil,
            mass_flow_foroilmin=mass_flow_foroilmin,
            MOR_correction=MOR_correction,
            MOR_correctionmin=MOR_correctionmin,
            MOR_correction2=MOR_correction2,
        )

        if MOR_full_flow is None:
            MinCaps = ""
        else:
            MinCaps = MOR_full_flow * evap_capacity_kw / 100

        if MOR_large is None:
            MaxCaps = ""
        else:
            large_duty = M_largeprop * evap_capacity_kw
            MaxCaps = MOR_large * large_duty / 100
    
    if velocity_m_sfinal:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if double_trouble:
                if MOR_full_flow is None:
                    st.metric("Minimum Capacity (Secondary)", "")
                else:
                    st.metric("Minimum Capacity (Secondary)", f"{MinCaps:.4f}kW")
            else:
                if MORfinal == "":
                    st.metric("Minimum Capacity", "")
                else:                
                    st.metric("Minimum Capacity", f"{MinCap:.4f}kW")

        with col2:
            if double_trouble:
                if MOR_full_flow is None:
                    st.metric("Minimum Oil Return", "")
                else:
                    st.metric("Minimum Oil Return", f"{MOR_full_flow:.1f}%")
            else:
                if MORfinal == "":
                    st.metric("Minimum Oil Return", "")
                else:
                    st.metric("Minimum Oil Return", f"{MORfinal:.1f}%")
        
        with col3:
            if double_trouble:
                if MOR_large is None:
                    st.metric("Minimum Capacity (Primary)", "")
                else:
                    st.metric("Minimum Capacity (Primary)", f"{MaxCaps:.4f}kW")
                    
        with col4:
            if double_trouble:
                if MOR_large is None:
                    st.metric("Maximum Oil Return", "")
                else:
                    st.metric("Maximum Oil Return", f"{MOR_large:.1f}%")

    if isinstance(MORfinal, (int, float)):
        is_ok, message = (True, "✅ OK") if required_oil_duty_pct >= MORfinal else (False, "❌ Insufficient flow")
    else:
        is_ok, message = (False, "")

    if is_ok:
        st.success(f"{message}")
    else:
        st.error(f"{message}")

elif tool_selection == "Manual Calculation":

    mode = st.radio("", ["Dry Suction", "Liquid", "Discharge", "Drain", "Pumped Liquid", "Wet Suction"], index=0, horizontal=True, label_visibility="collapsed")

    def circuit_for_manual_mode(mode: str) -> str:
        return {
            "Dry Suction": "Suction",
            "Wet Suction": "Suction",
            "Discharge": "Discharge",
            "Liquid": "Liquid",
            "Pumped Liquid": "Pumped",
            "Drain": "Liquid",  # main/branch low-side liquid
        }[mode]

    st.session_state["circuit"] = circuit_for_manual_mode(mode)
    circuit = st.session_state["circuit"]

    colx, cola, colb, colc = st.columns(4)

    with colx:
        refrigerant = st.selectbox("Refrigerant", [
            "R404A", "R134a", "R407F", "R744", "R744 TC", "R410A",
            "R407C", "R507A", "R448A", "R449A", "R22", "R32", "R454A", "R454C", "R455A", "R407A",
            "R290", "R1270", "R600a", "R717", "R1234ze", "R1234yf", "R12", "R11", "R454B", "R450A", "R513A", "R23", "R508B", "R502"
        ])

    with cola:
        dp_standard = st.selectbox(
            "Design Pressure Standard",
            ["BS EN 378", "ASME B31.5 - 2006"],
            index=0,
            key="manual_dp_standard",
        )

    with colb:
        copper_calc = st.selectbox(
            "Copper MWP Calculation Standard",
            ["BS1306", "DKI"],
            index=0,
            key="manual_copper_calc",
        )

    ctx = pressure_checker_inputs(
        refrigerant=refrigerant,
        circuit=circuit,
        dp_standard=dp_standard,
    )
    refrigerant_eff = ctx["refrigerant"]
    mwp_temp_c = ctx["mwp_temp_c"]

    with colc:
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
                    min_value=ctx["range_min_low"],
                    max_value=ctx["range_max_low"],
                    value=ctx["default_low_dt"],
                    step=1.0,
                    key="manual_design_temp_low",
                )
            else:
                design_temp_c = st.number_input(
                    "Design Temperature (°C)",
                    min_value=ctx["range_min_high"],
                    max_value=ctx["range_max_high"],
                    value=ctx["default_high_dt"],
                    step=1.0,
                    key="manual_design_temp_high",
                )
                
            r744_tc_pressure_bar_g = None
    
    if mode == "Dry Suction":
        
        # Load pipe data
        pipe_data = pd.read_csv("data/pipe_pressure_ratings_full.csv")
    
        # --- helpers ---
        def _nps_inch_to_mm(nps_str: str) -> float:
            # e.g. "1-1/8", '1"', "3/8"
            s = str(nps_str).replace('"', '').strip()
            if not s:
                return float('nan')
            parts = s.split('-')
            tot_in = 0.0
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                if '/' in p:
                    num, den = p.split('/')
                    tot_in += float(num) / float(den)
                else:
                    tot_in += float(p)
            return tot_in * 25.4  # mm
    
        ss = st.session_state
    
        # 1) Pipe material

        col1, col2, col3, col4 = st.columns(4)
        with col2:
            if refrigerant == "R717":
                excluded_materials = ["Copper ASTM", " Copper EN12735", "K65 Copper", "Reflok Aluminium"]
                pipe_materials = sorted(m for m in pipe_data["Material"].dropna().unique()
                                        if m not in excluded_materials)
            else:
                pipe_materials = sorted(pipe_data["Material"].dropna().unique())
    
            selected_material = st.selectbox("Pipe Material", pipe_materials, key="material")
        
        # detect material change
        material_changed = ss.get("last_material") is not None and ss.last_material != selected_material
        ss.last_material = selected_material
    
        # 2) Sizes for selected material (de-duped)
        material_df = pipe_data[pipe_data["Material"] == selected_material].copy()
    
        sizes_df = (
            material_df[["Nominal Size (inch)", "Nominal Size (mm)"]]
            .dropna(subset=["Nominal Size (inch)"])
            .assign(**{
                "Nominal Size (inch)": lambda d: d["Nominal Size (inch)"].astype(str).str.strip(),
            })
            .drop_duplicates(subset=["Nominal Size (inch)"], keep="first")
        )
    
        # make sure we have a numeric mm per nominal (fallback: parse the inch string)
        sizes_df["mm_num"] = pd.to_numeric(sizes_df.get("Nominal Size (mm)"), errors="coerce")
        sizes_df.loc[sizes_df["mm_num"].isna(), "mm_num"] = sizes_df.loc[sizes_df["mm_num"].isna(), "Nominal Size (inch)"].apply(_nps_inch_to_mm)
    
        pipe_sizes = sizes_df["Nominal Size (inch)"].tolist()
        mm_map = dict(zip(sizes_df["Nominal Size (inch)"], sizes_df["mm_num"]))
    
        # choose default index
        def _closest_index(target_mm: float) -> int:
            mm_list = [mm_map[s] for s in pipe_sizes]
            return min(range(len(mm_list)), key=lambda i: abs(mm_list[i] - target_mm)) if mm_list else 0

        # --- Handle deferred pipe selection (from "Select Optimal Pipe Size" button) ---
        if "_next_selected_size" in st.session_state:
            new_val = st.session_state["_next_selected_size"]
            # ✅ directly set the selectbox value itself
            st.session_state["selected_size"] = new_val
            # clean up the temporary flag
            del st.session_state["_next_selected_size"]

        if "_next_double_riser" in st.session_state:
            st.session_state["double_trouble"] = True
            st.session_state["manual_small"] = st.session_state["_next_manual_small"]
            st.session_state["manual_large"] = st.session_state["_next_manual_large"]
        
            del st.session_state["_next_double_riser"]
            del st.session_state["_next_manual_small"]
            del st.session_state["_next_manual_large"]

        default_index = 0
        override_val = st.session_state.get("selected_size_override")
        if override_val and override_val in pipe_sizes:
            default_index = pipe_sizes.index(override_val)
        elif material_changed and "prev_pipe_mm" in ss:
            default_index = _closest_index(ss.prev_pipe_mm)
        elif selected_material == " Copper EN12735" and ("1-1/8" in pipe_sizes or '1-1/8"' in pipe_sizes):
            want = "1-1/8" if "1-1/8" in pipe_sizes else '1-1/8"'
            default_index = pipe_sizes.index(want)
        elif "selected_size" in ss and ss.selected_size in pipe_sizes:
            default_index = pipe_sizes.index(ss.selected_size)

        disable_valves = st.session_state.get("double_trouble", False)
        
        with col1:
            selected_size = st.selectbox(
                "Nominal Pipe Size (inch)",
                pipe_sizes,
                index=default_index,
                key="selected_size",
                disabled=disable_valves,
            )

        ss.prev_pipe_mm = float(mm_map.get(selected_size, float("nan")))
    
        # 3) Gauge (if applicable)
        gauge_options = material_df[material_df["Nominal Size (inch)"].astype(str).str.strip() == selected_size]

        if "_next_gauge_main" in st.session_state:
            g = st.session_state.pop("_next_gauge_main")
        
            if "Gauge" in gauge_options.columns:
                valid = set(gauge_options["Gauge"].dropna())
                if g in valid:
                    st.session_state["gauge"] = g
        
        if "Gauge" in gauge_options.columns and gauge_options["Gauge"].notna().any():
            gauges = sorted(gauge_options["Gauge"].dropna().unique())
            with col2:
                selected_gauge = st.selectbox("Copper Gauge", gauges, key="gauge", disabled=disable_valves)
            selected_pipe_row = gauge_options[gauge_options["Gauge"] == selected_gauge].iloc[0]
        else:
            selected_pipe_row = gauge_options.iloc[0]

        pipe_index = material_to_pipe_index(selected_material)
        
        # you already have selected_gauge sometimes; otherwise None
        gauge = st.session_state.get("gauge")  # or whatever that mode uses
        od_mm, id_mm = get_dimensions_for_row(material_df, selected_size, gauge)
    
        # Pipe parameters
        pipe_size_inch = selected_pipe_row["Nominal Size (inch)"]
        ID_mm = selected_pipe_row["ID_mm"]

        def gauges_for_size(size_inch: str):
            rows = material_df[material_df["Nominal Size (inch)"].astype(str).str.strip() == str(size_inch)]
            if "Gauge" in rows.columns and rows["Gauge"].notna().any():
                return sorted(rows["Gauge"].dropna().unique())
            return []
    
        with col1:
            evap_capacity_kw = st.number_input("Evaporator Capacity (kW)", min_value=0.03, max_value=20000.0, value=10.0, step=1.0)
    
            # --- Base ranges per refrigerant ---
            if refrigerant in ("R23", "R508B"):
                evap_min, evap_max, evap_default = -100.0, -20.0, -80.0
                maxliq_min, maxliq_max, maxliq_default = -100.0, 10.0, -30.0
                minliq_min, minliq_max, minliq_default = -100.0, 10.0, -40.0
            elif refrigerant == "R744":
                evap_min, evap_max, evap_default = -50.0, 20.0, -10.0
                maxliq_min, maxliq_max, maxliq_default = -50.0, 30.0, 15.0
                minliq_min, minliq_max, minliq_default = -50.0, 30.0, 10.0
            else:
                evap_min, evap_max, evap_default = -50.0, 30.0, -10.0
                maxliq_min, maxliq_max, maxliq_default = -50.0, 60.0, 40.0
                minliq_min, minliq_max, minliq_default = -50.0, 60.0, 20.0
    
            # --- Init state (widget-backed) ---
            ss = st.session_state
    
            if "last_refrigerant" not in ss or ss.last_refrigerant != refrigerant:
                ss.maxliq_temp   = maxliq_default
                ss.minliq_temp = minliq_default
                ss.evap_temp   = evap_default
                ss.last_refrigerant = refrigerant
            
            ss.setdefault("maxliq_temp",   maxliq_default)
            ss.setdefault("minliq_temp", minliq_default)
            ss.setdefault("evap_temp",   evap_default)

            if "minliq_temp" in ss and "maxliq_temp" in ss:
                ss.minliq_temp = min(ss.maxliq_temp, ss.minliq_temp)

            if "minliq_temp" in ss and "maxliq_temp" in ss and "evap_temp" in ss:
                ss.evap_temp = min(ss.maxliq_temp, ss.minliq_temp, ss.evap_temp)
    
            # --- Callbacks implementing your downstream clamping logic ---
            def on_change_maxliq():
                # When cond changes: clamp minliq down to cond, then evap down to minliq
                ss.minliq_temp = min(ss.minliq_temp, ss.maxliq_temp)
                ss.evap_temp   = min(ss.evap_temp,   ss.minliq_temp)
    
            def on_change_minliq():
                # When minliq changes: clamp minliq down to cond, then evap down to minliq
                ss.minliq_temp = min(ss.minliq_temp, ss.maxliq_temp)
                ss.evap_temp   = min(ss.evap_temp,   ss.minliq_temp)
    
            def on_change_evap():
                # When evap changes: clamp evap down to minliq
                ss.evap_temp   = min(ss.evap_temp,   ss.minliq_temp)
    
            # --- Inputs with inclusive caps (≤), same order as your code ---
            if refrigerant == "R744 TC":
                # --- Split Max conditions into half-width boxes ---
                max_col1, max_col2 = st.columns(2)
            
                with max_col1:
                    ss.setdefault("gc_max_temp", 38.0)
                    gc_max_temp = st.number_input(
                        "Max GC Out Temp (°C)",
                        min_value=-50.0, max_value=50.0,
                        value=ss.gc_max_temp, step=1.0, key="gc_max_temp"
                    )
            
                with max_col2:
                    ss.setdefault("gc_max_pres", 93.7)
                    gc_max_pres = st.number_input(
                        "Max GC Out Pressure (bar(a))",
                        min_value=73.8, max_value=150.0,
                        value=ss.gc_max_pres, step=1.0, key="gc_max_pres"
                    )
            
                # --- Split Min conditions into half-width boxes ---
                min_col1, min_col2 = st.columns(2)
            
                with min_col1:
                    ss.setdefault("gc_min_temp", 5.0)
                    gc_min_temp = st.number_input(
                        "Min GC Out Temp (°C)",
                        min_value=-50.0, max_value=50.0,
                        value=ss.gc_min_temp, step=1.0, key="gc_min_temp"
                    )
            
                with min_col2:
                    ss.setdefault("gc_min_pres", 75.0)
                    gc_min_pres = st.number_input(
                        "Min GC Out Pressure (bar(a))",
                        min_value=40.0, max_value=150.0,
                        value=ss.gc_min_pres, step=1.0, key="gc_min_pres"
                    )
            
                # These assignments replace the old "maxliq_temp" and "minliq_temp"
                maxliq_temp = gc_max_temp
                minliq_temp = gc_min_temp

                ss.evap_temp = min(ss.evap_temp, minliq_temp, maxliq_temp)
            
            else:
                # --- Original inputs for normal refrigerants ---
                maxliq_temp = st.number_input(
                    "Max Liquid Temperature (°C)",
                    min_value=maxliq_min, max_value=maxliq_max,
                    value=ss.maxliq_temp, step=1.0, key="maxliq_temp",
                    on_change=on_change_maxliq,
                )
            
                minliq_temp = st.number_input(
                    "Min Liquid Temperature (°C)",
                    min_value=minliq_min, max_value=min(maxliq_temp, minliq_max),
                    value=ss.minliq_temp, step=1.0, key="minliq_temp",
                    on_change=on_change_minliq,
                )

            evaporating_temp = st.number_input(
                "Evaporating Temperature (°C)",
                min_value=evap_min, max_value=min(minliq_temp, evap_max),
                value=ss.evap_temp, step=1.0, key="evap_temp",
                on_change=on_change_evap,
            )
    
        with col2:
            superheat_K = st.number_input("Superheat (K)", min_value=0.0, max_value=60.0, value=5.0, step=1.0)
            max_penalty = st.number_input("Max Penalty (K)", min_value=0.0, max_value=6.0, value=1.0, step=0.1)
            required_oil_duty_pct = st.number_input("Required Oil Return Duty (%)", min_value=0.0, max_value=100.0, value=100.0, step=5.0)
    
        with col3:
            L = st.number_input("Pipe Length (m)", min_value=0.1, max_value=300.0, value=10.0, step=1.0, key="L")
            LRB = st.number_input("Long Radius Bends", min_value=0, max_value=50, value=0, step=1, key="LRB")
            SRB = st.number_input("Short Radius Bends", min_value=0, max_value=50, value=0, step=1, key="SRB")
            _45 = st.number_input("45° Bends", min_value=0, max_value=50, value=0, step=1, key="_45")
            MAC = st.number_input("Machine Bends", min_value=0, max_value=50, value=0, step=1, key="MAC")

        if st.session_state.get("double_trouble"):
            st.session_state.ball = 0
            st.session_state.globe = 0

        with col4:
            ptrap = st.number_input("P Traps", min_value=0, max_value=10, value=0, step=1, key="ptrap")
            ubend = st.number_input("U Bends", min_value=0, max_value=10, value=0, step=1, key="ubend")
            ball = st.number_input("Ball Valves", min_value=0, max_value=20, value=0, step=1, key="ball", disabled=disable_valves)
            globe = st.number_input("Globe Valves", min_value=0, max_value=20, value=0, step=1, key="globe", disabled=disable_valves)
            PLF = st.number_input("Pressure Loss Factors", min_value=0.0, max_value=20.0, value=0.0, step=0.1, key="PLF")
        
        disable_pipes = not st.session_state.get("double_trouble", False)
        
        def size_mm(size):
            return mm_map.get(size, 0.0)
        
        def on_change_large():
            if size_mm(ss.manual_large) < size_mm(ss.manual_small):
                ss.manual_small = ss.manual_large
        
        def on_change_small():
            if size_mm(ss.manual_small) > size_mm(ss.manual_large):
                ss.manual_large = ss.manual_small

        if "_next_gauge_large" in st.session_state:
            g = st.session_state.pop("_next_gauge_large")
        
            rows_large = material_df[
                material_df["Nominal Size (inch)"].astype(str).str.strip()
                == str(st.session_state.get("manual_large", selected_size))
            ]
        
            if "Gauge" in rows_large.columns:
                valid = set(rows_large["Gauge"].dropna())
                if g in valid:
                    st.session_state["gauge_large"] = g
        
        if "_next_gauge_small" in st.session_state:
            g = st.session_state.pop("_next_gauge_small")
        
            rows_small = material_df[
                material_df["Nominal Size (inch)"].astype(str).str.strip()
                == str(st.session_state.get("manual_small", selected_size))
            ]
        
            if "Gauge" in rows_small.columns:
                valid = set(rows_small["Gauge"].dropna())
                if g in valid:
                    st.session_state["gauge_small"] = g
        
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            manual_large = st.selectbox(
                "Large Riser Size",
                pipe_sizes,
                index=max(pipe_sizes.index(selected_size), 0),
                key="manual_large",
                on_change=on_change_large,
                disabled=disable_pipes
            )
        with col2:
            g_large_opts = gauges_for_size(manual_large)
            gauge_large = None
            if g_large_opts:
                gauge_large = st.selectbox("Large Riser Gauge", g_large_opts, key="gauge_large", disabled=disable_pipes)
        with col3:
            manual_small = st.selectbox(
                "Small Riser Size",
                pipe_sizes,
                index=max(pipe_sizes.index(selected_size) - 2, 0),
                key="manual_small",
                on_change=on_change_small,
                disabled=disable_pipes
            )
        with col4:
            g_small_opts = gauges_for_size(manual_small)
            gauge_small = None
            if g_small_opts:
                gauge_small = st.selectbox("Small Riser Gauge", g_small_opts, key="gauge_small", disabled=disable_pipes)

        # build selected_pipe_row_large
        rows_large = material_df[material_df["Nominal Size (inch)"].astype(str).str.strip() == str(manual_large)]
        
        if "Gauge" in rows_large.columns and rows_large["Gauge"].notna().any():
            row_large = rows_large[rows_large["Gauge"] == gauge_large].iloc[0]
        else:
            row_large = rows_large.iloc[0]
        
        # build selected_pipe_row_small
        rows_small = material_df[material_df["Nominal Size (inch)"].astype(str).str.strip() == str(manual_small)]
        
        if "Gauge" in rows_small.columns and rows_small["Gauge"].notna().any():
            row_small = rows_small[rows_small["Gauge"] == gauge_small].iloc[0]
        else:
            row_small = rows_small.iloc[0]
        
        pipe_index_large = material_to_pipe_index(selected_material)
        pipe_index_small = pipe_index_large  # same material
        
        od_large, id_large = get_dimensions_for_row(material_df, manual_large, gauge_large)
        od_small, id_small = get_dimensions_for_row(material_df, manual_small, gauge_small)

        if st.session_state.get("double_trouble"):
            result = system_pressure_check_double_riser(
                refrigerant=refrigerant,
                design_temp_c=design_temp_c,
                mwp_temp_c=mwp_temp_c,
                circuit="Suction",
                dp_standard=dp_standard,
            
                pipe_index_a=pipe_index_large,
                od_mm_a=od_large,
                id_mm_a=id_large,
                gauge_a=gauge_large,
            
                pipe_index_b=pipe_index_small,
                od_mm_b=od_small,
                id_mm_b=id_small,
                gauge_b=gauge_small,
            
                copper_calc=copper_calc,
                r744_tc_pressure_bar_g=r744_tc_pressure_bar_g,
            )
        else:
            result = system_pressure_check(
                refrigerant=refrigerant,
                design_temp_c=design_temp_c,
                mwp_temp_c=mwp_temp_c,
                circuit=circuit,
                pipe_index=pipe_index,
                od_mm=od_mm,
                id_mm=id_mm,
                gauge=gauge,
                copper_calc=copper_calc,
                r744_tc_pressure_bar_g=r744_tc_pressure_bar_g,
                dp_standard=dp_standard,
            )
        render_pressure_result(result)
        
        from utils.refrigerant_properties import RefrigerantProperties
        from utils.refrigerant_densities import RefrigerantDensities
        from utils.refrigerant_viscosities import RefrigerantViscosities
        from utils.supercompliq_co2 import RefrigerantProps
        from utils.oil_return_checker import check_oil_return
    
        T_evap = evaporating_temp
        T_cond = maxliq_temp

        props_sup = RefrigerantProps()
        props = RefrigerantProperties()
        
        if refrigerant == "R744 TC":
            
            h_in = props_sup.get_enthalpy_sup(gc_max_pres, maxliq_temp)
            if gc_min_pres >= 73.8: 
                h_inmin = props_sup.get_enthalpy_sup(gc_min_pres, minliq_temp)
            elif gc_min_pres <= 72.13:
                h_inmin = props.get_properties("R744", minliq_temp)["enthalpy_liquid2"]
            else:
                st.error("This pressure range (72.13–73.8 bar) is not allowed. Please choose another value.")
                st.stop()
            h_inlet = h_in
            h_inletmin = h_inmin
            h_evap = props.get_properties("R744", T_evap)["enthalpy_vapor"]
            h_10K = props.get_properties("R744", T_evap)["enthalpy_super"]
    
        else:
            
            h_in = props.get_properties(refrigerant, T_cond)["enthalpy_liquid2"]
            #st.write("h_in:", h_in)
            # for velocity
            h_inmin = props.get_properties(refrigerant, minliq_temp)["enthalpy_liquid2"]
            #st.write("h_inmin:", h_inmin)
            h_inlet = props.get_properties(refrigerant, T_cond)["enthalpy_liquid"]
            #st.write("h_inlet:", h_inlet)
            h_inletmin = props.get_properties(refrigerant, minliq_temp)["enthalpy_liquid"]
            #st.write("h_inletmin:", h_inletmin)
            h_evap = props.get_properties(refrigerant, T_evap)["enthalpy_vapor"]
            #st.write("h_evap:", h_evap)
            h_10K = props.get_properties(refrigerant, T_evap)["enthalpy_super"]
            #st.write("h_10K:", h_10K)
        
        hdiff_10K = h_10K - h_evap
        #st.write("hdiff_10K:", hdiff_10K)
        hdiff_custom = hdiff_10K * min(max(superheat_K, 5), 30) / 10
        #st.write("hdiff_custom:", hdiff_custom)
        h_super = h_evap + hdiff_custom
        #st.write("h_super:", h_super)
        h_foroil = (h_evap + h_super) / 2
        #st.write("h_foroil:", h_foroil)
        
        delta_h = h_evap - h_in
        #st.write("delta_h:", delta_h)
        delta_hmin = h_evap - h_inmin
        #st.write("delta_hmin:", delta_hmin)
        
        delta_h_foroil = h_foroil - h_inlet
        #st.write("delta_h_foroil:", delta_h_foroil)
        delta_h_foroilmin = h_foroil - h_inletmin
        #st.write("delta_h_foroilmin:", delta_h_foroilmin)
    
        mass_flow_kg_s = evap_capacity_kw / delta_h if delta_h > 0 else 0.01
        #st.write("mass_flow_kg_s:", mass_flow_kg_s)
        mass_flow_kg_smin = evap_capacity_kw / delta_hmin if delta_hmin > 0 else 0.01
        #st.write("mass_flow_kg_smin:", mass_flow_kg_smin)
        M_total = max(mass_flow_kg_s, mass_flow_kg_smin)
    
        mass_flow_foroil = evap_capacity_kw / delta_h_foroil if delta_h_foroil > 0 else 0.01
        #st.write("mass_flow_foroil:", mass_flow_foroil)
        mass_flow_foroilmin = evap_capacity_kw / delta_h_foroilmin if delta_h_foroilmin > 0 else 0.01
        #st.write("mass_flow_foroilmin:", mass_flow_foroilmin)
    
        # Calculate velocity for transparency
        if ID_mm is not None:
            ID_m = ID_mm / 1000.0
            #st.write("ID_mm:", ID_mm)
            #st.write("ID_m:", ID_m)
            area_m2 = math.pi * (ID_m / 2) ** 2
            #st.write("area_m2:", area_m2)
            
            if refrigerant == "R744 TC":
                density_super = RefrigerantDensities().get_density("R744", T_evap - max_penalty + 273.15, superheat_K)
                density_super2a = RefrigerantDensities().get_density("R744", T_evap + 273.15, ((superheat_K + 5) / 2))
                density_super2b = RefrigerantDensities().get_density("R744", T_evap - max_penalty + 273.15, ((superheat_K + 5) / 2))
                density_super2 = (density_super2a + density_super2b) / 2
                density_super_foroil = RefrigerantDensities().get_density("R744", T_evap + 273.15, min(max(superheat_K, 5), 30))
                density_sat = RefrigerantProperties().get_properties("R744", T_evap)["density_vapor"]
                density_5K = RefrigerantDensities().get_density("R744", T_evap + 273.15, 5)    
        
            else:
            
                density_super = RefrigerantDensities().get_density(refrigerant, T_evap - max_penalty + 273.15, superheat_K)
                #st.write("density_super:", density_super)
                density_super2a = RefrigerantDensities().get_density(refrigerant, T_evap + 273.15, ((superheat_K + 5) / 2))
                #st.write("density_super2a:", density_super2a)
                density_super2b = RefrigerantDensities().get_density(refrigerant, T_evap - max_penalty + 273.15, ((superheat_K + 5) / 2))
                #st.write("density_super2b:", density_super2b)
                density_super2 = (density_super2a + density_super2b) / 2
                #st.write("density_super2:", density_super2)
                density_super_foroil = RefrigerantDensities().get_density(refrigerant, T_evap + 273.15, min(max(superheat_K, 5), 30))
                #st.write("density_super_foroil:", density_super_foroil)
                density_sat = RefrigerantProperties().get_properties(refrigerant, T_evap)["density_vapor"]
                #st.write("density_sat:", density_sat)
                density_5K = RefrigerantDensities().get_density(refrigerant, T_evap + 273.15, 5)
                #st.write("density_5K:", density_5K)
            
            density = (density_super + density_5K) / 2
            #st.write("density:", density)
            density_foroil = (density_super_foroil + density_sat) / 2
            #st.write("density_foroil:", density_foroil)
            velocity_m_s1 = mass_flow_kg_s / (area_m2 * density)
            #st.write("velocity_m_s1:", velocity_m_s1)
            velocity_m_s1min = mass_flow_kg_smin / (area_m2 * density)
            #st.write("velocity_m_s1min:", velocity_m_s1min)
            velocity_m_s2 = mass_flow_kg_s / (area_m2 * density_super2)
            #st.write("velocity_m_s2:", velocity_m_s2)
            velocity_m_s2min = mass_flow_kg_smin / (area_m2 * density_super2)
            #st.write("velocity_m_s2min:", velocity_m_s2min)
            if refrigerant == "R744": velocity1_prop = 1
            elif refrigerant == "R744 TC": velocity1_prop = 1
            elif refrigerant == "R404A":
                if superheat_K > 45: velocity1_prop = (0.0328330590542629 * superheat_K) - 1.47748765744183
                else: velocity1_prop = 0
            elif refrigerant == "R134a":
                if superheat_K > 30: velocity1_prop = (-0.000566085879684639 * (superheat_K ** 2)) + (0.075049554857083 * superheat_K) - 1.74200935399632
                else: velocity1_prop = 0
            elif refrigerant == "R407F": velocity1_prop = 1
            elif refrigerant == "R407A": velocity1_prop = 1
            elif refrigerant == "R410A": velocity1_prop = 1
            elif refrigerant == "R407C": velocity1_prop = 0
            elif refrigerant == "R22": velocity1_prop = 1
            elif refrigerant == "R502": velocity1_prop = 1
            elif refrigerant == "R507A": velocity1_prop = 1
            elif refrigerant == "R448A": velocity1_prop = 1
            elif refrigerant == "R449A": velocity1_prop = 1
            elif refrigerant == "R717": velocity1_prop = 1
            else:
                if superheat_K > 30: velocity1_prop = (0.0000406422632403154 * (superheat_K ** 2)) - (0.000541007136813307 * superheat_K) + 0.748882946418884
                else: velocity1_prop = 0.769230769230769
            # if refrigerant == "R744": velocity1_prop = (-0.0142814388381874 * max(superheat_K, 5)) + 1.07140719419094
            # else: velocity1_prop = (-0.00280805561137312 * max(superheat_K, 5)) + 1.01404027805687
            #st.write("velocity1_prop:", velocity1_prop)
            velocity_m_s = (velocity_m_s1 * velocity1_prop) + (velocity_m_s2 * (1 - velocity1_prop))
            #st.write("velocity_m_s:", velocity_m_s)
            velocity_m_smin = (velocity_m_s1min * velocity1_prop) + (velocity_m_s2min * (1 - velocity1_prop))
            #st.write("velocity_m_smin:", velocity_m_smin)
            if refrigerant in ["R23", "R508B"]:
                oil_density_sat = (-0.853841209044878 * T_evap) + 999.190772536527
                oil_density_super = (-0.853841209044878 * (T_evap + min(max(superheat_K, 5), 30))) + 999.190772536527
            else:
                oil_density_sat = (-0.00356060606060549 * (T_evap ** 2)) - (0.957878787878808 * T_evap) + 963.595454545455
                oil_density_super = (-0.00356060606060549 * ((T_evap + min(max(superheat_K, 5), 30)) ** 2)) - (0.957878787878808 * (T_evap + min(max(superheat_K, 5), 30))) + 963.595454545455
            #st.write("oil_density_sat:", oil_density_sat)
            #st.write("oil_density_super:", oil_density_super)
            oil_density = (oil_density_sat + oil_density_super) / 2
            #st.write("oil_density:", oil_density)
            
            if refrigerant == "R404A": jg_half = 0.860772464072673
            elif refrigerant == "R134a": jg_half = 0.869986729796935
            elif refrigerant == "R407F": jg_half = 0.869042493641944
            elif refrigerant == "R744": jg_half = 0.877950613678719
            elif refrigerant == "R744 TC": jg_half = 0.877950613678719
            elif refrigerant == "R407A": jg_half = 0.867374311574041
            elif refrigerant == "R410A": jg_half = 0.8904423325365
            elif refrigerant == "R407C": jg_half = 0.858592104849471
            elif refrigerant == "R22": jg_half = 0.860563058394146
            elif refrigerant == "R502": jg_half = 0.858236706656266
            elif refrigerant == "R507A": jg_half = 0.887709710291009
            elif refrigerant == "R449A": jg_half = 0.867980496631757
            elif refrigerant == "R448A": jg_half = 0.86578818145833
            elif refrigerant == "R717": jg_half = 0.854957410951708
            elif refrigerant == "R290": jg_half = 0.844975139695726
            elif refrigerant == "R1270": jg_half = 0.849089717732815
            elif refrigerant == "R600a": jg_half = 0.84339338979887
            elif refrigerant == "R1234ze": jg_half = 0.867821375349728
            elif refrigerant == "R1234yf": jg_half = 0.860767472602571
            elif refrigerant == "R12": jg_half = 0.8735441986466
            elif refrigerant == "R11": jg_half = 0.864493203834913
            elif refrigerant == "R454B": jg_half = 0.869102255850291
            elif refrigerant == "R450A": jg_half = 0.865387140496035
            elif refrigerant == "R513A": jg_half = 0.861251244627232
            elif refrigerant == "R454A": jg_half = 0.868161104592492
            elif refrigerant == "R455A": jg_half = 0.865687329727713
            elif refrigerant == "R454C": jg_half = 0.866423016875524
            elif refrigerant == "R32": jg_half = 0.875213309852597
            elif refrigerant == "R23": jg_half = 0.865673418568001
            elif refrigerant == "R508B": jg_half = 0.864305626845382
            #st.write("jg_half:", jg_half)
            
            MinMassFlux = (jg_half ** 2) * ((density_foroil * 9.81 * ID_m * (oil_density - density_foroil)) ** 0.5)
            #st.write("MinMassFluxy:", MinMassFlux)
            MinMassFlow = MinMassFlux * area_m2
            #st.write("MinMassFlow:", MinMassFlow)
            MOR_pre = (MinMassFlow / mass_flow_foroil) * 100
            #st.write("MOR_pre:", MOR_pre)
            MOR_premin = (MinMassFlow / mass_flow_foroilmin) * 100
            #st.write("MOR_premin:", MOR_premin)
    
            if refrigerant in ["R23", "R508B"]:
                MOR_correctliq = T_cond + 47.03
            else:
                MOR_correctliq = T_cond
            if refrigerant in ["R23", "R508B"]:
                MOR_correctliqmin = minliq_temp + 47.03
            else:
                MOR_correctliqmin = minliq_temp
            if refrigerant in ["R23", "R508B"]:
                evapoil = T_evap + 46.14
            else:
                evapoil = T_evap
            #st.write("MOR_correctliq:", MOR_correctliq)
            #st.write("evapoil:", evapoil)
            if refrigerant == "R744": MOR_correction = (0.000225755013421421 * MOR_correctliq) - 0.00280879370374927
            elif refrigerant == "R744 TC": MOR_correction = (0.0000603336117708171 * h_in) - 0.0142318718120024
            elif refrigerant == "R407A": MOR_correction = (0.00000414431651323856 * (MOR_correctliq ** 2)) + (0.000381908525139781 * MOR_correctliq) - 0.0163450053041212
            elif refrigerant == "R449A": MOR_correction = (0.00000414431651323856 * (MOR_correctliq ** 2)) + (0.000381908525139781 * MOR_correctliq) - 0.0163450053041212
            elif refrigerant == "R448A": MOR_correction = (0.00000414431651323856 * (MOR_correctliq ** 2)) + (0.000381908525139781 * MOR_correctliq) - 0.0163450053041212
            elif refrigerant == "R502": MOR_correction = (0.00000414431651323856 * (MOR_correctliq ** 2)) + (0.000381908525139781 * MOR_correctliq) - 0.0163450053041212
            elif refrigerant == "R507A": MOR_correction = (0.000302619054048837 * MOR_correctliq) - 0.00930188913363997
            elif refrigerant == "R22": MOR_correction = (0.000108153843367715 * MOR_correctliq) - 0.00329248681202757
            elif refrigerant == "R407C": MOR_correction = (0.00000420322918839302 * (max(MOR_correctliq, -32.0716410083429) ** 2)) + (0.000269608915211859 * max(MOR_correctliq, -32.0716410083429)) - 0.0134546663857195
            elif refrigerant == "R410A": MOR_correction = 0
            elif refrigerant == "R407F": MOR_correction = (0.00000347332380289385 * (max(MOR_correctliq, -34.4346433150568) ** 2)) + (0.000239205332540693 * max(MOR_correctliq, -34.4346433150568)) - 0.0121545316131988
            elif refrigerant == "R134a": MOR_correction = (0.000195224660107459 * MOR_correctliq) - 0.00591757011487048
            elif refrigerant == "R404A": MOR_correction = (0.0000156507169104918 * (max(MOR_correctliq, -22.031637377024) ** 2)) + (0.000689621839324826 * max(MOR_correctliq, -22.031637377024)) - 0.0392
            else: MOR_correction = (0.00000461020482461793 * (max(MOR_correctliq, -23.6334996273983) ** 2)) + (0.000217910548009675 * max(MOR_correctliq, -23.6334996273983)) - 0.012074621594626
            #st.write("MOR_correction:", MOR_correction)
    
            if refrigerant == "R744": MOR_correctionmin = (0.000225755013421421 * MOR_correctliqmin) - 0.00280879370374927
            elif refrigerant == "R744 TC": MOR_correctionmin = (0.0000603336117708171 * h_inmin) - 0.0142318718120024
            elif refrigerant == "R407A": MOR_correctionmin = (0.00000414431651323856 * (MOR_correctliqmin ** 2)) + (0.000381908525139781 * MOR_correctliqmin) - 0.0163450053041212
            elif refrigerant == "R449A": MOR_correctionmin = (0.00000414431651323856 * (MOR_correctliqmin ** 2)) + (0.000381908525139781 * MOR_correctliqmin) - 0.0163450053041212
            elif refrigerant == "R448A": MOR_correctionmin = (0.00000414431651323856 * (MOR_correctliqmin ** 2)) + (0.000381908525139781 * MOR_correctliqmin) - 0.0163450053041212
            elif refrigerant == "R502": MOR_correctionmin = (0.00000414431651323856 * (MOR_correctliqmin ** 2)) + (0.000381908525139781 * MOR_correctliqmin) - 0.0163450053041212
            elif refrigerant == "R507A": MOR_correctionmin = (0.000302619054048837 * MOR_correctliqmin) - 0.00930188913363997
            elif refrigerant == "R22": MOR_correctionmin = (0.000108153843367715 * MOR_correctliqmin) - 0.00329248681202757
            elif refrigerant == "R407C": MOR_correctionmin = (0.00000420322918839302 * (max(MOR_correctliqmin, -32.0716410083429) ** 2)) + (0.000269608915211859 * max(MOR_correctliqmin, -32.0716410083429)) - 0.0134546663857195
            elif refrigerant == "R410A": MOR_correctionmin = 0
            elif refrigerant == "R407F": MOR_correctionmin = (0.00000347332380289385 * (max(MOR_correctliqmin, -34.4346433150568) ** 2)) + (0.000239205332540693 * max(MOR_correctliqmin, -34.4346433150568)) - 0.0121545316131988
            elif refrigerant == "R134a": MOR_correctionmin = (0.000195224660107459 * MOR_correctliqmin) - 0.00591757011487048
            elif refrigerant == "R404A": MOR_correctionmin = (0.0000156507169104918 * (max(MOR_correctliqmin, -22.031637377024) ** 2)) + (0.000689621839324826 * max(MOR_correctliqmin, -22.031637377024)) - 0.0392
            else: MOR_correctionmin = (0.00000461020482461793 * (max(MOR_correctliqmin, -23.6334996273983) ** 2)) + (0.000217910548009675 * max(MOR_correctliqmin, -23.6334996273983)) - 0.012074621594626
            #st.write("MOR_correctionmin:", MOR_correctionmin)
    
            if refrigerant == "R744": MOR_correction2 = (-0.0000176412848988908 * (evapoil ** 2)) - (0.00164308248808803 * evapoil) - 0.0184308798286039
            elif refrigerant == "R744 TC": MOR_correction2 = (-0.0000176412848988908 * (evapoil ** 2)) - (0.00164308248808803 * evapoil) - 0.0184308798286039
            elif refrigerant == "R407A": MOR_correction2 = (-0.000864076433837511 * evapoil) - 0.0145018190416687
            elif refrigerant == "R449A": MOR_correction2 = (-0.000835375233693285 * evapoil) - 0.0138846063856621
            elif refrigerant == "R448A": MOR_correction2 = (0.00000171366802431428 * (evapoil ** 2)) - (0.000865528727278154 * evapoil) - 0.0152961902042161
            elif refrigerant == "R502": MOR_correction2 = (0.00000484734071020993 * (evapoil ** 2)) - (0.000624822304716683 * evapoil) - 0.0128725684240106
            elif refrigerant == "R507A": MOR_correction2 = (-0.000701333343440148 * evapoil) - 0.0114900933623056
            elif refrigerant == "R22": MOR_correction2 = (0.00000636798209134899 * (evapoil ** 2)) - (0.000157783204337396 * evapoil) - 0.00575251626397381
            elif refrigerant == "R407C": MOR_correction2 = (-0.00000665735727676349 * (evapoil ** 2)) - (0.000894860288947537 * evapoil) - 0.0116054361757929
            elif refrigerant == "R410A": MOR_correction2 = (-0.000672268853990701 * evapoil) - 0.0111802230098585
            elif refrigerant == "R407F": MOR_correction2 = (0.00000263731418614519 * (evapoil ** 2)) - (0.000683997257738699 * evapoil) - 0.0126005968942147
            elif refrigerant == "R134a": MOR_correction2 = (-0.00000823045532174214 * (evapoil ** 2)) - (0.00108063672211041 * evapoil) - 0.0217411206961643
            elif refrigerant == "R404A": MOR_correction2 = (0.00000342378568620316 * (evapoil ** 2)) - (0.000329572335134041 * evapoil) - 0.00706087606597149
            else: MOR_correction2 = (-0.000711441807827186 * evapoil) - 0.0118194116436425
            #st.write("MOR_correction2:", MOR_correction2)
            
            if refrigerant in ["R23", "R508B"]:
                if T_evap < -86:
                    MOR = ""
                    MORmin = ""
                    MORfinal = ""
                elif T_evap > -42:
                    MOR = ""
                    MORmin = ""
                    MORfinal = ""
                else:
                    MOR = (1 - MOR_correction) * (1 - MOR_correction2) * MOR_pre
                    MORmin = (1 - MOR_correctionmin) * (1 - MOR_correction2) * MOR_premin
                    MORfinal = max(MOR, MORmin)
            else:    
                if T_evap < -40:
                    MOR = ""
                    MORmin = ""
                    MORfinal = ""
                elif T_evap > 4:
                    MOR = ""
                    MORmin = ""
                    MORfinal = ""
                else:
                    MOR = (1 - MOR_correction) * (1 - MOR_correction2) * MOR_pre
                    MORmin = (1 - MOR_correctionmin) * (1 - MOR_correction2) * MOR_premin
                    MORfinal = max(MOR, MORmin)
            #st.write("MOR:", MOR)
            #st.write("MORmin:", MORmin)
            #st.write("MORfinal:", MORfinal)
            velocity_m_sfinal = max(velocity_m_s, velocity_m_smin)
            #st.write("velocity_m_sfinal:", velocity_m_sfinal)
        else:
            velocity_m_s = None
            velocity_m_smin = None
            velocity_m_sfinal = None
    
        # Oil return check
        adjusted_duty_kw = evap_capacity_kw * (required_oil_duty_pct / 100.0)
        #st.write("adjusted_duty_kw:", adjusted_duty_kw)
    
        density_recalc = mass_flow_kg_s / (velocity_m_s * area_m2)
        #st.write("density_recalc:", density_recalc)
    
        if refrigerant == "R744 TC":
            
            viscosity_super = RefrigerantViscosities().get_viscosity("R744", T_evap - max_penalty + 273.15, superheat_K)
            viscosity_super2a = RefrigerantViscosities().get_viscosity("R744", T_evap + 273.15, ((superheat_K + 5) / 2))
            viscosity_super2b = RefrigerantViscosities().get_viscosity("R744", T_evap - max_penalty + 273.15, ((superheat_K + 5) / 2))
            viscosity_super2 = (viscosity_super2a + viscosity_super2b) / 2
            viscosity_sat = RefrigerantViscosities().get_viscosity("R744", T_evap + 273.15, 0)
            viscosity_5K = RefrigerantViscosities().get_viscosity("R744", T_evap + 273.15, 5)

        else:

            viscosity_super = RefrigerantViscosities().get_viscosity(refrigerant, T_evap - max_penalty + 273.15, superheat_K)
            #st.write("viscosity_super:", viscosity_super)
            viscosity_super2a = RefrigerantViscosities().get_viscosity(refrigerant, T_evap + 273.15, ((superheat_K + 5) / 2))
            #st.write("viscosity_super2a:", viscosity_super2a)
            viscosity_super2b = RefrigerantViscosities().get_viscosity(refrigerant, T_evap - max_penalty + 273.15, ((superheat_K + 5) / 2))
            #st.write("viscosity_super2b:", viscosity_super2b)
            viscosity_super2 = (viscosity_super2a + viscosity_super2b) / 2
            #st.write("viscosity_super2:", viscosity_super2)
            viscosity_sat = RefrigerantViscosities().get_viscosity(refrigerant, T_evap + 273.15, 0)
            #st.write("viscosity_sat:", viscosity_sat)
            viscosity_5K = RefrigerantViscosities().get_viscosity(refrigerant, T_evap + 273.15, 5)
            #st.write("viscosity_5K:", viscosity_5K)
        
        viscosity = (viscosity_super + viscosity_5K) / 2
        #st.write("viscosity:", viscosity)
        viscosity_final = (viscosity * velocity1_prop) + (viscosity_super2 * (1 - velocity1_prop))
        #st.write("viscosity_final:", viscosity_final)
    
        # density for reynolds and col2 display needs density_super2 factoring in!
        reynolds = (density_recalc * velocity_m_sfinal * ID_m) / (viscosity_final / 1000000)
        #st.write("reynolds:", reynolds)
    
        if selected_material in ["Steel SCH40", "Steel SCH80"]:
            eps = 0.00004572 #0.00015
        else:
            eps = 0.000001524 #0.000005
        
        tol = 1e-5
        max_iter = 60
        
        if reynolds < 2000.0:
            f = 64.0 / reynolds
        else:
            flo, fhi = 1e-5, 0.1
            def balance(gg):
                s = math.sqrt(gg)
                lhs = 1.0 / s
                rhs = -2.0 * math.log10((eps / (3.7 * ID_m)) + 2.51 / (reynolds * s))
                return lhs, rhs
    
            f = 0.5 * (flo + fhi)
            for _ in range(max_iter):
                f = 0.5 * (flo + fhi)
                lhs, rhs = balance(f)
                if abs(1.0 - lhs/rhs) < tol:
                    break
                # decide side using sign of (lhs - rhs)
                if (lhs - rhs) > 0.0:
                    flo = f
                else:
                    fhi = f
        
        # dynamic (velocity) pressure, kPa
        q_kPa = 0.5 * density_recalc * (velocity_m_sfinal ** 2) / 1000.0
    
        # 1) straight pipe only
        dp_pipe_kPa = f * (L / ID_m) * q_kPa
        
        dp_plf_kPa = q_kPa * PLF
    
        required_cols = ["SRB", "LRB", "BALL", "GLOBE"]
        missing = [c for c in required_cols if c not in selected_pipe_row.index]
        if missing:
            st.error(f"CSV missing required K columns: {missing}")
            st.stop()
    
        # Convert to floats and check NaNs
        try:
            K_SRB  = float(selected_pipe_row["SRB"])
            K_LRB  = float(selected_pipe_row["LRB"])
            K_BALL = float(selected_pipe_row["BALL"])
            K_GLOBE= float(selected_pipe_row["GLOBE"])
        except Exception as e:
            st.error(f"Failed to parse K-factors as numbers: {e}")
            st.stop()
    
        if any(pd.isna([K_SRB, K_LRB, K_BALL, K_GLOBE])):
            st.error("One or more K-factors are NaN in the CSV row.")
            st.stop()
        
        B_SRB = SRB + 0.5 * _45 + 2.0 * ubend + 3.0 * ptrap
        B_LRB = LRB + MAC
    
        dp_fittings_kPa = q_kPa * (
        K_SRB   * B_SRB +
        K_LRB   * B_LRB
        )

        dp_valves_kPa = q_kPa * (
        K_BALL  * ball +
        K_GLOBE * globe
        )
        
        dp_total_kPa = dp_pipe_kPa + dp_fittings_kPa + dp_valves_kPa + dp_plf_kPa

        converter = PressureTemperatureConverter()

        if refrigerant == "R744 TC":
            evappres = converter.temp_to_pressure("R744", T_evap)

        else:
            evappres = converter.temp_to_pressure(refrigerant, T_evap)

        postcirc = evappres - (dp_total_kPa / 100)
        
        if refrigerant == "R744 TC":
            postcirctemp = converter.pressure_to_temp("R744", postcirc)
        
        else:
            postcirctemp = converter.pressure_to_temp(refrigerant, postcirc)

        dt = T_evap - postcirctemp
        #st.write("dt:", dt)

        maxmass = max(mass_flow_kg_s, mass_flow_kg_smin)

        volflow = maxmass / density_recalc

        if MORfinal == "":
            MinCap = ""
        else:
            MinCap = MORfinal * evap_capacity_kw / 100
        
        def _pipe_row_for_size(size_inch: str, gauge: str | None = None):
            rows = material_df[
                material_df["Nominal Size (inch)"].astype(str).str.strip() == str(size_inch)
            ]
        
            if "Gauge" in rows.columns and rows["Gauge"].notna().any():
                if gauge is not None:
                    rows_g = rows[rows["Gauge"] == gauge]
                    if not rows_g.empty:
                        return rows_g.iloc[0]
                # fallback if gauge is None or invalid
                return rows.iloc[0]
        
            return rows.iloc[0]

        from utils.double_riser import balance_double_riser
        from utils.dry_suction_engine import (
            DrySuctionInputs,
            make_pipe_row_for_size,
            make_ctx,
            make_get_pipe_results,
        )
        
        # Only meaningful for R744 TC
        gc_max = gc_max_pres if refrigerant == "R744 TC" else None
        gc_min = gc_min_pres if refrigerant == "R744 TC" else None
        
        # --- shared dry suction engine (single-pipe and network use the same code) ---
        ds_inputs = DrySuctionInputs(
            refrigerant=refrigerant,
            T_evap=T_evap,
            T_cond=T_cond,
            minliq_temp=minliq_temp,
            superheat_K=superheat_K,
            max_penalty_K=max_penalty,
            evap_capacity_kw=evap_capacity_kw,

            L=L,
            SRB=SRB,
            LRB=LRB,
            bends_45=_45,
            MAC=MAC,
            ptrap=ptrap,
            ubend=ubend,
            ball=0,
            globe=0,
            PLF=PLF,

            selected_material=selected_material,
            gc_max_pres=gc_max,
            gc_min_pres=gc_min,
        )

        pipe_row_for_size = make_pipe_row_for_size(material_df)
        ctx = make_ctx(inputs=ds_inputs, pipe_row_for_size=pipe_row_for_size)
        get_pipe_results = make_get_pipe_results(inputs=ds_inputs, pipe_row_for_size=pipe_row_for_size)
        def MOR_full_cached(size):
            MOR_s, _ = get_pipe_results(size)
            return MOR_s
        
        
        @st.cache_data(show_spinner=False)
        def _cached_double_riser(
            size_small,
            size_large,
            M_total_kg_s,
            _ctx,
            gauge_small,
            gauge_large,
        ):
            return balance_double_riser(
                size_small=size_small,
                size_large=size_large,
                M_total_kg_s=M_total_kg_s,
                ctx=_ctx,
                gauge_small=gauge_small,
                gauge_large=gauge_large,
            )

        col1, col2, col3, col4, col5, spacer = st.columns([0.1, 0.1, 0.1, 0.1, 0.1, 0.4])
        
        # holders for messages to show later (full width)
        error_message = None
        debug_errors = None
        
        with col1:
            st.write("Auto-select")
        
        with col2:
            if st.button("Horizontal"):
                st.session_state.double_trouble = False
                results, errors = [], []
            
                for ps in pipe_sizes:
                    try:
                        rows = material_df[
                            material_df["Nominal Size (inch)"].astype(str).str.strip() == str(ps)
                        ]
            
                        # 🔹 force strongest gauge for this size
                        if "Gauge" in rows.columns and rows["Gauge"].notna().any():
                            test_gauge = max(rows["Gauge"].dropna().unique())
                        else:
                            test_gauge = None
            
                        MOR_i, dt_i = get_pipe_results(ps, gauge_override=test_gauge)
            
                        if math.isfinite(dt_i):
                            results.append({"size": ps, "dt": dt_i})
                        else:
                            errors.append((ps, "Non-numeric ΔT"))
            
                    except Exception as e:
                        errors.append((ps, str(e)))
            
                if not results:
                    debug_errors = errors
                    error_message = "No valid pipe sizes found. Check inputs and CSV rows."
                else:
                    valid = [r for r in results if r["dt"] <= max_penalty]
            
                    if valid:
                        best = min(valid, key=lambda x: mm_map[x["size"]])
                        best_size = best["size"]
            
                        st.session_state["_next_selected_size"] = best_size
            
                        rows = material_df[
                            material_df["Nominal Size (inch)"].astype(str).str.strip() == str(best_size)
                        ]
            
                        if "Gauge" in rows.columns and rows["Gauge"].notna().any():
                            gauges = sorted(rows["Gauge"].dropna().unique())
            
                            best_gauge = _auto_select_copper_gauge(
                                material_df=material_df,
                                size_inch=best_size,
                                gauges=gauges,
                                design_pressure=result["design_pressure_bar_g"],
                                refrigerant=refrigerant,
                                design_temp_c=design_temp_c,
                                mwp_temp_c=mwp_temp_c,
                                circuit=circuit,
                                pipe_index=pipe_index,
                                copper_calc=copper_calc,
                                dp_standard=dp_standard,
                                r744_tc_pressure_bar_g=r744_tc_pressure_bar_g,
                            )
            
                            st.session_state["_next_gauge_main"] = best_gauge
            
                        st.success(
                            f"✅ Selected low-ΔT pipe size: **{best_size}**  \n"
                            f"ΔT: {best['dt']:.3f} K (limit {max_penalty:.3f} K)"
                        )
                        st.rerun()
            
                    else:
                        best_dt = min(r["dt"] for r in results)
                        error_message = (
                            f"❌ No pipe satisfies the ΔT limit.\n"
                            f"Best achievable ΔT is **{best_dt:.3f} K**, "
                            f"but limit = **{max_penalty:.3f} K**."
                        )
        
        with col3:
            if st.button("Single Riser"):
                st.session_state.double_trouble = False
                results, errors = [], []
            
                for ps in pipe_sizes:
                    try:
                        rows = material_df[
                            material_df["Nominal Size (inch)"].astype(str).str.strip() == str(ps)
                        ]
            
                        # 🔹 force strongest gauge for this size (same pattern as Horizontal)
                        if "Gauge" in rows.columns and rows["Gauge"].notna().any():
                            test_gauge = max(rows["Gauge"].dropna().unique())
                        else:
                            test_gauge = None
            
                        MOR_i, dt_i = get_pipe_results(ps, gauge_override=test_gauge)
            
                        if math.isfinite(MOR_i) and math.isfinite(dt_i):
                            results.append(
                                {
                                    "size": ps,
                                    "MORfinal": MOR_i,
                                    "dt": dt_i,
                                }
                            )
                        else:
                            errors.append((ps, "Non-numeric MOR or ΔT"))
            
                    except Exception as e:
                        errors.append((ps, str(e)))
            
                if not results:
                    debug_errors = errors
                    error_message = "No valid pipe size results. Check inputs and CSV rows."
                else:
                    valid = [
                        r for r in results
                        if (r["MORfinal"] <= required_oil_duty_pct)
                        and (r["dt"] <= max_penalty)
                    ]
            
                    if valid:
                        best = min(valid, key=lambda x: mm_map[x["size"]])
                        best_size = best["size"]
            
                        # ✅ defer pipe size
                        st.session_state["_next_selected_size"] = best_size
            
                        # --- auto-select copper gauge (unchanged logic, correct order) ---
                        rows = material_df[
                            material_df["Nominal Size (inch)"].astype(str).str.strip() == str(best_size)
                        ]
            
                        if "Gauge" in rows.columns and rows["Gauge"].notna().any():
                            gauges = sorted(rows["Gauge"].dropna().unique())
            
                            best_gauge = _auto_select_copper_gauge(
                                material_df=material_df,
                                size_inch=best_size,
                                gauges=gauges,
                                design_pressure=result["design_pressure_bar_g"],
                                refrigerant=refrigerant,
                                design_temp_c=design_temp_c,
                                mwp_temp_c=mwp_temp_c,
                                circuit=circuit,
                                pipe_index=pipe_index,
                                copper_calc=copper_calc,
                                dp_standard=dp_standard,
                                r744_tc_pressure_bar_g=r744_tc_pressure_bar_g,
                            )
            
                            # ✅ defer gauge (never touch widget directly)
                            st.session_state["_next_gauge_main"] = best_gauge
            
                        st.success(
                            f"✅ Selected optimal pipe size: **{best_size}**  \n"
                            f"MOR: {best['MORfinal']:.1f}% | ΔT: {best['dt']:.2f} K"
                        )
                        st.rerun()
                    else:
                        error_message = (
                            "❌ No pipe meets both limits simultaneously.  \n"
                            "➡ Please relax one or more input limits."
                        )

        with col4:
            double_trouble = st.checkbox("Double Riser Mode", key="double_trouble")

        if double_trouble:
            dr = balance_double_riser(
                manual_small,
                manual_large,
                M_total,
                ctx,
                gauge_small=gauge_small,
                gauge_large=gauge_large,
            )

            rs = dr.small_result
            rl = dr.large_result

            from utils.double_riser import compute_double_riser_oil_metrics

            MOR_full_flow, MOR_large, SST, M_largeprop = compute_double_riser_oil_metrics(
                dr=dr,
                refrigerant=refrigerant,
                T_evap=T_evap,
                density_foroil=density_foroil,
                oil_density=oil_density,
                jg_half=jg_half,
                mass_flow_foroil=mass_flow_foroil,
                mass_flow_foroilmin=mass_flow_foroilmin,
                MOR_correction=MOR_correction,
                MOR_correctionmin=MOR_correctionmin,
                MOR_correction2=MOR_correction2,
            )
        
        with col5:
            if st.button("Double Riser") and double_trouble:
                sizes_asc = sorted(pipe_sizes, key=lambda s: mm_map[s])
            
                small_candidates = []
            
                for s in sizes_asc:
                    MOR_s = MOR_full_cached(s)
            
                    if not math.isfinite(MOR_s):
                        continue
            
                    if MOR_s <= min(required_oil_duty_pct, 50.0):
                        small_candidates.append(s)
            
                if not small_candidates:
                    st.error("❌ No pipe size satisfies full-flow oil return duty.")
                    st.stop()
            
                small = max(small_candidates, key=lambda s: mm_map[s])
            
                def resolve_large(small, min_large_mm):
                    for large in sizes_asc:
                        if mm_map[large] < min_large_mm:
                            continue
                
                        dr, MOR_full, MOR_large = eval_pair_cached(small, large)
                
                        if MOR_full is None or MOR_large is None:
                            continue
                
                        if dr.DT_K > max_penalty:
                            continue
                
                        # CHANGE #1 (Option C): MOR_large is NOT checked here anymore
                        # if MOR_large > 100.0:
                        #     continue
                
                        return large, dr, MOR_full, MOR_large
                
                    return None, None, None, None
            
                large, dr, MOR_full, MOR_large = resolve_large(small, mm_map[small])
            
                # CHANGE #2 (Option C-1): if MOR_large fails, do NOT stop; just treat as "no best yet"
                best_small = None
                best_large = None
                best_dr = None
                best_MOR_full = None
                best_MOR_large = None
            
                if large is not None and MOR_large is not None and MOR_large <= 100.0:
                    best_small = small
                    best_large = large
                    best_dr = dr
                    best_MOR_full = MOR_full
                    best_MOR_large = MOR_large
            
                prev_large_mm = mm_map[best_large] if best_large is not None else None
            
                idx = sizes_asc.index(small)
            
                while idx > 0:
                    candidate_small = sizes_asc[idx - 1]
            
                    MOR_s = MOR_full_cached(candidate_small)
                    if not math.isfinite(MOR_s) or MOR_s > min(required_oil_duty_pct, 50.0):
                        break
            
                    candidate_large, dr_c, MOR_f, MOR_l = resolve_large(
                        candidate_small,
                        min_large_mm=mm_map[candidate_small],
                    )
            
                    if candidate_large is None:
                        break
            
                    if prev_large_mm is not None and mm_map[candidate_large] > prev_large_mm:
                        break
            
                    # CHANGE #2 (Option C-1): post-convergence MOR_large gate with "continue"
                    if MOR_l is None or MOR_l > 100.0:
                        idx -= 1
                        continue
            
                    best_small = candidate_small
                    best_large = candidate_large
                    best_dr = dr_c
                    best_MOR_full = MOR_f
                    best_MOR_large = MOR_l
            
                    prev_large_mm = mm_map[candidate_large]
                    idx -= 1
            
                if best_large is None:
                    st.error("❌ No valid large riser meets ΔT and MOR limits.")
                    st.stop()
            
                st.session_state["_next_double_riser"] = True
                st.session_state["_next_manual_small"] = best_small
                st.session_state["_next_manual_large"] = best_large
            
                rows_small = material_df[
                    material_df["Nominal Size (inch)"].astype(str).str.strip() == str(best_small)
                ]
            
                if "Gauge" in rows_small.columns and rows_small["Gauge"].notna().any():
                    gauges_small = sorted(rows_small["Gauge"].dropna().unique())
            
                    best_gauge_small = _auto_select_copper_gauge(
                        material_df=material_df,
                        size_inch=best_small,
                        gauges=gauges_small,
                        design_pressure=result["design_pressure_bar_g"],
                        refrigerant=refrigerant,
                        design_temp_c=design_temp_c,
                        mwp_temp_c=mwp_temp_c,
                        circuit=circuit,
                        pipe_index=pipe_index_small,
                        copper_calc=copper_calc,
                        dp_standard=dp_standard,
                        r744_tc_pressure_bar_g=r744_tc_pressure_bar_g,
                    )
            
                    st.session_state["_next_gauge_small"] = best_gauge_small
            
                rows_large = material_df[
                    material_df["Nominal Size (inch)"].astype(str).str.strip() == str(best_large)
                ]
            
                if "Gauge" in rows_large.columns and rows_large["Gauge"].notna().any():
                    gauges_large = sorted(rows_large["Gauge"].dropna().unique())
            
                    best_gauge_large = _auto_select_copper_gauge(
                        material_df=material_df,
                        size_inch=best_large,
                        gauges=gauges_large,
                        design_pressure=result["design_pressure_bar_g"],
                        refrigerant=refrigerant,
                        design_temp_c=design_temp_c,
                        mwp_temp_c=mwp_temp_c,
                        circuit=circuit,
                        pipe_index=pipe_index_large,
                        copper_calc=copper_calc,
                        dp_standard=dp_standard,
                        r744_tc_pressure_bar_g=r744_tc_pressure_bar_g,
                    )
            
                    st.session_state["_next_gauge_large"] = best_gauge_large
            
                st.success(
                    f"✅ **Double Riser Selected**\n\n"
                    f"Small: **{best_small}** | Large: **{best_large}**\n"
                    f"MOR (full flow): {best_MOR_full:.1f}%\n"
                    f"MOR (large riser): {best_MOR_large:.1f}%\n"
                    f"Temperature penalty: {best_dr.DT_K:.2f} K"
                )
            
                st.rerun()
        
        with spacer:
            st.empty()
        
        if debug_errors:
            with st.expander("⚠️ Pipe selection debug details", expanded=True):
                for ps, msg in debug_errors:
                    st.write(f"❌ {ps}: {msg}")
        
        if error_message:
            st.error(error_message)
    
        if velocity_m_sfinal:
            col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    
            with col1:
                if double_trouble:
                    st.metric("Refrigerant Velocity", f"{rl.velocity_m_s:.2f}m/s")
                else:
                    st.metric("Refrigerant Velocity", f"{velocity_m_sfinal:.2f}m/s")
    
            with col2:
                st.metric("Suction Density", f"{density_recalc:.2f}kg/m³")
    
            with col3:
                if double_trouble:
                    if MOR_full_flow is None:
                        st.metric("Min Oil Return", "")
                    else:
                        st.metric("Min Oil Return", f"{MOR_full_flow:.1f}%")
                else:
                    if MORfinal == "":
                        st.metric("MOR (%)", "")
                    else:
                        st.metric("MOR (%)", f"{MORfinal:.1f}%")
    
            with col4:
                if double_trouble:
                    st.metric("Pressure Drop", f"{dr.DP_kPa:.2f}kPa")
                else:
                    st.metric("Pressure Drop", f"{dp_total_kPa:.2f}kPa")
    
            with col5:
                if double_trouble:
                    st.metric("Temp Penalty", f"{dr.DT_K:.2f}K")
                else:
                    st.metric("Temp Penalty", f"{dt:.3f}K")

            with col6:
                if double_trouble:
                    st.metric("SST", f"{SST:.2f}°C")
                else:
                    st.metric("SST", f"{postcirctemp:.2f}°C")

            with col7:
                st.metric("Evaporating Pressure", f"{evappres:.2f}bar(a)")

            col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    
            with col1:
                st.metric("Mass Flow Rate", f"{maxmass:.5f}kg/s")
    
            with col2:
                st.metric("Volumetric Flow Rate", f"{volflow:.5f}m³/s")
    
            with col3:
                if double_trouble:
                    if MOR_large is None:
                        st.metric("Max Oil Return", "")
                    else:
                        st.metric("Max Oil Return", f"{MOR_large:.1f}%")
                else:
                    if MORfinal == "":
                        st.metric("Minimum Capacity", "")
                    else:
                        st.metric("Minimum Capacity", f"{MinCap:.4f}kW")
    
            with col4:
                if double_trouble:
                    st.metric("Pipe PD", f"{dr.dp_pipe:.2f}kPa")
                else:
                    st.metric("Pipe PD", f"{dp_pipe_kPa:.2f}kPa")
    
            with col5:
                if double_trouble:
                    st.metric("Fittings PD", f"{dr.dp_fit:.2f}kPa")
                else:
                    st.metric("Fittings PD", f"{dp_fittings_kPa:.2f}kPa")

            with col6:
                if double_trouble:
                    st.metric("Valves PD", f"{dr.dp_valve:.2f}kPa")
                else:
                    st.metric("Valves PD", f"{dp_valves_kPa:.2f}kPa")

            with col7:
                if double_trouble:
                    st.metric("Velocity Pressure PD", f"{dr.dp_plf:.2f}kPa")
                else:
                    st.metric("Velocity Pressure PD", f"{dp_plf_kPa:.2f}kPa")

        if double_trouble:
            if isinstance(MOR_full_flow, (int, float)):
                is_ok, message = (True, "✅ OK") if required_oil_duty_pct >= MOR_full_flow else (False, "❌ Insufficient flow")
            else:
                is_ok, message = (False, "")
        
            if is_ok:
                st.success(f"{message}")
            else:
                st.error(f"{message}")   
        else:
            if isinstance(MORfinal, (int, float)):
                is_ok, message = (True, "✅ OK") if required_oil_duty_pct >= MORfinal else (False, "❌ Insufficient flow")
            else:
                is_ok, message = (False, "")
        
            if is_ok:
                st.success(f"{message}")
            else:
                st.error(f"{message}")
    
    if mode == "Liquid":
        
        # Load pipe data
        pipe_data = pd.read_csv("data/pipe_pressure_ratings_full.csv")
    
        # --- helpers ---
        def _nps_inch_to_mm(nps_str: str) -> float:
            # e.g. "1-1/8", '1"', "3/8"
            s = str(nps_str).replace('"', '').strip()
            if not s:
                return float('nan')
            parts = s.split('-')
            tot_in = 0.0
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                if '/' in p:
                    num, den = p.split('/')
                    tot_in += float(num) / float(den)
                else:
                    tot_in += float(p)
            return tot_in * 25.4  # mm
    
        ss = st.session_state
    
        # 1) Pipe material
        col1, col2, col3, col4 = st.columns(4)
        with col2:
            if refrigerant == "R717":
                excluded_materials = ["Copper ASTM", " Copper EN12735", "K65 Copper", "Reflok Aluminium"]
                pipe_materials = sorted(m for m in pipe_data["Material"].dropna().unique()
                                        if m not in excluded_materials)
            else:
                pipe_materials = sorted(pipe_data["Material"].dropna().unique())
    
            selected_material = st.selectbox("Pipe Material", pipe_materials, key="material")
        
        # detect material change
        material_changed = ss.get("last_material") is not None and ss.last_material != selected_material
        ss.last_material = selected_material
    
        # 2) Sizes for selected material (de-duped)
        material_df = pipe_data[pipe_data["Material"] == selected_material].copy()
    
        sizes_df = (
            material_df[["Nominal Size (inch)", "Nominal Size (mm)"]]
            .dropna(subset=["Nominal Size (inch)"])
            .assign(**{
                "Nominal Size (inch)": lambda d: d["Nominal Size (inch)"].astype(str).str.strip(),
            })
            .drop_duplicates(subset=["Nominal Size (inch)"], keep="first")
        )
    
        # make sure we have a numeric mm per nominal (fallback: parse the inch string)
        sizes_df["mm_num"] = pd.to_numeric(sizes_df.get("Nominal Size (mm)"), errors="coerce")
        sizes_df.loc[sizes_df["mm_num"].isna(), "mm_num"] = sizes_df.loc[sizes_df["mm_num"].isna(), "Nominal Size (inch)"].apply(_nps_inch_to_mm)
    
        pipe_sizes = sizes_df["Nominal Size (inch)"].tolist()
        mm_map = dict(zip(sizes_df["Nominal Size (inch)"], sizes_df["mm_num"]))
    
        # choose default index
        def _closest_index(target_mm: float) -> int:
            mm_list = [mm_map[s] for s in pipe_sizes]
            return min(range(len(mm_list)), key=lambda i: abs(mm_list[i] - target_mm)) if mm_list else 0
    
        # --- consume any deferred selection from the button ---
        if "_next_selected_size" in st.session_state:
            new_val = st.session_state.pop("_next_selected_size")
            # only accept if the option exists for the current material
            if new_val in pipe_sizes:
                # force the widget to pick this value on next render
                st.session_state["selected_size"] = new_val
                # optional: keep a one-shot override flag so we can clean up later
                st.session_state["_selected_size_just_set"] = True
        
        default_index = 0
        if material_changed and "prev_pipe_mm" in ss:
            default_index = _closest_index(ss.prev_pipe_mm)
        elif selected_material == " Copper EN12735" and ("1/2" in pipe_sizes or '1/2"' in pipe_sizes):
            # first load or no previous selection → prefer 1-1/8" for  Copper EN12735
            want = "1/2" if "1/2" in pipe_sizes else '1/2"'
            default_index = pipe_sizes.index(want)
        elif "selected_size" in ss and ss.selected_size in pipe_sizes:
            # if Streamlit kept the selection, use it
            default_index = pipe_sizes.index(ss.selected_size)
        
        with col1:
            selected_size = st.selectbox(
                "Nominal Pipe Size (inch)",
                pipe_sizes,
                index=default_index,
                key="selected_size",
            )
    
        # remove the one-shot flag so future reruns don't keep forcing it
        if st.session_state.get("_selected_size_just_set"):
            del st.session_state["_selected_size_just_set"]
        
        # remember the selected size in mm for next material change
        ss.prev_pipe_mm = float(mm_map.get(selected_size, float("nan")))

        # --- consume any deferred gauge from Auto-select ---
        if "_next_gauge" in st.session_state:
            g = st.session_state.pop("_next_gauge")
        
            # Only apply if the current size actually has that gauge option
            rows = material_df[
                material_df["Nominal Size (inch)"].astype(str).str.strip() == str(selected_size)
            ]
            if "Gauge" in rows.columns and rows["Gauge"].notna().any():
                valid_gauges = set(rows["Gauge"].dropna().unique())
                if g in valid_gauges:
                    st.session_state["gauge"] = g
    
        # 3) Gauge (if applicable)
        gauge_options = material_df[material_df["Nominal Size (inch)"].astype(str).str.strip() == selected_size]
        if "Gauge" in gauge_options.columns and gauge_options["Gauge"].notna().any():
            gauges = sorted(gauge_options["Gauge"].dropna().unique())
            with col2:
                selected_gauge = st.selectbox("Copper Gauge", gauges, key="gauge")
            selected_pipe_row = gauge_options[gauge_options["Gauge"] == selected_gauge].iloc[0]
        else:
            selected_pipe_row = gauge_options.iloc[0]

        pipe_index = material_to_pipe_index(selected_material)
        
        # you already have selected_gauge sometimes; otherwise None
        gauge = st.session_state.get("gauge")  # or whatever that mode uses
        od_mm, id_mm = get_dimensions_for_row(material_df, selected_size, gauge)
        
        result = system_pressure_check(
            refrigerant=refrigerant,
            design_temp_c=design_temp_c,
            mwp_temp_c=mwp_temp_c,
            circuit=circuit,
            pipe_index=pipe_index,
            od_mm=od_mm,
            id_mm=id_mm,
            gauge=gauge,
            copper_calc=copper_calc,
            r744_tc_pressure_bar_g=r744_tc_pressure_bar_g,
            dp_standard=dp_standard,
        )
        
        render_pressure_result(result)
    
        # Pipe parameters
        pipe_size_inch = selected_pipe_row["Nominal Size (inch)"]
        ID_mm = selected_pipe_row["ID_mm"]
    
        with col1:
            
            evap_capacity_kw = st.number_input("Evaporator Capacity (kW)", min_value=0.03, max_value=20000.0, value=10.0, step=1.0)
            
            # --- Base ranges per refrigerant ---
            if refrigerant in ("R23", "R508B"):
                evap_min, evap_max, evap_default = -100.0, -20.0, -80.0
                cond_min, cond_max, cond_default = -100.0, 10.0, -30.0
                maxliq_min, maxliq_max, maxliq_default = -100.0, 10.0, -40.0
            elif refrigerant == "R744":
                evap_min, evap_max, evap_default = -50.0, 20.0, -10.0
                cond_min, cond_max, cond_default = -23.0, 30.0, 15.0
                maxliq_min, maxliq_max, maxliq_default = -50.0, 30.0, 10.0
            else:
                evap_min, evap_max, evap_default = -50.0, 30.0, -10.0
                cond_min, cond_max, cond_default = -23.0, 60.0, 43.0
                maxliq_min, maxliq_max, maxliq_default = -50.0, 60.0, 40.0
    
            # --- Init state (widget-backed) ---
            ss = st.session_state
    
            if "last_refrigerant" not in ss or ss.last_refrigerant != refrigerant:
                ss.cond_temp   = cond_default
                ss.maxliq_temp = maxliq_default
                ss.evap_temp   = evap_default
                ss.last_refrigerant = refrigerant
            
            ss.setdefault("cond_temp",   cond_default)
            ss.setdefault("maxliq_temp", maxliq_default)
            ss.setdefault("evap_temp",   evap_default)

            if "maxliq_temp" in ss and "cond_temp" in ss and "evap_temp" in ss:
                ss.cond_temp = min(max(ss.cond_temp, ss.maxliq_temp, ss.evap_temp), cond_max)
    
            if "cond_temp" in ss and "maxliq_temp" in ss and "evap_temp" in ss:
                ss.evap_temp = min(ss.maxliq_temp, ss.cond_temp, ss.evap_temp)
            
            # --- Callbacks implementing your downstream clamping logic ---
            def on_change_cond():
                # When cond changes: clamp maxliq down to cond, then evap down to maxliq
                ss.maxliq_temp = min(ss.maxliq_temp, ss.cond_temp)
                ss.evap_temp   = min(ss.evap_temp,   ss.maxliq_temp)
    
            def on_change_maxliq():
                # When maxliq changes: clamp maxliq down to cond, then evap down to maxliq
                ss.maxliq_temp = min(ss.maxliq_temp, ss.cond_temp)
                ss.evap_temp   = min(ss.evap_temp,   ss.maxliq_temp)
    
            def on_change_evap():
                # When evap changes: clamp evap down to maxliq
                if refrigerant == "R744 TC":
                    ss.evap_temp   = min(ss.evap_temp,   maxliq_temp)
                else:
                    ss.evap_temp   = min(ss.evap_temp,   ss.maxliq_temp)
    
            if refrigerant == "R744 TC":     
                ss.setdefault("gc_max_temp", 38.0)
                gc_max_temp = st.number_input(
                    "Gas Cooler Outlet Temp (°C)",
                    min_value=-50.0, max_value=50.0,
                    value=ss.gc_max_temp, step=1.0, key="gc_max_temp"
                )
        
                ss.setdefault("gc_max_pres", 93.7)
                gc_max_pres = st.number_input(
                    "Gas Cooler Outlet Pressure (bar(a))",
                    min_value=73.8, max_value=150.0,
                    value=ss.gc_max_pres, step=1.0, key="gc_max_pres"
                )
            
                maxliq_temp = gc_max_temp

                ss.evap_temp = min(ss.evap_temp, maxliq_temp)
            
            else:        
                condensing_temp = st.number_input(
                    "Condensing Temperature (°C)",
                    min_value=cond_min, max_value=cond_max,
                    value=ss.cond_temp, step=1.0, key="cond_temp",
                    on_change=on_change_cond,
                )
        
                maxliq_temp = st.number_input(
                    "Liquid Temperature (°C)",
                    min_value=maxliq_min, max_value=min(condensing_temp, maxliq_max),
                    value=ss.maxliq_temp, step=1.0, key="maxliq_temp",
                    on_change=on_change_maxliq,
                )

        with col2:
            
            evaporating_temp = st.number_input(
                "Evaporating Temperature (°C)",
                min_value=evap_min, max_value=min(maxliq_temp, evap_max),
                value=ss.evap_temp, step=1.0, key="evap_temp",
                on_change=on_change_evap,
            )
    
        with col2:
            if refrigerant == "R744 TC":
                risem = st.number_input("Liquid Line Rise (m)", min_value=0.0, max_value=30.0, value=0.0, step=1.0, disabled=True)
            else:
                risem = st.number_input("Liquid Line Rise (m)", min_value=0.0, max_value=30.0, value=0.0, step=1.0)
            if refrigerant == "R744 TC":
                max_lineloss = st.number_input("Max Pressure Drop (kPa)", min_value=0.0, max_value=250.0, value=15.0, step=1.0)
            else:
                max_penalty = st.number_input("Max Penalty (K)", min_value=0.0, max_value=6.0, value=1.0, step=0.1)
    
        with col3:
            L = st.number_input("Pipe Length (m)", min_value=0.1, max_value=300.0, value=10.0, step=1.0)
            LRB = st.number_input("Long Radius Bends", min_value=0, max_value=50, value=0, step=1)
            SRB = st.number_input("Short Radius Bends", min_value=0, max_value=50, value=0, step=1)
            _45 = st.number_input("45° Bends", min_value=0, max_value=50, value=0, step=1)
            MAC = st.number_input("Machine Bends", min_value=0, max_value=50, value=0, step=1)
    
        with col4:
            ptrap = st.number_input("P Traps", min_value=0, max_value=10, value=0, step=1)
            ubend = st.number_input("U Bends", min_value=0, max_value=10, value=0, step=1)
            ball = st.number_input("Ball Valves", min_value=0, max_value=20, value=0, step=1, key="ball")
            globe = st.number_input("Globe Valves", min_value=0, max_value=20, value=0, step=1, key="globe")
            PLF = st.number_input("Pressure Loss Factors", min_value=0.0, max_value=20.0, value=0.0, step=0.1)
        
        from utils.refrigerant_properties import RefrigerantProperties
        from utils.refrigerant_densities import RefrigerantDensities
        from utils.refrigerant_viscosities import RefrigerantViscosities
        from utils.supercompliq_co2 import RefrigerantProps
    
        if refrigerant == "R744 TC":
            T_evap = evaporating_temp
            T_liq = maxliq_temp
        
        else:
            T_evap = evaporating_temp
            T_liq = maxliq_temp
            T_cond = condensing_temp
    
        props = RefrigerantProperties()
        props_sup = RefrigerantProps()
        
        if refrigerant == "R744 TC":
            h_in = props_sup.get_enthalpy_sup(gc_max_pres, maxliq_temp)
            h_evap = props.get_properties("R744", T_evap)["enthalpy_vapor"]
    
        else:
            h_in = props.get_properties(refrigerant, T_liq)["enthalpy_liquid2"]
            h_evap = props.get_properties(refrigerant, T_evap)["enthalpy_vapor"]
        
        delta_h = h_evap - h_in

        mass_flow_kg_s = evap_capacity_kw / delta_h if delta_h > 0 else 0.01
    
        if ID_mm is not None:
            ID_m = ID_mm / 1000.0

            area_m2 = math.pi * (ID_m / 2) ** 2

            if refrigerant == "R744 TC":
                density = props_sup.get_density_sup(gc_max_pres, maxliq_temp)
            else:
                density = RefrigerantProperties().get_properties(refrigerant, T_liq)["density_liquid2"]

            velocity_m_s = mass_flow_kg_s / (area_m2 * density)

        else:
            velocity_m_s = None

        if refrigerant == "R744 TC":
            viscosity = props_sup.get_viscosity_sup(gc_max_pres, maxliq_temp)
        else:
            viscosity = RefrigerantProperties().get_properties(refrigerant, T_liq)["viscosity_liquid"]
    
        reynolds = (density * velocity_m_s * ID_m) / (viscosity / 1000000)
    
        if selected_material in ["Steel SCH40", "Steel SCH80"]:
            eps = 0.00004572 #0.00015
        else:
            eps = 0.000001524 #0.000005
        
        tol = 1e-5
        max_iter = 60
        
        if reynolds < 2000.0:
            f = 64.0 / reynolds
        else:
            flo, fhi = 1e-5, 0.1
            def balance(gg):
                s = math.sqrt(gg)
                lhs = 1.0 / s
                rhs = -2.0 * math.log10((eps / (3.7 * ID_m)) + 2.51 / (reynolds * s))
                return lhs, rhs
    
            f = 0.5 * (flo + fhi)
            for _ in range(max_iter):
                f = 0.5 * (flo + fhi)
                lhs, rhs = balance(f)
                if abs(1.0 - lhs/rhs) < tol:
                    break
                # decide side using sign of (lhs - rhs)
                if (lhs - rhs) > 0.0:
                    flo = f
                else:
                    fhi = f
        
        # dynamic (velocity) pressure, kPa
        q_kPa = 0.5 * density * (velocity_m_s ** 2) / 1000.0
    
        # 1) straight pipe only
        dp_pipe_kPa = f * (L / ID_m) * q_kPa
        
        dp_plf_kPa = q_kPa * PLF
    
        required_cols = ["SRB", "LRB", "BALL", "GLOBE"]
        missing = [c for c in required_cols if c not in selected_pipe_row.index]
        if missing:
            st.error(f"CSV missing required K columns: {missing}")
            st.stop()
    
        # Convert to floats and check NaNs
        try:
            K_SRB  = float(selected_pipe_row["SRB"])
            K_LRB  = float(selected_pipe_row["LRB"])
            K_BALL = float(selected_pipe_row["BALL"])
            K_GLOBE= float(selected_pipe_row["GLOBE"])
        except Exception as e:
            st.error(f"Failed to parse K-factors as numbers: {e}")
            st.stop()
    
        if any(pd.isna([K_SRB, K_LRB, K_BALL, K_GLOBE])):
            st.error("One or more K-factors are NaN in the CSV row.")
            st.stop()
        
        B_SRB = SRB + 0.5 * _45 + 2.0 * ubend + 3.0 * ptrap
        B_LRB = LRB + MAC
    
        dp_fittings_kPa = q_kPa * (
        K_SRB   * B_SRB +
        K_LRB   * B_LRB
        )

        dp_valves_kPa = q_kPa * (
        K_BALL  * ball +
        K_GLOBE * globe
        )
        
        dp_total_kPa = dp_pipe_kPa + dp_fittings_kPa + dp_valves_kPa + dp_plf_kPa
        
        if refrigerant == "R744 TC":
            converter = PressureTemperatureConverter()
            condpres = gc_max_pres
            evappres = converter.temp_to_pressure("R744", T_evap)
        
        else:
            converter = PressureTemperatureConverter()
            condpres = converter.temp_to_pressure(refrigerant, T_cond)
            postcirc = condpres - (dp_total_kPa / 100)
            postcirctemp = converter.pressure_to_temp(refrigerant, postcirc)
            
            dt = T_cond - postcirctemp
    
            head = 9.80665 * risem * density / 1000
            
            dp_withhead = dp_total_kPa + head
    
            postall = condpres - (dp_withhead / 100)
            postalltemp = converter.pressure_to_temp(refrigerant, postall)
            
            tall = T_cond - postalltemp
    
            exsub = T_cond - T_liq
    
            addsub = max(tall - exsub, 0)
    
            evappres = converter.temp_to_pressure(refrigerant, T_evap)

        volflow = mass_flow_kg_s / density

        compratio = condpres / evappres

        def _pipe_row_for_size(size_inch: str, gauge_override=None):
            """
            Return the CSV row for a given nominal size.
            If gauge_override is provided, force that gauge.
            Otherwise, fall back to current UI-selected gauge (if any).
            """
            rows = material_df[
                material_df["Nominal Size (inch)"].astype(str).str.strip() == str(size_inch)
            ]
            if rows.empty:
                return None
        
            if "Gauge" in rows.columns and rows["Gauge"].notna().any():
                if gauge_override is not None:
                    rows_g = rows[rows["Gauge"] == gauge_override]
                    if not rows_g.empty:
                        return rows_g.iloc[0]
        
                if "gauge" in st.session_state:
                    g = st.session_state["gauge"]
                    rows_g = rows[rows["Gauge"] == g]
                    if not rows_g.empty:
                        return rows_g.iloc[0]
        
                return rows.iloc[0]
        
            return rows.iloc[0]
        
        def get_liquid_dt_for_size(size_inch: str, gauge_override=None) -> float:
            """
            Compute ΔT (dt) for a given pipe size using the SAME logic as your main Liquid calc
            (without static head — use 'dt', not 'tall').
            Returns float('nan') on any failure.
            """
            try:
                pipe_row = _pipe_row_for_size(size_inch, gauge_override)
                if pipe_row is None:
                    return float("nan")
        
                ID_mm_local = float(pipe_row["ID_mm"])
                ID_m_local = ID_mm_local / 1000.0
                area_m2_local = math.pi * (ID_m_local / 2) ** 2
        
                # Properties at liquid temperature (size-independent)
                if refrigerant == "R744 TC":
                    density_liq = props_sup.get_density_sup(gc_max_pres, maxliq_temp)
                    visc_liq = props_sup.get_viscosity_sup(gc_max_pres, maxliq_temp)
                else:
                    density_liq = RefrigerantProperties().get_properties(refrigerant, T_liq)["density_liquid2"]
                    visc_liq = RefrigerantProperties().get_properties(refrigerant, T_liq)["viscosity_liquid"]
                
                # Mass flow already computed outside (size-independent)
                v_local = mass_flow_kg_s / (area_m2_local * density_liq)
        
                # Reynolds
                Re = (density_liq * v_local * ID_m_local) / (visc_liq / 1_000_000)
        
                # Roughness by material
                eps = 0.00004572 if selected_material in ["Steel SCH40", "Steel SCH80"] else 0.000001524
        
                # Friction factor (laminar vs Colebrook)
                if Re < 2000.0 and Re > 0:
                    f_local = 64.0 / Re
                else:
                    flo, fhi = 1e-5, 0.1
                    tol, max_iter = 1e-5, 60
        
                    def bal(gg):
                        s = math.sqrt(gg)
                        lhs = 1.0 / s
                        rhs = -2.0 * math.log10((eps / (3.7 * ID_m_local)) + (2.51 / (Re * s)))
                        return lhs, rhs
        
                    f_local = 0.5 * (flo + fhi)
                    for _ in range(max_iter):
                        f_try = 0.5 * (flo + fhi)
                        lhs, rhs = bal(f_try)
                        if abs(1.0 - lhs / rhs) < tol:
                            f_local = f_try
                            break
                        if (lhs - rhs) > 0.0:
                            flo = f_try
                        else:
                            fhi = f_try
                    else:
                        f_local = 0.5 * (flo + fhi)
        
                # Dynamic pressure (kPa)
                q_kPa_local = 0.5 * density_liq * (v_local ** 2) / 1000.0
        
                # K-factors for this size
                for c in ["SRB", "LRB", "BALL", "GLOBE"]:
                    if c not in pipe_row.index or pd.isna(pipe_row[c]):
                        return float("nan")
                K_SRB   = float(pipe_row["SRB"])
                K_LRB   = float(pipe_row["LRB"])
                K_BALL  = float(pipe_row["BALL"])
                K_GLOBE = float(pipe_row["GLOBE"])
        
                # Bend/valve counts
                B_SRB = SRB + 0.5 * _45 + 2.0 * ubend + 3.0 * ptrap
                B_LRB = LRB + MAC
        
                # Pressure drops (kPa)
                dp_pipe_kPa_local    = f_local * (L / ID_m_local) * q_kPa_local
                dp_plf_kPa_local     = q_kPa_local * PLF
                dp_fittings_kPa_local= q_kPa_local * (K_SRB * B_SRB + K_LRB * B_LRB)
                dp_valves_kPa_local  = q_kPa_local * (K_BALL * ball    + K_GLOBE * globe)
        
                dp_total_kPa_local = dp_pipe_kPa_local + dp_fittings_kPa_local + dp_valves_kPa_local + dp_plf_kPa_local
        
                # Convert DP to post-circ temperature and get ΔT
                if refrigerant == "R744 TC":
                    dt_local = dp_total_kPa_local
                else:
                    conv = PressureTemperatureConverter()
                    condpres_local = conv.temp_to_pressure(refrigerant, T_cond)
                    postcirc_local = condpres_local - (dp_total_kPa_local / 100.0)  # kPa -> bar: /100
                    postcirctemp_local = conv.pressure_to_temp(refrigerant, postcirc_local)
                    dt_local = T_cond - postcirctemp_local
        
                return float(dt_local)
            except Exception:
                return float("nan")

        def _auto_select_copper_gauge(
            *,
            material_df,
            size_inch,
            gauges,
            design_pressure,
            refrigerant,
            design_temp_c,
            mwp_temp_c,
            circuit,
            pipe_index,
            copper_calc,
            dp_standard,
            r744_tc_pressure_bar_g,
        ):
            """
            Return:
            - highest gauge where MWP >= design pressure
            - else lowest gauge
            """
        
            passing = []
        
            for g in sorted(gauges):
                try:
                    od_mm, id_mm = get_dimensions_for_row(material_df, size_inch, g)
        
                    res = system_pressure_check(
                        refrigerant=refrigerant,
                        design_temp_c=design_temp_c,
                        mwp_temp_c=mwp_temp_c,
                        circuit=circuit,
                        pipe_index=pipe_index,
                        od_mm=od_mm,
                        id_mm=id_mm,
                        gauge=g,
                        copper_calc=copper_calc,
                        r744_tc_pressure_bar_g=r744_tc_pressure_bar_g,
                        dp_standard=dp_standard,
                    )
        
                    mwp_val = governing_mwp(res["mwp_bar"])
        
                    if mwp_val >= design_pressure:
                        passing.append(g)
        
                except Exception:
                    continue
        
            if passing:
                return max(passing)
            else:
                return min(gauges)
        
        if st.button("Auto-select"):
            if refrigerant == "R744 TC":
                results, errors = [], []
            
                for ps in pipe_sizes:
                    rows = material_df[
                        material_df["Nominal Size (inch)"].astype(str).str.strip() == str(ps)
                    ]
        
                    # 🔹 force strongest gauge for this size (evaluation only)
                    if "Gauge" in rows.columns and rows["Gauge"].notna().any():
                        test_gauge = max(rows["Gauge"].dropna().unique())
                    else:
                        test_gauge = None
        
                    dt_i = get_liquid_dt_for_size(ps, gauge_override=test_gauge)
        
                    if math.isfinite(dt_i):
                        results.append({"size": ps, "dt": dt_i})
                    else:
                        errors.append((ps, "Non-numeric ΔT"))
            
                if not results:
                    with st.expander("⚠️ Liquid selection debug details", expanded=True):
                        for ps, msg in errors:
                            st.write(f"❌ {ps}: {msg}")
                    st.error("No valid pipe size results. Check inputs and CSV rows.")
                else:
                    # Keep sizes that satisfy the dt limit, pick the smallest OD (mm)
                    valid = [r for r in results if r["dt"] <= max_lineloss]
            
                    if valid:
                        best = min(valid, key=lambda x: mm_map[x["size"]])
                        st.session_state["_next_selected_size"] = best["size"]
        
                        # 🔹 Auto-select gauge for Copper EN12735
                        if selected_material.strip() == "Copper EN12735":
                            rows = material_df[
                                material_df["Nominal Size (inch)"].astype(str).str.strip() == best["size"]
                            ]
                    
                            if "Gauge" in rows.columns and rows["Gauge"].notna().any():
                                gauges = sorted(rows["Gauge"].dropna().unique())
                    
                                best_gauge = _auto_select_copper_gauge(
                                    material_df=material_df,
                                    size_inch=best["size"],
                                    gauges=gauges,
                                    design_pressure=result["design_pressure_bar_g"],
                                    refrigerant=refrigerant,
                                    design_temp_c=design_temp_c,
                                    mwp_temp_c=mwp_temp_c,
                                    circuit=circuit,
                                    pipe_index=pipe_index,
                                    copper_calc=copper_calc,
                                    dp_standard=dp_standard,
                                    r744_tc_pressure_bar_g=r744_tc_pressure_bar_g,
                                )
                    
                                st.session_state["_next_gauge"] = best_gauge
                        
                        st.success(
                            f"✅ Selected liquid pipe size: **{best['size']}**  \n"
                            f"ΔT: {best['dt']:.2f} kPa (limit {max_lineloss:.2f} kPa)"
                        )
                        st.rerun()
                    else:
                        best_dt = min(r["dt"] for r in results if math.isfinite(r["dt"]))
                        st.error(
                            "❌ No pipe meets the PD limit.  \n"
                            f"Best achievable PD = {best_dt:.2f} kPa (must be ≤ {max_lineloss:.2f} kPa)  \n"
                            "➡ Relax the Max PD or choose a different material/length/fittings."
                        )
            else:
                results, errors = [], []
            
                for ps in pipe_sizes:
                    rows = material_df[
                        material_df["Nominal Size (inch)"].astype(str).str.strip() == str(ps)
                    ]
        
                    # 🔹 force strongest gauge for this size (evaluation only)
                    if "Gauge" in rows.columns and rows["Gauge"].notna().any():
                        test_gauge = max(rows["Gauge"].dropna().unique())
                    else:
                        test_gauge = None
        
                    dt_i = get_liquid_dt_for_size(ps, gauge_override=test_gauge)
        
                    if math.isfinite(dt_i):
                        results.append({"size": ps, "dt": dt_i})
                    else:
                        errors.append((ps, "Non-numeric ΔT"))
            
                if not results:
                    with st.expander("⚠️ Liquid selection debug details", expanded=True):
                        for ps, msg in errors:
                            st.write(f"❌ {ps}: {msg}")
                    st.error("No valid pipe size results. Check inputs and CSV rows.")
                else:
                    # Keep sizes that satisfy the dt limit, pick the smallest OD (mm)
                    valid = [r for r in results if r["dt"] <= max_penalty]
            
                    if valid:
                        best = min(valid, key=lambda x: mm_map[x["size"]])
                        st.session_state["_next_selected_size"] = best["size"]
        
                        # 🔹 Auto-select gauge for Copper EN12735
                        if selected_material.strip() == "Copper EN12735":
                            rows = material_df[
                                material_df["Nominal Size (inch)"].astype(str).str.strip() == best["size"]
                            ]
                    
                            if "Gauge" in rows.columns and rows["Gauge"].notna().any():
                                gauges = sorted(rows["Gauge"].dropna().unique())
                    
                                best_gauge = _auto_select_copper_gauge(
                                    material_df=material_df,
                                    size_inch=best["size"],
                                    gauges=gauges,
                                    design_pressure=result["design_pressure_bar_g"],
                                    refrigerant=refrigerant,
                                    design_temp_c=design_temp_c,
                                    mwp_temp_c=mwp_temp_c,
                                    circuit=circuit,
                                    pipe_index=pipe_index,
                                    copper_calc=copper_calc,
                                    dp_standard=dp_standard,
                                    r744_tc_pressure_bar_g=r744_tc_pressure_bar_g,
                                )
                    
                                st.session_state["_next_gauge"] = best_gauge
                        
                        st.success(
                            f"✅ Selected liquid pipe size: **{best['size']}**  \n"
                            f"ΔT: {best['dt']:.3f} K (limit {max_penalty:.3f} K)"
                        )
                        st.rerun()
                    else:
                        best_dt = min(r["dt"] for r in results if math.isfinite(r["dt"]))
                        st.error(
                            "❌ No pipe meets the ΔT limit.  \n"
                            f"Best achievable ΔT = {best_dt:.3f} K (must be ≤ {max_penalty:.3f} K)  \n"
                            "➡ Relax the Max Penalty or choose a different material/length/fittings."
                        )
    
        if velocity_m_s:
            col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    
            with col1:
                st.metric("Refrigerant Velocity", f"{velocity_m_s:.2f}m/s")
    
            with col2:
                st.metric("Liquid Density", f"{density:.1f}kg/m³")
    
            with col3:
                st.metric("Pressure Drop", f"{dp_total_kPa:.2f}kPa")
    
            with col4:
                if refrigerant == "R744 TC":
                    st.metric("Temp Penalty", "")
                else:
                    st.metric("Temp Penalty", f"{dt:.2f}K")

            with col5:
                if refrigerant == "R744 TC":
                    st.metric("Additional Subcooling Required", "")
                else:
                    st.metric("Additional Subcooling Required", f"{addsub:.2f}K")

            with col6:
                st.metric("Evaporating Pressure", f"{evappres:.2f}bar(a)")

            with col7:
                st.metric("Condensing Pressure", f"{condpres:.2f}bar(a)")

            # correcting default values between cond, max liq, and min liq between liquid calcs and dry suction calcs
            
            col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    
            with col1:
                st.metric("Mass Flow Rate", f"{mass_flow_kg_s:.5f}kg/s")
    
            with col2:
                st.metric("Volumetric Flow Rate", f"{volflow:.5f}m³/s")
    
            with col3:
                st.metric("Pipe PD", f"{dp_pipe_kPa:.2f}kPa")
    
            with col4:
                st.metric("Fittings PD", f"{dp_fittings_kPa:.2f}kPa")

            with col5:
                st.metric("Valves PD", f"{dp_valves_kPa:.2f}kPa")

            with col6:
                st.metric("Velocity Pressure PD", f"{dp_plf_kPa:.2f}kPa")

            with col7:
                st.metric("Compression Ratio", f"{compratio:.2f}")

    if mode == "Discharge":

        from utils.refrigerant_properties import RefrigerantProperties
        from utils.refrigerant_densities import RefrigerantDensities
        from utils.refrigerant_viscosities import RefrigerantViscosities
        from utils.refrigerant_entropies import RefrigerantEntropies
        from utils.refrigerant_enthalpies import RefrigerantEnthalpies
        from utils.supercompliq_co2 import RefrigerantProps
        
        # Load pipe data
        pipe_data = pd.read_csv("data/pipe_pressure_ratings_full.csv")
    
        # --- helpers ---
        def _nps_inch_to_mm(nps_str: str) -> float:
            # e.g. "1-1/8", '1"', "3/8"
            s = str(nps_str).replace('"', '').strip()
            if not s:
                return float('nan')
            parts = s.split('-')
            tot_in = 0.0
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                if '/' in p:
                    num, den = p.split('/')
                    tot_in += float(num) / float(den)
                else:
                    tot_in += float(p)
            return tot_in * 25.4  # mm
    
        ss = st.session_state
    
        # 1) Pipe material
        col1, col2, col3, col4 = st.columns(4)
        with col2:
            if refrigerant == "R717":
                excluded_materials = ["Copper ASTM", " Copper EN12735", "K65 Copper", "Reflok Aluminium"]
                pipe_materials = sorted(m for m in pipe_data["Material"].dropna().unique()
                                        if m not in excluded_materials)
            else:
                pipe_materials = sorted(pipe_data["Material"].dropna().unique())
    
            selected_material = st.selectbox("Pipe Material", pipe_materials, key="material")
        
        # detect material change
        material_changed = ss.get("last_material") is not None and ss.last_material != selected_material
        ss.last_material = selected_material
    
        # 2) Sizes for selected material (de-duped)
        material_df = pipe_data[pipe_data["Material"] == selected_material].copy()
    
        sizes_df = (
            material_df[["Nominal Size (inch)", "Nominal Size (mm)"]]
            .dropna(subset=["Nominal Size (inch)"])
            .assign(**{
                "Nominal Size (inch)": lambda d: d["Nominal Size (inch)"].astype(str).str.strip(),
            })
            .drop_duplicates(subset=["Nominal Size (inch)"], keep="first")
        )
    
        # make sure we have a numeric mm per nominal (fallback: parse the inch string)
        sizes_df["mm_num"] = pd.to_numeric(sizes_df.get("Nominal Size (mm)"), errors="coerce")
        sizes_df.loc[sizes_df["mm_num"].isna(), "mm_num"] = sizes_df.loc[sizes_df["mm_num"].isna(), "Nominal Size (inch)"].apply(_nps_inch_to_mm)
    
        pipe_sizes = sizes_df["Nominal Size (inch)"].tolist()
        mm_map = dict(zip(sizes_df["Nominal Size (inch)"], sizes_df["mm_num"]))
    
        # choose default index
        def _closest_index(target_mm: float) -> int:
            mm_list = [mm_map[s] for s in pipe_sizes]
            return min(range(len(mm_list)), key=lambda i: abs(mm_list[i] - target_mm)) if mm_list else 0
    
        # --- consume any deferred selection from the button ---
        if "_next_selected_size" in st.session_state:
            new_val = st.session_state.pop("_next_selected_size")
            if new_val in pipe_sizes:
                st.session_state["selected_size"] = new_val
                st.session_state["_selected_size_just_set"] = True
        
        default_index = 0
        if material_changed and "prev_pipe_mm" in ss:
            default_index = _closest_index(ss.prev_pipe_mm)
        elif selected_material == " Copper EN12735" and ("5/8" in pipe_sizes or '5/8"' in pipe_sizes):
            # first load or no previous selection → prefer 1-1/8" for  Copper EN12735
            want = "5/8" if "5/8" in pipe_sizes else '5/8"'
            default_index = pipe_sizes.index(want)
        elif "selected_size" in ss and ss.selected_size in pipe_sizes:
            # if Streamlit kept the selection, use it
            default_index = pipe_sizes.index(ss.selected_size)
    
        with col1:
            selected_size = st.selectbox(
                "Nominal Pipe Size (inch)",
                pipe_sizes,
                index=default_index,
                key="selected_size",
            )
    
        if st.session_state.get("_selected_size_just_set"):
            del st.session_state["_selected_size_just_set"]
        
        # remember the selected size in mm for next material change
        ss.prev_pipe_mm = float(mm_map.get(selected_size, float("nan")))

        # --- consume any deferred gauge from Auto-select (Discharge mode) ---
        if "_next_gauge" in st.session_state:
            g = st.session_state.pop("_next_gauge")
        
            rows = material_df[
                material_df["Nominal Size (inch)"].astype(str).str.strip() == str(selected_size)
            ]
        
            if "Gauge" in rows.columns and rows["Gauge"].notna().any():
                valid_gauges = set(rows["Gauge"].dropna().unique())
                if g in valid_gauges:
                    st.session_state["gauge"] = g
    
        # 3) Gauge (if applicable)
        gauge_options = material_df[material_df["Nominal Size (inch)"].astype(str).str.strip() == selected_size]
        if "Gauge" in gauge_options.columns and gauge_options["Gauge"].notna().any():
            gauges = sorted(gauge_options["Gauge"].dropna().unique())
            with col2:
                selected_gauge = st.selectbox("Copper Gauge", gauges, key="gauge")
            selected_pipe_row = gauge_options[gauge_options["Gauge"] == selected_gauge].iloc[0]
        else:
            selected_pipe_row = gauge_options.iloc[0]

        pipe_index = material_to_pipe_index(selected_material)
        
        # you already have selected_gauge sometimes; otherwise None
        gauge = st.session_state.get("gauge")  # or whatever that mode uses
        od_mm, id_mm = get_dimensions_for_row(material_df, selected_size, gauge)
        
        result = system_pressure_check(
            refrigerant=refrigerant,
            design_temp_c=design_temp_c,
            mwp_temp_c=mwp_temp_c,
            circuit=circuit,
            pipe_index=pipe_index,
            od_mm=od_mm,
            id_mm=id_mm,
            gauge=gauge,
            copper_calc=copper_calc,
            r744_tc_pressure_bar_g=r744_tc_pressure_bar_g,
            dp_standard=dp_standard,
        )
        
        render_pressure_result(result)
    
        # Pipe parameters
        pipe_size_inch = selected_pipe_row["Nominal Size (inch)"]
        ID_mm = selected_pipe_row["ID_mm"]

        with col1:
            
            evap_capacity_kw = st.number_input("Evaporator Capacity (kW)", min_value=0.03, max_value=20000.0, value=10.0, step=1.0)
            
            # --- Base ranges per refrigerant ---
            if refrigerant in ("R23", "R508B"):
                evap_min, evap_max, evap_default = -100.0, -20.0, -80.0
                cond_min, cond_max, cond_default = -100.0, 10.0, -30.0
                maxliq_min, maxliq_max, maxliq_default = -100.0, 10.0, -40.0
            elif refrigerant == "R744":
                evap_min, evap_max, evap_default = -50.0, 20.0, -10.0
                cond_min, cond_max, cond_default = -23.0, 30.0, 15.0
                maxliq_min, maxliq_max, maxliq_default = -50.0, 30.0, 10.0
            else:
                evap_min, evap_max, evap_default = -50.0, 30.0, -10.0
                cond_min, cond_max, cond_default = -23.0, 60.0, 43.0
                maxliq_min, maxliq_max, maxliq_default = -50.0, 60.0, 40.0
    
            # --- Init state (widget-backed) ---
            ss = st.session_state
    
            if "last_refrigerant" not in ss or ss.last_refrigerant != refrigerant:
                ss.cond_temp   = cond_default
                ss.maxliq_temp = maxliq_default
                ss.evap_temp   = evap_default
                ss.last_refrigerant = refrigerant
            
            ss.setdefault("cond_temp",   cond_default)
            ss.setdefault("maxliq_temp", maxliq_default)
            ss.setdefault("evap_temp",   evap_default)

            if "maxliq_temp" in ss and "cond_temp" in ss and "evap_temp" in ss:
                ss.cond_temp = min(max(ss.cond_temp, ss.maxliq_temp, ss.evap_temp), cond_max)
    
            if "cond_temp" in ss and "maxliq_temp" in ss and "evap_temp" in ss:
                ss.evap_temp = min(ss.maxliq_temp, ss.cond_temp, ss.evap_temp)
            
            # --- Callbacks implementing your downstream clamping logic ---
            def on_change_cond():
                # When cond changes: clamp maxliq down to cond, then evap down to maxliq
                ss.maxliq_temp = min(ss.maxliq_temp, ss.cond_temp)
                ss.evap_temp   = min(ss.evap_temp,   ss.maxliq_temp)
    
            def on_change_maxliq():
                # When maxliq changes: clamp maxliq down to cond, then evap down to maxliq
                ss.maxliq_temp = min(ss.maxliq_temp, ss.cond_temp)
                ss.evap_temp   = min(ss.evap_temp,   ss.maxliq_temp)
    
            def on_change_evap():
                # When evap changes: clamp evap down to maxliq
                if refrigerant == "R744 TC":
                    ss.evap_temp   = min(ss.evap_temp,   maxliq_temp)
                else:
                    ss.evap_temp   = min(ss.evap_temp,   ss.maxliq_temp)

            if refrigerant == "R744 TC":
                ss.setdefault("gc_max_temp", 38.0)
                gc_max_temp = st.number_input(
                    "Gas Cooler Outlet Temp (°C)",
                    min_value=-50.0, max_value=50.0,
                    value=ss.gc_max_temp, step=1.0, key="gc_max_temp"
                )
        
                ss.setdefault("gc_max_pres", 93.7)
                gc_max_pres = st.number_input(
                    "Gas Cooler Outlet Pressure (bar(a))",
                    min_value=73.8, max_value=150.0,
                    value=ss.gc_max_pres, step=1.0, key="gc_max_pres"
                )
            
                maxliq_temp = gc_max_temp

                ss.evap_temp = min(ss.evap_temp, maxliq_temp)
            
            else:
                condensing_temp = st.number_input(
                    "Condensing Temperature (°C)",
                    min_value=cond_min, max_value=cond_max,
                    value=ss.cond_temp, step=1.0, key="cond_temp",
                    on_change=on_change_cond,
                )
        
                maxliq_temp = st.number_input(
                    "Liquid Temperature (°C)",
                    min_value=maxliq_min, max_value=min(condensing_temp, maxliq_max),
                    value=ss.maxliq_temp, step=1.0, key="maxliq_temp",
                    on_change=on_change_maxliq,
                )

            evaporating_temp = st.number_input(
                "Evaporating Temperature (°C)",
                min_value=evap_min, max_value=min(maxliq_temp, evap_max),
                value=ss.evap_temp, step=1.0, key="evap_temp",
                on_change=on_change_evap,
            )
        
        with col2:
            isen = st.number_input("Isentropic Efficiency (%)", min_value=50.0, max_value=90.0, value=75.0, step=5.0)
            superheat_K = st.number_input("Superheat (K)", min_value=0.0, max_value=60.0, value=5.0, step=1.0)
            if refrigerant == "R744 TC":
                max_linelosss = st.number_input("Max Pressure Drop (kPa)", min_value=0.0, max_value=250.0, value=40.0, step=5.0)
            else:
                max_penalty = st.number_input("Max Penalty (K)", min_value=0.0, max_value=6.0, value=1.0, step=0.1)
    
        with col3:
            L = st.number_input("Pipe Length (m)", min_value=0.1, max_value=300.0, value=10.0, step=1.0)
            LRB = st.number_input("Long Radius Bends", min_value=0, max_value=50, value=0, step=1)
            SRB = st.number_input("Short Radius Bends", min_value=0, max_value=50, value=0, step=1)
            _45 = st.number_input("45° Bends", min_value=0, max_value=50, value=0, step=1)
            MAC = st.number_input("Machine Bends", min_value=0, max_value=50, value=0, step=1)
    
        with col4:
            ptrap = st.number_input("P Traps", min_value=0, max_value=10, value=0, step=1)
            ubend = st.number_input("U Bends", min_value=0, max_value=10, value=0, step=1)
            ball = st.number_input("Ball Valves", min_value=0, max_value=20, value=0, step=1, key="ball")
            globe = st.number_input("Globe Valves", min_value=0, max_value=20, value=0, step=1, key="globe")
            PLF = st.number_input("Pressure Loss Factors", min_value=0.0, max_value=20.0, value=0.0, step=0.1)

        if refrigerant == "R744 TC":
            T_evap = evaporating_temp
            T_liq = maxliq_temp
        else:
            T_evap = evaporating_temp
            T_liq = maxliq_temp
            T_cond = condensing_temp
    
        props = RefrigerantProperties()
        props_sup = RefrigerantProps()

        if refrigerant == "R744 TC":
            h_in = props_sup.get_enthalpy_sup(gc_max_pres, maxliq_temp)
            h_evap = props.get_properties("R744", T_evap)["enthalpy_vapor"]
        else:
            h_in = props.get_properties(refrigerant, T_liq)["enthalpy_liquid2"]
            h_evap = props.get_properties(refrigerant, T_evap)["enthalpy_vapor"]
        
        delta_h = h_evap - h_in

        mass_flow_kg_s = evap_capacity_kw / delta_h if delta_h > 0 else 0.01
    
        if ID_mm is not None:
            ID_m = ID_mm / 1000.0

            area_m2 = math.pi * (ID_m / 2) ** 2

            if refrigerant == "R744 TC":
                suc_ent = RefrigerantEntropies().get_entropy("R744", T_evap + 273.15, superheat_K)
                isen_sup = props_sup.get_temperature_from_entropy(gc_max_pres, suc_ent)
                isen_enth = props_sup.get_enthalpy_sup(gc_max_pres, isen_sup)
                suc_enth = RefrigerantEnthalpies().get_enthalpy("R744", T_evap + 273.15, superheat_K)
            else:
                suc_ent = RefrigerantEntropies().get_entropy(refrigerant, T_evap + 273.15, superheat_K)
                isen_sup = RefrigerantEntropies().get_superheat_from_entropy(refrigerant, T_cond + 273.15, suc_ent)
                isen_enth = RefrigerantEnthalpies().get_enthalpy(refrigerant, T_cond + 273.15, isen_sup)
                suc_enth = RefrigerantEnthalpies().get_enthalpy(refrigerant, T_evap + 273.15, superheat_K)

            isen_change = isen_enth - suc_enth

            enth_change = isen_change / (isen / 100)

            dis_enth = suc_enth + enth_change

            if refrigerant == "R744 TC":
                dis_t = props_sup.get_temperature_from_enthalpy(gc_max_pres, dis_enth)
            else:
                dis_sup = RefrigerantEnthalpies().get_superheat_from_enthalpy(refrigerant, T_cond + 273.15, dis_enth)
                dis_t = T_cond + dis_sup
            
            if refrigerant == "R744 TC":
                dis_dens = props_sup.get_density_sup(gc_max_pres, dis_t)
                dis_visc = props_sup.get_viscosity_sup(gc_max_pres, dis_t)
            else:
                dis_dens = RefrigerantDensities().get_density(refrigerant, T_cond + 273.15, dis_sup)
                dis_visc = RefrigerantViscosities().get_viscosity(refrigerant, T_cond + 273.15, dis_sup)

            velocity_m_s = mass_flow_kg_s / (area_m2 * dis_dens)
            
        else:
            velocity_m_s = None

        reynolds = (dis_dens * velocity_m_s * ID_m) / (dis_visc / 1000000)

        if selected_material in ["Steel SCH40", "Steel SCH80"]:
            eps = 0.00004572 #0.00015
        else:
            eps = 0.000001524 #0.000005
        
        tol = 1e-5
        max_iter = 60
        
        if reynolds < 2000.0:
            f = 64.0 / reynolds
        else:
            flo, fhi = 1e-5, 0.1
            def balance(gg):
                s = math.sqrt(gg)
                lhs = 1.0 / s
                rhs = -2.0 * math.log10((eps / (3.7 * ID_m)) + 2.51 / (reynolds * s))
                return lhs, rhs
    
            f = 0.5 * (flo + fhi)
            for _ in range(max_iter):
                f = 0.5 * (flo + fhi)
                lhs, rhs = balance(f)
                if abs(1.0 - lhs/rhs) < tol:
                    break
                # decide side using sign of (lhs - rhs)
                if (lhs - rhs) > 0.0:
                    flo = f
                else:
                    fhi = f

        q_kPa = 0.5 * dis_dens * (velocity_m_s ** 2) / 1000.0

        dp_pipe_kPa = f * (L / ID_m) * q_kPa
        
        dp_plf_kPa = q_kPa * PLF
    
        required_cols = ["SRB", "LRB", "BALL", "GLOBE"]
        missing = [c for c in required_cols if c not in selected_pipe_row.index]
        if missing:
            st.error(f"CSV missing required K columns: {missing}")
            st.stop()
    
        # Convert to floats and check NaNs
        try:
            K_SRB  = float(selected_pipe_row["SRB"])
            K_LRB  = float(selected_pipe_row["LRB"])
            K_BALL = float(selected_pipe_row["BALL"])
            K_GLOBE= float(selected_pipe_row["GLOBE"])
        except Exception as e:
            st.error(f"Failed to parse K-factors as numbers: {e}")
            st.stop()
    
        if any(pd.isna([K_SRB, K_LRB, K_BALL, K_GLOBE])):
            st.error("One or more K-factors are NaN in the CSV row.")
            st.stop()
        
        B_SRB = SRB + 0.5 * _45 + 2.0 * ubend + 3.0 * ptrap
        B_LRB = LRB + MAC
    
        dp_fittings_kPa = q_kPa * (
        K_SRB   * B_SRB +
        K_LRB   * B_LRB
        )

        dp_valves_kPa = q_kPa * (
        K_BALL  * ball +
        K_GLOBE * globe
        )
        
        dp_total_kPa = dp_pipe_kPa + dp_fittings_kPa + dp_valves_kPa + dp_plf_kPa
        
        if refrigerant == "R744 TC":
            converter = PressureTemperatureConverter()
            condpres = gc_max_pres
            evappres = converter.temp_to_pressure("R744", T_evap)
        
        else:
            converter = PressureTemperatureConverter()
            condpres = converter.temp_to_pressure(refrigerant, T_cond)
            postcirc = condpres - (dp_total_kPa / 100)
            postcirctemp = converter.pressure_to_temp(refrigerant, postcirc)
            dt = T_cond - postcirctemp
            evappres = converter.temp_to_pressure(refrigerant, T_evap)

        volflow = mass_flow_kg_s / dis_dens

        compratio = condpres / evappres
        
        def _pipe_row_for_size(size_inch: str, gauge_override=None):
            rows = material_df[
                material_df["Nominal Size (inch)"].astype(str).str.strip() == str(size_inch)
            ]
            if rows.empty:
                return None
        
            if "Gauge" in rows.columns and rows["Gauge"].notna().any():
        
                # 🔹 1️⃣ evaluation-time override (auto-select logic)
                if gauge_override is not None:
                    rows_g = rows[rows["Gauge"] == gauge_override]
                    if not rows_g.empty:
                        return rows_g.iloc[0]
        
                # 🔹 2️⃣ UI-selected gauge (manual mode)
                if "gauge" in st.session_state:
                    g = st.session_state["gauge"]
                    rows_g = rows[rows["Gauge"] == g]
                    if not rows_g.empty:
                        return rows_g.iloc[0]
        
                # 🔹 3️⃣ fallback (first row)
                return rows.iloc[0]
        
            return rows.iloc[0]

        def get_discharge_dt_for_size(size_inch: str, gauge_override=None) -> float:
            """
            Compute ΔT (dt) for a given discharge pipe size using the SAME method as the main block:
            1) Isentropic chain → discharge superheat, enthalpy, temperature
            2) Discharge density/viscosity at (T_cond, dis_sup)
            3) Velocity/Reynolds → friction factor (Colebrook or laminar)
            4) q_kPa → dp's (pipe + fittings + valves + PLF)
            5) Convert dp to post-circ temperature → dt = T_cond - postcirc_temp
            Returns NaN on failure.
            """
            try:
                pipe_row = _pipe_row_for_size(size_inch, gauge_override)
                if pipe_row is None:
                    return float("nan")
        
                # Geometry
                ID_mm_local = float(pipe_row["ID_mm"])
                ID_m_local  = ID_mm_local / 1000.0
                area_m2     = math.pi * (ID_m_local / 2) ** 2
        
                # 1) Isentropic chain – size independent, but we recompute to be safe
                if refrigerant == "R744 TC":
                    suc_ent    = RefrigerantEntropies().get_entropy("R744", T_evap + 273.15, superheat_K)
                    isen_sup   = props_sup.get_temperature_from_entropy(gc_max_pres, suc_ent)
                    isen_enth  = props_sup.get_enthalpy_sup(gc_max_pres, isen_sup)
                    suc_enth   = RefrigerantEnthalpies().get_enthalpy("R744", T_evap + 273.15, superheat_K)
                else:
                    suc_ent    = RefrigerantEntropies().get_entropy(refrigerant, T_evap + 273.15, superheat_K)
                    isen_sup   = RefrigerantEntropies().get_superheat_from_entropy(refrigerant, T_cond + 273.15, suc_ent)
                    isen_enth  = RefrigerantEnthalpies().get_enthalpy(refrigerant, T_cond + 273.15, isen_sup)
                    suc_enth   = RefrigerantEnthalpies().get_enthalpy(refrigerant, T_evap + 273.15, superheat_K)
                
                isen_change = isen_enth - suc_enth
                enth_change = isen_change / (isen / 100.0)
                dis_enth    = suc_enth + enth_change
                
                if refrigerant == "R744 TC":
                    dis_t = props_sup.get_temperature_from_enthalpy(gc_max_pres, dis_enth)
                else:
                    dis_sup     = RefrigerantEnthalpies().get_superheat_from_enthalpy(refrigerant, T_cond + 273.15, dis_enth)
        
                # Discharge properties at (T_cond, dis_sup)
                if refrigerant == "R744 TC":
                    dis_dens = props_sup.get_density_sup(gc_max_pres, dis_t)
                    dis_visc = props_sup.get_viscosity_sup(gc_max_pres, dis_t)
                else:
                    dis_dens = RefrigerantDensities().get_density(refrigerant, T_cond + 273.15, dis_sup)
                    dis_visc = RefrigerantViscosities().get_viscosity(refrigerant, T_cond + 273.15, dis_sup)
        
                # Mass flow is size-independent (already computed in main code)
                v = mass_flow_kg_s / (area_m2 * dis_dens)
        
                # 2) Reynolds
                Re = (dis_dens * v * ID_m_local) / (dis_visc / 1_000_000)
        
                # 3) Roughness and friction factor
                eps = 0.00004572 if selected_material in ["Steel SCH40", "Steel SCH80"] else 0.000001524
                if Re < 2000.0 and Re > 0:
                    f_local = 64.0 / Re
                else:
                    flo, fhi = 1e-5, 0.1
                    tol, max_iter = 1e-5, 60
                    def balance(gg):
                        s = math.sqrt(gg)
                        lhs = 1.0 / s
                        rhs = -2.0 * math.log10((eps / (3.7 * ID_m_local)) + 2.51 / (Re * s))
                        return lhs, rhs
                    f_local = 0.5 * (flo + fhi)
                    for _ in range(max_iter):
                        f_try = 0.5 * (flo + fhi)
                        lhs, rhs = balance(f_try)
                        if abs(1.0 - lhs / rhs) < tol:
                            f_local = f_try
                            break
                        if (lhs - rhs) > 0.0:
                            flo = f_try
                        else:
                            fhi = f_try
                    else:
                        f_local = 0.5 * (flo + fhi)
        
                # 4) Dynamic pressure and K-based losses
                q_kPa = 0.5 * dis_dens * (v ** 2) / 1000.0
        
                for c in ["SRB", "LRB", "BALL", "GLOBE"]:
                    if c not in pipe_row.index or pd.isna(pipe_row[c]):
                        return float("nan")
                K_SRB   = float(pipe_row["SRB"])
                K_LRB   = float(pipe_row["LRB"])
                K_BALL  = float(pipe_row["BALL"])
                K_GLOBE = float(pipe_row["GLOBE"])
        
                B_SRB = SRB + 0.5 * _45 + 2.0 * ubend + 3.0 * ptrap
                B_LRB = LRB + MAC
        
                dp_pipe_kPa = f_local * (L / ID_m_local) * q_kPa
                dp_plf_kPa  = q_kPa * PLF
                dp_fit_kPa  = q_kPa * (K_SRB * B_SRB + K_LRB * B_LRB)
                dp_val_kPa  = q_kPa * (K_BALL * ball + K_GLOBE * globe)
        
                dp_total_kPa_local = dp_pipe_kPa + dp_fit_kPa + dp_val_kPa + dp_plf_kPa
        
                # 5) Convert Δp → ΔT at condenser side
                if refrigerant == "R744 TC":
                    dt_local = dp_total_kPa_local
                else:
                    conv = PressureTemperatureConverter()
                    condpres_local   = conv.temp_to_pressure(refrigerant, T_cond)
                    postcirc_local   = condpres_local - (dp_total_kPa_local / 100.0)  # kPa→bar
                    postcirctemp_loc = conv.pressure_to_temp(refrigerant, postcirc_local)
                    dt_local         = T_cond - postcirctemp_loc
        
                return float(dt_local)
        
            except Exception:
                return float("nan")

        def _auto_select_copper_gauge(
            *,
            material_df,
            size_inch,
            gauges,
            design_pressure,
            refrigerant,
            design_temp_c,
            mwp_temp_c,
            circuit,
            pipe_index,
            copper_calc,
            dp_standard,
            r744_tc_pressure_bar_g,
        ):
            """
            Return:
            - highest gauge where MWP >= design pressure
            - else lowest gauge
            """
        
            passing = []
        
            for g in sorted(gauges):
                try:
                    od_mm, id_mm = get_dimensions_for_row(material_df, size_inch, g)
        
                    res = system_pressure_check(
                        refrigerant=refrigerant,
                        design_temp_c=design_temp_c,
                        mwp_temp_c=mwp_temp_c,
                        circuit=circuit,
                        pipe_index=pipe_index,
                        od_mm=od_mm,
                        id_mm=id_mm,
                        gauge=g,
                        copper_calc=copper_calc,
                        r744_tc_pressure_bar_g=r744_tc_pressure_bar_g,
                        dp_standard=dp_standard,
                    )
        
                    mwp_val = governing_mwp(res["mwp_bar"])
        
                    if mwp_val >= design_pressure:
                        passing.append(g)
        
                except Exception:
                    continue
        
            if passing:
                return max(passing)
            else:
                return min(gauges)
        
        if st.button("Auto-select"):
            if refrigerant == "R744 TC":
                results, errors = [], []
        
                for ps in pipe_sizes:
                    rows = material_df[
                        material_df["Nominal Size (inch)"].astype(str).str.strip() == ps
                    ]
        
                    # 🔹 force strongest gauge for evaluation (discharge)
                    gauge_override = None
                    if "Gauge" in rows.columns and rows["Gauge"].notna().any():
                        gauge_override = max(rows["Gauge"].dropna().unique())
        
                    dt_i = get_discharge_dt_for_size(ps, gauge_override=gauge_override)
        
                    if math.isfinite(dt_i):
                        results.append({"size": ps, "dt": dt_i})
                    else:
                        errors.append((ps, "Non-numeric ΔT"))
        
                if not results:
                    with st.expander("⚠️ Discharge selection debug details", expanded=True):
                        for ps, msg in errors:
                            st.write(f"❌ {ps}: {msg}")
                    st.error("No valid pipe size results. Check inputs and CSV rows.")
                else:
                    valid = [r for r in results if r["dt"] <= max_linelosss]
        
                    if valid:
                        best = min(valid, key=lambda x: mm_map[x["size"]])  # smallest OD that passes
                        st.session_state["_next_selected_size"] = best["size"]
        
                        # 🔹 Auto-select gauge for Copper EN12735
                        if selected_material.strip() == "Copper EN12735":
                            rows = material_df[
                                material_df["Nominal Size (inch)"].astype(str).str.strip() == best["size"]
                            ]
        
                            if "Gauge" in rows.columns and rows["Gauge"].notna().any():
                                gauges = sorted(rows["Gauge"].dropna().unique())
        
                                best_gauge = _auto_select_copper_gauge(
                                    material_df=material_df,
                                    size_inch=best["size"],
                                    gauges=gauges,
                                    design_pressure=result["design_pressure_bar_g"],
                                    refrigerant=refrigerant,
                                    design_temp_c=design_temp_c,
                                    mwp_temp_c=mwp_temp_c,
                                    circuit=circuit,
                                    pipe_index=pipe_index,
                                    copper_calc=copper_calc,
                                    dp_standard=dp_standard,
                                    r744_tc_pressure_bar_g=r744_tc_pressure_bar_g,
                                )
        
                                st.session_state["_next_gauge"] = best_gauge
        
                        st.success(
                            f"✅ Selected discharge pipe size: **{best['size']}**  \n"
                            f"ΔT: {best['dt']:.2f} kPa (limit {max_linelosss:.2f} kPa)"
                        )
                        st.rerun()
                    else:
                        best_dt = min(r["dt"] for r in results if math.isfinite(r["dt"]))
                        st.error(
                            "❌ No pipe meets the PD limit.  \n"
                            f"Best achievable PD = {best_dt:.2f} kPa (must be ≤ {max_linelosss:.2f} kPa)  \n"
                            "➡ Relax the Max PD or change material/length/fittings."
                        )
        
            else:
                results, errors = [], []
        
                for ps in pipe_sizes:
                    rows = material_df[
                        material_df["Nominal Size (inch)"].astype(str).str.strip() == ps
                    ]
        
                    # 🔹 force strongest gauge for evaluation (discharge)
                    gauge_override = None
                    if "Gauge" in rows.columns and rows["Gauge"].notna().any():
                        gauge_override = max(rows["Gauge"].dropna().unique())
        
                    dt_i = get_discharge_dt_for_size(ps, gauge_override=gauge_override)
        
                    if math.isfinite(dt_i):
                        results.append({"size": ps, "dt": dt_i})
                    else:
                        errors.append((ps, "Non-numeric ΔT"))
        
                if not results:
                    with st.expander("⚠️ Discharge selection debug details", expanded=True):
                        for ps, msg in errors:
                            st.write(f"❌ {ps}: {msg}")
                    st.error("No valid pipe size results. Check inputs and CSV rows.")
                else:
                    valid = [r for r in results if r["dt"] <= max_penalty]
        
                    if valid:
                        best = min(valid, key=lambda x: mm_map[x["size"]])  # smallest OD that passes
                        st.session_state["_next_selected_size"] = best["size"]
        
                        # 🔹 Auto-select gauge for Copper EN12735
                        if selected_material.strip() == "Copper EN12735":
                            rows = material_df[
                                material_df["Nominal Size (inch)"].astype(str).str.strip() == best["size"]
                            ]
        
                            if "Gauge" in rows.columns and rows["Gauge"].notna().any():
                                gauges = sorted(rows["Gauge"].dropna().unique())
        
                                best_gauge = _auto_select_copper_gauge(
                                    material_df=material_df,
                                    size_inch=best["size"],
                                    gauges=gauges,
                                    design_pressure=result["design_pressure_bar_g"],
                                    refrigerant=refrigerant,
                                    design_temp_c=design_temp_c,
                                    mwp_temp_c=mwp_temp_c,
                                    circuit=circuit,
                                    pipe_index=pipe_index,
                                    copper_calc=copper_calc,
                                    dp_standard=dp_standard,
                                    r744_tc_pressure_bar_g=r744_tc_pressure_bar_g,
                                )
        
                                st.session_state["_next_gauge"] = best_gauge
        
                        st.success(
                            f"✅ Selected discharge pipe size: **{best['size']}**  \n"
                            f"ΔT: {best['dt']:.3f} K (limit {max_penalty:.3f} K)"
                        )
                        st.rerun()
                    else:
                        best_dt = min(r["dt"] for r in results if math.isfinite(r["dt"]))
                        st.error(
                            "❌ No pipe meets the ΔT limit.  \n"
                            f"Best achievable ΔT = {best_dt:.3f} K (must be ≤ {max_penalty:.3f} K)  \n"
                            "➡ Relax the Max Penalty or change material/length/fittings."
                        )
    
        if velocity_m_s:
            col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    
            with col1:
                st.metric("Refrigerant Velocity", f"{velocity_m_s:.2f}m/s")
    
            with col2:
                st.metric("Discharge Density", f"{dis_dens:.2f}kg/m³")
    
            with col3:
                st.metric("Pressure Drop", f"{dp_total_kPa:.2f}kPa")
    
            with col4:
                if refrigerant == "R744 TC":
                    st.metric("Temp Penalty", "")
                else:
                    st.metric("Temp Penalty", f"{dt:.2f}K")

            with col5:
                st.metric("Discharge Temperature", f"{dis_t:.2f}°C")

            with col6:
                st.metric("Evaporating Pressure", f"{evappres:.2f}bar(a)")

            with col7:
                st.metric("Condensing Pressure", f"{condpres:.2f}bar(a)")
            
            col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    
            with col1:
                st.metric("Mass Flow Rate", f"{mass_flow_kg_s:.5f}kg/s")
    
            with col2:
                st.metric("Volumetric Flow Rate", f"{volflow:.5f}m³/s")
    
            with col3:
                st.metric("Pipe PD", f"{dp_pipe_kPa:.2f}kPa")
    
            with col4:
                st.metric("Fittings PD", f"{dp_fittings_kPa:.2f}kPa")

            with col5:
                st.metric("Valves PD", f"{dp_valves_kPa:.2f}kPa")

            with col6:
                st.metric("Velocity Pressure PD", f"{dp_plf_kPa:.2f}kPa")

            with col7:
                st.metric("Compression Ratio", f"{compratio:.2f}")

    if mode == "Drain":

        from utils.refrigerant_properties import RefrigerantProperties
        from utils.refrigerant_densities import RefrigerantDensities
        from utils.refrigerant_viscosities import RefrigerantViscosities
        from utils.refrigerant_entropies import RefrigerantEntropies
        from utils.refrigerant_enthalpies import RefrigerantEnthalpies
        
        # Load pipe data
        pipe_data = pd.read_csv("data/pipe_pressure_ratings_full.csv")
    
        # --- helpers ---
        def _nps_inch_to_mm(nps_str: str) -> float:
            # e.g. "1-1/8", '1"', "3/8"
            s = str(nps_str).replace('"', '').strip()
            if not s:
                return float('nan')
            parts = s.split('-')
            tot_in = 0.0
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                if '/' in p:
                    num, den = p.split('/')
                    tot_in += float(num) / float(den)
                else:
                    tot_in += float(p)
            return tot_in * 25.4  # mm
    
        ss = st.session_state
    
        # 1) Pipe material
        col1, col2, col3, col4 = st.columns(4)
        with col2:
            if refrigerant == "R717":
                excluded_materials = ["Copper ASTM", " Copper EN12735", "K65 Copper", "Reflok Aluminium"]
                pipe_materials = sorted(m for m in pipe_data["Material"].dropna().unique()
                                        if m not in excluded_materials)
            else:
                pipe_materials = sorted(pipe_data["Material"].dropna().unique())
    
            selected_material = st.selectbox("Pipe Material", pipe_materials, key="material", disabled=True)
        
        # detect material change
        material_changed = ss.get("last_material") is not None and ss.last_material != selected_material
        ss.last_material = selected_material
    
        # 2) Sizes for selected material (de-duped)
        material_df = pipe_data[pipe_data["Material"] == selected_material].copy()
    
        sizes_df = (
            material_df[["Nominal Size (inch)", "Nominal Size (mm)"]]
            .dropna(subset=["Nominal Size (inch)"])
            .assign(**{
                "Nominal Size (inch)": lambda d: d["Nominal Size (inch)"].astype(str).str.strip(),
            })
            .drop_duplicates(subset=["Nominal Size (inch)"], keep="first")
        )
    
        # make sure we have a numeric mm per nominal (fallback: parse the inch string)
        sizes_df["mm_num"] = pd.to_numeric(sizes_df.get("Nominal Size (mm)"), errors="coerce")
        sizes_df.loc[sizes_df["mm_num"].isna(), "mm_num"] = sizes_df.loc[sizes_df["mm_num"].isna(), "Nominal Size (inch)"].apply(_nps_inch_to_mm)
    
        pipe_sizes = sizes_df["Nominal Size (inch)"].tolist()
        mm_map = dict(zip(sizes_df["Nominal Size (inch)"], sizes_df["mm_num"]))
    
        # choose default index
        def _closest_index(target_mm: float) -> int:
            mm_list = [mm_map[s] for s in pipe_sizes]
            return min(range(len(mm_list)), key=lambda i: abs(mm_list[i] - target_mm)) if mm_list else 0

        default_index = 0
        if material_changed and "prev_pipe_mm" in ss:
            default_index = _closest_index(ss.prev_pipe_mm)
        elif "selected_size" in ss and ss.selected_size in pipe_sizes:
            default_index = pipe_sizes.index(ss.selected_size)
        
        # Apply auto-selected value if present
        if "auto_selected_main" in ss and ss.auto_selected_main in pipe_sizes:
            default_index = pipe_sizes.index(ss.auto_selected_main)
            del ss.auto_selected_main
        
        with col1:

            # --- Before main pipe selectbox ---
            if "auto_selected_main" in st.session_state and st.session_state.auto_selected_main in pipe_sizes:
                default_index = pipe_sizes.index(st.session_state.auto_selected_main)
                del st.session_state.auto_selected_main  # clear after use

            selected_size = st.selectbox(
                "Main Pipe Size (inch)",
                pipe_sizes,
                index=default_index,
                key="selected_size",
            )
    
        # remember the selected size in mm for next material change
        ss.prev_pipe_mm = float(mm_map.get(selected_size, float("nan")))
    
        # 3) Gauge (if applicable)
        gauge_options = material_df[material_df["Nominal Size (inch)"].astype(str).str.strip() == selected_size]

        if "_next_gauge_main" in st.session_state:
            g = st.session_state.pop("_next_gauge_main")
            rows = gauge_options
            if "Gauge" in rows.columns:
                valid = set(rows["Gauge"].dropna())
                if g in valid:
                    st.session_state["gauge"] = g
        
        if "Gauge" in gauge_options.columns and gauge_options["Gauge"].notna().any():
            gauges = sorted(gauge_options["Gauge"].dropna().unique())
            with col2:
                selected_gauge = st.selectbox("Main Copper Gauge", gauges, key="gauge")
            selected_pipe_row = gauge_options[gauge_options["Gauge"] == selected_gauge].iloc[0]
        else:
            selected_pipe_row = gauge_options.iloc[0]
    
        # Pipe parameters
        pipe_size_inch = selected_pipe_row["Nominal Size (inch)"]
        ID_mm = selected_pipe_row["ID_mm"]

        # --- helpers (reuse or keep separate if you want clarity) ---
        def _nps_inch_to_mm_2(nps_str: str) -> float:
            s = str(nps_str).replace('"', '').strip()
            if not s:
                return float('nan')
            parts = s.split('-')
            tot_in = 0.0
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                if '/' in p:
                    num, den = p.split('/')
                    tot_in += float(num) / float(den)
                else:
                    tot_in += float(p)
            return tot_in * 25.4  # mm

        ss = st.session_state

        # 1️⃣ Use same pipe material from main pipe
        selected_material_2 = ss.get("last_material", None)
        if not selected_material_2:
            st.warning("⚠️ Please select a pipe material in the main input first.")
            st.stop()

        # 2️⃣ Filter data for that material only
        material_df_2 = pipe_data[pipe_data["Material"] == selected_material_2].copy()

        sizes_df_2 = (
            material_df_2[["Nominal Size (inch)", "Nominal Size (mm)"]]
            .dropna(subset=["Nominal Size (inch)"])
            .assign(**{
                "Nominal Size (inch)": lambda d: d["Nominal Size (inch)"].astype(str).str.strip(),
            })
            .drop_duplicates(subset=["Nominal Size (inch)"], keep="first")
        )

        # Convert inch → mm if needed
        sizes_df_2["mm_num"] = pd.to_numeric(sizes_df_2.get("Nominal Size (mm)"), errors="coerce")
        sizes_df_2.loc[sizes_df_2["mm_num"].isna(), "mm_num"] = sizes_df_2.loc[
            sizes_df_2["mm_num"].isna(), "Nominal Size (inch)"
        ].apply(_nps_inch_to_mm_2)

        pipe_sizes_2 = sizes_df_2["Nominal Size (inch)"].tolist()
        mm_map_2 = dict(zip(sizes_df_2["Nominal Size (inch)"], sizes_df_2["mm_num"]))

        # 3️⃣ Choose default index
        def _closest_index_2(target_mm: float) -> int:
            mm_list = [mm_map_2[s] for s in pipe_sizes_2]
            return min(range(len(mm_list)), key=lambda i: abs(mm_list[i] - target_mm)) if mm_list else 0

        default_index_2 = 0
        if "prev_pipe_mm_2" in ss:
            default_index_2 = _closest_index_2(ss.prev_pipe_mm_2)
        elif "selected_size_2" in ss and ss.selected_size_2 in pipe_sizes_2:
            default_index_2 = pipe_sizes_2.index(ss.selected_size_2)

        # --- Apply auto-selected branch size ---
        if "_next_selected_branch" in st.session_state:
            v = st.session_state.pop("_next_selected_branch")
            if v in pipe_sizes_2:
                st.session_state["selected_size_2"] = v

        # 4️⃣ Size selector
        with col1:

            selected_size_2 = st.selectbox(
                "Branch Pipe Size (inch)",
                pipe_sizes_2,
                index=default_index_2,
                key="selected_size_2",
            )

        ss.prev_pipe_mm_2 = float(mm_map_2.get(selected_size_2, float("nan")))

        # 5️⃣ Gauge selector (if applicable)
        gauge_options_2 = material_df_2[
            material_df_2["Nominal Size (inch)"].astype(str).str.strip() == selected_size_2
        ]

        if "_next_gauge_branch" in st.session_state:
            g = st.session_state.pop("_next_gauge_branch")
            rows = gauge_options_2
            if "Gauge" in rows.columns:
                valid = set(rows["Gauge"].dropna())
                if g in valid:
                    st.session_state["gauge_2"] = g

        if "Gauge" in gauge_options_2.columns and gauge_options_2["Gauge"].notna().any():
            gauges_2 = sorted(gauge_options_2["Gauge"].dropna().unique())
            with col2:
                selected_gauge_2 = st.selectbox("Branch Copper Gauge", gauges_2, key="gauge_2")
            selected_pipe_row_2 = gauge_options_2[gauge_options_2["Gauge"] == selected_gauge_2].iloc[0]
        else:
            selected_pipe_row_2 = gauge_options_2.iloc[0]

        # 6️⃣ Output parameters for secondary pipe
        pipe_size_inch_2 = selected_pipe_row_2["Nominal Size (inch)"]
        ID_mm_2 = selected_pipe_row_2["ID_mm"]
        
        # --- Rule: Main pipe must be >= Branch pipe ---
        invalid_pipe_selection = False
        if ID_mm < ID_mm_2:
            st.error(
                f"🚫 Invalid selection: The main pipe ({selected_size} – {ID_mm:.2f} mm ID) "
                f"is smaller than the branch pipe ({selected_size_2} – {ID_mm_2:.2f} mm ID). "
                "Please choose a larger main pipe or a smaller branch pipe."
            )
            invalid_pipe_selection = True

        if not invalid_pipe_selection:
        
            pipe_index = material_to_pipe_index(selected_material)
        
            g_main = selected_gauge if "selected_gauge" in locals() else None
            g_branch = selected_gauge_2 if "selected_gauge_2" in locals() else None
        
            od_main, id_main = get_dimensions_for_row(material_df, selected_size, g_main)
            od_branch, id_branch = get_dimensions_for_row(material_df_2, selected_size_2, g_branch)
        
            res_main = system_pressure_check(
                refrigerant=refrigerant_eff if "refrigerant_eff" in locals() else refrigerant,
                design_temp_c=design_temp_c,
                mwp_temp_c=mwp_temp_c,
                circuit=circuit,
                pipe_index=pipe_index,
                od_mm=od_main,
                id_mm=id_main,
                gauge=g_main,
                copper_calc=copper_calc,
                r744_tc_pressure_bar_g=r744_tc_pressure_bar_g,
                dp_standard=dp_standard,
            )
        
            res_branch = system_pressure_check(
                refrigerant=refrigerant_eff if "refrigerant_eff" in locals() else refrigerant,
                design_temp_c=design_temp_c,
                mwp_temp_c=mwp_temp_c,
                circuit=circuit,
                pipe_index=pipe_index,
                od_mm=od_branch,
                id_mm=id_branch,
                gauge=g_branch,
                copper_calc=copper_calc,
                r744_tc_pressure_bar_g=r744_tc_pressure_bar_g,
                dp_standard=dp_standard,
            )
        
            combined = dict(res_main)
            combined["branch_a"] = res_main
            combined["branch_b"] = res_branch
            render_pressure_result(combined)
        
        with col3:
            
            evap_capacity_kw = st.number_input("Evaporator Capacity (kW)", min_value=0.03, max_value=20000.0, value=10.0, step=1.0)
            no_branch = st.number_input("No. of Branches", min_value=2, max_value=10, value=2, step=1)
            
            # --- Base ranges per refrigerant ---
            if refrigerant in ("R23", "R508B"):
                evap_min, evap_max, evap_default = -100.0, -20.0, -80.0
                cond_min, cond_max, cond_default = -100.0, 10.0, -30.0
                maxliq_min, maxliq_max, maxliq_default = -100.0, 10.0, -40.0
            elif refrigerant == "R744":
                evap_min, evap_max, evap_default = -50.0, 20.0, -10.0
                cond_min, cond_max, cond_default = -23.0, 30.0, 15.0
                maxliq_min, maxliq_max, maxliq_default = -50.0, 30.0, 10.0
            else:
                evap_min, evap_max, evap_default = -50.0, 30.0, -10.0
                cond_min, cond_max, cond_default = -23.0, 60.0, 43.0
                maxliq_min, maxliq_max, maxliq_default = -50.0, 60.0, 40.0
    
            # --- Init state (widget-backed) ---
            ss = st.session_state
    
            if "last_refrigerant" not in ss or ss.last_refrigerant != refrigerant:
                ss.cond_temp   = cond_default
                ss.maxliq_temp = maxliq_default
                ss.evap_temp   = evap_default
                ss.last_refrigerant = refrigerant
            
            ss.setdefault("cond_temp",   cond_default)
            ss.setdefault("maxliq_temp", maxliq_default)
            ss.setdefault("evap_temp",   evap_default)

            if "maxliq_temp" in ss and "cond_temp" in ss and "evap_temp" in ss:
                ss.cond_temp = min(max(ss.cond_temp, ss.maxliq_temp, ss.evap_temp), cond_max)
    
            if "cond_temp" in ss and "maxliq_temp" in ss and "evap_temp" in ss:
                ss.evap_temp = min(ss.maxliq_temp, ss.cond_temp, ss.evap_temp)
            
            # --- Callbacks implementing your downstream clamping logic ---
            def on_change_cond():
                # When cond changes: clamp maxliq down to cond, then evap down to maxliq
                ss.maxliq_temp = min(ss.maxliq_temp, ss.cond_temp)
                ss.evap_temp   = min(ss.evap_temp,   ss.maxliq_temp)
    
            def on_change_maxliq():
                # When maxliq changes: clamp maxliq down to cond, then evap down to maxliq
                ss.maxliq_temp = min(ss.maxliq_temp, ss.cond_temp)
                ss.evap_temp   = min(ss.evap_temp,   ss.maxliq_temp)
    
            def on_change_evap():
                # When evap changes: clamp evap down to maxliq
                ss.evap_temp   = min(ss.evap_temp,   ss.maxliq_temp)
    
        with col4:
            
            # --- Inputs with inclusive caps (≤), same order as your code ---
            condensing_temp = st.number_input(
                "Condensing Temperature (°C)",
                min_value=cond_min, max_value=cond_max,
                value=ss.cond_temp, step=1.0, key="cond_temp",
                on_change=on_change_cond,
                disabled=True,
            )
    
            maxliq_temp = st.number_input(
                "Liquid Temperature (°C)",
                min_value=maxliq_min, max_value=min(condensing_temp, maxliq_max),
                value=ss.maxliq_temp, step=1.0, key="maxliq_temp",
                on_change=on_change_maxliq,
                disabled=True,
            )

            evaporating_temp = st.number_input(
                "Evaporating Temperature (°C)",
                min_value=evap_min, max_value=min(maxliq_temp, evap_max),
                value=ss.evap_temp, step=1.0, key="evap_temp",
                on_change=on_change_evap,
                disabled=True,
            )

        if not invalid_pipe_selection:
        
            T_evap = evaporating_temp
            T_liq = maxliq_temp
            T_cond = condensing_temp
        
            props = RefrigerantProperties()
    
            h_in = props.get_properties(refrigerant, T_liq)["enthalpy_liquid2"]
    
            h_evap = props.get_properties(refrigerant, T_evap)["enthalpy_vapor"]
            
            delta_h = h_evap - h_in
    
            mass_flow_kg_s = evap_capacity_kw / delta_h if delta_h > 0 else 0.01
    
            if ID_mm is not None:
                ID_m = ID_mm / 1000.0
    
                area_m2 = math.pi * (ID_m / 2) ** 2
    
                density1 = RefrigerantProperties().get_properties(refrigerant, T_liq)["density_liquid2"]
    
                density2 = RefrigerantProperties().get_properties(refrigerant, T_cond)["density_liquid"]
    
                density = min(density1, density2)
    
                velocity_m_s = mass_flow_kg_s / (area_m2 * density)
    
            else:
                velocity_m_s = None
    
            mf_branch = mass_flow_kg_s / no_branch
    
            ID_m_2 = ID_mm_2 / 1000.0
    
            area_m2_2 = math.pi * (ID_m_2 / 2) ** 2
    
            vel_branch = mf_branch / (area_m2_2 * density)

            def _auto_select_copper_gauge(
                *,
                material_df,
                size_inch,
                gauges,
                design_pressure,
                refrigerant,
                design_temp_c,
                mwp_temp_c,
                circuit,
                pipe_index,
                copper_calc,
                dp_standard,
                r744_tc_pressure_bar_g,
            ):
                """
                Return:
                - highest gauge where MWP >= design pressure
                - else lowest gauge
                """
            
                passing = []
            
                for g in sorted(gauges):
                    try:
                        od_mm, id_mm = get_dimensions_for_row(material_df, size_inch, g)
            
                        res = system_pressure_check(
                            refrigerant=refrigerant,
                            design_temp_c=design_temp_c,
                            mwp_temp_c=mwp_temp_c,
                            circuit=circuit,
                            pipe_index=pipe_index,
                            od_mm=od_mm,
                            id_mm=id_mm,
                            gauge=g,
                            copper_calc=copper_calc,
                            r744_tc_pressure_bar_g=r744_tc_pressure_bar_g,
                            dp_standard=dp_standard,
                        )
            
                        mwp_val = governing_mwp(res["mwp_bar"])
            
                        if mwp_val >= design_pressure:
                            passing.append(g)
            
                    except Exception:
                        continue
            
                if passing:
                    return max(passing)
                else:
                    return min(gauges)
    
            # --- Automatic pipe size selection button ---
            if st.button("Auto-select"):
                target_velocity = 0.55  # m/s
            
                # --- helper: get ID_mm for a size using the strongest gauge row (if gauge exists) ---
                def _id_mm_strongest_for_size(df, size_inch: str):
                    rows = df[df["Nominal Size (inch)"].astype(str).str.strip() == str(size_inch)]
                    if rows.empty:
                        return None
            
                    if "Gauge" in rows.columns and rows["Gauge"].notna().any():
                        g_strong = max(rows["Gauge"].dropna().unique())
                        rows_g = rows[rows["Gauge"] == g_strong]
                        if not rows_g.empty:
                            return rows_g.iloc[0]["ID_mm"]
                    return rows.iloc[0]["ID_mm"]
            
                # Compute branch pipe size FIRST (so we can enforce main >= branch)
                best_branch = None
                best_branch_id_mm = None
            
                for size in pipe_sizes_2:
                    ID_branch_mm = _id_mm_strongest_for_size(material_df_2, size)
                    if ID_branch_mm is None:
                        continue
            
                    ID_branch_m = ID_branch_mm / 1000.0
                    area_branch = math.pi * (ID_branch_m / 2) ** 2
                    vel_branch_calc = mf_branch / (area_branch * density)
            
                    if vel_branch_calc <= target_velocity:
                        best_branch = size
                        best_branch_id_mm = ID_branch_mm
                        break
            
                # Compute main pipe size (must satisfy velocity AND be >= branch ID if branch was found)
                best_main = None
            
                for size in pipe_sizes:
                    ID_main_mm = _id_mm_strongest_for_size(material_df, size)
                    if ID_main_mm is None:
                        continue
            
                    if best_branch_id_mm is not None and ID_main_mm < best_branch_id_mm:
                        continue
            
                    ID_main_m = ID_main_mm / 1000.0
                    area_main = math.pi * (ID_main_m / 2) ** 2
                    vel_main = mass_flow_kg_s / (area_main * density)
            
                    if vel_main <= target_velocity:
                        best_main = size
                        break
            
                if best_main:
                    gauges_main = sorted(
                        material_df[
                            material_df["Nominal Size (inch)"].astype(str).str.strip() == best_main
                        ]["Gauge"].dropna().unique()
                    )
            
                    if gauges_main:
                        best_gauge_main = _auto_select_copper_gauge(
                            material_df=material_df,
                            size_inch=best_main,
                            gauges=gauges_main,
                            design_pressure=res_main["design_pressure_bar_g"],
                            refrigerant=refrigerant,
                            design_temp_c=design_temp_c,
                            mwp_temp_c=mwp_temp_c,
                            circuit=circuit,
                            pipe_index=pipe_index,
                            copper_calc=copper_calc,
                            dp_standard=dp_standard,
                            r744_tc_pressure_bar_g=r744_tc_pressure_bar_g,
                        )
            
                        st.session_state["_next_gauge_main"] = best_gauge_main
            
                if best_branch:
                    gauges_branch = sorted(
                        material_df_2[
                            material_df_2["Nominal Size (inch)"].astype(str).str.strip() == best_branch
                        ]["Gauge"].dropna().unique()
                    )
            
                    if gauges_branch:
                        best_gauge_branch = _auto_select_copper_gauge(
                            material_df=material_df_2,
                            size_inch=best_branch,
                            gauges=gauges_branch,
                            design_pressure=res_main["design_pressure_bar_g"],
                            refrigerant=refrigerant,
                            design_temp_c=design_temp_c,
                            mwp_temp_c=mwp_temp_c,
                            circuit=circuit,
                            pipe_index=pipe_index,
                            copper_calc=copper_calc,
                            dp_standard=dp_standard,
                            r744_tc_pressure_bar_g=r744_tc_pressure_bar_g,
                        )
            
                        st.session_state["_next_gauge_branch"] = best_gauge_branch
            
                # ✅ Store in temp state (safe names that are not bound to widgets)
                if best_main:
                    st.session_state["auto_selected_main"] = best_main
                    st.success(f"✅ Main pipe set to {best_main} (velocity ≤ {target_velocity} m/s)")
                else:
                    st.warning("⚠️ No main pipe found with velocity ≤ 0.55 m/s")
            
                if best_branch:
                    st.session_state["_next_selected_branch"] = best_branch
                    st.success(f"✅ Branch pipe set to {best_branch} (velocity ≤ {target_velocity} m/s)")
                else:
                    st.warning("⚠️ No branch pipe found with velocity ≤ 0.55 m/s")
            
                st.rerun()
    
            col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
            
            with col1:
    
                st.metric("Main Velocity", f"{velocity_m_s:.2f}m/s")
    
            with col2:
    
                st.metric("Branch Velocity", f"{vel_branch:.2f}m/s")
    
            with col3:
                st.empty()
            with col4:
                st.empty()
            with col5:
                st.empty()
            with col6:
                st.empty()
            with col7:
                st.empty()

        else:
            st.warning("Pipe size validation failed — calculations skipped.")

    if mode == "Wet Suction":

        def find_pipe_diameter(PD, Vis, Den, MassF, choice, surface_roughness):
        
            import math
        
            VEA = MassF / Den
            RenoEA = Vis
        
            # Hi2 / Lo2 bounds
            Hi2 = 0.3048
            Lo2 = 0.0003048
        
            for _ in range(200):
                PipeDia = (Hi2 + Lo2) / 2.0
                PipeArea = math.pi * (PipeDia * 0.5) ** 2
        
                Vel = VEA / PipeArea
                VP = 0.5 * Den * Vel**2
                Reno = Den * PipeDia * Vel / RenoEA
        
                # Friction factor
                if Reno < 2000:
                    FF = 64.0 / Reno
                else:
                    # Colebrook bisection exactly like VB
                    Hi = 0.1
                    Lo = 0.00001
                    FF = None
                    for _ in range(60):
                        FF_try = (Hi + Lo) / 2.0
                        LHS = 1.0 / math.sqrt(FF_try)
                        RHS = -2 * (math.log10((surface_roughness / (PipeDia * 3.7)) +
                                               (2.51 / (Reno * math.sqrt(FF_try)))))
                        if LHS > RHS:
                            Lo = FF_try
                        else:
                            Hi = FF_try
                    FF = (Hi + Lo) / 2.0
        
                # Pipe pressure drop
                PPD = FF * 30.48 / PipeDia * VP
        
                if PPD > PD:
                    Lo2 = PipeDia
                else:
                    Hi2 = PipeDia
        
                if abs(1 - (PPD / PD)) < 0.00001:
                    break
        
            return PipeDia
        
        # Load pipe data
        pipe_data = pd.read_csv("data/pipe_pressure_ratings_full.csv")
    
        # --- helpers ---
        def _nps_inch_to_mm(nps_str: str) -> float:
            # e.g. "1-1/8", '1"', "3/8"
            s = str(nps_str).replace('"', '').strip()
            if not s:
                return float('nan')
            parts = s.split('-')
            tot_in = 0.0
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                if '/' in p:
                    num, den = p.split('/')
                    tot_in += float(num) / float(den)
                else:
                    tot_in += float(p)
            return tot_in * 25.4  # mm
    
        ss = st.session_state
    
        # 1) Pipe material
        col1, col2, col3, col4 = st.columns(4)
        with col2:
            if refrigerant == "R717":
                excluded_materials = ["Copper ASTM", " Copper EN12735", "K65 Copper", "Reflok Aluminium"]
                pipe_materials = sorted(m for m in pipe_data["Material"].dropna().unique()
                                        if m not in excluded_materials)
            else:
                pipe_materials = sorted(pipe_data["Material"].dropna().unique())
    
            selected_material = st.selectbox("Pipe Material", pipe_materials, key="material")
        
        # detect material change
        material_changed = ss.get("last_material") is not None and ss.last_material != selected_material
        ss.last_material = selected_material
    
        # 2) Sizes for selected material (de-duped)
        material_df = pipe_data[pipe_data["Material"] == selected_material].copy()
    
        sizes_df = (
            material_df[["Nominal Size (inch)", "Nominal Size (mm)"]]
            .dropna(subset=["Nominal Size (inch)"])
            .assign(**{
                "Nominal Size (inch)": lambda d: d["Nominal Size (inch)"].astype(str).str.strip(),
            })
            .drop_duplicates(subset=["Nominal Size (inch)"], keep="first")
        )
    
        # make sure we have a numeric mm per nominal (fallback: parse the inch string)
        sizes_df["mm_num"] = pd.to_numeric(sizes_df.get("Nominal Size (mm)"), errors="coerce")
        sizes_df.loc[sizes_df["mm_num"].isna(), "mm_num"] = sizes_df.loc[sizes_df["mm_num"].isna(), "Nominal Size (inch)"].apply(_nps_inch_to_mm)
    
        pipe_sizes = sizes_df["Nominal Size (inch)"].tolist()
        mm_map = dict(zip(sizes_df["Nominal Size (inch)"], sizes_df["mm_num"]))

        def _pipe_row_for_size(size_inch: str, gauge_override=None):
            rows = material_df[
                material_df["Nominal Size (inch)"].astype(str).str.strip() == str(size_inch)
            ]
            if rows.empty:
                return None
        
            if "Gauge" in rows.columns and rows["Gauge"].notna().any():
                # 🔹 override takes priority
                if gauge_override is not None:
                    rows_g = rows[rows["Gauge"] == gauge_override]
                    if not rows_g.empty:
                        return rows_g.iloc[0]
        
                # 🔹 otherwise fall back to UI-selected gauge
                if "gauge" in st.session_state:
                    g = st.session_state["gauge"]
                    rows_g = rows[rows["Gauge"] == g]
                    if not rows_g.empty:
                        return rows_g.iloc[0]
        
                return rows.iloc[0]
        
            return rows.iloc[0]
    
        # choose default index
        def _closest_index(target_mm: float) -> int:
            mm_list = [mm_map[s] for s in pipe_sizes]
            return min(range(len(mm_list)), key=lambda i: abs(mm_list[i] - target_mm)) if mm_list else 0
    
        # --- consume any deferred selection from Auto-select button ---
        if "_next_selected_size" in st.session_state:
            new_val = st.session_state.pop("_next_selected_size")
            if new_val in pipe_sizes:
                st.session_state["selected_size"] = new_val
                st.session_state["_selected_size_just_set"] = True
        
        default_index = 0
        if material_changed and "prev_pipe_mm" in ss:
            default_index = _closest_index(ss.prev_pipe_mm)
        elif selected_material == " Copper EN12735" and ("7/8" in pipe_sizes or '7/8"' in pipe_sizes):
            # first load or no previous selection → prefer 1-1/8" for  Copper EN12735
            want = "7/8" if "7/8" in pipe_sizes else '7/8"'
            default_index = pipe_sizes.index(want)
        elif "selected_size" in ss and ss.selected_size in pipe_sizes:
            # if Streamlit kept the selection, use it
            default_index = pipe_sizes.index(ss.selected_size)
        
        with col1:
            selected_size = st.selectbox(
                "Nominal Pipe Size (inch)",
                pipe_sizes,
                index=default_index,
                key="selected_size",
            )

        # --- clean up one-shot flag ---
        if st.session_state.get("_selected_size_just_set"):
            del st.session_state["_selected_size_just_set"]
    
        # remember the selected size in mm for next material change
        ss.prev_pipe_mm = float(mm_map.get(selected_size, float("nan")))

        # --- consume any deferred gauge from Auto-select (Wet Suction) ---
        if "_next_gauge" in st.session_state:
            g = st.session_state.pop("_next_gauge")
        
            rows = material_df[
                material_df["Nominal Size (inch)"].astype(str).str.strip() == str(selected_size)
            ]
        
            if "Gauge" in rows.columns and rows["Gauge"].notna().any():
                valid_gauges = set(rows["Gauge"].dropna().unique())
                if g in valid_gauges:
                    st.session_state["gauge"] = g
    
        # 3) Gauge (if applicable)
        gauge_options = material_df[material_df["Nominal Size (inch)"].astype(str).str.strip() == selected_size]
        if "Gauge" in gauge_options.columns and gauge_options["Gauge"].notna().any():
            gauges = sorted(gauge_options["Gauge"].dropna().unique())
            with col2:
                selected_gauge = st.selectbox("Copper Gauge", gauges, key="gauge")
            selected_pipe_row = gauge_options[gauge_options["Gauge"] == selected_gauge].iloc[0]
        else:
            selected_pipe_row = gauge_options.iloc[0]

        pipe_index = material_to_pipe_index(selected_material)
        
        # you already have selected_gauge sometimes; otherwise None
        gauge = st.session_state.get("gauge")  # or whatever that mode uses
        od_mm, id_mm = get_dimensions_for_row(material_df, selected_size, gauge)
        
        result = system_pressure_check(
            refrigerant=refrigerant,
            design_temp_c=design_temp_c,
            mwp_temp_c=mwp_temp_c,
            circuit=circuit,
            pipe_index=pipe_index,
            od_mm=od_mm,
            id_mm=id_mm,
            gauge=gauge,
            copper_calc=copper_calc,
            r744_tc_pressure_bar_g=r744_tc_pressure_bar_g,
            dp_standard=dp_standard,
        )
        
        render_pressure_result(result)
    
        # Pipe parameters
        pipe_size_inch = selected_pipe_row["Nominal Size (inch)"]
        ID_mm = selected_pipe_row["ID_mm"]
    
        with col1:
            
            evap_capacity_kw = st.number_input("Evaporator Capacity (kW)", min_value=0.03, max_value=20000.0, value=10.0, step=1.0)
            
            # --- Base ranges per refrigerant ---
            if refrigerant in ("R23", "R508B"):
                evap_min, evap_max, evap_default = -100.0, -20.0, -80.0
            elif refrigerant == "R744":
                evap_min, evap_max, evap_default = -50.0, 20.0, -10.0
            else:
                evap_min, evap_max, evap_default = -50.0, 30.0, -10.0
    
            # --- Init state (widget-backed) ---
            ss = st.session_state
    
            if "last_refrigerant" not in ss or ss.last_refrigerant != refrigerant:
                ss.evap_temp   = evap_default
                ss.last_refrigerant = refrigerant
        
            ss.setdefault("evap_temp",   evap_default)

            liq_oq = st.number_input("Liquid Overfeed Quantity (%)", min_value=0.0, max_value=2000.0, value=100.0, step=25.0)

        with col2:
            
            evaporating_temp = st.number_input(
                "Evaporating Temperature (°C)",
                min_value=evap_min, max_value=evap_max,
                value=ss.evap_temp, step=1.0, key="evap_temp",
            )
    
        with col2:
            WetSucPenaltyFactor = st.number_input("Max Liquid Hold-Up Penalty Bias", min_value=1.0, max_value=3.0, value=1.5, step=0.1)
            max_penalty = st.number_input("Max Penalty (K)", min_value=0.0, max_value=6.0, value=1.0, step=0.1)
    
        with col3:
            L = st.number_input("Pipe Length (m)", min_value=0.1, max_value=300.0, value=10.0, step=1.0)
            LRB = st.number_input("Long Radius Bends", min_value=0, max_value=50, value=0, step=1)
            SRB = st.number_input("Short Radius Bends", min_value=0, max_value=50, value=0, step=1)
            _45 = st.number_input("45° Bends", min_value=0, max_value=50, value=0, step=1)
            MAC = st.number_input("Machine Bends", min_value=0, max_value=50, value=0, step=1)
    
        with col4:
            ptrap = st.number_input("P Traps", min_value=0, max_value=10, value=0, step=1)
            ubend = st.number_input("U Bends", min_value=0, max_value=10, value=0, step=1)
            ball = st.number_input("Ball Valves", min_value=0, max_value=20, value=0, step=1, key="ball")
            globe = st.number_input("Globe Valves", min_value=0, max_value=20, value=0, step=1, key="globe")
            PLF = st.number_input("Pressure Loss Factors", min_value=0.0, max_value=20.0, value=0.0, step=0.1)
        
        from utils.refrigerant_properties import RefrigerantProperties
        from utils.refrigerant_densities import RefrigerantDensities
        from utils.refrigerant_viscosities import RefrigerantViscosities
    
        D_int = ID_mm / 1000
        A_total = math.pi * (D_int / 2)**2
        
        T_evap = evaporating_temp
    
        props = RefrigerantProperties()

        h_in = props.get_properties(refrigerant, T_evap)["enthalpy_liquid"]
        h_out = props.get_properties(refrigerant, T_evap)["enthalpy_vapor"]
        deltah = h_out - h_in

        base_massflow = evap_capacity_kw / deltah
        BMR_massflow = 293.07107017224996 / deltah
        overfeed_ratio = 1 + liq_oq / 100
        m_g = base_massflow                # vapour mass
        m_l = base_massflow * (overfeed_ratio - 1)   # liquid mass
        m_gplusl = m_g + m_l

        d_liq1 = props.get_properties(refrigerant, T_evap)["density_liquid"]
        d_vap1 = props.get_properties(refrigerant, T_evap)["density_vapor"]

        v_liq1 = props.get_properties(refrigerant, T_evap)["viscosity_liquid3"] / 1000000
        v_vap1 = RefrigerantViscosities().get_viscosity(refrigerant, T_evap + 273.15, 0) / 1000000

        d_liq2 = props.get_properties(refrigerant, T_evap - max_penalty)["density_liquid"]
        d_vap2 = props.get_properties(refrigerant, T_evap - max_penalty)["density_vapor"]

        v_liq2 = props.get_properties(refrigerant, T_evap - max_penalty)["viscosity_liquid3"] / 1000000
        v_vap2 = RefrigerantViscosities().get_viscosity(refrigerant, T_evap + 273.15 - max_penalty, 0) / 1000000

        d_liq = (d_liq1 + d_liq2) / 2
        d_vap = (d_vap1 + d_vap2) / 2
        v_liq = (v_liq1 + v_liq2) / 2
        v_vap = (v_vap1 + v_vap2) / 2

        Q_g = m_g / d_vap
        Q_l = m_l / d_liq if liq_oq > 0 else 0

        if liq_oq <= 0 or overfeed_ratio <= 1:
            # Dry suction – behave like your old code's dry branch
            liquid_ratio = 0.0
            A_gas = A_total
            gas_velocity = Q_g / A_gas if A_gas > 0 else 0.0
            D_h = D_int
        else:
            # --- VB6 Wet Suction Logic (faithful) ---
            
            # Determine surface roughness exactly like VB
            if selected_material in ["Steel SCH40", "Steel SCH80"]:
                surface_roughness = 0.00004572
            else:
                surface_roughness = 0.000001524
            
            # Mass flow terms for VB
            D = BMR_massflow   # VB scaling
            
            # VB-equivalent diameters for gas and liquid
            A_diam = find_pipe_diameter(689.476, v_vap, d_vap, D, 1, surface_roughness)
            B_diam = find_pipe_diameter(689.476, v_liq, d_liq, D * (overfeed_ratio - 1), 2, surface_roughness)
            
            # Convert diameters to areas (VB logic)
            A_area = math.pi * (A_diam / 2)**2
            B_area = math.pi * (B_diam / 2)**2
            C_area = A_area + B_area
            
            # VB liquid ratio
            liquid_ratio = B_area / C_area if C_area > 0 else 0
            
            # --- Geometry (VB strata model) ---
            Radius = D_int / 2
            TotalArea = math.pi * Radius**2
            LiqArea = TotalArea * liquid_ratio
            SucArea = TotalArea - LiqArea  # gas area
            
            # Solve Angle exactly like VB
            DegCon = 57.2957795130824
            Angle = (LiqArea / (Radius**2 * 0.5)) * DegCon
            
            # Chord / Arc / Perimeter (VB)
            Chord = math.sin((Angle / DegCon) / 2) * Radius * 2
            Arc = ((360 - Angle) * math.pi) / (360 / (Radius * 2))
            Perimeter = Chord + Arc
            
            # Hydraulic diameter
            if Perimeter > 0:
                D_h = 4 * TotalArea / Perimeter
            else:
                D_h = D_int  # dry fallback
            
            # Gas velocity using VB suction area
            A_gas = SucArea
            gas_velocity = Q_g / A_gas if A_gas > 0 else 0
    
        if v_vap > 0:
            Re = d_vap * gas_velocity * D_h / v_vap
        else:
            Re = 0
    
        if selected_material in ["Steel SCH40", "Steel SCH80"]:
            eps = 0.00004572
        else:
            eps = 0.000001524
    
        if Re <= 0:
            f = 0
        elif Re < 2000:
            f = 64 / Re
        else:
            rel = eps / D_h
            f = 0.02
            for _ in range(60):
                rhs = -2.0 * math.log10((rel / 3.7) + (2.51 / (Re * math.sqrt(f))))
                f_new = 1.0 / (rhs * rhs)
                if abs(f_new - f) / f < 1e-5:
                    f = f_new
                    break
                f = f_new
    
        dyn = 0.5 * d_vap * gas_velocity**2 / 1000
    
        dp_pipe = f * (L / D_h) * dyn
        dp_plf = dyn * PLF
    
        K_SRB = float(selected_pipe_row["SRB"])
        K_LRB = float(selected_pipe_row["LRB"])
        K_BALL = float(selected_pipe_row["BALL"])
        K_GLOBE = float(selected_pipe_row["GLOBE"])
    
        B_SRB = SRB + 0.5 * _45 + 2 * ubend + 3 * ptrap
        B_LRB = LRB + MAC
    
        dp_fittings = dyn * (K_SRB * B_SRB + K_LRB * B_LRB)
        dp_valves = dyn * (K_BALL * ball + K_GLOBE * globe)
        
        if refrigerant == "R404A": C_ref = 0.77
        elif refrigerant == "R502": C_ref = 0.76
        elif refrigerant == "R717": C_ref = 0.64
        elif refrigerant == "R134a": C_ref = 0.71
        else: C_ref = 0.73
    
        WetSucFactor = 1 + (WetSucPenaltyFactor - 1) * (liquid_ratio / C_ref)
        if WetSucFactor < 1:
            WetSucFactor = 1
    
        dp_pipe_ws = dp_pipe * WetSucFactor
        dp_fittings_ws = dp_fittings * WetSucFactor
        dp_valves_ws = dp_valves * WetSucFactor
        dp_plf_ws = dp_plf * WetSucFactor
    
        dp_total_ws = dp_pipe_ws + dp_fittings_ws + dp_valves_ws + dp_plf_ws
        
        converter = PressureTemperatureConverter()
        evappres = converter.temp_to_pressure(refrigerant, T_evap)

        postcirc = evappres - (dp_total_ws / 100)
        
        postcirctemp = converter.pressure_to_temp(refrigerant, postcirc)

        dt = T_evap - postcirctemp

        def get_wet_suction_dt_for_size(size_inch: str, gauge_override=None) -> float:
            """Compute ΔT for a given Wet Suction pipe size using identical logic to the main block."""
            try:
                pipe_row = _pipe_row_for_size(size_inch, gauge_override)
                if pipe_row is None:
                    return float("nan")
        
                # Geometry
                ID_mm_local = float(pipe_row["ID_mm"])
                D_int = ID_mm_local / 1000
                A_total = math.pi * (D_int / 2) ** 2
        
                # --- identical property setup ---
                T_evap_local = T_evap
                props = RefrigerantProperties()
                h_in = props.get_properties(refrigerant, T_evap_local)["enthalpy_liquid"]
                h_out = props.get_properties(refrigerant, T_evap_local)["enthalpy_vapor"]
                deltah = h_out - h_in
                base_massflow = evap_capacity_kw / deltah
                BMR_massflow = 293.07107017224996 / deltah
                overfeed_ratio = 1 + liq_oq / 100
                m_g = base_massflow
                m_l = base_massflow * (overfeed_ratio - 1)
                m_gplusl = m_g + m_l
        
                d_liq1 = props.get_properties(refrigerant, T_evap_local)["density_liquid"]
                d_vap1 = props.get_properties(refrigerant, T_evap_local)["density_vapor"]
                v_liq1 = props.get_properties(refrigerant, T_evap_local)["viscosity_liquid3"] / 1_000_000
                v_vap1 = RefrigerantViscosities().get_viscosity(refrigerant, T_evap_local + 273.15, 0) / 1_000_000
                d_liq2 = props.get_properties(refrigerant, T_evap_local - max_penalty)["density_liquid"]
                d_vap2 = props.get_properties(refrigerant, T_evap_local - max_penalty)["density_vapor"]
                v_liq2 = props.get_properties(refrigerant, T_evap_local - max_penalty)["viscosity_liquid3"] / 1_000_000
                v_vap2 = RefrigerantViscosities().get_viscosity(refrigerant, T_evap_local + 273.15 - max_penalty, 0) / 1_000_000
        
                d_liq = (d_liq1 + d_liq2) / 2
                d_vap = (d_vap1 + d_vap2) / 2
                v_liq = (v_liq1 + v_liq2) / 2
                v_vap = (v_vap1 + v_vap2) / 2
        
                Q_g = m_g / d_vap
                Q_l = m_l / d_liq if liq_oq > 0 else 0
        
                # --- identical wet suction geometry logic ---
                if liq_oq <= 0 or overfeed_ratio <= 1:
                    liquid_ratio = 0.0
                    A_gas = A_total
                    gas_velocity = Q_g / A_gas if A_gas > 0 else 0.0
                    D_h = D_int
                else:
                    # surface roughness
                    surface_roughness = 0.00004572 if selected_material in ["Steel SCH40", "Steel SCH80"] else 0.000001524
        
                    D = BMR_massflow
                    A_diam = find_pipe_diameter(689.476, v_vap, d_vap, D, 1, surface_roughness)
                    B_diam = find_pipe_diameter(689.476, v_liq, d_liq, D * (overfeed_ratio - 1), 2, surface_roughness)
                    A_area = math.pi * (A_diam / 2) ** 2
                    B_area = math.pi * (B_diam / 2) ** 2
                    C_area = A_area + B_area
                    liquid_ratio = B_area / C_area if C_area > 0 else 0
                    Radius = D_int / 2
                    TotalArea = math.pi * Radius**2
                    LiqArea = TotalArea * liquid_ratio
                    SucArea = TotalArea - LiqArea
                    DegCon = 57.2957795130824
                    Angle = (LiqArea / (Radius**2 * 0.5)) * DegCon
                    Chord = math.sin((Angle / DegCon) / 2) * Radius * 2
                    Arc = ((360 - Angle) * math.pi) / (360 / (Radius * 2))
                    Perimeter = Chord + Arc
                    D_h = 4 * TotalArea / Perimeter if Perimeter > 0 else D_int
                    A_gas = SucArea
                    gas_velocity = Q_g / A_gas if A_gas > 0 else 0
        
                # --- identical friction and dp chain ---
                Re = d_vap * gas_velocity * D_h / v_vap if v_vap > 0 else 0
                eps = 0.00004572 if selected_material in ["Steel SCH40", "Steel SCH80"] else 0.000001524
        
                if Re <= 0:
                    f = 0
                elif Re < 2000:
                    f = 64 / Re
                else:
                    rel = eps / D_h
                    f = 0.02
                    for _ in range(60):
                        rhs = -2.0 * math.log10((rel / 3.7) + (2.51 / (Re * math.sqrt(f))))
                        f_new = 1.0 / (rhs * rhs)
                        if abs(f_new - f) / f < 1e-5:
                            f = f_new
                            break
                        f = f_new
        
                dyn = 0.5 * d_vap * gas_velocity**2 / 1000
                dp_pipe = f * (L / D_h) * dyn
                dp_plf = dyn * PLF
                K_SRB = float(pipe_row["SRB"])
                K_LRB = float(pipe_row["LRB"])
                K_BALL = float(pipe_row["BALL"])
                K_GLOBE = float(pipe_row["GLOBE"])
                B_SRB = SRB + 0.5 * _45 + 2 * ubend + 3 * ptrap
                B_LRB = LRB + MAC
                dp_fittings = dyn * (K_SRB * B_SRB + K_LRB * B_LRB)
                dp_valves = dyn * (K_BALL * ball + K_GLOBE * globe)
        
                # --- identical wet suction correction ---
                if refrigerant == "R404A": C_ref = 0.77
                elif refrigerant == "R502": C_ref = 0.76
                elif refrigerant == "R717": C_ref = 0.64
                elif refrigerant == "R134a": C_ref = 0.71
                else: C_ref = 0.73
        
                WetSucFactor = 1 + (WetSucPenaltyFactor - 1) * (liquid_ratio / C_ref)
                if WetSucFactor < 1:
                    WetSucFactor = 1
        
                dp_total_ws = (dp_pipe + dp_fittings + dp_valves + dp_plf) * WetSucFactor
        
                conv = PressureTemperatureConverter()
                evappres = conv.temp_to_pressure(refrigerant, T_evap_local)
                postcirc = evappres - (dp_total_ws / 100)
                postcirctemp = conv.pressure_to_temp(refrigerant, postcirc)
                dt_local = T_evap_local - postcirctemp
        
                return float(dt_local)
        
            except Exception:
                return float("nan")

        def _auto_select_copper_gauge(
            *,
            material_df,
            size_inch,
            gauges,
            design_pressure,
            refrigerant,
            design_temp_c,
            mwp_temp_c,
            circuit,
            pipe_index,
            copper_calc,
            dp_standard,
            r744_tc_pressure_bar_g,
        ):
            """
            Return:
            - highest gauge where MWP >= design pressure
            - else lowest gauge
            """
        
            passing = []
        
            for g in sorted(gauges):
                try:
                    od_mm, id_mm = get_dimensions_for_row(material_df, size_inch, g)
        
                    res = system_pressure_check(
                        refrigerant=refrigerant,
                        design_temp_c=design_temp_c,
                        mwp_temp_c=mwp_temp_c,
                        circuit=circuit,
                        pipe_index=pipe_index,
                        od_mm=od_mm,
                        id_mm=id_mm,
                        gauge=g,
                        copper_calc=copper_calc,
                        r744_tc_pressure_bar_g=r744_tc_pressure_bar_g,
                        dp_standard=dp_standard,
                    )
        
                    mwp_val = governing_mwp(res["mwp_bar"])
        
                    if mwp_val >= design_pressure:
                        passing.append(g)
        
                except Exception:
                    continue
        
            if passing:
                return max(passing)
            else:
                return min(gauges)

        if st.button("Auto-select"):
            results, errors = [], []
        
            for ps in pipe_sizes:
                rows = material_df[
                    material_df["Nominal Size (inch)"].astype(str).str.strip() == ps
                ]
        
                # 🔹 force strongest gauge for this size during evaluation
                gauge_override = None
                if "Gauge" in rows.columns and rows["Gauge"].notna().any():
                    gauge_override = max(rows["Gauge"].dropna().unique())
        
                dt_i = get_wet_suction_dt_for_size(ps, gauge_override=gauge_override)
        
                if math.isfinite(dt_i):
                    results.append({"size": ps, "dt": dt_i})
                else:
                    errors.append((ps, "failed or non-numeric ΔT"))
        
            if not results:
                with st.expander("⚠️ Debug Details", expanded=True):
                    for ps, msg in errors:
                        st.write(f"❌ {ps}: {msg}")
                st.error("No valid results for any pipe size — check CSV or inputs.")
            else:
                valid = [r for r in results if r["dt"] <= max_penalty]
        
                if valid:
                    best = min(valid, key=lambda x: mm_map[x["size"]])  # smallest ID that passes
                    st.session_state["_next_selected_size"] = best["size"]
        
                    # 🔹 Auto-select gauge for Copper EN12735
                    if selected_material.strip() == "Copper EN12735":
                        rows = material_df[
                            material_df["Nominal Size (inch)"].astype(str).str.strip() == best["size"]
                        ]
        
                        if "Gauge" in rows.columns and rows["Gauge"].notna().any():
                            gauges = sorted(rows["Gauge"].dropna().unique())
        
                            best_gauge = _auto_select_copper_gauge(
                                material_df=material_df,
                                size_inch=best["size"],
                                gauges=gauges,
                                design_pressure=result["design_pressure_bar_g"],
                                refrigerant=refrigerant,
                                design_temp_c=design_temp_c,
                                mwp_temp_c=mwp_temp_c,
                                circuit=circuit,
                                pipe_index=pipe_index,
                                copper_calc=copper_calc,
                                dp_standard=dp_standard,
                                r744_tc_pressure_bar_g=r744_tc_pressure_bar_g,
                            )
        
                            st.session_state["_next_gauge"] = best_gauge
        
                    st.success(
                        f"✅ Auto-selected pipe: **{best['size']}**  \n"
                        f"ΔT = {best['dt']:.3f} K ≤ {max_penalty:.3f} K"
                    )
                    st.rerun()
                else:
                    best_dt = min(r["dt"] for r in results if math.isfinite(r["dt"]))
                    st.error(
                        f"❌ No pipe meets ΔT ≤ {max_penalty:.3f} K.  \n"
                        f"Best achievable ΔT = {best_dt:.3f} K."
                    )
    
        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    
        with col1:
            st.metric("Refrigerant Velocity", f"{gas_velocity:.2f}m/s")
    
        with col2:
            st.metric("Vapour Density", f"{d_vap:.2f}kg/m³")
    
        with col3:
            st.metric("Vapour Volumetric Flow", f"{Q_g:.5f}m³/s")
    
        with col4:
            st.metric("Pressure Drop", f"{dp_total_ws:.2f}kPa")

        with col5:
            st.metric("Temp Penalty", f"{dt:.2f}K")

        with col6:
            st.metric("Saturated Temperature", f"{postcirctemp:.2f}°C")

        with col7:
            st.metric("Evaporating Pressure", f"{evappres:.2f}bar(a)")
            
        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    
        with col1:
            st.metric("Mass Flow Rate", f"{m_gplusl:.5f}kg/s")
    
        with col2:
            st.metric("Liquid Density", f"{d_liq:.1f}kg/m³")
    
        with col3:
            st.metric("Liquid Volumetric Flow", f"{Q_l:.5f}m³/s")
    
        with col4:
            st.metric("Pipe PD", f"{dp_pipe_ws:.2f}kPa")

        with col5:
            st.metric("Fittings PD", f"{dp_fittings_ws:.2f}kPa")

        with col6:
            st.metric("Valves PD", f"{dp_valves_ws:.2f}kPa")

        with col7:
            st.metric("Velocity Pressure PD", f"{dp_plf_ws:.2f}kPa")

    if mode == "Pumped Liquid":

        # Load pipe data
        pipe_data = pd.read_csv("data/pipe_pressure_ratings_full.csv")
    
        # --- helpers ---
        def _nps_inch_to_mm(nps_str: str) -> float:
            # e.g. "1-1/8", '1"', "3/8"
            s = str(nps_str).replace('"', '').strip()
            if not s:
                return float('nan')
            parts = s.split('-')
            tot_in = 0.0
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                if '/' in p:
                    num, den = p.split('/')
                    tot_in += float(num) / float(den)
                else:
                    tot_in += float(p)
            return tot_in * 25.4  # mm
    
        ss = st.session_state
    
        # 1) Pipe material
        col1, col2, col3, col4 = st.columns(4)
        with col2:
            if refrigerant == "R717":
                excluded_materials = ["Copper ASTM", " Copper EN12735", "K65 Copper", "Reflok Aluminium"]
                pipe_materials = sorted(m for m in pipe_data["Material"].dropna().unique()
                                        if m not in excluded_materials)
            else:
                pipe_materials = sorted(pipe_data["Material"].dropna().unique())
    
            selected_material = st.selectbox("Pipe Material", pipe_materials, key="material")
        
        # detect material change
        material_changed = ss.get("last_material") is not None and ss.last_material != selected_material
        ss.last_material = selected_material
    
        # 2) Sizes for selected material (de-duped)
        material_df = pipe_data[pipe_data["Material"] == selected_material].copy()
    
        sizes_df = (
            material_df[["Nominal Size (inch)", "Nominal Size (mm)"]]
            .dropna(subset=["Nominal Size (inch)"])
            .assign(**{
                "Nominal Size (inch)": lambda d: d["Nominal Size (inch)"].astype(str).str.strip(),
            })
            .drop_duplicates(subset=["Nominal Size (inch)"], keep="first")
        )
    
        # make sure we have a numeric mm per nominal (fallback: parse the inch string)
        sizes_df["mm_num"] = pd.to_numeric(sizes_df.get("Nominal Size (mm)"), errors="coerce")
        sizes_df.loc[sizes_df["mm_num"].isna(), "mm_num"] = sizes_df.loc[sizes_df["mm_num"].isna(), "Nominal Size (inch)"].apply(_nps_inch_to_mm)
    
        pipe_sizes = sizes_df["Nominal Size (inch)"].tolist()
        mm_map = dict(zip(sizes_df["Nominal Size (inch)"], sizes_df["mm_num"]))

        # -------- helper: get CSV row for a given pipe size --------
        def _pipe_row_for_size(size_inch: str, gauge_override=None):
            rows = material_df[
                material_df["Nominal Size (inch)"].astype(str).str.strip() == str(size_inch)
            ]
            if rows.empty:
                return None
        
            if "Gauge" in rows.columns and rows["Gauge"].notna().any():
                # 🔹 override takes priority
                if gauge_override is not None:
                    rows_g = rows[rows["Gauge"] == gauge_override]
                    if not rows_g.empty:
                        return rows_g.iloc[0]
        
                # 🔹 otherwise fall back to UI-selected gauge
                if "gauge" in st.session_state:
                    g = st.session_state["gauge"]
                    rows_g = rows[rows["Gauge"] == g]
                    if not rows_g.empty:
                        return rows_g.iloc[0]
        
                return rows.iloc[0]
        
            return rows.iloc[0]

        # -------- helper: recompute dp_total_kPa for any pipe size --------
        def get_pumped_dp_for_size(size_inch: str, gauge_override=None) -> float:
            """
            Returns dp_total_kPa for a given size using the EXACT SAME PHYSICS AND EQUATIONS
            as the Pumped Liquid main calculation block.
            """
        
            try:
                row = _pipe_row_for_size(size_inch, gauge_override)
                if row is None:
                    return float("nan")
        
                # Extract diameter
                ID_mm_local = float(row["ID_mm"])
                ID_m_local = ID_mm_local / 1000.0
                A_local = math.pi * (ID_m_local / 2)**2
        
                # SAME liquid density & viscosity
                rho = RefrigerantProperties().get_properties(refrigerant, T_evap)["density_liquid2"]
                visc = RefrigerantProperties().get_properties(refrigerant, T_evap)["viscosity_liquid"]
        
                # SAME mass flow
                m_dot = (
                    evap_capacity_kw / deltah * (1 + liq_oq / 100)
                    if deltah > 0 else 0.01
                )
        
                # Velocity
                v = m_dot / (A_local * rho)
        
                # Reynolds
                Re = (rho * v * ID_m_local) / (visc / 1_000_000)
        
                # Roughness
                eps = 0.00004572 if selected_material in ["Steel SCH40", "Steel SCH80"] else 0.000001524
        
                # Friction factor identical method
                if Re < 2000:
                    f_local = 64.0 / Re
                else:
                    flo, fhi = 1e-5, 0.1
                    tol, max_iter = 1e-5, 60
        
                    def bal(ff):
                        s = math.sqrt(ff)
                        lhs = 1.0 / s
                        rhs = -2.0 * math.log10((eps / (3.7 * ID_m_local)) + 2.51 / (Re * s))
                        return lhs, rhs
        
                    f_local = 0.5 * (flo + fhi)
                    for _ in range(max_iter):
                        f_try = 0.5 * (flo + fhi)
                        lhs, rhs = bal(f_try)
                        if abs(1 - lhs / rhs) < tol:
                            f_local = f_try
                            break
        
                        if (lhs - rhs) > 0:
                            flo = f_try
                        else:
                            fhi = f_try
        
                # Dynamic pressure (kPa)
                q_kPa_local = 0.5 * rho * v**2 / 1000.0
        
                # K-factors
                try:
                    K_SRB   = float(row["SRB"])
                    K_LRB   = float(row["LRB"])
                    K_BALL  = float(row["BALL"])
                    K_GLOBE = float(row["GLOBE"])
                except:
                    return float("nan")
        
                # Bend factors identical to main block
                B_SRB = SRB + 0.5 * _45 + 2*ubend + 3*ptrap
                B_LRB = LRB + MAC
        
                # Pressure drops
                dp_pipe_local = f_local * (L / ID_m_local) * q_kPa_local
                dp_plf_local  = q_kPa_local * PLF
                dp_fit_local  = q_kPa_local * (K_SRB * B_SRB + K_LRB * B_LRB)
                dp_valv_local = q_kPa_local * (K_BALL * ball + K_GLOBE * globe)
        
                dp_total_local = dp_pipe_local + dp_fit_local + dp_valv_local + dp_plf_local
        
                return float(dp_total_local)
        
            except Exception:
                return float("nan")
        
        # choose default index
        def _closest_index(target_mm: float) -> int:
            mm_list = [mm_map[s] for s in pipe_sizes]
            return min(range(len(mm_list)), key=lambda i: abs(mm_list[i] - target_mm)) if mm_list else 0
        
        default_index = 0
        if material_changed and "prev_pipe_mm" in ss:
            default_index = _closest_index(ss.prev_pipe_mm)
        elif selected_material == " Copper EN12735" and ("1/2" in pipe_sizes or '1/2"' in pipe_sizes):
            # first load or no previous selection → prefer 1-1/8" for  Copper EN12735
            want = "1/2" if "1/2" in pipe_sizes else '1/2"'
            default_index = pipe_sizes.index(want)
        elif "selected_size" in ss and ss.selected_size in pipe_sizes:
            # if Streamlit kept the selection, use it
            default_index = pipe_sizes.index(ss.selected_size)
        
        if "_next_selected_size" in ss:
            if ss["_next_selected_size"] in pipe_sizes:
                ss.selected_size = ss["_next_selected_size"]
            del ss["_next_selected_size"]
        
        with col1:
            selected_size = st.selectbox(
                "Nominal Pipe Size (inch)",
                pipe_sizes,
                index=default_index,
                key="selected_size",
            )
        
        # remember the selected size in mm for next material change
        ss.prev_pipe_mm = float(mm_map.get(selected_size, float("nan")))

        # --- consume any deferred gauge from Auto-select (Pumped Liquid) ---
        if "_next_gauge" in st.session_state:
            g = st.session_state.pop("_next_gauge")
        
            rows = material_df[
                material_df["Nominal Size (inch)"].astype(str).str.strip() == str(selected_size)
            ]
        
            if "Gauge" in rows.columns and rows["Gauge"].notna().any():
                valid_gauges = set(rows["Gauge"].dropna().unique())
                if g in valid_gauges:
                    st.session_state["gauge"] = g
    
        # 3) Gauge (if applicable)
        gauge_options = material_df[material_df["Nominal Size (inch)"].astype(str).str.strip() == selected_size]
        if "Gauge" in gauge_options.columns and gauge_options["Gauge"].notna().any():
            gauges = sorted(gauge_options["Gauge"].dropna().unique())
            with col2:
                selected_gauge = st.selectbox("Copper Gauge", gauges, key="gauge")
            selected_pipe_row = gauge_options[gauge_options["Gauge"] == selected_gauge].iloc[0]
        else:
            selected_pipe_row = gauge_options.iloc[0]

        pipe_index = material_to_pipe_index(selected_material)
        
        # you already have selected_gauge sometimes; otherwise None
        gauge = st.session_state.get("gauge")  # or whatever that mode uses
        od_mm, id_mm = get_dimensions_for_row(material_df, selected_size, gauge)
        
        result = system_pressure_check(
            refrigerant=refrigerant,
            design_temp_c=design_temp_c,
            mwp_temp_c=mwp_temp_c,
            circuit=circuit,
            pipe_index=pipe_index,
            od_mm=od_mm,
            id_mm=id_mm,
            gauge=gauge,
            copper_calc=copper_calc,
            r744_tc_pressure_bar_g=r744_tc_pressure_bar_g,
            dp_standard=dp_standard,
        )
        
        render_pressure_result(result)
    
        # Pipe parameters
        pipe_size_inch = selected_pipe_row["Nominal Size (inch)"]
        ID_mm = selected_pipe_row["ID_mm"]
    
        with col1:
            
            evap_capacity_kw = st.number_input("Evaporator Capacity (kW)", min_value=0.03, max_value=20000.0, value=10.0, step=1.0)
            
            # --- Base ranges per refrigerant ---
            if refrigerant in ("R23", "R508B"):
                evap_min, evap_max, evap_default = -100.0, -20.0, -80.0
            elif refrigerant == "R744":
                evap_min, evap_max, evap_default = -50.0, 20.0, -10.0
            else:
                evap_min, evap_max, evap_default = -50.0, 30.0, -10.0
    
            # --- Init state (widget-backed) ---
            ss = st.session_state
    
            if "last_refrigerant" not in ss or ss.last_refrigerant != refrigerant:
                ss.evap_temp   = evap_default
                ss.last_refrigerant = refrigerant
        
            ss.setdefault("evap_temp",   evap_default)

            liq_oq = st.number_input("Liquid Overfeed Quantity (%)", min_value=0.0, max_value=2000.0, value=100.0, step=25.0)

        with col2:
            
            evaporating_temp = st.number_input(
                "Pumped Liquid Temperature (°C)",
                min_value=evap_min, max_value=evap_max,
                value=ss.evap_temp, step=1.0, key="evap_temp",
            )
    
        with col2:
            risem = st.number_input("Liquid Lift (m)", min_value=0.0, max_value=30.0, value=0.0, step=1.0)
            max_ppd = st.number_input("Max Pressure Drop (bar)", min_value=0.01, max_value=2.0, value=0.5, step=0.1)
    
        with col3:
            L = st.number_input("Pipe Length (m)", min_value=0.1, max_value=300.0, value=10.0, step=1.0)
            LRB = st.number_input("Long Radius Bends", min_value=0, max_value=50, value=0, step=1)
            SRB = st.number_input("Short Radius Bends", min_value=0, max_value=50, value=0, step=1)
            _45 = st.number_input("45° Bends", min_value=0, max_value=50, value=0, step=1)
            MAC = st.number_input("Machine Bends", min_value=0, max_value=50, value=0, step=1)
    
        with col4:
            ptrap = st.number_input("P Traps", min_value=0, max_value=10, value=0, step=1)
            ubend = st.number_input("U Bends", min_value=0, max_value=10, value=0, step=1)
            ball = st.number_input("Ball Valves", min_value=0, max_value=20, value=0, step=1, key="ball")
            globe = st.number_input("Globe Valves", min_value=0, max_value=20, value=0, step=1, key="globe")
            PLF = st.number_input("Pressure Loss Factors", min_value=0.0, max_value=20.0, value=0.0, step=0.1)
        
        from utils.refrigerant_properties import RefrigerantProperties
        from utils.refrigerant_densities import RefrigerantDensities
        from utils.refrigerant_viscosities import RefrigerantViscosities
    
        T_evap = evaporating_temp
    
        props = RefrigerantProperties()

        h_in = props.get_properties(refrigerant, T_evap)["enthalpy_liquid2"]
        h_out = props.get_properties(refrigerant, T_evap)["enthalpy_vapor"]
        deltah = h_out - h_in

        mass_flow_kg_s = (evap_capacity_kw / deltah) * (1 + (liq_oq / 100)) if deltah > 0 else 0.01
    
        if ID_mm is not None:
            ID_m = ID_mm / 1000.0

            area_m2 = math.pi * (ID_m / 2) ** 2

            density = RefrigerantProperties().get_properties(refrigerant, T_evap)["density_liquid2"]

            velocity_m_s = mass_flow_kg_s / (area_m2 * density)

        else:
            velocity_m_s = None

        viscosity = RefrigerantProperties().get_properties(refrigerant, T_evap)["viscosity_liquid"]
    
        reynolds = (density * velocity_m_s * ID_m) / (viscosity / 1000000)
    
        if selected_material in ["Steel SCH40", "Steel SCH80"]:
            eps = 0.00004572 #0.00015
        else:
            eps = 0.000001524 #0.000005
        
        tol = 1e-5
        max_iter = 60
        
        if reynolds < 2000.0:
            f = 64.0 / reynolds
        else:
            flo, fhi = 1e-5, 0.1
            def balance(gg):
                s = math.sqrt(gg)
                lhs = 1.0 / s
                rhs = -2.0 * math.log10((eps / (3.7 * ID_m)) + 2.51 / (reynolds * s))
                return lhs, rhs
    
            f = 0.5 * (flo + fhi)
            for _ in range(max_iter):
                f = 0.5 * (flo + fhi)
                lhs, rhs = balance(f)
                if abs(1.0 - lhs/rhs) < tol:
                    break
                # decide side using sign of (lhs - rhs)
                if (lhs - rhs) > 0.0:
                    flo = f
                else:
                    fhi = f
        
        # dynamic (velocity) pressure, kPa
        q_kPa = 0.5 * density * (velocity_m_s ** 2) / 1000.0
    
        # 1) straight pipe only
        dp_pipe_kPa = f * (L / ID_m) * q_kPa
        
        dp_plf_kPa = q_kPa * PLF
    
        required_cols = ["SRB", "LRB", "BALL", "GLOBE"]
        missing = [c for c in required_cols if c not in selected_pipe_row.index]
        if missing:
            st.error(f"CSV missing required K columns: {missing}")
            st.stop()
    
        # Convert to floats and check NaNs
        try:
            K_SRB  = float(selected_pipe_row["SRB"])
            K_LRB  = float(selected_pipe_row["LRB"])
            K_BALL = float(selected_pipe_row["BALL"])
            K_GLOBE= float(selected_pipe_row["GLOBE"])
        except Exception as e:
            st.error(f"Failed to parse K-factors as numbers: {e}")
            st.stop()
    
        if any(pd.isna([K_SRB, K_LRB, K_BALL, K_GLOBE])):
            st.error("One or more K-factors are NaN in the CSV row.")
            st.stop()
        
        B_SRB = SRB + 0.5 * _45 + 2.0 * ubend + 3.0 * ptrap
        B_LRB = LRB + MAC
    
        dp_fittings_kPa = q_kPa * (
        K_SRB   * B_SRB +
        K_LRB   * B_LRB
        )

        dp_valves_kPa = q_kPa * (
        K_BALL  * ball +
        K_GLOBE * globe
        )
        
        dp_total_kPa = dp_pipe_kPa + dp_fittings_kPa + dp_valves_kPa + dp_plf_kPa
        
        converter = PressureTemperatureConverter()
        evappres = converter.temp_to_pressure2(refrigerant, T_evap)
        postcirc = evappres - (dp_total_kPa / 100)

        head = 9.80665 * risem * density / 1000
        
        dp_withhead = dp_total_kPa + head

        dp_withhead_bar = dp_withhead / 100

        postall = evappres - (dp_withhead / 100)

        volflow = mass_flow_kg_s / density

        mf3600 = mass_flow_kg_s * 3600

        vf3600 = volflow * 3600

        vf_lpm = volflow * 60000

        max_ppd_kpa = max_ppd * 100

        def _auto_select_copper_gauge(
            *,
            material_df,
            size_inch,
            gauges,
            design_pressure,
            refrigerant,
            design_temp_c,
            mwp_temp_c,
            circuit,
            pipe_index,
            copper_calc,
            dp_standard,
            r744_tc_pressure_bar_g,
        ):
            """
            Return:
            - highest gauge where MWP >= design pressure
            - else lowest gauge
            """
        
            passing = []
        
            for g in sorted(gauges):
                try:
                    od_mm, id_mm = get_dimensions_for_row(material_df, size_inch, g)
        
                    res = system_pressure_check(
                        refrigerant=refrigerant,
                        design_temp_c=design_temp_c,
                        mwp_temp_c=mwp_temp_c,
                        circuit=circuit,
                        pipe_index=pipe_index,
                        od_mm=od_mm,
                        id_mm=id_mm,
                        gauge=g,
                        copper_calc=copper_calc,
                        r744_tc_pressure_bar_g=r744_tc_pressure_bar_g,
                        dp_standard=dp_standard,
                    )
        
                    mwp_val = governing_mwp(res["mwp_bar"])
        
                    if mwp_val >= design_pressure:
                        passing.append(g)
        
                except Exception:
                    continue
        
            if passing:
                return max(passing)
            else:
                return min(gauges)

        # -------- Auto-select smallest pipe size meeting dp ≤ max_ppd_kpa --------
        if st.button("Auto-select"):
            results = []
            errors = []
        
            for ps in pipe_sizes:
                rows = material_df[
                    material_df["Nominal Size (inch)"].astype(str).str.strip() == ps
                ]
        
                # 🔹 force strongest gauge for this size during evaluation
                gauge_override = None
                if "Gauge" in rows.columns and rows["Gauge"].notna().any():
                    gauge_override = max(rows["Gauge"].dropna().unique())
        
                dp_i = get_pumped_dp_for_size(ps, gauge_override=gauge_override)
        
                if math.isfinite(dp_i):
                    results.append({"size": ps, "dp": dp_i})
                else:
                    errors.append((ps, "Non-numeric DP"))
        
            if not results:
                st.error("No valid pipe results. Check CSV or input data.")
            else:
                valid = [r for r in results if r["dp"] <= max_ppd_kpa]
        
                if valid:
                    best = min(valid, key=lambda x: mm_map[x["size"]])
                    st.session_state["_next_selected_size"] = best["size"]
        
                    # 🔹 Auto-select gauge for Copper EN12735
                    if selected_material.strip() == "Copper EN12735":
                        rows = material_df[
                            material_df["Nominal Size (inch)"].astype(str).str.strip() == best["size"]
                        ]
        
                        if "Gauge" in rows.columns and rows["Gauge"].notna().any():
                            gauges = sorted(rows["Gauge"].dropna().unique())
        
                            best_gauge = _auto_select_copper_gauge(
                                material_df=material_df,
                                size_inch=best["size"],
                                gauges=gauges,
                                design_pressure=result["design_pressure_bar_g"],
                                refrigerant=refrigerant,
                                design_temp_c=design_temp_c,
                                mwp_temp_c=mwp_temp_c,
                                circuit=circuit,
                                pipe_index=pipe_index,
                                copper_calc=copper_calc,
                                dp_standard=dp_standard,
                                r744_tc_pressure_bar_g=r744_tc_pressure_bar_g,
                            )
        
                            st.session_state["_next_gauge"] = best_gauge
        
                    st.success(
                        f"✅ Selected pumped liquid pipe size: **{best['size']}**\n"
                        f"Pressure Drop: {best['dp']:.2f} kPa (limit {max_ppd_kpa:.2f} kPa)"
                    )
                    st.rerun()
                else:
                    best_dp = min(r["dp"] for r in results)
                    st.error(
                        f"❌ No pipe meets the pressure-drop limit.\n"
                        f"Best achievable: {best_dp:.2f} kPa (limit {max_ppd_kpa:.2f} kPa)."
                    )
    
        if velocity_m_s:
            col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    
            with col1:
                st.metric("Refrigerant Velocity", f"{velocity_m_s:.2f}m/s")
    
            with col2:
                st.metric("Liquid Density", f"{density:.1f}kg/m³")

            with col3:
                st.metric("Volumetric Flow Rate", f"{volflow:.5f}m³/s")
    
            with col4:
                st.metric("Volumetric Flow Rate", f"{vf_lpm:.2f}lpm")

            with col5:
                st.metric("Pressure Drop", f"{dp_total_kPa:.2f}kPa")

            with col6:
                st.metric("Liquid Pressure", f"{evappres:.2f}bar(a)")

            with col7:
                st.metric("System Pump Head", f"{dp_withhead_bar:.3f}bar")

            # correcting default values between cond, max liq, and min liq between liquid calcs and dry suction calcs
            
            col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    
            with col1:
                st.metric("Mass Flow Rate", f"{mass_flow_kg_s:.5f}kg/s")
    
            with col2:
                st.metric("Mass Flow Rate", f"{mf3600:.0f}kg/hr")
    
            with col3:
                st.metric("Volumetric Flow Rate", f"{vf3600:.3f}m³/hr")

            with col4:
                st.metric("Pipe PD", f"{dp_pipe_kPa:.2f}kPa")

            with col5:
                st.metric("Fittings PD", f"{dp_fittings_kPa:.2f}kPa")

            with col6:
                st.metric("Valves PD", f"{dp_valves_kPa:.2f}kPa")
                
            with col7:
                st.metric("Velocity Pressure PD", f"{dp_plf_kPa:.2f}kPa")

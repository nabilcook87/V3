import math
from utils.refrigerant_properties import RefrigerantProperties

def get_correction_factor(pipe_size_inch):
    correction_factors = {
        "1/4": 0.0542353548542213, "3/8": 0.132622007740968, "1/2": 0.252329374817354, "5/8": 0.417271385372936,
        "3/4": 0.65465093095418, "7/8": 0.965351624641057, "1-1/8": 1.88505511366633, "1-3/8": 3.2762154102682,
        "1-5/8": 5.08940466376191, "2-1/8": 10.2848883188943, "2-5/8": 17.2409820795164, "3-1/8": 27.1560625075574,
        "3-5/8": 38.9318438501782, "4-1/8": 54.5202012640502
    }
    return correction_factors.get(pipe_size_inch.strip())

def get_base_min_duty_kw(refrigerant):
    return {
        "R22": 1.0, "R134a": 1.0, "R404A": 1.2, "R407C": 1.0,
        "R407F": 1.2, "R410A": 1.1, "R507A": 1.2, "R744": 1.5,
        "R448A": 1.2, "R449A": 1.2, "R32": 1.0, "R454A": 1.0
    }.get(refrigerant.strip(), 1.0)

def get_scaling_factor(refrigerant):
    return {
        "R22": 1.00, "R134a": 0.941, "R404A": 1.04, "R407C": 1.00,
        "R407F": 0.7595, "R410A": 0.608, "R507A": 1.04, "R744": 1.20,
        "R448A": 1.04, "R449A": 1.04, "R32": 1.00, "R454A": 1.00
    }.get(refrigerant.strip(), 1.0)

def check_oil_return(pipe_size_inch, refrigerant, evap_capacity_kw, duty_pct,
                     evap_temp, cond_temp, superheat, subcool):

    cf = get_correction_factor(pipe_size_inch)
    if cf is None:
        return False, f"❌ No correction factor for pipe size {pipe_size_inch}"

    base_min_kw = get_base_min_duty_kw(refrigerant)
    scaling = get_scaling_factor(refrigerant)

    props = RefrigerantProperties()

    try:
        h_liq = props.get_properties(refrigerant, cond_temp - subcool)["enthalpy_liquid"]
        h_vap = props.get_properties(refrigerant, evap_temp)["enthalpy_vapor"]
        h_vap_plus10 = props.get_properties(refrigerant, evap_temp)["enthalpy_super"]
    except Exception:
        return False, "❌ Error reading refrigerant enthalpies"

    Cp = (h_vap_plus10 - h_vap) / 10
    h_out = h_vap + superheat * Cp
    delta_h = h_out - h_liq

    if delta_h <= 0:
        return False, "❌ Invalid enthalpy values (Δh ≤ 0)"

    delta_h2 = 70
                
    # 🔥 Step 1: calculate min required mass flow at 100% duty
    min_mass_flow = ((base_min_kw * cf) / scaling) / delta_h2

    # 🔥 Step 2: scale down actual flow for duty percentage
    actual_mass_flow = (evap_capacity_kw * (duty_pct / 100.0)) / delta_h

    full_mass_flow = evap_capacity_kw / delta_h if delta_h > 0 else 0.01

    min_oil_return = (min_mass_flow / full_mass_flow) * 100

    # 🔥 Step 3: compare
    if actual_mass_flow >= min_mass_flow:
        return True, f"✅ OK: {actual_mass_flow:.3f} kg/s ≥ {min_mass_flow:.3f} kg/s (min required)", min_oil_return
    else:
        return False, f"❌ Insufficient flow: {actual_mass_flow:.3f} < {min_mass_flow:.3f} kg/s (min required)", min_oil_return

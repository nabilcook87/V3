# utils/friction_calculations.py

import math

def darcy_friction_factor(Re):
    """
    Calculate Darcy friction factor.
    Uses Blasius approximation for turbulent flow and laminar equation if Re < 2000.
    """
    if Re < 2000:
        return 64 / Re  # Laminar flow
    else:
        return 0.3164 * Re ** -0.25  # Blasius for turbulent flow

def pressure_drop_per_meter(rho, velocity, diameter_mm):
    """
    Calculate pressure drop per meter of straight pipe using Darcy-Weisbach (Pa/m).

    Parameters:
        rho: fluid density (kg/m³)
        velocity: fluid velocity (m/s)
        diameter_mm: internal diameter of pipe (mm)
    """
    D = diameter_mm / 1000.0  # convert to meters
    mu = 1.2e-5  # Approximate dynamic viscosity of refrigerant (Pa·s)

    Re = (rho * velocity * D) / mu
    f = darcy_friction_factor(Re)

    dp_per_m = f * (rho * velocity**2) / (2 * D)
    return dp_per_m  # Pa/m

def get_equivalent_length(fittings_list, pipe_diameter_mm):
    """
    Calculate the total equivalent length from a list of fittings.

    Parameters:
        fittings_list: list of dicts like:
            [
                {"type": "Long Radius Elbow", "count": 4},
                {"type": "Tee (through)", "count": 1},
                {"type": "Reducer", "count": 2}
            ]
        pipe_diameter_mm: used to scale certain fittings like reducers if needed

    Returns:
        total equivalent length in meters
    """
    # Multipliers expressed in pipe diameters (D)
    fitting_equiv_multipliers = {
        "Long Radius Elbow": 30,
        "Short Radius Elbow": 60,
        "Tee (through)": 20,
        "Tee (branch)": 60,
        "Gate Valve": 8,
        "Ball Valve": 10,
        "Globe Valve": 340,
        "Check Valve": 100,
        "Strainer": 100,
        "Expansion Loop": 500,
        "Reducer": 10,  # Typical value for short transition reducers
        "Miscellaneous": 50  # Fallback for unknowns
    }

    total_equiv_length_m = 0.0

    for fitting in fittings_list:
        f_type = fitting.get("type", "Miscellaneous")
        count = fitting.get("count", 0)
        multiplier = fitting_equiv_multipliers.get(f_type, fitting_equiv_multipliers["Miscellaneous"])
        length_per_fitting_m = (multiplier / 1000.0) * pipe_diameter_mm
        total_equiv_length_m += length_per_fitting_m * count

    return total_equiv_length_m
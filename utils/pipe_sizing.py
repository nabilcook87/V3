from utils.refrigerant_properties import RefrigerantProperties
from utils.friction_calculations import get_equivalent_length
from utils.system_pressure_checker import system_pressure_check
from utils.oil_return_checker import check_oil_return
from utils.pipe_length_volume_calc import calculate_pipe_volume_liters
import math
import pandas as pd

class PipeSizer:
    def __init__(self):
        self.refrigerant_props = RefrigerantProperties()
        self.pipe_table = self.load_pipe_table()

    def load_pipe_table(self):
        df = pd.read_csv("data/pipe_pressure_ratings_full.csv")
        return df.to_dict(orient="records")

    def size_pipe(self, refrigerant, pipe_type, T_evap, T_cond, superheat_K, subcooling_K,
                  pipe_length_m, evap_capacity_kw, fixed_pipe_size, has_riser=False,
                  fittings_equivalent_length_m=0.0):

        props = self.refrigerant_props.get_properties(refrigerant, T_evap)
        h_vap = props["enthalpy_vapor"]
        h_vap_plus = self.refrigerant_props.get_properties(refrigerant, T_evap)["enthalpy_super"]
        Cp_vapor = (h_vap_plus - h_vap) / 10
        h_subcooled = self.refrigerant_props.get_properties(refrigerant, T_cond)["enthalpy_liquid"]

        Δh_kj_per_kg = (
            Cp_vapor * superheat_K
            if pipe_type in ["Dry Suction", "Discharge"]
            else h_subcooled - self.refrigerant_props.get_properties(refrigerant, T_cond - subcooling_K)["enthalpy_liquid"]
        )
        m_dot_kg_s = (evap_capacity_kw / Δh_kj_per_kg) if Δh_kj_per_kg > 0 else 0.01

        candidates = self.pipe_table if not fixed_pipe_size else [
            pipe for pipe in self.pipe_table if pipe.get("Nominal Size (inch)") == fixed_pipe_size
        ]

        best_pipe = None
        for pipe in candidates:
            ID_mm = pipe.get("ID_mm")
            friction = pipe.get("Friction Factor")
            pipe_size_inch = pipe.get("Nominal Size (inch)")

            if ID_mm is None or friction is None or pipe_size_inch is None:
                continue  # skip incomplete data

            ID_m = ID_mm / 1000.0
            area_m2 = math.pi * (ID_m / 2) ** 2
            density = props["density_vapor"] if pipe_type in ["Dry Suction", "Discharge"] else props["density_liquid"]
            velocity = m_dot_kg_s / (area_m2 * density)

            total_equiv_length = pipe_length_m + fittings_equivalent_length_m
            pressure_drop_fric = (
                friction * (total_equiv_length / ID_m) * (density * velocity ** 2 / 2)
            ) / 1000  # kPa

            pressure_drop_total = pressure_drop_fric

            # ✅ NEW: Use mass flow-based oil return logic from legacy VB code
            passes_velocity, _ = check_oil_return(pipe_size_inch, refrigerant, m_dot_kg_s)
            rating_ok = check_pipe_rating(pd.Series(pipe), T_cond, pressure_drop_total)

            if passes_velocity and rating_ok:
                best_pipe = {
                    "selected_pipe": pipe,
                    "velocity_m_s": velocity,
                    "pressure_drop_total_kpa": pressure_drop_total,
                    "mass_flow_kg_s": m_dot_kg_s
                }
                break

        if best_pipe:
            return best_pipe
        else:
            raise ValueError("No suitable pipe found for given conditions.")

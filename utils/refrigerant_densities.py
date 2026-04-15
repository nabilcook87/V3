# utils/refrigerant_densities.py

import json
import numpy as np
import os

class RefrigerantDensities:
    def __init__(self):
        base_path = os.path.dirname(os.path.dirname(__file__))
        data_path = os.path.join(base_path, 'data', 'refrigerant_densities.json')
        with open(data_path, 'r') as file:
            self.tables = json.load(file)

    def get_density(self, refrigerant, evap_temp_K, superheat_K):
        """
        2D log-linear interpolation using linear input axes and log-transformed densities.
        """
        table = self.tables.get(refrigerant)
        if table is None:
            raise ValueError(f"Refrigerant '{refrigerant}' not found.")

        # Superheat axis and evap temp axis
        superheat_axis = np.array(table["superheat"], dtype=np.float64)
        evap_keys = [k for k in table if k != "superheat"]
        evap_vals = np.array(sorted([float(k) for k in evap_keys]), dtype=np.float64)

        # Data matrix (densities), log-transformed
        data_matrix = np.array([table[k] for k in map(str, evap_vals)], dtype=np.float64)
        log_data = np.log(data_matrix)

        # First interpolate along superheat (x-direction)
        interp_log_z = np.array([
            np.interp(superheat_K, superheat_axis, row)
            for row in log_data
        ])

        # Then interpolate along evap temp (y-direction)
        final_log_density = np.interp(evap_temp_K, evap_vals, interp_log_z)

        return float(np.exp(final_log_density))

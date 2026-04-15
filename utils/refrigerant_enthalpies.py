# utils/refrigerant_enthalpies.py

import json
import numpy as np
import os

class RefrigerantEnthalpies:
    def __init__(self):
        base_path = os.path.dirname(os.path.dirname(__file__))
        data_path = os.path.join(base_path, 'data', 'refrigerant_enthalpies.json')
        with open(data_path, 'r') as file:
            self.tables = json.load(file)

    def get_enthalpy(self, refrigerant, evap_temp_K, superheat_K):
        """
        2D log-linear interpolation using linear input axes and log-transformed enthalpies.
        """
        table = self.tables.get(refrigerant)
        if table is None:
            raise ValueError(f"Refrigerant '{refrigerant}' not found.")

        # Superheat axis and evap temp axis
        superheat_axis = np.array(table["superheat"], dtype=np.float64)
        evap_keys = [k for k in table if k != "superheat"]
        evap_vals = np.array(sorted([float(k) for k in evap_keys]), dtype=np.float64)

        # Data matrix (enthalpies), log-transformed
        data_matrix = np.array([table[k] for k in map(str, evap_vals)], dtype=np.float64)
        log_data = np.log(data_matrix)

        # First interpolate along superheat (x-direction)
        interp_log_z = np.array([
            np.interp(superheat_K, superheat_axis, row)
            for row in log_data
        ])

        # Then interpolate along evap temp (y-direction)
        final_log_enthalpy = np.interp(evap_temp_K, evap_vals, interp_log_z)

        return float(np.exp(final_log_enthalpy))

    def get_superheat_from_enthalpy(self, refrigerant, evap_temp_K, enthalpy):

        table = self.tables.get(refrigerant)
        if table is None:
            raise ValueError(f"Refrigerant '{refrigerant}' not found.")

        superheat_axis = np.array(table["superheat"], dtype=np.float64)
        evap_keys = [k for k in table if k != "superheat"]
        evap_vals = np.array(sorted([float(k) for k in evap_keys]), dtype=np.float64)

        data_matrix = np.array([table[k] for k in map(str, evap_vals)], dtype=np.float64)
        log_data = np.log(data_matrix)

        target_log_h = float(np.log(enthalpy))

        j = int(np.searchsorted(evap_vals, evap_temp_K))
        j0 = max(j - 1, 0)
        j1 = min(j, len(evap_vals) - 1)

        def inverse_interp_logh_to_superheat(log_row, target_log_h):

            xp = np.asarray(log_row, dtype=np.float64)
            fp = np.asarray(superheat_axis, dtype=np.float64)

            if xp[0] > xp[-1]:
                xp = xp[::-1]
                fp = fp[::-1]

            return float(np.interp(target_log_h, xp, fp))

        sh0 = inverse_interp_logh_to_superheat(log_data[j0, :], target_log_h)
        if j0 == j1:

            return sh0

        sh1 = inverse_interp_logh_to_superheat(log_data[j1, :], target_log_h)

        y0, y1 = evap_vals[j0], evap_vals[j1]

        t = (evap_temp_K - y0) / (y1 - y0) if y1 != y0 else 0.0
        superheat = (1.0 - t) * sh0 + t * sh1

        return float(superheat)

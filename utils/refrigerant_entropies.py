# utils/refrigerant_entropies.py

import json
import numpy as np
import os

class RefrigerantEntropies:
    def __init__(self):
        base_path = os.path.dirname(os.path.dirname(__file__))
        data_path = os.path.join(base_path, 'data', 'refrigerant_entropies.json')
        with open(data_path, 'r') as file:
            self.tables = json.load(file)

    def get_entropy(self, refrigerant, evap_temp_K, superheat_K):
        """
        2D log-linear interpolation using linear input axes and log-transformed entropies.
        """
        table = self.tables.get(refrigerant)
        if table is None:
            raise ValueError(f"Refrigerant '{refrigerant}' not found.")

        # Superheat axis and evap temp axis
        superheat_axis = np.array(table["superheat"], dtype=np.float64)
        evap_keys = [k for k in table if k != "superheat"]
        evap_vals = np.array(sorted([float(k) for k in evap_keys]), dtype=np.float64)

        # Data matrix (entropies), log-transformed
        data_matrix = np.array([table[k] for k in map(str, evap_vals)], dtype=np.float64)
        log_data = np.log(data_matrix)

        # First interpolate along superheat (x-direction)
        interp_log_z = np.array([
            np.interp(superheat_K, superheat_axis, row)
            for row in log_data
        ])

        # Then interpolate along evap temp (y-direction)
        final_log_entropy = np.interp(evap_temp_K, evap_vals, interp_log_z)

        return float(np.exp(final_log_entropy))

    def get_superheat_from_entropy(self, refrigerant, evap_temp_K, entropy):
        """
        Invert the log-linear surface to get superheat given entropy and evaporator temperature.

        Strategy:
        - Work in log-space for entropy to stay consistent with get_entropy().
        - For the two evap-temp rows that bracket `evap_temp_K`, do a 1D inverse interpolation:
            log_entropy(row) -> superheat
          (i.e., swap the roles in np.interp so we interpolate on the x-axis = log(entropy).)
        - Then linearly interpolate the two superheat values across evap_temp to the target `evap_temp_K`.

        Notes:
        - Assumes monotonic log-entropy vs superheat along each row (typical physically).
        - Out-of-range values are clamped to the edge by np.interp, mirroring get_entropy().
        """
        table = self.tables.get(refrigerant)
        if table is None:
            raise ValueError(f"Refrigerant '{refrigerant}' not found.")

        # Axes
        superheat_axis = np.array(table["superheat"], dtype=np.float64)  # x
        evap_keys = [k for k in table if k != "superheat"]
        evap_vals = np.array(sorted([float(k) for k in evap_keys]), dtype=np.float64)  # y

        # Data (z = entropy), stored as rows per evap temp, columns per superheat
        data_matrix = np.array([table[k] for k in map(str, evap_vals)], dtype=np.float64)
        log_data = np.log(data_matrix)

        target_log_s = float(np.log(entropy))

        # Find bracketing evap-temp indices
        j = int(np.searchsorted(evap_vals, evap_temp_K))
        j0 = max(j - 1, 0)
        j1 = min(j, len(evap_vals) - 1)

        def inverse_interp_logS_to_superheat(log_row, target_log_s):
            """
            Given one evap-temp row of log(entropy) over superheat, invert to get superheat.
            Handles non-ascending rows by sorting (shouldn't happen, but be robust).
            """
            xp = np.asarray(log_row, dtype=np.float64)         # x = log(entropy)
            fp = np.asarray(superheat_axis, dtype=np.float64)  # y = superheat

            # Ensure xp is ascending for np.interp; if descending, flip both.
            if xp[0] > xp[-1]:
                xp = xp[::-1]
                fp = fp[::-1]

            # If there are any equal consecutive xp (flat segments), np.interp is fine,
            # but to be extra safe we can do a stable argsort and unique. Keep it simple:
            return float(np.interp(target_log_s, xp, fp))

        # Invert on the two bracketing rows
        sh0 = inverse_interp_logS_to_superheat(log_data[j0, :], target_log_s)
        if j0 == j1:
            # Exact match or out of bounds -> single row result
            return sh0

        sh1 = inverse_interp_logS_to_superheat(log_data[j1, :], target_log_s)

        # Linear blend across evap temp
        y0, y1 = evap_vals[j0], evap_vals[j1]
        # Protect against division by zero (shouldn't happen since j0 != j1 here)
        t = (evap_temp_K - y0) / (y1 - y0) if y1 != y0 else 0.0
        superheat = (1.0 - t) * sh0 + t * sh1

        return float(superheat)

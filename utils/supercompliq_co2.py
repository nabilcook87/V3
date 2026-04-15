# utils/supercompliq_co2.py
import json
import os
from typing import Literal, Dict, Any, List
import numpy as np

PropertyName = Literal["density", "enthalpy", "viscosity", "entropy"]

class RefrigerantProps:
    """
    2D log-linear interpolation on a rectangular grid:
    - x-axis: temperature in °C (linear)
    - y-axis: pressure in bar(a) (linear)
    - z-values (properties): log-transformed, then bilinear interpolation.

    JSON format (per property):
    {
      "temperature": [... °C ...],
      "73.8": [...],
      "75":   [...],
      ...
    }
    """

    def __init__(self, data_path: str = None):
        if data_path is None:
            base_path = os.path.dirname(os.path.dirname(__file__))
            data_path = os.path.join(base_path, "data", "supercompliq_co2.json")
        with open(data_path, "r") as f:
            self.data: Dict[str, Any] = json.load(f)

        # Validate available properties
        self._props_available = set(self.data.keys())

    # ---------- Public API ----------

    def get(
        self,
        prop: PropertyName,
        pressure_bar_a: float,
        temperature_C: float,
        *,
        clip: bool = False,
    ) -> float:
        """
        Return the requested property using 2D log-linear interpolation.

        Parameters
        ----------
        prop : one of "density", "enthalpy", "viscosity", "entropy"
        pressure_bar_a : pressure (bar absolute)
        temperature_C : temperature (°C)
        clip : if True, values outside the tabulated ranges are clipped
               to the nearest grid edge instead of raising.

        Returns
        -------
        float
        """
        table = self._get_property_table(prop)

        temps = np.asarray(table["temperature"], dtype=np.float64)  # x-axis
        # y-axis (pressures) from string keys -> float, sorted ascending
        pressure_keys = sorted(
            (k for k in table.keys() if k != "temperature"),
            key=lambda s: float(s),
        )
        pressures = np.asarray([float(k) for k in pressure_keys], dtype=np.float64)

        # Build data matrix shape [n_pressures, n_temps]
        data_matrix = np.vstack(
            [np.asarray(table[k], dtype=np.float64) for k in pressure_keys]
        )

        # Guard: positive values for log
        if np.any(data_matrix <= 0.0):
            raise ValueError(
                f"Non-positive values found in '{prop}' table; cannot log-transform."
            )

        x = float(temperature_C)
        y = float(pressure_bar_a)

        # Range checks / clipping
        x_min, x_max = temps[0], temps[-1]
        y_min, y_max = pressures[0], pressures[-1]
        if clip:
            x = float(np.clip(x, x_min, x_max))
            y = float(np.clip(y, y_min, y_max))
        else:
            if not (x_min <= x <= x_max):
                raise ValueError(
                    f"Temperature {temperature_C} °C outside table range [{x_min}, {x_max}]."
                )
            if not (y_min <= y <= y_max):
                raise ValueError(
                    f"Pressure {pressure_bar_a} bar(a) outside table range [{y_min}, {y_max}]."
                )

        # ---- 2-stage interpolation with log(z) ----
        logZ = np.log(data_matrix)

        # 1) interpolate along temperature (x) for each pressure row
        #    -> vector of logZ at target temperature, length n_pressures
        logZ_vs_y = np.array([np.interp(x, temps, row) for row in logZ])

        # 2) interpolate along pressure (y) across those values
        logZ_final = float(np.interp(y, pressures, logZ_vs_y))

        return float(np.exp(logZ_final))

    def get_density_sup(self, pressure_bar_a: float, temperature_C: float, **kw) -> float:
        return self.get("density", pressure_bar_a, temperature_C, **kw)

    def get_enthalpy_sup(self, pressure_bar_a: float, temperature_C: float, **kw) -> float:
        return self.get("enthalpy", pressure_bar_a, temperature_C, **kw)

    def get_viscosity_sup(self, pressure_bar_a: float, temperature_C: float, **kw) -> float:
        return self.get("viscosity", pressure_bar_a, temperature_C, **kw)

    def get_entropy_sup(self, pressure_bar_a: float, temperature_C: float, **kw) -> float:
        return self.get("entropy", pressure_bar_a, temperature_C, **kw)

    # ---------- Helpers ----------

    def temperatures(self, prop: PropertyName) -> List[float]:
        return list(self._get_property_table(prop)["temperature"])

    def pressures(self, prop: PropertyName) -> List[float]:
        table = self._get_property_table(prop)
        return sorted([float(k) for k in table.keys() if k != "temperature"])

    def _get_property_table(self, prop: PropertyName) -> Dict[str, Any]:
        if prop not in self._props_available:
            raise ValueError(
                f"Property '{prop}' not found. Available: {sorted(self._props_available)}"
            )

        return self.data[prop]

    def get_temperature_from_property(
        self,
        prop: Literal["entropy", "enthalpy"],
        pressure_bar_a: float,
        target_value: float,
        *,
        clip: bool = False,
    ) -> float:
        """
        Invert the 2D log-linear surface to obtain temperature from the given property value
        (entropy or enthalpy) at a specific pressure.

        Parameters
        ----------
        prop : "entropy" or "enthalpy"
            The property to invert.
        pressure_bar_a : float
            Pressure in bar(a).
        target_value : float
            Target property value (same units as the table).
        clip : bool
            If True, clamp pressure and temperature range; otherwise raise if out of range.

        Returns
        -------
        float
            Temperature in °C corresponding to the target property at the given pressure.
        """
        if prop not in {"entropy", "enthalpy"}:
            raise ValueError("Only 'entropy' or 'enthalpy' can be inverted to temperature.")

        table = self._get_property_table(prop)
        temps = np.asarray(table["temperature"], dtype=np.float64)
        pressure_keys = sorted((k for k in table.keys() if k != "temperature"), key=lambda s: float(s))
        pressures = np.asarray([float(k) for k in pressure_keys], dtype=np.float64)
        data_matrix = np.vstack([np.asarray(table[k], dtype=np.float64) for k in pressure_keys])

        if np.any(data_matrix <= 0.0):
            raise ValueError(f"Non-positive values in '{prop}' table; cannot log-transform.")

        log_data = np.log(data_matrix)
        target_log = float(np.log(target_value))

        # Handle pressure range
        y = float(pressure_bar_a)
        y_min, y_max = pressures[0], pressures[-1]
        if clip:
            y = float(np.clip(y, y_min, y_max))
        elif not (y_min <= y <= y_max):
            raise ValueError(f"Pressure {y} bar(a) outside range [{y_min}, {y_max}].")

        # --- Find bracketing pressures for 2D interpolation ---
        j = int(np.searchsorted(pressures, y))
        j0 = max(j - 1, 0)
        j1 = min(j, len(pressures) - 1)

        def inverse_interp_logZ_to_T(log_row, target_log):
            """
            Invert one 1D curve log(property) vs temperature → get temperature.
            Assumes monotonic log(property) along temperature.
            """
            xp = np.asarray(log_row, dtype=np.float64)
            fp = np.asarray(temps, dtype=np.float64)
            if xp[0] > xp[-1]:  # ensure ascending xp for np.interp
                xp = xp[::-1]
                fp = fp[::-1]
            return float(np.interp(target_log, xp, fp))

        # Invert along temperature for the two bracketing pressure rows
        T0 = inverse_interp_logZ_to_T(log_data[j0, :], target_log)
        if j0 == j1:
            return T0
        T1 = inverse_interp_logZ_to_T(log_data[j1, :], target_log)

        # Linear interpolation across pressure
        p0, p1 = pressures[j0], pressures[j1]
        t = (y - p0) / (p1 - p0) if p1 != p0 else 0.0
        return float((1.0 - t) * T0 + t * T1)

    # Convenience wrappers
    def get_temperature_from_entropy(self, pressure_bar_a: float, entropy: float, **kw) -> float:
        """Return temperature (°C) for given pressure and entropy."""
        return self.get_temperature_from_property("entropy", pressure_bar_a, entropy, **kw)

    def get_temperature_from_enthalpy(self, pressure_bar_a: float, enthalpy: float, **kw) -> float:
        """Return temperature (°C) for given pressure and enthalpy."""
        return self.get_temperature_from_property("enthalpy", pressure_bar_a, enthalpy, **kw)

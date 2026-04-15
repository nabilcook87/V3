# /utils/pressure_temp_converter.py

import numpy as np
import math
from utils.refrigerant_properties import RefrigerantProperties

class PressureTemperatureConverter:
    def __init__(self):
        self.refrigerant_props = RefrigerantProperties()

    def pressure_to_temp(self, refrigerant, target_pressure_bar):
        """
        Find saturation temperature for a given pressure using ln interpolation.
        """
        data = self.refrigerant_props.tables[refrigerant]
        pressures = data["pressure_bar"]
        temperatures = data["temperature_C"]

        for i in range(len(pressures) - 1):
            if pressures[i] <= target_pressure_bar <= pressures[i + 1]:
                x1, x2 = pressures[i], pressures[i + 1]
                y1, y2 = temperatures[i], temperatures[i + 1]

                import math
                ln_x1, ln_x2 = math.log(x1), math.log(x2)
                ln_target = math.log(target_pressure_bar)
                slope = (y2 - y1) / (ln_x2 - ln_x1)
                return y1 + slope * (ln_target - ln_x1)

        # Outside range — clamp to min or max
        if target_pressure_bar < pressures[0]:
            return temperatures[0]
        else:
            return temperatures[-1]

    def temp_to_pressure(self, refrigerant, temperature_C):
        """
        Find saturation pressure for a given temperature using ln interpolation.
        """
        import math

        data = self.refrigerant_props.tables[refrigerant]
        pressures = data["pressure_bar"]
        temperatures = data["temperature_C"]

        for i in range(len(temperatures) - 1):
            if temperatures[i] <= temperature_C <= temperatures[i + 1]:
                y1, y2 = temperatures[i], temperatures[i + 1]
                x1, x2 = pressures[i], pressures[i + 1]

                ln_x1, ln_x2 = math.log(x1), math.log(x2)
                slope = (y2 - y1) / (ln_x2 - ln_x1)

                # Rearranged to get ln(P) from T: ln(P) = (T - y1)/slope + ln(x1)
                ln_target = (temperature_C - y1) / slope + math.log(x1)
                return math.exp(ln_target)

        # Outside range — clamp to min or max
        if temperature_C < temperatures[0]:
            return pressures[0]
        else:
            return pressures[-1]

    def pressure_drop_to_temp_penalty(self, refrigerant, sat_temp_C, pressure_drop_kPa):
        data = self.refrigerant_props.tables[refrigerant]
        temps = np.array(data["temperature_C"])
        pressures_Pa = np.array(data["pressure_bar"]) * 1e5  # Convert to Pascals

        if not (temps[0] <= sat_temp_C <= temps[-1]):
            return 0.0

        # Compute log slopes
        lnP = np.log(pressures_Pa)
        dlnP_dT = np.diff(lnP) / np.diff(temps)
        slope_temps = temps[:-1]

        slope = np.interp(sat_temp_C, slope_temps, dlnP_dT)

        if abs(slope) < 1e-12:
            return 0.0

        delta_lnP = pressure_drop_kPa * 1e3 / np.interp(sat_temp_C, temps, pressures_Pa)
        delta_T = delta_lnP / slope

        return delta_T

    def temp_penalty_to_pressure_drop(self, refrigerant, sat_temp_C, temp_penalty_K):
        data = self.refrigerant_props.tables[refrigerant]
        temps = np.array(data["temperature_C"])
        pressures_Pa = np.array(data["pressure_bar"]) * 1e5  # Pascals

        if not (temps[0] <= sat_temp_C <= temps[-1]):
            return 0.0

        lnP = np.log(pressures_Pa)
        dlnP_dT = np.diff(lnP) / np.diff(temps)
        slope_temps = temps[:-1]

        slope = np.interp(sat_temp_C, slope_temps, dlnP_dT)

        if abs(slope) < 1e-12:
            return 0.0

        P = np.interp(sat_temp_C, temps, pressures_Pa)
        delta_lnP = slope * temp_penalty_K
        delta_P = P * (np.exp(delta_lnP) - 1)

        return delta_P / 1e3  # back to kPa

    def pressure2_to_temp(self, refrigerant, target_pressure_bar):
        """
        Convert pressure_bar2 → temperature using ln interpolation.
        Uses: bubblepoint_C vs pressure_bar
        """
        data = self.refrigerant_props.tables[refrigerant]
        temps = np.array(data["bubblepoint_C"])
        pressures = np.array(data["pressure_bar"])

        for i in range(len(pressures) - 1):
            if pressures[i] <= target_pressure_bar <= pressures[i + 1]:
                x1, x2 = pressures[i], pressures[i + 1]
                y1, y2 = temps[i], temps[i + 1]

                ln_x1, ln_x2 = math.log(x1), math.log(x2)
                ln_target = math.log(target_pressure_bar)
                slope = (y2 - y1) / (ln_x2 - ln_x1)

                return y1 + slope * (ln_target - ln_x1)

        # Clamp
        if target_pressure_bar < pressures[0]:
            return temps[0]
        else:
            return temps[-1]


    def temp_to_pressure2(self, refrigerant, temperature_C):
        """
        Convert temperature → pressure_bar2 using ln interpolation.
        Uses: bubblepoint_C vs pressure_bar
        """
        data = self.refrigerant_props.tables[refrigerant]
        temps = np.array(data["bubblepoint_C"])
        pressures = np.array(data["pressure_bar"])

        for i in range(len(temps) - 1):
            if temps[i] <= temperature_C <= temps[i + 1]:
                y1, y2 = temps[i], temps[i + 1]
                x1, x2 = pressures[i], pressures[i + 1]

                ln_x1, ln_x2 = math.log(x1), math.log(x2)
                slope = (y2 - y1) / (ln_x2 - ln_x1)

                ln_target = (temperature_C - y1) / slope + ln_x1
                return math.exp(ln_target)

        # Clamp
        if temperature_C < temps[0]:
            return pressures[0]
        else:
            return pressures[-1]


    # --------------------------------------------------------
    # ΔP2 ↔ ΔT penalties (pressure_bar2 version)
    # --------------------------------------------------------

    def pressure2_drop_to_temp_penalty(self, refrigerant, sat_temp_C, pressure_drop_kPa):
        """
        Convert pressure_bar2 drop → temperature penalty.
        Uses: bubblepoint_C vs pressure_bar
        """
        data = self.refrigerant_props.tables[refrigerant]
        temps = np.array(data["bubblepoint_C"])
        pressures_Pa = np.array(data["pressure_bar"]) * 1e5

        if not (temps[0] <= sat_temp_C <= temps[-1]):
            return 0.0

        lnP = np.log(pressures_Pa)
        dlnP_dT = np.diff(lnP) / np.diff(temps)
        slope_temps = temps[:-1]

        slope = np.interp(sat_temp_C, slope_temps, dlnP_dT)
        if abs(slope) < 1e-12:
            return 0.0

        P = np.interp(sat_temp_C, temps, pressures_Pa)
        delta_lnP = (pressure_drop_kPa * 1e3) / P

        return delta_lnP / slope


    def temp_penalty_to_pressure2_drop(self, refrigerant, sat_temp_C, temp_penalty_K):
        """
        Convert temperature penalty → pressure_bar2 drop.
        Uses: bubblepoint_C vs pressure_bar
        """
        data = self.refrigerant_props.tables[refrigerant]
        temps = np.array(data["bubblepoint_C"])
        pressures_Pa = np.array(data["pressure_bar"]) * 1e5

        if not (temps[0] <= sat_temp_C <= temps[-1]):
            return 0.0

        lnP = np.log(pressures_Pa)
        dlnP_dT = np.diff(lnP) / np.diff(temps)
        slope_temps = temps[:-1]

        slope = np.interp(sat_temp_C, slope_temps, dlnP_dT)
        if abs(slope) < 1e-12:
            return 0.0

        P = np.interp(sat_temp_C, temps, pressures_Pa)

        delta_lnP = slope * temp_penalty_K
        delta_P = P * (math.exp(delta_lnP) - 1)

        return delta_P / 1e3   # Pa → kPa

# utils/pipe_length_volume_calc.py

import math

def calculate_pipe_volume_liters(diameter_mm, length_m):
    """
    Calculate internal volume of a cylindrical pipe in liters.

    Parameters:
        diameter_mm (float): Internal diameter in millimeters
        length_m (float): Length of the pipe in meters

    Returns:
        float: Volume in liters
    """
    diameter_m = diameter_mm / 1000.0
    radius_m = diameter_m / 2
    area_m2 = math.pi * radius_m ** 2
    volume_m3 = area_m2 * length_m
    return volume_m3 * 1000  # Convert to liters

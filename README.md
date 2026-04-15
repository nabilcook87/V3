# Micropipe - Refrigeration & HVAC Pipe Sizing Tool

**Micropipe** is a professional-grade engineering tool for designing and verifying refrigeration pipe networks.  
It calculates mass flow, selects pipe sizes, evaluates oil return, pressure drops, refrigerant mass, and more.

---

## 🚀 Features

- Size complete refrigeration piping networks:  
  _Dry Suction_, _Wet Suction_, _Discharge_, _Liquid_, _Condenser Drain_, _Pumped Liquid_
- Compute mass flow from evaporator load using enthalpy + Cp-based superheat logic
- Dynamic refrigerant properties: pressure, temperature, enthalpy, density
- Pipe sizing by velocity + pressure drop, with override option
- Velocity and oil return checks (horizontal ≥ 4 m/s, vertical ≥ 7 m/s)
- Total pressure drop including friction + fittings
- Equivalent length calculator with elbows, tees, valves, reducers
- Pressure ↔ Temperature and Pressure Drop ↔ Temperature Penalty converters
- System pressure rating checks per material and pipe size
- Calculate internal volume and refrigerant mass in each pipe
- Fully save/load entire network design files in JSON
- Clean Streamlit UI with circuit editor and tools

---

## 📂 Folder Structure

/micropipe/ ├── app.py # Main Streamlit app ├── requirements.txt # Required Python packages ├── README.md # This file ├── /data/ │ ├── refrigerant_tables.json # Thermodynamic data │ └── pipe_pressure_ratings_full.csv # Pipe sizes + pressure ratings ├── /utils/ │ ├── refrigerant_properties.py # Refrigerant lookup/interpolation │ ├── pipe_sizing.py # Mass flow + pipe size logic │ ├── oil_return_checker.py # Oil return rules │ ├── friction_calculations.py # Fitting equivalent lengths │ ├── system_pressure_checker.py # Pressure rating checks │ ├── pressure_temp_converter.py # ΔP⇄ΔT + sat pressure/temp tool │ ├── pipe_length_volume_calc.py # Internal volume calculator │ ├── save_load_manager.py # JSON project save/load │ └── network_builder.py # Frontend circuit network logic

---

## ⚙️ How to Run

pip install -r requirements.txt
streamlit run app.py

Then open the app in your browser at:
http://localhost:8501

---

## ❄️ Supported Refrigerants

- R404A
- R134a
- R407F
- R744 (CO₂)
- R410A
- R407C
- R507A
- R448A
- R449A
- R22
- R32
- R454A

✅ All include temperature-dependent: enthalpy, density, pressure, specific heat

---

## 📌 Notes

- Based on professional refrigeration physics (no toy assumptions)
- Cp-based superheat enthalpy logic for correct mass flow
- Oil return + velocity rules align with industrial norms
- Compatible with metric and imperial pipe standards
- Fully replicates and modernizes old VB tool logic

---

## 💾 Save/Load

- Projects are saved as JSON files, preserving all circuit data, temperatures, and refrigerant selections.
- You can build a full network and return to it later.

---

## ✅ Status

- All core and utility tools now implemented.
- Streamlit front end complete.
- Codebase ready for packaging or deployment.

---

## 🛠️ Built With

- Python 3.9+
- Streamlit
- NumPy + Pandas

---

© 2025 – Built with brutal thermodynamic honesty.

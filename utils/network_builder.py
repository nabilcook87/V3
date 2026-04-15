import streamlit as st
from utils.refrigerant_properties import RefrigerantProperties
from utils.pipe_sizing import PipeSizer
from utils.pipe_length_volume_calc import calculate_pipe_volume_liters
from utils.save_load_manager import SaveLoadManager

class NetworkBuilder:
    def __init__(self):
        if "circuits" not in st.session_state:
            st.session_state.circuits = []

        self.circuits = st.session_state.circuits
        self.network_name = ""
        self.network_type = "Dry Suction"
        self.refrigerant = "R404A"
        self.evaporating_temp = -10.0
        self.condensing_temp = 40.0
        self.superheat = 5.0
        self.subcooling = 3.0
        self.max_temp_penalty_K = 1.0
        self.pipe_sizer = PipeSizer()
        self.refrigerant_props = RefrigerantProperties()

    def run(self):
        st.subheader("Pipe Network Configuration")
        self.network_name = st.text_input("Network Name", value=self.network_name)
        self.network_type = st.selectbox("Network Type", [
            "Dry Suction", "Wet Suction", "Discharge", "Liquid",
            "Condenser Drain", "Pumped Refrigerant Liquid"
        ])
        self.refrigerant = st.selectbox("Refrigerant", [
            "R404A", "R134a", "R407F", "R744", "R410A",
            "R407C", "R507A", "R448A", "R449A", "R22", "R32", "R454A"
        ])
        self.evaporating_temp = st.number_input("Evaporating Temperature (°C)", value=self.evaporating_temp)
        self.condensing_temp = st.number_input("Condensing Temperature (°C)", value=self.condensing_temp)
        self.superheat = st.number_input("Superheat (K)", value=self.superheat)
        self.subcooling = st.number_input("Subcooling (K)", value=self.subcooling)
        self.max_temp_penalty_K = st.number_input("Max Temperature Penalty (K)", value=self.max_temp_penalty_K)

        st.markdown("---")
        st.subheader("Circuits in Network")

        if st.button("Add New Circuit"):
            self.add_circuit()

        total_refrigerant_mass = 0.0

        for idx, circuit in enumerate(self.circuits):
            st.markdown(f"#### Circuit {idx+1}")
            updated_circuit = self.edit_circuit(circuit, idx)
            st.session_state.circuits[idx] = updated_circuit

            try:
                results = self.pipe_sizer.size_pipe(
                    self.refrigerant,
                    self.network_type,
                    self.evaporating_temp,
                    self.condensing_temp,
                    self.superheat,
                    self.subcooling,
                    updated_circuit["length_m"],
                    updated_circuit["evap_capacity_kw"],
                    updated_circuit["fixed_pipe_size"],
                    updated_circuit["has_riser"],
                    updated_circuit["fittings_equivalent_length_m"]
                )
                updated_circuit["results"] = results

                pipe_id_mm = results.get("selected_pipe", {}).get("ID_mm")
                if pipe_id_mm is None:
                    raise ValueError("Missing ID_mm in selected pipe.")

                volume_liters = calculate_pipe_volume_liters(pipe_id_mm, updated_circuit["length_m"])
                props = self.refrigerant_props.get_properties(self.refrigerant, self.evaporating_temp)
                density = props["density_vapor"] if self.network_type in ["Dry Suction", "Discharge"] else props["density_liquid"]
                updated_circuit["refrigerant_mass_kg"] = volume_liters / 1000 * density
                total_refrigerant_mass += updated_circuit["refrigerant_mass_kg"]

                st.success(f"✅ Pipe: {results['selected_pipe']['Nominal Size (inch)']} | Velocity: {results['velocity_m_s']:.2f} m/s | ΔP: {results['pressure_drop_total_kpa']:.2f} kPa")
                st.write(f"🔸 Refrigerant Mass in Pipe: **{updated_circuit['refrigerant_mass_kg']:.2f} kg**")

            except Exception as e:
                st.error(f"Error: {e}")

        st.markdown("---")
        st.write(f"🔷 **Total Refrigerant Mass in Network:** {total_refrigerant_mass:.2f} kg")

        if st.button("Save Project"):
            SaveLoadManager().save_project(self, self.network_name)

        if st.button("Clear All Circuits"):
            st.session_state.circuits = []

    def add_circuit(self):
        st.session_state.circuits.append({
            "name": f"Circuit {len(self.circuits)+1}",
            "pipe_type": self.network_type,
            "length_m": 10.0,
            "evap_capacity_kw": 10.0,
            "fixed_pipe_size": None,
            "has_riser": False,
            "fittings_equivalent_length_m": 2.0,
            "circuit_notes": "",
            "refrigerant_mass_kg": 0.0
        })

    def edit_circuit(self, circuit, idx):
        circuit["name"] = st.text_input("Circuit Name", circuit["name"], key=f"name_{idx}")
        circuit["length_m"] = st.number_input("Straight Length (m)", value=circuit["length_m"], key=f"length_{idx}")
        circuit["evap_capacity_kw"] = st.number_input("Evaporator Capacity (kW)", value=circuit["evap_capacity_kw"], key=f"capacity_{idx}")
        circuit["has_riser"] = st.checkbox("Riser", value=circuit.get("has_riser", False), key=f"riser_{idx}")
        circuit["fittings_equivalent_length_m"] = st.number_input("Fittings Equivalent Length (m)", value=circuit["fittings_equivalent_length_m"], key=f"fittings_{idx}")

        manual_size = st.checkbox("Manually Fix Pipe Size?", value=circuit["fixed_pipe_size"] is not None, key=f"manual_{idx}")
        if manual_size:
            circuit["fixed_pipe_size"] = st.text_input("Fixed Pipe Size (inches)", circuit.get("fixed_pipe_size", ""), key=f"fixed_size_{idx}")
        else:
            circuit["fixed_pipe_size"] = None

        circuit["circuit_notes"] = st.text_area("Circuit Notes", value=circuit["circuit_notes"], key=f"notes_{idx}")
        return circuit

"""
Ordinary differential equation system for compartmental modeling.

Implements the cable equation and ionic current models for
mesh-based neuronal compartments.
"""

import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Any
from scipy.integrate import solve_ivp
from .core import Neuron, Compartment
import warnings


class ODESystem:
    """
    Implements the ODE system for compartmental neuronal modeling.
    """

    def __init__(self, neuron: Neuron):
        self.neuron = neuron
        self.compartment_ids = list(neuron.compartment_graph.compartments.keys())
        self.num_compartments = len(self.compartment_ids)
        self.id_to_index = {comp_id: i for i, comp_id in enumerate(self.compartment_ids)}

        # Default biophysical parameters
        self.default_params = {
            "temperature": 6.3,  # °C
            "capacitance": 1.0,  # µF/cm²
            "leak_conductance": 0.0003,  # S/cm²
            "leak_reversal": -54.3,  # mV
            "na_conductance": 0.12,  # S/cm²
            "na_reversal": 50.0,  # mV
            "k_conductance": 0.036,  # S/cm²
            "k_reversal": -77.0,  # mV
        }

        # State variables: [V1, V2, ..., Vn, m1, h1, n1, m2, h2, n2, ...]
        # Each compartment has V (voltage) + 3 gating variables (m, h, n)
        self.state_size = 4 * self.num_compartments

        # Stimulation protocols
        self.stimuli = []

    def set_parameters(self, **params):
        """Set biophysical parameters."""
        self.default_params.update(params)

    def add_stimulus(
        self, compartment_id: str, start_time: float, duration: float, amplitude: float, stimulus_type: str = "current"
    ):
        """
        Add a stimulus to a compartment.

        Args:
            compartment_id: Target compartment ID
            start_time: Stimulus start time (ms)
            duration: Stimulus duration (ms)
            amplitude: Stimulus amplitude (nA for current, mV for voltage)
            stimulus_type: 'current' or 'voltage'
        """
        if compartment_id not in self.compartment_ids:
            raise ValueError(f"Compartment {compartment_id} not found")

        stimulus = {
            "compartment_id": compartment_id,
            "compartment_index": self.id_to_index[compartment_id],
            "start_time": start_time,
            "end_time": start_time + duration,
            "amplitude": amplitude,
            "type": stimulus_type,
        }

        self.stimuli.append(stimulus)

    def get_initial_conditions(self) -> np.ndarray:
        """Get initial conditions for the ODE system."""
        y0 = np.zeros(self.state_size)

        # Set initial voltages
        for i, comp_id in enumerate(self.compartment_ids):
            compartment = self.neuron.get_compartment(comp_id)
            y0[i] = compartment.membrane_potential

        # Set initial gating variables (steady-state at rest)
        v_rest = -65.0  # mV
        for i in range(self.num_compartments):
            voltage_idx = i
            gating_start_idx = self.num_compartments + 3 * i

            # Use resting potential for initial gating variables
            m_inf = self._m_inf(v_rest)
            h_inf = self._h_inf(v_rest)
            n_inf = self._n_inf(v_rest)

            y0[gating_start_idx] = m_inf  # m
            y0[gating_start_idx + 1] = h_inf  # h
            y0[gating_start_idx + 2] = n_inf  # n

        return y0

    def ode_function(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        ODE function for the compartmental model.

        Args:
            t: Current time (ms)
            y: State vector [V1, V2, ..., Vn, m1, h1, n1, m2, h2, n2, ...]

        Returns:
            Time derivatives dy/dt
        """
        dydt = np.zeros_like(y)

        # Extract voltages and gating variables
        voltages = y[: self.num_compartments]
        gating_vars = y[self.num_compartments :].reshape(self.num_compartments, 3)

        # Compute voltage derivatives (cable equation)
        for i, comp_id in enumerate(self.compartment_ids):
            compartment = self.neuron.get_compartment(comp_id)

            # Membrane currents
            V = voltages[i]
            m, h, n = gating_vars[i]

            # Ionic currents (Hodgkin-Huxley)
            I_Na = self._sodium_current(V, m, h, compartment.area)
            I_K = self._potassium_current(V, n, compartment.area)
            I_leak = self._leak_current(V, compartment.area)
            I_ion = I_Na + I_K + I_leak

            # Axial currents from neighboring compartments
            I_axial = 0.0
            neighbors = self.neuron.compartment_graph.get_neighbors(comp_id)
            for neighbor_id in neighbors:
                neighbor_idx = self.id_to_index[neighbor_id]
                V_neighbor = voltages[neighbor_idx]
                conductance = self.neuron.compartment_graph.get_connection_conductance(comp_id, neighbor_id)
                I_axial += conductance * (V_neighbor - V)

            # Applied stimulus
            I_stim = self._get_stimulus_current(t, comp_id)

            # Membrane equation: C_m * dV/dt = -I_ion + I_axial + I_stim
            capacitance = self.default_params["capacitance"] * compartment.area * 1e-8  # µF
            dydt[i] = (-I_ion + I_axial + I_stim) / capacitance

        # Compute gating variable derivatives
        for i in range(self.num_compartments):
            V = voltages[i]
            m, h, n = gating_vars[i]

            gating_start_idx = self.num_compartments + 3 * i

            # Sodium activation (m)
            dydt[gating_start_idx] = (self._m_inf(V) - m) / self._tau_m(V)

            # Sodium inactivation (h)
            dydt[gating_start_idx + 1] = (self._h_inf(V) - h) / self._tau_h(V)

            # Potassium activation (n)
            dydt[gating_start_idx + 2] = (self._n_inf(V) - n) / self._tau_n(V)

        return dydt

    def _sodium_current(self, V: float, m: float, h: float, area: float) -> float:
        """Sodium current (nA)."""
        g_Na = self.default_params["na_conductance"] * area * 1e-8  # S
        E_Na = self.default_params["na_reversal"]
        return g_Na * m**3 * h * (V - E_Na) * 1e9  # nA

    def _potassium_current(self, V: float, n: float, area: float) -> float:
        """Potassium current (nA)."""
        g_K = self.default_params["k_conductance"] * area * 1e-8  # S
        E_K = self.default_params["k_reversal"]
        return g_K * n**4 * (V - E_K) * 1e9  # nA

    def _leak_current(self, V: float, area: float) -> float:
        """Leak current (nA)."""
        g_leak = self.default_params["leak_conductance"] * area * 1e-8  # S
        E_leak = self.default_params["leak_reversal"]
        return g_leak * (V - E_leak) * 1e9  # nA

    def _get_stimulus_current(self, t: float, compartment_id: str) -> float:
        """Get stimulus current for a compartment at time t."""
        I_stim = 0.0

        for stimulus in self.stimuli:
            if (
                stimulus["compartment_id"] == compartment_id
                and stimulus["start_time"] <= t <= stimulus["end_time"]
                and stimulus["type"] == "current"
            ):
                I_stim += stimulus["amplitude"]

        return I_stim

    # Hodgkin-Huxley gating variables (temperature-corrected)
    def _alpha_m(self, V: float) -> float:
        """Sodium activation rate constant."""
        temp_factor = 3.0 ** ((self.default_params["temperature"] - 6.3) / 10.0)
        return temp_factor * 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))

    def _beta_m(self, V: float) -> float:
        """Sodium activation rate constant."""
        temp_factor = 3.0 ** ((self.default_params["temperature"] - 6.3) / 10.0)
        return temp_factor * 4.0 * np.exp(-(V + 65.0) / 18.0)

    def _alpha_h(self, V: float) -> float:
        """Sodium inactivation rate constant."""
        temp_factor = 3.0 ** ((self.default_params["temperature"] - 6.3) / 10.0)
        return temp_factor * 0.07 * np.exp(-(V + 65.0) / 20.0)

    def _beta_h(self, V: float) -> float:
        """Sodium inactivation rate constant."""
        temp_factor = 3.0 ** ((self.default_params["temperature"] - 6.3) / 10.0)
        return temp_factor * 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))

    def _alpha_n(self, V: float) -> float:
        """Potassium activation rate constant."""
        temp_factor = 3.0 ** ((self.default_params["temperature"] - 6.3) / 10.0)
        return temp_factor * 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))

    def _beta_n(self, V: float) -> float:
        """Potassium activation rate constant."""
        temp_factor = 3.0 ** ((self.default_params["temperature"] - 6.3) / 10.0)
        return temp_factor * 0.125 * np.exp(-(V + 65.0) / 80.0)

    def _m_inf(self, V: float) -> float:
        """Steady-state sodium activation."""
        alpha = self._alpha_m(V)
        beta = self._beta_m(V)
        return alpha / (alpha + beta)

    def _h_inf(self, V: float) -> float:
        """Steady-state sodium inactivation."""
        alpha = self._alpha_h(V)
        beta = self._beta_h(V)
        return alpha / (alpha + beta)

    def _n_inf(self, V: float) -> float:
        """Steady-state potassium activation."""
        alpha = self._alpha_n(V)
        beta = self._beta_n(V)
        return alpha / (alpha + beta)

    def _tau_m(self, V: float) -> float:
        """Sodium activation time constant."""
        alpha = self._alpha_m(V)
        beta = self._beta_m(V)
        return 1.0 / (alpha + beta)

    def _tau_h(self, V: float) -> float:
        """Sodium inactivation time constant."""
        alpha = self._alpha_h(V)
        beta = self._beta_h(V)
        return 1.0 / (alpha + beta)

    def _tau_n(self, V: float) -> float:
        """Potassium activation time constant."""
        alpha = self._alpha_n(V)
        beta = self._beta_n(V)
        return 1.0 / (alpha + beta)

    def get_steady_state_voltages(self) -> Dict[str, float]:
        """Compute steady-state voltages for all compartments."""
        # For now, return resting potential
        # Could be improved with steady-state analysis
        return {comp_id: -65.0 for comp_id in self.compartment_ids}

    def validate_system(self) -> Dict[str, List[str]]:
        """Validate the ODE system setup."""
        issues = {"errors": [], "warnings": []}

        # Check for compartments without connections
        isolated_comps = []
        for comp_id in self.compartment_ids:
            neighbors = self.neuron.compartment_graph.get_neighbors(comp_id)
            if not neighbors:
                isolated_comps.append(comp_id)

        if isolated_comps:
            issues["warnings"].append(f"Found {len(isolated_comps)} isolated compartments")

        # Check for very small compartments
        small_comps = []
        for comp_id in self.compartment_ids:
            comp = self.neuron.get_compartment(comp_id)
            if comp.area < 0.01:  # µm²
                small_comps.append(comp_id)

        if small_comps:
            issues["warnings"].append(f"Found {len(small_comps)} very small compartments")

        # Check stimulus targets
        for stimulus in self.stimuli:
            if stimulus["compartment_id"] not in self.compartment_ids:
                issues["errors"].append(f"Stimulus targets non-existent compartment: {stimulus['compartment_id']}")

        return issues

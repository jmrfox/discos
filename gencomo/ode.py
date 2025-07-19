"""
Ordinary differential equation system for compartmental modeling.

Implements the cable equation and ionic current models for
mesh-based neuronal compartments.
"""

import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Any
from scipy.integrate import solve_ivp
import warnings


class ODESystem:
    """
    Implements the ODE system for compartmental neuronal modeling.
    """

    def __init__(self, graph):
        self.graph = graph
        self.compartment_ids = list(graph.nodes())
        self.num_compartments = len(self.compartment_ids)
        self.id_to_index = {comp_id: i for i, comp_id in enumerate(self.compartment_ids)}

        # Default biophysical parameters
        self.default_params = {
            "temperature": 279.45,  # K (6.3°C + 273.15)
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
    
    def __str__(self) -> str:
        """String representation of ODESystem."""
        return f"ODESystem({self.num_compartments} compartments, {len(self.stimuli)} stimuli)"
    
    def __repr__(self) -> str:
        """Detailed representation of ODESystem."""
        return self.__str__()
    
    def print_summary(self) -> None:
        """Print a detailed summary of the ODE system."""
        print(f"🧠 ODESystem Summary")
        print(f"   Compartments: {self.num_compartments}")
        print(f"   State variables: {self.state_size} (4 per compartment: V, m, h, n)")
        print(f"   Stimuli: {len(self.stimuli)}")
        print(f"   Graph edges: {len(self.graph.edges())}")
        
        print(f"\n📊 Biophysical Parameters:")
        for param, value in self.default_params.items():
            print(f"   {param}: {value}")
        
        if self.stimuli:
            print(f"\n⚡ Stimuli:")
            for i, stim in enumerate(self.stimuli):
                print(f"   {i+1}. {stim['compartment_id']}: {stim['amplitude']} {stim['type']} from {stim['start_time']}-{stim['end_time']} ms")
    
    def get_compartment_info(self, compartment_id: str) -> Dict[str, Any]:
        """Get information about a specific compartment."""
        if compartment_id not in self.compartment_ids:
            raise ValueError(f"Compartment {compartment_id} not found")
        
        # Get properties from the graph node
        props = dict(self.graph.nodes[compartment_id])
        
        # Add computed properties
        neighbors = list(self.graph.neighbors(compartment_id))
        props['neighbors'] = neighbors
        props['num_neighbors'] = len(neighbors)
        props['index'] = self.id_to_index[compartment_id]
        
        return props

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

    def get_initial_conditions(self, verbose: bool = False) -> np.ndarray:
        """Get initial conditions for the ODE system."""
        if verbose:
            print("   Computing initial conditions...")

        y0 = np.zeros(self.state_size)

        # Set initial voltages
        for i, comp_id in enumerate(self.compartment_ids):
            # Get membrane potential from graph node properties, default to -65 mV if not set
            node_props = self.graph.nodes[comp_id]
            membrane_potential = node_props.get('membrane_potential', -65.0)
            y0[i] = membrane_potential
            if verbose:
                print(f"     Compartment {comp_id}: V₀ = {membrane_potential:.1f} mV")

        # Set initial gating variables (steady-state at rest)
        v_rest = -65.0  # mV
        if verbose:
            print(f"     Computing gating variables at V_rest = {v_rest} mV")

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

            if verbose:
                comp_id = self.compartment_ids[i]
                print(f"     Compartment {comp_id}: m₀={m_inf:.3f}, h₀={h_inf:.3f}, n₀={n_inf:.3f}")

        if verbose:
            print(f"   Initial state vector shape: {y0.shape}")

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

        # Check for numerical issues
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            print(f"⚠️  WARNING: NaN or Inf detected in state at t={t:.3f}")
            print(f"   State: {y}")
            return dydt  # Return zeros to avoid propagating NaN/Inf

        # Extract voltages and gating variables
        voltages = y[: self.num_compartments]
        gating_vars = y[self.num_compartments :].reshape(self.num_compartments, 3)

        # Check for extreme voltages that might cause numerical issues
        if np.any(np.abs(voltages) > 200):  # More than ±200 mV
            print(f"⚠️  WARNING: Extreme voltage detected at t={t:.3f}: {voltages}")

        # Compute voltage derivatives (cable equation)
        for i, comp_id in enumerate(self.compartment_ids):
            # Get compartment properties from graph node
            node_props = self.graph.nodes[comp_id]
            area = node_props.get('external_surface_area', 1.0)  # Default area if not specified

            # Membrane currents
            V = voltages[i]
            m, h, n = gating_vars[i]

            # Check gating variable bounds
            if not (0 <= m <= 1 and 0 <= h <= 1 and 0 <= n <= 1):
                print(f"⚠️  WARNING: Gating variables out of bounds at t={t:.3f}")
                print(f"   Compartment {comp_id}: m={m:.3f}, h={h:.3f}, n={n:.3f}")
                # Clip to valid range
                m = np.clip(m, 0, 1)
                h = np.clip(h, 0, 1)
                n = np.clip(n, 0, 1)

            try:
                # Ionic currents (Hodgkin-Huxley)
                I_Na = self._sodium_current(V, m, h, area)
                I_K = self._potassium_current(V, n, area)
                I_leak = self._leak_current(V, area)
                I_ion = I_Na + I_K + I_leak

                # Check for extreme currents
                if abs(I_ion) > 1e6:  # More than 1 nA
                    print(f"⚠️  WARNING: Large ionic current at t={t:.3f}: {I_ion:.2e} pA")

                # Axial currents from neighboring compartments
                I_axial = 0.0
                neighbors = list(self.graph.neighbors(comp_id))
                for neighbor_id in neighbors:
                    neighbor_idx = self.id_to_index[neighbor_id]
                    V_neighbor = voltages[neighbor_idx]
                    # Default conductance if not specified in edge data
                    edge_data = self.graph.get_edge_data(comp_id, neighbor_id, {})
                    conductance = edge_data.get('conductance', 1e-6)  # Default conductance in S
                    # Convert conductance to appropriate units if needed
                    I_axial += (
                        conductance * (V_neighbor - V) * 1000
                    )  # Assume conductance gives nA, convert to pA                # Applied stimulus
                I_stim = self._get_stimulus_current(t, comp_id)

                # Membrane equation: C_m * dV/dt = -I_ion + I_axial + I_stim
                # Units: area (μm²), capacitance (μF/cm²)
                # Convert area to cm²: μm² × 1e-8
                # Convert capacitance: μF/cm² × cm² = μF = 1e-6 F
                # Final capacitance in pF: μF × 1e6 = pF
                capacitance = self.default_params["capacitance"] * area * 1e-2  # pF

                if capacitance <= 0:
                    print(f"⚠️  WARNING: Zero or negative capacitance for compartment {comp_id}")
                    capacitance = 1.0  # Minimum capacitance in pF

                # I_total in pA, capacitance in pF, dV/dt in mV/ms
                I_total = -I_ion + I_axial + I_stim
                dVdt = I_total / capacitance  # mV/ms

                # Check for unrealistic membrane voltages (not derivatives - those can be large!)
                if abs(V) > 1000:  # More than ±1000 mV is unrealistic
                    print(f"⚠️  WARNING: Extreme membrane voltage at t={t:.3f}: V={V:.1f} mV")
                    print(f"   I_total={I_total:.2e} pA, C={capacitance:.2e} pF")

                # For debugging: show voltage info only if both derivative AND voltage are concerning
                if abs(dVdt) > 10000 and abs(V) > 200:  # Only warn if both derivative AND voltage are large
                    print(
                        f"🔍 DEBUG: Large derivative with high voltage at t={t:.3f}: dV/dt={dVdt:.1e} mV/ms, V={V:.1f} mV"
                    )

                dydt[i] = dVdt

            except Exception as e:
                print(f"❌ ERROR in voltage calculation for compartment {comp_id} at t={t:.3f}: {e}")
                dydt[i] = 0.0

        # Compute gating variable derivatives
        for i in range(self.num_compartments):
            V = voltages[i]
            m, h, n = gating_vars[i]

            gating_start_idx = self.num_compartments + 3 * i

            try:
                # Sodium activation (m)
                tau_m = self._tau_m(V)
                if tau_m > 0:
                    dydt[gating_start_idx] = (self._m_inf(V) - m) / tau_m
                else:
                    dydt[gating_start_idx] = 0.0

                # Sodium inactivation (h)
                tau_h = self._tau_h(V)
                if tau_h > 0:
                    dydt[gating_start_idx + 1] = (self._h_inf(V) - h) / tau_h
                else:
                    dydt[gating_start_idx + 1] = 0.0

                # Potassium activation (n)
                tau_n = self._tau_n(V)
                if tau_n > 0:
                    dydt[gating_start_idx + 2] = (self._n_inf(V) - n) / tau_n
                else:
                    dydt[gating_start_idx + 2] = 0.0

            except Exception as e:
                print(f"❌ ERROR in gating variable calculation for compartment {i} at t={t:.3f}: {e}")
                dydt[gating_start_idx] = 0.0
                dydt[gating_start_idx + 1] = 0.0
                dydt[gating_start_idx + 2] = 0.0

        # Final check for numerical issues in derivatives
        if np.any(np.isnan(dydt)) or np.any(np.isinf(dydt)):
            print(f"❌ ERROR: NaN or Inf in derivatives at t={t:.3f}")
            print(f"   Derivatives: {dydt}")
            dydt = np.zeros_like(dydt)  # Return zeros to avoid propagating NaN/Inf

        return dydt

    def _sodium_current(self, V: float, m: float, h: float, area: float) -> float:
        """Sodium current (pA)."""
        # area is in μm², conductance in S/cm²
        # Convert: μm² to cm² (×1e-8), then to pA (×1e12)
        g_Na = self.default_params["na_conductance"] * area * 1e-8  # S
        E_Na = self.default_params["na_reversal"]
        return g_Na * m**3 * h * (V - E_Na) * 1e12  # pA

    def _potassium_current(self, V: float, n: float, area: float) -> float:
        """Potassium current (pA)."""
        g_K = self.default_params["k_conductance"] * area * 1e-8  # S
        E_K = self.default_params["k_reversal"]
        return g_K * n**4 * (V - E_K) * 1e12  # pA

    def _leak_current(self, V: float, area: float) -> float:
        """Leak current (pA)."""
        g_leak = self.default_params["leak_conductance"] * area * 1e-8  # S
        E_leak = self.default_params["leak_reversal"]
        return g_leak * (V - E_leak) * 1e12  # pA

    def _get_stimulus_current(self, t: float, compartment_id: str) -> float:
        """Get stimulus current for a compartment at time t (pA)."""
        I_stim = 0.0

        for stimulus in self.stimuli:
            # Check for None values and ensure proper types
            start_time = stimulus.get("start_time", 0.0)
            end_time = stimulus.get("end_time", 0.0)
            amplitude = stimulus.get("amplitude", 0.0)
            stim_type = stimulus.get("type", "current")

            if (
                stimulus["compartment_id"] == compartment_id
                and start_time is not None
                and end_time is not None
                and start_time <= t <= end_time
                and stim_type == "current"
            ):
                # Convert stimulus from nA to pA
                I_stim += amplitude * 1000  # nA to pA

        return I_stim

    # Hodgkin-Huxley gating variables (temperature-corrected)
    def _alpha_m(self, V: float) -> float:
        """Sodium activation rate constant."""
        temp_factor = 3.0 ** ((self.default_params["temperature"] - 279.45) / 10.0)
        return temp_factor * 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))

    def _beta_m(self, V: float) -> float:
        """Sodium activation rate constant."""
        temp_factor = 3.0 ** ((self.default_params["temperature"] - 279.45) / 10.0)
        return temp_factor * 4.0 * np.exp(-(V + 65.0) / 18.0)

    def _alpha_h(self, V: float) -> float:
        """Sodium inactivation rate constant."""
        temp_factor = 3.0 ** ((self.default_params["temperature"] - 279.45) / 10.0)
        return temp_factor * 0.07 * np.exp(-(V + 65.0) / 20.0)

    def _beta_h(self, V: float) -> float:
        """Sodium inactivation rate constant."""
        temp_factor = 3.0 ** ((self.default_params["temperature"] - 279.45) / 10.0)
        return temp_factor * 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))

    def _alpha_n(self, V: float) -> float:
        """Potassium activation rate constant."""
        temp_factor = 3.0 ** ((self.default_params["temperature"] - 279.45) / 10.0)
        return temp_factor * 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))

    def _beta_n(self, V: float) -> float:
        """Potassium activation rate constant."""
        temp_factor = 3.0 ** ((self.default_params["temperature"] - 279.45) / 10.0)
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

        # Check graph connectivity
        if self.num_compartments == 0:
            issues["errors"].append("No compartments found in the neuron")
        elif self.num_compartments == 1:
            # Single compartment is valid (simplest case)
            pass
        else:
            # For multi-compartment systems, check connectivity
            import networkx as nx

            # The graph is already a NetworkX graph, so we can check connectivity directly
            if not nx.is_connected(self.graph):
                num_components = nx.number_connected_components(self.graph)
                components = list(nx.connected_components(self.graph))
                issues["errors"].append(
                    f"Graph has {num_components} disconnected components. "
                    f"All compartments must be connected. Components: {[list(comp) for comp in components]}"
                )

        # Check for very small compartments
        small_comps = []
        for comp_id in self.compartment_ids:
            node_props = self.graph.nodes[comp_id]
            area = node_props.get('external_surface_area', 1.0)
            if area < 0.01:  # µm²
                small_comps.append(comp_id)

        if small_comps:
            issues["warnings"].append(f"Found {len(small_comps)} very small compartments")

        # Check stimulus targets
        for stimulus in self.stimuli:
            if stimulus["compartment_id"] not in self.compartment_ids:
                issues["errors"].append(f"Stimulus targets non-existent compartment: {stimulus['compartment_id']}")

        return issues
    
    def solve_forward_euler(self, t_span: Tuple[float, float], dt: float = 0.01, 
                           verbose: bool = False) -> Dict[str, Any]:
        """
        Solve the ODE system using forward Euler method.
        
        Args:
            t_span: Time span (t_start, t_end) in ms
            dt: Time step in ms
            verbose: Print progress information
            
        Returns:
            Dictionary containing solution data
        """
        t_start, t_end = t_span
        t_points = np.arange(t_start, t_end + dt, dt)
        n_points = len(t_points)
        
        if verbose:
            print(f"🔄 Forward Euler solver")
            print(f"   Time span: {t_start} to {t_end} ms")
            print(f"   Time step: {dt} ms")
            print(f"   Total points: {n_points}")
        
        # Initialize solution arrays
        y_solution = np.zeros((n_points, self.state_size))
        y_solution[0] = self.get_initial_conditions(verbose=verbose)
        
        # Forward Euler integration
        for i in range(n_points - 1):
            t = t_points[i]
            y = y_solution[i]
            
            # Compute derivative
            dydt = self.ode_function(t, y)
            
            # Forward Euler step: y_{n+1} = y_n + dt * f(t_n, y_n)
            y_solution[i + 1] = y + dt * dydt
            
            if verbose and (i + 1) % max(1, n_points // 10) == 0:
                progress = (i + 1) / n_points * 100
                print(f"   Progress: {progress:.1f}%")
        
        # Extract voltages and gating variables
        voltages = y_solution[:, :self.num_compartments]
        gating_vars = y_solution[:, self.num_compartments:].reshape(n_points, self.num_compartments, 3)
        
        result = {
            'method': 'forward_euler',
            't': t_points,
            'y': y_solution,
            'voltages': voltages,
            'gating_variables': gating_vars,
            'compartment_ids': self.compartment_ids,
            'success': True,
            'message': f'Forward Euler integration completed with {n_points} points'
        }
        
        if verbose:
            print(f"✅ Integration completed successfully")
            print(f"   Final time: {t_points[-1]:.3f} ms")
            print(f"   Voltage range: {voltages.min():.1f} to {voltages.max():.1f} mV")
        
        return result
    
    def solve_scipy(self, t_span: Tuple[float, float], method: str = 'RK45', 
                   max_step: float = 0.1, rtol: float = 1e-6, atol: float = 1e-9,
                   verbose: bool = False) -> Dict[str, Any]:
        """
        Solve the ODE system using scipy.integrate.solve_ivp.
        
        Args:
            t_span: Time span (t_start, t_end) in ms
            method: Integration method ('RK45', 'DOP853', 'Radau', etc.)
            max_step: Maximum step size in ms
            rtol: Relative tolerance
            atol: Absolute tolerance
            verbose: Print progress information
            
        Returns:
            Dictionary containing solution data
        """
        if verbose:
            print(f"🔄 SciPy solver ({method})")
            print(f"   Time span: {t_span[0]} to {t_span[1]} ms")
            print(f"   Max step: {max_step} ms")
            print(f"   Tolerances: rtol={rtol}, atol={atol}")
        
        y0 = self.get_initial_conditions(verbose=verbose)
        
        # Solve using scipy
        sol = solve_ivp(
            self.ode_function, 
            t_span, 
            y0, 
            method=method,
            max_step=max_step,
            rtol=rtol,
            atol=atol,
            dense_output=True
        )
        
        if not sol.success:
            if verbose:
                print(f"❌ Integration failed: {sol.message}")
            return {
                'method': method,
                'success': False,
                'message': sol.message
            }
        
        # Extract solution
        t_points = sol.t
        y_solution = sol.y.T  # Transpose to match forward_euler format
        n_points = len(t_points)
        
        # Extract voltages and gating variables
        voltages = y_solution[:, :self.num_compartments]
        gating_vars = y_solution[:, self.num_compartments:].reshape(n_points, self.num_compartments, 3)
        
        result = {
            'method': method,
            't': t_points,
            'y': y_solution,
            'voltages': voltages,
            'gating_variables': gating_vars,
            'compartment_ids': self.compartment_ids,
            'success': True,
            'message': sol.message,
            'nfev': sol.nfev,  # Number of function evaluations
            'njev': sol.njev,  # Number of Jacobian evaluations
            'nlu': sol.nlu     # Number of LU decompositions
        }
        
        if verbose:
            print(f"✅ Integration completed successfully")
            print(f"   Final time: {t_points[-1]:.3f} ms")
            print(f"   Function evaluations: {sol.nfev}")
            print(f"   Voltage range: {voltages.min():.1f} to {voltages.max():.1f} mV")
        
        return result

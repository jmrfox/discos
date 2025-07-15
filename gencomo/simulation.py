"""
Simulation engine for GenCoMo compartmental models.

Provides high-level interface for running simulations and analyzing results.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from scipy.integrate import solve_ivp
from .core import Neuron
from .ode import ODESystem
import warnings
from dataclasses import dataclass
import time

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Note: tqdm not available, falling back to simple progress reporting")


@dataclass
class SimulationResult:
    """Container for simulation results."""

    time: np.ndarray
    voltages: Dict[str, np.ndarray]  # compartment_id -> voltage trace
    gating_variables: Dict[str, Dict[str, np.ndarray]]  # compartment_id -> {'m': array, 'h': array, 'n': array}
    success: bool
    message: str
    simulation_time: float
    parameters: Dict[str, Any]


class Simulator:
    """
    High-level simulation interface for GenCoMo models.
    """

    def __init__(self, neuron: Neuron):
        self.neuron = neuron
        self.ode_system = ODESystem(neuron)
        self.results = None

    def set_biophysics(self, **params):
        """Set biophysical parameters."""
        self.ode_system.set_parameters(**params)

    def add_stimulus(
        self, compartment_id: str, start_time: float, duration: float, amplitude: float, stimulus_type: str = "current"
    ):
        """Add stimulus to the simulation."""
        self.ode_system.add_stimulus(compartment_id, start_time, duration, amplitude, stimulus_type)

    def run_simulation(
        self,
        duration: float,
        dt: float = 0.025,
        method: str = "Radau",  # Better default for stiff HH equations
        rtol: float = 1e-5,  # More appropriate for HH
        atol: float = 1e-8,  # More appropriate for HH
        max_step: Optional[float] = None,
        progress_callback: Optional[callable] = None,
        verbose: bool = True,
    ) -> SimulationResult:
        """
        Run the compartmental simulation.

        Args:
            duration: Simulation duration (ms)
            dt: Time step for output (ms)
            method: Integration method ('RK45', 'DOP853', 'Radau', etc.)
            rtol: Relative tolerance
            atol: Absolute tolerance
            max_step: Maximum step size (ms)
            progress_callback: Function to call with progress updates
            verbose: Whether to print detailed progress information

        Returns:
            SimulationResult object
        """
        if verbose:
            print("=" * 80)
            print("🧠 GENCOMO SIMULATION STARTING")
            print("=" * 80)
            print(f"📊 Simulation Parameters:")
            print(f"   Duration: {duration} ms")
            print(f"   Time step: {dt} ms")
            print(f"   Method: {method}")
            print(f"   Compartments: {len(self.ode_system.compartment_ids)}")
            print(f"   State variables: {self.ode_system.state_size}")
            print(f"   Stimuli: {len(self.ode_system.stimuli)}")
            print(f"   Tolerances: rtol={rtol}, atol={atol}")
            if max_step:
                print(f"   Max step: {max_step} ms")

        start_time = time.time()

        # Validate system
        if verbose:
            print("\n🔍 Validating system...")
        validation = self.ode_system.validate_system()
        if validation["errors"]:
            if verbose:
                print(f"❌ Validation failed: {validation['errors']}")
            return SimulationResult(
                time=np.array([]),
                voltages={},
                gating_variables={},
                success=False,
                message=f"Validation errors: {validation['errors']}",
                simulation_time=0.0,
                parameters={},
            )

        if validation["warnings"]:
            for warning in validation["warnings"]:
                if verbose:
                    print(f"⚠️  Warning: {warning}")
                warnings.warn(warning)

        if verbose:
            print("✅ System validation passed")

        # Set up time points
        t_span = (0.0, duration)
        t_eval = np.arange(0.0, duration + dt, dt)

        if verbose:
            print(f"\n⏱️  Time setup:")
            print(f"   Time span: {t_span}")
            print(f"   Output points: {len(t_eval)}")
            print(f"   First few times: {t_eval[:5]}")

        # Get initial conditions
        if verbose:
            print("\n🎯 Computing initial conditions...")
        y0 = self.ode_system.get_initial_conditions(verbose=verbose)
        if verbose:
            print(f"   Initial state shape: {y0.shape}")
            print(f"   Initial voltages: {y0[:len(self.ode_system.compartment_ids)]}")

        # Test ODE function once
        if verbose:
            print("\n🧪 Testing ODE function...")
        try:
            dydt_test = self.ode_system.ode_function(0.0, y0)
            if verbose:
                print(f"   ODE derivatives shape: {dydt_test.shape}")
                print(f"   Initial voltage derivatives: {dydt_test[:len(self.ode_system.compartment_ids)]}")
                print("✅ ODE function test passed")
        except Exception as e:
            if verbose:
                print(f"❌ ODE function test failed: {e}")
            return SimulationResult(
                time=np.array([]),
                voltages={},
                gating_variables={},
                success=False,
                message=f"ODE function test failed: {str(e)}",
                simulation_time=0.0,
                parameters={},
            )

        # Progress tracking setup
        progress_bar = None
        last_progress_time = time.time()
        progress_step = 0

        if verbose and TQDM_AVAILABLE:
            progress_bar = tqdm(total=100, desc="🔬 Integration", unit="%", ncols=80)

        try:
            if verbose:
                print(f"\n🚀 Starting integration with {method}...")
                print(f"   This may take a while for large systems...")

            # Run integration with timeout monitoring
            kwargs = {
                "fun": self.ode_system.ode_function,
                "t_span": t_span,
                "y0": y0,
                "t_eval": t_eval,
                "method": method,
                "rtol": rtol,
                "atol": atol,
            }

            # Only add max_step if it's not None
            if max_step is not None:
                kwargs["max_step"] = max_step
            else:
                # Set a reasonable default max_step for HH equations
                # Typical action potential timescale is ~1ms, so limit steps to 0.1ms
                default_max_step = min(0.1, duration / 500)  # 0.1ms or 0.2% of duration
                kwargs["max_step"] = default_max_step
                if verbose:
                    print(f"   Using default max_step: {default_max_step:.4f} ms")

            # Monitor integration progress
            integration_start = time.time()

            if verbose:
                print("   Integration starting...")

            sol = solve_ivp(**kwargs)

            integration_end = time.time()
            integration_time = integration_end - integration_start

            if verbose:
                print(f"   Integration completed in {integration_time:.2f} seconds")

            simulation_time = time.time() - start_time

            if progress_bar:
                progress_bar.close()

            if verbose:
                print(f"\n✅ Integration completed!")
                print(f"   Success: {sol.success}")
                print(f"   Message: {sol.message}")
                print(f"   Total time: {simulation_time:.3f} seconds")
                print(f"   Solution shape: {sol.y.shape}")
                print(f"   Time points: {len(sol.t)}")

            if not sol.success:
                if verbose:
                    print(f"❌ Integration failed: {sol.message}")
                return SimulationResult(
                    time=np.array([]),
                    voltages={},
                    gating_variables={},
                    success=False,
                    message=f"Integration failed: {sol.message}",
                    simulation_time=simulation_time,
                    parameters={},
                )

            # Parse results
            if verbose:
                print("\n📊 Parsing solution...")
            voltages, gating_vars = self._parse_solution(sol)

            if verbose:
                print(f"   Extracted {len(voltages)} voltage traces")
                print(f"   Extracted gating variables for {len(gating_vars)} compartments")

            result = SimulationResult(
                time=sol.t,
                voltages=voltages,
                gating_variables=gating_vars,
                success=True,
                message="Simulation completed successfully",
                simulation_time=simulation_time,
                parameters={
                    "duration": duration,
                    "dt": dt,
                    "method": method,
                    "num_compartments": len(self.ode_system.compartment_ids),
                    "num_stimuli": len(self.ode_system.stimuli),
                },
            )

            self.results = result

            if verbose:
                print(f"\n🎉 Simulation completed successfully in {simulation_time:.3f} seconds!")
                print("=" * 80)

            return result

        except Exception as e:
            simulation_time = time.time() - start_time
            if progress_bar:
                progress_bar.close()
            if verbose:
                print(f"\n❌ Simulation failed: {str(e)}")
                import traceback

                traceback.print_exc()
            return SimulationResult(
                time=np.array([]),
                voltages={},
                gating_variables={},
                success=False,
                message=f"Simulation error: {str(e)}",
                simulation_time=simulation_time,
                parameters={},
            )

    def _parse_solution(self, sol) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]]:
        """Parse the solution into voltage and gating variable traces."""
        num_comps = self.ode_system.num_compartments

        # Extract voltages
        voltages = {}
        for i, comp_id in enumerate(self.ode_system.compartment_ids):
            voltages[comp_id] = sol.y[i, :]

        # Extract gating variables
        gating_variables = {}
        for i, comp_id in enumerate(self.ode_system.compartment_ids):
            gating_start_idx = num_comps + 3 * i
            gating_variables[comp_id] = {
                "m": sol.y[gating_start_idx, :],
                "h": sol.y[gating_start_idx + 1, :],
                "n": sol.y[gating_start_idx + 2, :],
            }

        return voltages, gating_variables

    def get_spike_times(self, compartment_id: str, threshold: float = 0.0) -> np.ndarray:
        """
        Detect spike times in a compartment.

        Args:
            compartment_id: Target compartment
            threshold: Spike detection threshold (mV)

        Returns:
            Array of spike times (ms)
        """
        if self.results is None or not self.results.success:
            raise ValueError("No successful simulation results available")

        if compartment_id not in self.results.voltages:
            raise ValueError(f"Compartment {compartment_id} not found in results")

        voltage = self.results.voltages[compartment_id]
        time = self.results.time

        # Find upward threshold crossings
        above_threshold = voltage > threshold
        spike_indices = np.where(np.diff(above_threshold.astype(int)) == 1)[0]

        return time[spike_indices] if len(spike_indices) > 0 else np.array([])

    def compute_propagation_velocity(
        self, start_compartment: str, end_compartment: str, threshold: float = 0.0
    ) -> Optional[float]:
        """
        Compute action potential propagation velocity between compartments.

        Args:
            start_compartment: Starting compartment ID
            end_compartment: Ending compartment ID
            threshold: Spike detection threshold (mV)

        Returns:
            Propagation velocity (m/s) or None if calculation fails
        """
        if self.results is None or not self.results.success:
            raise ValueError("No successful simulation results available")

        # Get spike times
        start_spikes = self.get_spike_times(start_compartment, threshold)
        end_spikes = self.get_spike_times(end_compartment, threshold)

        if len(start_spikes) == 0 or len(end_spikes) == 0:
            return None

        # Use first spike for velocity calculation
        start_time = start_spikes[0]
        end_time = end_spikes[0]

        # Get compartment positions
        start_comp = self.neuron.get_compartment(start_compartment)
        end_comp = self.neuron.get_compartment(end_compartment)

        # Compute distance
        distance = np.linalg.norm(end_comp.centroid - start_comp.centroid) * 1e-6  # µm to m

        # Compute velocity
        time_diff = (end_time - start_time) * 1e-3  # ms to s

        if time_diff <= 0:
            return None

        velocity = distance / time_diff  # m/s
        return velocity

    def analyze_results(self) -> Dict[str, Any]:
        """Analyze simulation results and return summary statistics."""
        if self.results is None or not self.results.success:
            raise ValueError("No successful simulation results available")

        analysis = {
            "simulation_info": {
                "duration": self.results.time[-1] if len(self.results.time) > 0 else 0,
                "num_time_points": len(self.results.time),
                "num_compartments": len(self.results.voltages),
                "simulation_time": self.results.simulation_time,
            },
            "voltage_stats": {},
            "spike_analysis": {},
            "compartment_analysis": {},
        }

        # Voltage statistics for each compartment
        for comp_id, voltage in self.results.voltages.items():
            analysis["voltage_stats"][comp_id] = {
                "mean": np.mean(voltage),
                "std": np.std(voltage),
                "min": np.min(voltage),
                "max": np.max(voltage),
                "range": np.max(voltage) - np.min(voltage),
            }

            # Spike analysis
            spikes = self.get_spike_times(comp_id)
            analysis["spike_analysis"][comp_id] = {
                "num_spikes": len(spikes),
                "spike_times": spikes.tolist() if len(spikes) > 0 else [],
                "mean_firing_rate": len(spikes) / (self.results.time[-1] * 1e-3) if len(self.results.time) > 0 else 0,
            }

        # Overall analysis
        all_spikes = []
        for comp_spikes in analysis["spike_analysis"].values():
            all_spikes.extend(comp_spikes["spike_times"])

        analysis["overall"] = {
            "total_spikes": len(all_spikes),
            "active_compartments": sum(1 for comp in analysis["spike_analysis"].values() if comp["num_spikes"] > 0),
            "max_voltage_range": (
                max(stats["range"] for stats in analysis["voltage_stats"].values()) if analysis["voltage_stats"] else 0
            ),
        }

        return analysis

    def save_results(self, filepath: str, format: str = "npz"):
        """
        Save simulation results to file.

        Args:
            filepath: Output file path
            format: File format ('npz', 'hdf5', 'csv')
        """
        if self.results is None or not self.results.success:
            raise ValueError("No successful simulation results available")

        if format == "npz":
            self._save_npz(filepath)
        elif format == "hdf5":
            self._save_hdf5(filepath)
        elif format == "csv":
            self._save_csv(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _save_npz(self, filepath: str):
        """Save results as NPZ file."""
        save_dict = {"time": self.results.time, "compartment_ids": list(self.results.voltages.keys())}

        # Add voltage traces
        for comp_id, voltage in self.results.voltages.items():
            save_dict[f"voltage_{comp_id}"] = voltage

        # Add gating variables
        for comp_id, gating in self.results.gating_variables.items():
            for var_name, var_data in gating.items():
                save_dict[f"{var_name}_{comp_id}"] = var_data

        np.savez_compressed(filepath, **save_dict)

    def _save_hdf5(self, filepath: str):
        """Save results as HDF5 file."""
        try:
            import h5py

            with h5py.File(filepath, "w") as f:
                f.create_dataset("time", data=self.results.time)

                # Voltage group
                voltage_group = f.create_group("voltages")
                for comp_id, voltage in self.results.voltages.items():
                    voltage_group.create_dataset(comp_id, data=voltage)

                # Gating variables group
                gating_group = f.create_group("gating_variables")
                for comp_id, gating in self.results.gating_variables.items():
                    comp_group = gating_group.create_group(comp_id)
                    for var_name, var_data in gating.items():
                        comp_group.create_dataset(var_name, data=var_data)

                # Metadata
                f.attrs["success"] = self.results.success
                f.attrs["message"] = self.results.message
                f.attrs["simulation_time"] = self.results.simulation_time

        except ImportError:
            raise ImportError("h5py required for HDF5 format")

    def _save_csv(self, filepath: str):
        """Save results as CSV file."""
        import pandas as pd

        # Create dataframe
        data = {"time": self.results.time}

        # Add voltage columns
        for comp_id, voltage in self.results.voltages.items():
            data[f"V_{comp_id}"] = voltage

        # Add gating variable columns
        for comp_id, gating in self.results.gating_variables.items():
            for var_name, var_data in gating.items():
                data[f"{var_name}_{comp_id}"] = var_data

        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)

    def load_results(self, filepath: str, format: str = "npz") -> SimulationResult:
        """Load simulation results from file."""
        if format == "npz":
            return self._load_npz(filepath)
        elif format == "hdf5":
            return self._load_hdf5(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _load_npz(self, filepath: str) -> SimulationResult:
        """Load results from NPZ file."""
        data = np.load(filepath)

        time = data["time"]
        compartment_ids = data["compartment_ids"].tolist()

        voltages = {}
        gating_variables = {}

        for comp_id in compartment_ids:
            voltages[comp_id] = data[f"voltage_{comp_id}"]
            gating_variables[comp_id] = {
                "m": data[f"m_{comp_id}"],
                "h": data[f"h_{comp_id}"],
                "n": data[f"n_{comp_id}"],
            }

        return SimulationResult(
            time=time,
            voltages=voltages,
            gating_variables=gating_variables,
            success=True,
            message="Loaded from file",
            simulation_time=0.0,
            parameters={},
        )

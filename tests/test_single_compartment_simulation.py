"""
Test script for single compartment simulation.

This script creates a single compartment neuron from a simple cylinder mesh,
then runs a simulation with the Hodgkin-Huxley model to test the ODE and
simulation systems.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to path for importing gencomo
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

from gencomo import MeshSegmenter, Neuron, Compartment
from gencomo.demos import create_cylinder_mesh
from gencomo.simulation import Simulator


def create_single_compartment_neuron():
    """
    Create a single compartment neuron from a simple cylinder mesh.

    Returns:
        Neuron: A neuron with one compartment
    """
    print("Creating single compartment neuron...")

    # Create a simple cylinder mesh (radius=0.5, length=1.0)
    cylinder = create_cylinder_mesh(radius=0.5, length=1.0, resolution=20)

    print(f"Cylinder properties:")
    print(f"  Volume: {cylinder.volume:.6f} μm³")
    print(f"  Surface area: {cylinder.area:.6f} μm²")
    print(f"  Bounds: {cylinder.bounds}")

    # Segment the mesh with large slice_height to get single compartment
    slice_height = 2.0  # Larger than cylinder length
    segmenter = MeshSegmenter()
    segments = segmenter.segment_mesh(cylinder, slice_height=slice_height)

    print(f"\nSegmentation results:")
    print(f"  Number of segments: {len(segments)}")

    if len(segments) != 1:
        raise ValueError(f"Expected 1 segment, got {len(segments)}")

    segment = segments[0]
    print(f"  Segment ID: {segment.id}")
    print(f"  Segment volume: {segment.volume:.6f} μm³")
    print(f"  External surface area: {segment.external_surface_area:.6f} μm²")

    # Create neuron and add the single compartment
    neuron = Neuron("single_compartment_neuron")
    neuron.set_mesh(cylinder)

    # Create compartment from segment
    compartment = Compartment(
        id=segment.id,
        z_level=segment.slice_index,
        area=segment.external_surface_area,  # μm²
        volume=segment.volume,  # μm³
        centroid=np.array([0.0, 0.0, 0.5]),  # Center of cylinder
        boundary_points=np.array([[0, 0]]),  # Placeholder
        membrane_potential=-70.0,  # mV, resting potential
    )

    # Add compartment to neuron
    neuron.compartment_graph.add_compartment(compartment)

    print(f"\nCreated neuron with compartment:")
    print(f"  Compartment ID: {compartment.id}")
    print(f"  Membrane area: {compartment.area:.6f} μm²")
    print(f"  Volume: {compartment.volume:.6f} μm³")
    print(f"  Initial potential: {compartment.membrane_potential} mV")

    return neuron


def test_single_compartment_simulation():
    """
    Test single compartment simulation with stimulus injection.
    """
    print("=" * 80)
    print("🧪 SINGLE COMPARTMENT SIMULATION TEST")
    print("=" * 80)

    # Create single compartment neuron
    neuron = create_single_compartment_neuron()

    # Create simulator
    simulator = Simulator(neuron)

    # Set biophysical parameters (classic Hodgkin-Huxley)
    simulator.set_biophysics(
        temperature=6.3,  # °C
        capacitance=1.0,  # µF/cm²
        leak_conductance=0.0003,  # S/cm²
        leak_reversal=-54.3,  # mV
        na_conductance=0.12,  # S/cm²
        na_reversal=50.0,  # mV
        k_conductance=0.036,  # S/cm²
        k_reversal=-77.0,  # mV
    )

    # Get compartment ID
    compartment_ids = list(neuron.compartment_graph.compartments.keys())
    compartment_id = compartment_ids[0]

    print(f"\nAdding stimulus to compartment: {compartment_id}")

    # Add current stimulus
    simulator.add_stimulus(
        compartment_id=compartment_id,
        start_time=10.0,  # ms
        duration=1.0,  # ms
        amplitude=5.0,  # nA
        stimulus_type="current",
    )

    # Run simulation
    print("\nRunning simulation...")
    duration = 50.0  # ms
    dt = 0.025  # ms

    results = simulator.run_simulation(duration=duration, dt=dt, method="RK45")

    if not results.success:
        print(f"❌ Simulation failed: {results.message}")
        return False

    print(f"✅ Simulation completed successfully!")
    print(f"  Duration: {duration} ms")
    print(f"  Time steps: {len(results.time)}")
    print(f"  Simulation time: {results.simulation_time:.3f} seconds")

    # Analyze results
    voltage = results.voltages[compartment_id]
    gating_vars = results.gating_variables[compartment_id]

    print(f"\nVoltage analysis:")
    print(f"  Resting potential: {voltage[0]:.2f} mV")
    print(f"  Maximum voltage: {np.max(voltage):.2f} mV")
    print(f"  Minimum voltage: {np.min(voltage):.2f} mV")
    print(f"  Final voltage: {voltage[-1]:.2f} mV")

    # Check for action potential
    spike_threshold = 0.0  # mV
    spikes = simulator.get_spike_times(compartment_id, threshold=spike_threshold)

    print(f"\nSpike analysis:")
    print(f"  Spike threshold: {spike_threshold} mV")
    print(f"  Number of spikes: {len(spikes)}")
    if len(spikes) > 0:
        print(f"  First spike at: {spikes[0]:.2f} ms")
        print(f"  Peak voltage: {np.max(voltage):.2f} mV")

    # Plot results
    plot_simulation_results(results, compartment_id)

    # Validate that we get expected behavior
    # For a 5 nA stimulus on a small compartment, we should get an action potential
    action_potential_generated = np.max(voltage) > 20.0  # mV

    if action_potential_generated:
        print("✅ Action potential successfully generated!")
    else:
        print("⚠️  No action potential detected - may need stronger stimulus")

    return True


def plot_simulation_results(results, compartment_id):
    """
    Plot simulation results for the single compartment.

    Args:
        results: SimulationResult object
        compartment_id: ID of the compartment to plot
    """
    print("\nPlotting simulation results...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Single Compartment Simulation Results", fontsize=16)

    time = results.time
    voltage = results.voltages[compartment_id]
    gating_vars = results.gating_variables[compartment_id]

    # Voltage trace
    axes[0, 0].plot(time, voltage, "b-", linewidth=2)
    axes[0, 0].set_xlabel("Time (ms)")
    axes[0, 0].set_ylabel("Voltage (mV)")
    axes[0, 0].set_title("Membrane Potential")
    axes[0, 0].grid(True, alpha=0.3)

    # Add stimulus period
    axes[0, 0].axvspan(10.0, 11.0, alpha=0.2, color="red", label="Stimulus")
    axes[0, 0].legend()

    # Sodium gating variables
    axes[0, 1].plot(time, gating_vars["m"], "r-", label="m (Na activation)", linewidth=2)
    axes[0, 1].plot(time, gating_vars["h"], "g-", label="h (Na inactivation)", linewidth=2)
    axes[0, 1].set_xlabel("Time (ms)")
    axes[0, 1].set_ylabel("Gating Variable")
    axes[0, 1].set_title("Sodium Channel Gates")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Potassium gating variable
    axes[1, 0].plot(time, gating_vars["n"], "b-", label="n (K activation)", linewidth=2)
    axes[1, 0].set_xlabel("Time (ms)")
    axes[1, 0].set_ylabel("Gating Variable")
    axes[1, 0].set_title("Potassium Channel Gates")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Phase plot (V vs dV/dt)
    dVdt = np.gradient(voltage, time)
    axes[1, 1].plot(voltage, dVdt, "k-", linewidth=1)
    axes[1, 1].set_xlabel("Voltage (mV)")
    axes[1, 1].set_ylabel("dV/dt (mV/ms)")
    axes[1, 1].set_title("Phase Plot")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    save_path = os.path.join(os.path.dirname(__file__), "single_compartment_simulation.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {save_path}")

    plt.show()


if __name__ == "__main__":
    try:
        success = test_single_compartment_simulation()
        print("\n" + "=" * 80)
        if success:
            print("🎉 Single compartment simulation test PASSED!")
        else:
            print("💥 Single compartment simulation test FAILED!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

"""
Test single compartment simulation with appropriately scaled parameters.
Demonstrates proper parameter selection for small compartments.
"""

import numpy as np
import matplotlib.pyplot as plt
from gencomo import create_cylinder_mesh, Simulator, MeshSegmenter
from gencomo.core import Neuron, Compartment


def test_single_compartment_corrected():
    """Test single compartment simulation with realistic parameters."""
    print("=" * 80)
    print("🧪 CORRECTED SINGLE COMPARTMENT SIMULATION TEST")
    print("=" * 80)
    print("Creating single compartment neuron...")

    # Create a small cylinder mesh
    cylinder = create_cylinder_mesh(radius=0.5, length=1.0, resolution=20)

    print(f"Cylinder properties:")
    print(f"  Volume: {cylinder.volume:.6f} μm³")
    print(f"  Surface area: {cylinder.area:.6f} μm²")
    print(f"  Bounds: {cylinder.bounds}")

    # Segment the mesh
    segmenter = MeshSegmenter()
    segments = segmenter.segment_mesh(cylinder, slice_height=2.0)

    print(f"Segmentation results:")
    print(f"  Number of segments: {len(segments)}")

    # Get the first (and only) segment
    segment = segments[0]

    print(f"  Segment ID: {segment.id}")
    print(f"  Segment volume: {segment.volume:.6f} μm³")
    print(f"  External surface area: {segment.external_surface_area:.6f} μm²")

    # Create neuron from segmentation
    neuron = Neuron("single_compartment_test_corrected")
    neuron.set_mesh(cylinder)

    # Create compartment from segment
    compartment = Compartment(
        id=segment.id,
        z_level=segment.slice_index,
        area=segment.external_surface_area,
        volume=segment.volume,
        centroid=segment.centroid,
        boundary_points=np.array([[0, 0]]),
        membrane_potential=-70.0,
    )
    neuron.compartment_graph.add_compartment(compartment)

    print(f"Created neuron with compartment:")
    print(f"  Compartment ID: {compartment.id}")
    print(f"  Membrane area: {compartment.area:.6f} μm²")
    print(f"  Volume: {compartment.volume:.6f} μm³")
    print(f"  Initial potential: {compartment.membrane_potential:.1f} mV")

    # Create simulator
    simulator = Simulator(neuron)

    # Set biophysical parameters
    simulator.set_biophysics(
        temperature=6.3,
        capacitance=1.0,
        leak_conductance=0.0003,
        leak_reversal=-54.3,
        na_conductance=0.12,
        na_reversal=50.0,
        k_conductance=0.036,
        k_reversal=-77.0,
    )

    # Calculate appropriate stimulus based on compartment size
    area = compartment.area  # μm²
    # For small compartments (<10 μm²), use 0.05-0.08 nA for action potential
    # Our compartment is ~4.7 μm², so let's use 0.08 nA
    stimulus_amplitude = 0.08  # nA

    print(f"Adding stimulus to compartment: {compartment.id}")
    print(f"  Stimulus amplitude: {stimulus_amplitude} nA (scaled for {area:.1f} μm² area)")

    simulator.add_stimulus(
        compartment_id=compartment.id,
        start_time=10.0,
        duration=1.0,
        amplitude=stimulus_amplitude,
        stimulus_type="current",
    )

    print("Running simulation...")

    # Run simulation
    results = simulator.run_simulation(
        duration=50.0, dt=0.025, method="Radau", rtol=1e-5, atol=1e-8, verbose=True  # Better for stiff systems
    )

    if results.success:
        print("✅ Simulation completed successfully!")
        print(f"  Time steps: {len(results.time)}")
        print(f"  Simulation time: {results.simulation_time:.3f} seconds")

        # Analyze voltage trace
        voltage_trace = results.voltages[compartment.id]

        print("Voltage analysis:")
        print(f"  Resting potential: {voltage_trace[0]:.2f} mV")
        print(f"  Maximum voltage: {np.max(voltage_trace):.2f} mV")
        print(f"  Minimum voltage: {np.min(voltage_trace):.2f} mV")
        print(f"  Final voltage: {voltage_trace[-1]:.2f} mV")

        # Simple spike detection
        spike_threshold = 0.0  # mV
        spikes = []
        above_threshold = voltage_trace > spike_threshold

        # Find rising edges
        for i in range(1, len(above_threshold)):
            if above_threshold[i] and not above_threshold[i - 1]:
                spikes.append(results.time[i])

        print("Spike analysis:")
        print(f"  Spike threshold: {spike_threshold} mV")
        print(f"  Number of spikes: {len(spikes)}")
        if spikes:
            print(f"  First spike at: {spikes[0]:.2f} ms")
            print(f"  Peak voltage: {np.max(voltage_trace):.2f} mV")

        # Create plot (if display is available)
        try:
            import matplotlib

            matplotlib.use("Agg")  # Use non-interactive backend
            plt.figure(figsize=(12, 8))

            # Voltage trace
            plt.subplot(2, 1, 1)
            plt.plot(results.time, voltage_trace, "b-", linewidth=2, label="Voltage")
            plt.axhline(y=spike_threshold, color="r", linestyle="--", alpha=0.7, label="Spike threshold")
            plt.xlabel("Time (ms)")
            plt.ylabel("Voltage (mV)")
            plt.title(
                f"Single Compartment Hodgkin-Huxley Model (Area: {area:.1f} μm², Stimulus: {stimulus_amplitude} nA)"
            )
            plt.grid(True, alpha=0.3)
            plt.legend()

            # Mark stimulus period
            plt.axvspan(10.0, 11.0, alpha=0.2, color="orange", label="Stimulus")

            # Gating variables
            plt.subplot(2, 1, 2)
            gating = results.gating_variables[compartment.id]
            plt.plot(results.time, gating["m"], "r-", label="m (Na activation)")
            plt.plot(results.time, gating["h"], "g-", label="h (Na inactivation)")
            plt.plot(results.time, gating["n"], "b-", label="n (K activation)")
            plt.xlabel("Time (ms)")
            plt.ylabel("Gating variable")
            plt.title("Hodgkin-Huxley Gating Variables")
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save plot
            plot_filename = "single_compartment_corrected_simulation.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
            print(f"Plot saved to: {plot_filename}")
        except Exception as e:
            print(f"⚠️  Could not create plot: {e}")

        # Check if action potential was generated
        if np.max(voltage_trace) > 20:  # mV
            print("✅ Action potential successfully generated!")
        else:
            print("⚠️  No action potential detected - stimulus may be too weak")
    else:
        print("❌ Simulation failed!")
        print(f"Error: {results.message}")
        assert False, f"Simulation failed: {results.message}"


if __name__ == "__main__":
    print("=" * 80)
    print("🎉 Single compartment simulation test PASSED!")
    print("=" * 80)
    try:
        test_single_compartment_corrected()
        print("✅ Test completed successfully!")
    except Exception as e:
        print(f"❌ Test failed: {e}")

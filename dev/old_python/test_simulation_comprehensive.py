"""
Test simulation functionality including debug scenarios and parameter scaling.
This test incorporates functionality from debug_integration.py and demonstrates
proper parameter scaling for different compartment sizes.
"""

import numpy as np
import pytest
from gencomo import MeshSegmenter, Neuron, Compartment
from gencomo.demos import create_cylinder_mesh
from gencomo.simulation import Simulator


class TestSimulation:
    """Test suite for simulation functionality."""

    
test_debug_integration_basic(self):
        """Test basic integration functionality (incorporates debug_integration.py)."""
        # Create single compartment neuron
        cylinder = create_cylinder_mesh(radius=0.5, length=1.0, resolution=20)
        segmenter = MeshSegmenter()
        segments = segmenter.segment_mesh(cylinder, slice_height=2.0)
        segment = segments[0]

        neuron = Neuron("debug_neuron")
        neuron.set_mesh(cylinder)

        compartment = Compartment(
            id=segment.id,
            z_level=segment.slice_index,
            area=segment.external_surface_area,
            volume=segment.volume,
            centroid=np.array([0.0, 0.0, 0.5]),
            boundary_points=np.array([[0, 0]]),
            membrane_potential=-70.0,
        )

        neuron.compartment_graph.add_compartment(compartment)

        # Create simulator
        simulator = Simulator(neuron)

        # Set biophysics
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

        # Use appropriately scaled stimulus for small compartment
        simulator.add_stimulus(
            compartment_id=compartment.id,
            start_time=2.0,
            duration=0.5,
            amplitude=0.005,  # Small stimulus for small compartment
            stimulus_type="current",
        )

        # Run short test simulation
        results = simulator.run_simulation(
            duration=5.0,
            dt=0.025,
            method="Radau",
            rtol=1e-5,
            atol=1e-8,
            max_step=0.1,
            verbose=False,  # Keep quiet for tests
        )

        # Verify success
        assert results.success, f"Simulation failed: {results.message}"

        # Check voltage ranges are reasonable
        comp_id = list(results.voltages.keys())[0]
        voltage = results.voltages[comp_id]

        assert np.min(voltage) > -100, "Voltage too negative"
        assert np.max(voltage) < 100, "Voltage too positive"
        assert len(voltage) > 0, "No voltage data returned"

    
test_parameter_scaling_small_compartment(self):
        """Test that small compartments work with appropriately scaled parameters."""
        cylinder = create_cylinder_mesh(radius=0.5, length=1.0, resolution=20)
        segmenter = MeshSegmenter()
        segments = segmenter.segment_mesh(cylinder, slice_height=2.0)
        segment = segments[0]

        neuron = Neuron("small_comp_test")
        neuron.set_mesh(cylinder)

        compartment = Compartment(
            id=segment.id,
            z_level=segment.slice_index,
            area=segment.external_surface_area,  # ~4.7 μm²
            volume=segment.volume,
            centroid=np.array([0.0, 0.0, 0.5]),
            boundary_points=np.array([[0, 0]]),
            membrane_potential=-70.0,
        )

        neuron.compartment_graph.add_compartment(compartment)
        simulator = Simulator(neuron)

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

        # Test that small stimulus (appropriate for area ~5 μm²) works
        simulator.add_stimulus(
            compartment_id=compartment.id,
            start_time=10.0,
            duration=1.0,
            amplitude=0.005,  # 0.005 nA - appropriate for small compartment
            stimulus_type="current",
        )

        results = simulator.run_simulation(duration=30.0, dt=0.025, method="Radau", rtol=1e-5, atol=1e-8, verbose=False)

        assert results.success

        # Check voltage stays in reasonable bounds
        comp_id = list(results.voltages.keys())[0]
        voltage = results.voltages[comp_id]

        # Should not have extreme voltages (>1000 mV indicates scaling problems)
        assert np.max(np.abs(voltage)) < 1000, f"Extreme voltage detected: {np.max(np.abs(voltage)):.1f} mV"

        # Should have some voltage variation (not stuck at resting potential)
        voltage_range = np.max(voltage) - np.min(voltage)
        assert voltage_range > 1.0, f"Insufficient voltage variation: {voltage_range:.2f} mV"

    
test_integration_method_radau(self):
        """Test that Radau integration method works for stiff HH equations."""
        cylinder = create_cylinder_mesh(radius=0.5, length=1.0, resolution=20)
        segmenter = MeshSegmenter()
        segments = segmenter.segment_mesh(cylinder, slice_height=2.0)
        segment = segments[0]

        neuron = Neuron("radau_test")
        neuron.set_mesh(cylinder)

        compartment = Compartment(
            id=segment.id,
            z_level=segment.slice_index,
            area=segment.external_surface_area,
            volume=segment.volume,
            centroid=np.array([0.0, 0.0, 0.5]),
            boundary_points=np.array([[0, 0]]),
            membrane_potential=-70.0,
        )

        neuron.compartment_graph.add_compartment(compartment)
        simulator = Simulator(neuron)

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

        simulator.add_stimulus(
            compartment_id=compartment.id,
            start_time=5.0,
            duration=1.0,
            amplitude=0.01,  # Slightly larger to test action potential
            stimulus_type="current",
        )

        # Test Radau method specifically
        results = simulator.run_simulation(
            duration=20.0,
            dt=0.025,
            method="Radau",  # Specifically test Radau
            rtol=1e-5,
            atol=1e-8,
            max_step=0.1,
            verbose=False,
        )

        assert results.success, f"Radau integration failed: {results.message}"
        assert len(results.voltages) > 0, "No voltage data returned"

    
test_stimulus_scaling_validation(self):
        """Test that stimulus scaling guidelines are followed."""
        # Test different stimulus amplitudes for small compartment
        # According to guidelines: Small (<10 μm²): 0.001-0.1 nA

        for amplitude in [0.001, 0.005, 0.01, 0.05]:  # Test range of appropriate values
            # Create fresh neuron for each test to avoid stimulus accumulation
            cylinder = create_cylinder_mesh(radius=0.5, length=1.0, resolution=20)
            segmenter = MeshSegmenter()
            segments = segmenter.segment_mesh(cylinder, slice_height=2.0)
            segment = segments[0]

            neuron = Neuron(f"scaling_test_{amplitude}")
            neuron.set_mesh(cylinder)

            area = segment.external_surface_area  # Should be ~4.7 μm²

            compartment = Compartment(
                id=segment.id,
                z_level=segment.slice_index,
                area=area,
                volume=segment.volume,
                centroid=np.array([0.0, 0.0, 0.5]),
                boundary_points=np.array([[0, 0]]),
                membrane_potential=-70.0,
            )

            neuron.compartment_graph.add_compartment(compartment)
            simulator = Simulator(neuron)

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

            simulator.add_stimulus(
                compartment_id=compartment.id,
                start_time=5.0,
                duration=0.5,
                amplitude=amplitude,
                stimulus_type="current",
            )

            results = simulator.run_simulation(
                duration=15.0, dt=0.025, method="Radau", rtol=1e-5, atol=1e-8, verbose=False
            )

            # Should succeed with appropriate stimulus amplitudes
            assert results.success, f"Failed with amplitude {amplitude} nA"

            # Voltage should stay reasonable
            comp_id = list(results.voltages.keys())[0]
            voltage = results.voltages[comp_id]
            max_abs_voltage = np.max(np.abs(voltage))

            # Should not exceed extreme values (indicates scaling problems)
            assert max_abs_voltage < 500, f"Extreme voltage {max_abs_voltage:.1f} mV with {amplitude} nA stimulus"



test_single_compartment_action_potential():
    """Test that we can generate a realistic action potential."""
    cylinder = create_cylinder_mesh(radius=0.5, length=1.0, resolution=20)
    segmenter = MeshSegmenter()
    segments = segmenter.segment_mesh(cylinder, slice_height=2.0)
    segment = segments[0]

    neuron = Neuron("ap_test")
    neuron.set_mesh(cylinder)

    compartment = Compartment(
        id=segment.id,
        z_level=segment.slice_index,
        area=segment.external_surface_area,
        volume=segment.volume,
        centroid=np.array([0.0, 0.0, 0.5]),
        boundary_points=np.array([[0, 0]]),
        membrane_potential=-70.0,
    )

    neuron.compartment_graph.add_compartment(compartment)
    simulator = Simulator(neuron)

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

    # Use stimulus that should generate action potential
    simulator.add_stimulus(
        compartment_id=compartment.id,
        start_time=10.0,
        duration=1.0,
        amplitude=0.08,  # Larger stimulus to ensure AP
        stimulus_type="current",
    )

    results = simulator.run_simulation(duration=50.0, dt=0.025, method="Radau", rtol=1e-5, atol=1e-8, verbose=False)

    assert results.success

    comp_id = list(results.voltages.keys())[0]
    voltage = results.voltages[comp_id]

    # Should have action potential-like behavior
    resting = voltage[0]  # Initial voltage
    peak = np.max(voltage)

    # Action potential should depolarize significantly
    depolarization = peak - resting
    assert depolarization > 50, f"Insufficient depolarization: {depolarization:.1f} mV"

    # Peak should be positive (overshooting action potential)
    assert peak > 0, f"No overshoot: peak = {peak:.1f} mV"

    # Should return toward resting
    final = voltage[-1]
    recovery = abs(final - resting)
    assert recovery < 20, f"Poor recovery: final voltage {final:.1f} mV vs resting {resting:.1f} mV"


if __name__ == "__main__":
    # Run tests when called directly
    test_suite = TestSimulation()

    print("Running simulation tests...")

    try:
        test_suite.test_debug_integration_basic()
        print("✅ Debug integration test passed")

        test_suite.test_parameter_scaling_small_compartment()
        print("✅ Parameter scaling test passed")

        test_suite.test_integration_method_radau()
        print("✅ Radau integration test passed")

        test_suite.test_stimulus_scaling_validation()
        print("✅ Stimulus scaling validation passed")

        test_single_compartment_action_potential()
        print("✅ Action potential test passed")

        print("\n🎉 All simulation tests passed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()

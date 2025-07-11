"""
Advanced example: Load a real neuronal mesh and perform detailed analysis.

This example shows more advanced GenCoMo features:
- Loading complex mesh geometries
- Advanced region detection and filtering
- Custom biophysical parameters
- Multiple stimulation protocols
- Detailed analysis and visualization
"""

import numpy as np
from pathlib import Path
from gencomo import MeshProcessor, ZAxisSlicer, RegionDetector, GraphBuilder, Neuron, Simulator


def load_and_preprocess_mesh(mesh_file: str):
    """Load and preprocess a neuronal mesh."""
    print(f"Loading mesh from: {mesh_file}")

    mesh_processor = MeshProcessor()

    try:
        mesh = mesh_processor.load_mesh(mesh_file)
    except Exception as e:
        print(f"Error loading mesh: {e}")
        print("Creating a more complex test mesh instead...")
        mesh = create_complex_test_mesh()
        mesh_processor.mesh = mesh
        mesh_processor.original_mesh = mesh.copy()
        mesh_processor.bounds = mesh_processor._compute_bounds()

    print("Preprocessing mesh...")

    # Repair mesh if needed
    if not mesh.is_watertight:
        print("  Repairing non-watertight mesh...")
        mesh_processor.repair_mesh()

    # Center and align
    print("  Centering mesh...")
    mesh_processor.center_mesh()

    print("  Aligning with z-axis...")
    mesh_processor.align_with_z_axis()

    # Smooth if needed
    print("  Applying smoothing...")
    mesh_processor.smooth_mesh(iterations=1)

    return mesh_processor


def create_complex_test_mesh():
    """Create a more complex test mesh with branching."""
    import trimesh

    # Create main trunk
    trunk = trimesh.creation.cylinder(radius=5.0, height=50.0, sections=16)
    trunk.vertices[:, 2] += 25.0  # Center at z=25

    # Create branches
    branch1 = trimesh.creation.cylinder(radius=2.0, height=30.0, sections=12)
    branch1.vertices[:, 2] += 15.0
    # Rotate and translate branch1
    branch1.vertices = branch1.vertices @ trimesh.transformations.rotation_matrix(np.pi / 4, [0, 1, 0])[:3, :3]
    branch1.vertices[:, 0] += 15.0
    branch1.vertices[:, 2] += 30.0

    # Create branch2
    branch2 = trimesh.creation.cylinder(radius=1.5, height=25.0, sections=12)
    branch2.vertices[:, 2] += 12.5
    # Rotate and translate branch2
    branch2.vertices = branch2.vertices @ trimesh.transformations.rotation_matrix(-np.pi / 6, [0, 1, 0])[:3, :3]
    branch2.vertices[:, 0] -= 10.0
    branch2.vertices[:, 2] += 35.0

    # Combine meshes
    combined = trimesh.util.concatenate([trunk, branch1, branch2])
    return combined


def advanced_region_detection(slices, mesh_processor):
    """Perform advanced region detection with filtering."""
    print("Advanced region detection...")

    # Initial detection
    detector = RegionDetector()
    regions = detector.detect_regions(slices, min_area=1.0, hole_detection=True)  # Larger minimum area

    print(f"  Initial regions detected: {len(regions)}")

    # Filter regions by size and properties
    filtered_regions = detector.filter_regions(
        min_area=2.0, max_area=1000.0, outer_only=True  # Remove very large artifacts  # Only keep outer boundaries
    )

    print(f"  Filtered regions: {len(filtered_regions)}")

    # Show region statistics
    stats = detector.compute_region_statistics()
    print(f"  Area range: {stats['area_stats']['min']:.1f} - {stats['area_stats']['max']:.1f} µm²")
    print(f"  Average perimeter: {stats['perimeter_stats']['mean']:.1f} µm")

    return filtered_regions


def setup_advanced_biophysics(simulator):
    """Set up advanced biophysical parameters."""
    print("Setting up biophysics...")

    # Temperature-dependent parameters
    simulator.set_biophysics(
        temperature=22.0,  # Room temperature
        capacitance=1.0,  # µF/cm²
        # Leak current
        leak_conductance=0.0001,  # Lower leak
        leak_reversal=-70.0,  # mV
        # Sodium current (reduced for stability)
        na_conductance=0.08,  # S/cm²
        na_reversal=55.0,  # mV
        # Potassium current
        k_conductance=0.024,  # S/cm²
        k_reversal=-80.0,  # mV
    )


def create_stimulation_protocol(simulator, compartment_ids):
    """Create a complex stimulation protocol."""
    print("Creating stimulation protocol...")

    if len(compartment_ids) < 1:
        print("  No compartments available for stimulation")
        return

    # Main stimulus at proximal compartment
    prox_comp = compartment_ids[0]
    simulator.add_stimulus(compartment_id=prox_comp, start_time=20.0, duration=2.0, amplitude=8.0)
    print(f"  Added main stimulus to {prox_comp}")

    # Secondary stimulus if enough compartments
    if len(compartment_ids) >= 3:
        mid_comp = compartment_ids[len(compartment_ids) // 2]
        simulator.add_stimulus(compartment_id=mid_comp, start_time=60.0, duration=1.0, amplitude=4.0)
        print(f"  Added secondary stimulus to {mid_comp}")

    # Brief test pulse
    if len(compartment_ids) >= 2:
        test_comp = compartment_ids[1]
        simulator.add_stimulus(compartment_id=test_comp, start_time=100.0, duration=0.5, amplitude=2.0)
        print(f"  Added test pulse to {test_comp}")


def detailed_analysis(simulator, results):
    """Perform detailed analysis of simulation results."""
    print("Detailed analysis...")

    analysis = simulator.analyze_results()

    # Overall statistics
    print(f"  Simulation duration: {analysis['simulation_info']['duration']:.1f} ms")
    print(f"  Simulation time: {analysis['simulation_info']['simulation_time']:.2f} s")
    print(f"  Total spikes: {analysis['overall']['total_spikes']}")
    print(f"  Active compartments: {analysis['overall']['active_compartments']}")

    # Voltage analysis
    print("\n  Voltage statistics:")
    voltage_ranges = []
    for comp_id, stats in analysis["voltage_stats"].items():
        voltage_range = stats["max"] - stats["min"]
        voltage_ranges.append(voltage_range)
        if voltage_range > 20:  # Significant activity
            print(f"    {comp_id}: {stats['min']:.1f} to {stats['max']:.1f} mV (range: {voltage_range:.1f})")

    if voltage_ranges:
        print(f"  Average voltage range: {np.mean(voltage_ranges):.1f} mV")
        print(f"  Max voltage range: {np.max(voltage_ranges):.1f} mV")

    # Spike analysis
    print("\n  Spike analysis:")
    active_comps = [
        comp_id for comp_id, spike_data in analysis["spike_analysis"].items() if spike_data["num_spikes"] > 0
    ]

    if active_comps:
        print(f"    Active compartments: {len(active_comps)}")
        for comp_id in active_comps[:5]:  # Show first 5
            spike_data = analysis["spike_analysis"][comp_id]
            print(
                f"      {comp_id}: {spike_data['num_spikes']} spikes, " f"rate: {spike_data['mean_firing_rate']:.1f} Hz"
            )

    # Propagation analysis
    if len(active_comps) >= 2:
        print("\n  Propagation analysis:")
        try:
            velocity = simulator.compute_propagation_velocity(active_comps[0], active_comps[1])
            if velocity:
                print(f"    Propagation velocity: {velocity:.2f} m/s")
        except Exception as e:
            print(f"    Could not compute propagation velocity: {e}")

    return analysis


def main():
    """Run the advanced example."""
    print("GenCoMo Advanced Example")
    print("========================")

    # You can specify a real mesh file here
    mesh_file = "path/to/your/neuron.stl"  # Change this to your mesh file

    # Step 1: Load and preprocess mesh
    print("\n1. Loading and preprocessing mesh...")
    mesh_processor = load_and_preprocess_mesh(mesh_file)

    # Step 2: Create high-resolution slices
    print("\n2. Creating high-resolution slices...")
    slicer = ZAxisSlicer(mesh_processor.mesh)

    # Use finer slicing for better resolution
    z_min, z_max = mesh_processor.get_z_range()
    slice_spacing = (z_max - z_min) / 100  # 100 slices
    slices = slicer.create_slices(slice_spacing=slice_spacing)

    print(f"   Created {len(slices)} slices with spacing {slice_spacing:.2f} µm")

    # Step 3: Advanced region detection
    print("\n3. Advanced region detection...")
    regions = advanced_region_detection(slices, mesh_processor)

    # Step 4: Build optimized connectivity
    print("\n4. Building optimized connectivity...")
    graph_builder = GraphBuilder()

    # Use hybrid connection method for better connectivity
    compartment_graph = graph_builder.build_compartment_graph(
        regions,
        connection_method="hybrid",
        min_overlap_ratio=0.05,  # More sensitive overlap detection
        max_connection_distance=10.0,  # µm
    )

    # Validate connectivity
    validation = graph_builder.validate_graph()
    if validation["warnings"]:
        print(f"   Connectivity warnings: {len(validation['warnings'])}")
    if validation["errors"]:
        print(f"   Connectivity errors: {len(validation['errors'])}")

    # Step 5: Create neuron model
    print("\n5. Creating advanced neuron model...")
    neuron = Neuron("advanced_neuron")
    neuron.set_mesh(mesh_processor.mesh)
    neuron.compartment_graph = compartment_graph

    # Step 6: Set up advanced simulation
    print("\n6. Setting up advanced simulation...")
    simulator = Simulator(neuron)

    setup_advanced_biophysics(simulator)
    create_stimulation_protocol(simulator, simulator.ode_system.compartment_ids)

    # Step 7: Run longer simulation
    print("\n7. Running simulation...")
    results = simulator.run_simulation(
        duration=150.0,  # Longer simulation
        dt=0.01,  # Finer time resolution
        method="DOP853",  # Higher-order method
        rtol=1e-8,  # Tighter tolerance
        atol=1e-11,
    )

    if results.success:
        print("   Simulation completed successfully!")

        # Step 8: Detailed analysis
        print("\n8. Detailed analysis...")
        analysis = detailed_analysis(simulator, results)

        # Step 9: Save results
        print("\n9. Saving results...")
        try:
            simulator.save_results("advanced_results.npz", format="npz")
            print("   Results saved to advanced_results.npz")
        except Exception as e:
            print(f"   Could not save results: {e}")

    else:
        print(f"   Simulation failed: {results.message}")
        return 1

    print("\nAdvanced example completed!")
    print("\nThis example demonstrated:")
    print("- Advanced mesh preprocessing and repair")
    print("- High-resolution slicing and region detection")
    print("- Hybrid connectivity detection")
    print("- Complex stimulation protocols")
    print("- Detailed analysis and propagation velocity")
    print("- Results saving and validation")

    return 0


if __name__ == "__main__":
    main()

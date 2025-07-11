"""
Basic example: Process a simple neuronal mesh and run a compartmental simulation.

This example demonstrates the complete GenCoMo workflow:
1. Load and preprocess a neuronal mesh
2. Slice the mesh along the z-axis
3. Detect closed regions in each slice
4. Build connectivity graph between regions
5. Set up and run a compartmental simulation
"""

import numpy as np
import trimesh
from gencomo import MeshProcessor, ZAxisSlicer, RegionDetector, GraphBuilder, Neuron, Simulator


def create_simple_cylinder_mesh(length=100.0, radius=5.0, num_segments=20):
    """
    Create a simple cylindrical mesh for testing.

    Args:
        length: Cylinder length (µm)
        radius: Cylinder radius (µm)
        num_segments: Number of circular segments

    Returns:
        trimesh.Trimesh object
    """
    # Create cylinder vertices
    theta = np.linspace(0, 2 * np.pi, num_segments, endpoint=False)
    z_levels = np.linspace(0, length, 50)

    vertices = []
    faces = []

    # Create vertices
    for z in z_levels:
        for angle in theta:
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            vertices.append([x, y, z])

    vertices = np.array(vertices)

    # Create faces (triangular mesh)
    for i in range(len(z_levels) - 1):
        for j in range(num_segments):
            # Current ring indices
            curr_base = i * num_segments
            next_base = (i + 1) * num_segments

            # Current vertices
            v1 = curr_base + j
            v2 = curr_base + (j + 1) % num_segments
            v3 = next_base + j
            v4 = next_base + (j + 1) % num_segments

            # Two triangles per quad
            faces.append([v1, v2, v3])
            faces.append([v2, v4, v3])

    faces = np.array(faces)

    # Create mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return mesh


def main():
    """Run the basic example."""
    print("GenCoMo Basic Example")
    print("====================")

    # Step 1: Create or load a mesh
    print("\n1. Creating test mesh...")
    mesh = create_simple_cylinder_mesh(length=50.0, radius=3.0, num_segments=12)
    print(f"   Created mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    # Step 2: Process the mesh
    print("\n2. Processing mesh...")
    mesh_processor = MeshProcessor()
    mesh_processor.mesh = mesh
    mesh_processor.original_mesh = mesh.copy()
    mesh_processor.bounds = mesh_processor._compute_bounds()

    # Center and align the mesh
    mesh_processor.center_mesh()
    mesh_processor.align_with_z_axis()

    print("   Mesh properties:")
    props = mesh_processor.compute_mesh_properties()
    for key, value in props.items():
        if isinstance(value, (int, float)):
            print(f"     {key}: {value:.2f}")
        else:
            print(f"     {key}: {value}")

    # Step 3: Slice the mesh
    print("\n3. Slicing mesh along z-axis...")
    slicer = ZAxisSlicer(mesh_processor.mesh)
    slices = slicer.create_slices(num_slices=20)
    print(f"   Created {len(slices)} slices")

    # Show slice summary
    summary = slicer.get_slice_summary()
    print(f"   Average contours per slice: {summary['avg_contours_per_slice']:.1f}")

    # Step 4: Detect regions
    print("\n4. Detecting regions...")
    region_detector = RegionDetector()
    regions = region_detector.detect_regions(slices, min_area=0.5)
    print(f"   Detected {len(regions)} regions")

    # Show region statistics
    stats = region_detector.compute_region_statistics()
    print(f"   Outer regions: {stats['outer_regions']}")
    print(f"   Average area: {stats['area_stats']['mean']:.2f} µm²")

    # Step 5: Build compartment graph
    print("\n5. Building compartment graph...")
    graph_builder = GraphBuilder()
    compartment_graph = graph_builder.build_compartment_graph(regions)

    # Show connectivity statistics
    conn_stats = graph_builder.get_connectivity_statistics()
    print(f"   Compartments: {conn_stats['num_compartments']}")
    print(f"   Connections: {conn_stats['num_connections']}")
    print(f"   Average degree: {conn_stats['avg_degree']:.1f}")

    # Step 6: Create neuron model
    print("\n6. Creating neuron model...")
    neuron = Neuron("test_neuron")
    neuron.set_mesh(mesh_processor.mesh)
    neuron.compartment_graph = compartment_graph

    # Step 7: Set up simulation
    print("\n7. Setting up simulation...")
    simulator = Simulator(neuron)

    # Set biophysical parameters
    simulator.set_biophysics(
        capacitance=1.0,  # µF/cm²
        leak_conductance=0.0003,  # S/cm²
        na_conductance=0.12,  # S/cm²
        k_conductance=0.036,  # S/cm²
    )

    # Add stimulus to first compartment
    if simulator.ode_system.compartment_ids:
        first_comp = simulator.ode_system.compartment_ids[0]
        simulator.add_stimulus(
            compartment_id=first_comp, start_time=10.0, duration=1.0, amplitude=5.0  # ms  # ms  # nA
        )
        print(f"   Added stimulus to compartment: {first_comp}")

    # Step 8: Run simulation
    print("\n8. Running simulation...")
    results = simulator.run_simulation(duration=50.0, dt=0.025)  # ms  # ms

    if results.success:
        print(f"   Simulation completed successfully!")
        print(f"   Simulation time: {results.simulation_time:.2f} seconds")

        # Step 9: Analyze results
        print("\n9. Analyzing results...")
        analysis = simulator.analyze_results()

        print(f"   Time points: {analysis['simulation_info']['num_time_points']}")
        print(f"   Total spikes: {analysis['overall']['total_spikes']}")
        print(f"   Active compartments: {analysis['overall']['active_compartments']}")

        # Show voltage statistics for first few compartments
        print("   Voltage statistics (first 3 compartments):")
        for i, (comp_id, stats) in enumerate(analysis["voltage_stats"].items()):
            if i >= 3:
                break
            print(f"     {comp_id}: {stats['min']:.1f} to {stats['max']:.1f} mV")

    else:
        print(f"   Simulation failed: {results.message}")
        return 1

    print("\nExample completed successfully!")
    print("\nNext steps:")
    print("- Load your own mesh files using MeshProcessor.load_mesh()")
    print("- Experiment with different slicing parameters")
    print("- Try different biophysical parameters")
    print("- Add multiple stimuli to different compartments")
    print("- Analyze spike propagation between compartments")

    return 0


if __name__ == "__main__":
    main()

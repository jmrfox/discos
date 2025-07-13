"""
Test script for mesh segmentation functionality.
"""

import sys
import os

# Add the parent directory to path for importing gencomo
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import trimesh
import networkx as nx

from gencomo import MeshSegmenter
from gencomo.demos import create_cylinder_mesh, create_torus_mesh, create_demo_neuron_mesh


def validate_connectivity_graph(segmenter, test_name="Test"):
    """Validate that the connectivity graph is a single connected component.

    Args:
        segmenter: MeshSegmenter instance with connectivity graph
        test_name: Name of the test for reporting

    Returns:
        bool: True if graph is properly connected, False otherwise
    """
    print(f"\n🔗 {test_name} - Connectivity Validation:")

    # Check if the connectivity graph is connected
    is_connected = nx.is_connected(segmenter.connectivity_graph)
    num_components = nx.number_connected_components(segmenter.connectivity_graph)

    print(f"  Graph is connected: {is_connected}")
    print(f"  Number of connected components: {num_components}")

    if is_connected:
        print(f"  ✅ SUCCESS: Single connected graph - all segments can communicate")

        # Additional connectivity metrics for complex graphs
        if len(segmenter.connectivity_graph.nodes) > 1:
            diameter = nx.diameter(segmenter.connectivity_graph)
            avg_path_length = nx.average_shortest_path_length(segmenter.connectivity_graph)
            print(f"  Graph diameter (max shortest path): {diameter}")
            print(f"  Average shortest path length: {avg_path_length:.2f}")
    else:
        print(f"  ❌ ERROR: Graph has {num_components} disconnected components!")
        print(f"  This indicates segmentation issues - some segments are isolated")

        # Show the sizes of each component
        components = list(nx.connected_components(segmenter.connectivity_graph))
        component_sizes = [len(comp) for comp in components]
        print(f"  Component sizes: {component_sizes}")

    return is_connected


def test_cylinder_segmentation(plot_graph: bool = False, show_plot: bool = True, include_3d_view: bool = False):
    """Test segmentation on a cylinder mesh with surface area validation.

    Args:
        plot_graph: Whether to generate and save connectivity graph plot
        show_plot: Whether to display the plot (only used if plot_graph=True)
        include_3d_view: Whether to include 3D view alongside network view (only used if plot_graph=True)
    """
    print("🔧 Testing Cylinder Segmentation with Surface Area Validation")
    print("=" * 70)

    # Create cylinder mesh using demos function
    radius = 0.5
    height = 2.0
    cylinder = create_cylinder_mesh(length=height, radius=radius, resolution=64)

    print(f"Created cylinder:")
    print(f"  Radius: {radius}, Height: {height}")
    print(f"  Vertices: {len(cylinder.vertices)}, Faces: {len(cylinder.faces)}")
    print(f"  Volume: {cylinder.volume:.6f}")
    print(f"  Surface area: {cylinder.area:.6f}")
    print(f"  Z-bounds: [{cylinder.bounds[0,2]:.2f}, {cylinder.bounds[1,2]:.2f}]")
    print(f"  Watertight: {cylinder.is_watertight}")

    # Theoretical calculations
    cap_area = np.pi * radius**2  # πr²
    total_side_area = 2 * np.pi * radius * height  # 2πrh
    total_theoretical_area = 2 * cap_area + total_side_area

    print(f"\nTheoretical surface areas:")
    print(f"  Cap area (πr²): {cap_area:.6f}")
    print(f"  Total side area (2πrh): {total_side_area:.6f}")
    print(f"  Total area: {total_theoretical_area:.6f}")

    # Segment the mesh
    slice_height = 0.1
    print(f"\nSegmenting with slice_height = {slice_height}...")

    segmenter = MeshSegmenter()
    segments = segmenter.segment_mesh(cylinder, slice_height=slice_height)

    print(f"Segmentation results:")
    print(f"  Number of segments: {len(segments)}")

    if len(segments) == 0:
        print("❌ No segments created! Segmentation failed.")
        return False

    # Validate volume conservation
    total_segment_volume = sum(seg.volume for seg in segments)
    volume_error = abs(total_segment_volume - cylinder.volume)
    volume_conservation = (total_segment_volume / cylinder.volume) * 100

    print(f"\n📊 Volume Conservation:")
    print(f"  Original volume: {cylinder.volume:.6f}")
    print(f"  Total segment volume: {total_segment_volume:.6f}")
    print(f"  Conservation: {volume_conservation:.1f}%")
    print(f"  Error: {volume_error:.6f}")

    # Validate surface areas
    print(f"\n📋 Surface Area Analysis:")

    # Expected surface areas per segment type
    side_area_per_slice = 2 * np.pi * radius * slice_height  # 2πrh for one slice

    total_ext_area = 0
    total_int_area = 0
    surface_area_valid = True

    for i, seg in enumerate(segments):
        # Determine expected areas based on segment position
        if i == 0:  # First segment (has bottom cap)
            expected_ext = cap_area + side_area_per_slice
            expected_int = cap_area  # One cut at top
            segment_type = "First (bottom cap)"
        elif i == len(segments) - 1:  # Last segment (has top cap)
            expected_ext = cap_area + side_area_per_slice
            expected_int = cap_area  # One cut at bottom
            segment_type = "Last (top cap)"
        else:  # Middle segments (no caps)
            expected_ext = side_area_per_slice
            expected_int = 2 * cap_area  # Two cuts (top and bottom)
            segment_type = "Middle (no caps)"

        ext_error = abs(seg.external_surface_area - expected_ext)
        int_error = abs(seg.internal_surface_area - expected_int)

        print(f"  Segment {i} ({segment_type}):")
        print(f"    Volume: {seg.volume:.6f}, Z: [{seg.z_min:.2f}, {seg.z_max:.2f}]")
        print(
            f"    External area: {seg.external_surface_area:.6f} (expected: {expected_ext:.6f}, error: {ext_error:.6f})"
        )
        print(
            f"    Internal area: {seg.internal_surface_area:.6f} (expected: {expected_int:.6f}, error: {int_error:.6f})"
        )

        if ext_error > 0.1 or int_error > 0.1:
            print(f"    ❌ Surface area mismatch!")
            surface_area_valid = False
        else:
            print(f"    ✅ Surface areas correct!")

        total_ext_area += seg.external_surface_area
        total_int_area += seg.internal_surface_area

    # Check total surface area conservation
    ext_area_error = abs(total_ext_area - cylinder.area)
    # Internal faces: each cut creates 2 internal faces (one per adjacent segment)
    num_cuts = len(segments) - 1
    expected_total_int = num_cuts * 2 * cap_area  # Each cut creates 2 internal faces
    int_area_error = abs(total_int_area - expected_total_int)

    print(f"\n🔬 Surface Area Conservation:")
    print(f"  Original mesh area: {cylinder.area:.6f}")
    print(f"  Total external area: {total_ext_area:.6f} (error: {ext_area_error:.6f})")
    print(f"  Total internal area: {total_int_area:.6f}")
    print(f"  Expected internal: {expected_total_int:.6f} (error: {int_area_error:.6f})")

    # Overall validation
    volume_valid = volume_error < 1e-6
    ext_area_valid = ext_area_error < 0.1
    int_area_valid = int_area_error < 0.1

    # Connectivity validation
    connectivity_valid = validate_connectivity_graph(segmenter, "Cylinder")

    print(f"\n🎯 Validation Results:")
    print(f"  Volume conservation: {'✅ PASS' if volume_valid else '❌ FAIL'}")
    print(f"  External area conservation: {'✅ PASS' if ext_area_valid else '❌ FAIL'}")
    print(f"  Internal area calculation: {'✅ PASS' if int_area_valid else '❌ FAIL'}")
    print(f"  Individual surface areas: {'✅ PASS' if surface_area_valid else '❌ FAIL'}")
    print(f"  Connectivity graph: {'✅ PASS' if connectivity_valid else '❌ FAIL'}")

    all_tests_pass = volume_valid and ext_area_valid and int_area_valid and surface_area_valid and connectivity_valid
    print(f"\n🏆 Overall Result: {'✅ ALL TESTS PASS' if all_tests_pass else '❌ SOME TESTS FAILED'}")

    # Validate connectivity graph
    connectivity_valid = validate_connectivity_graph(segmenter, test_name="Cylinder Segmentation")

    # Plot connectivity graph if requested
    if plot_graph:
        print(f"\n📊 Generating connectivity graph visualization...")
        graph_path = segmenter.visualize_connectivity_graph(
            save_path="tests/data/cylinder_connectivity_graph.png", show_plot=show_plot, include_3d_view=include_3d_view
        )
        if graph_path:
            print(f"  Connectivity graph saved to: {graph_path}")
        else:
            print(f"  Failed to save connectivity graph")

    return all_tests_pass and connectivity_valid


def test_torus_segmentation(plot_graph: bool = False, show_plot: bool = True, include_3d_view: bool = False):
    """Test segmentation on a torus mesh with surface area validation.

    Args:
        plot_graph: If True, generate and save connectivity graph visualization
        show_plot: Whether to display the plot (only used if plot_graph=True)
        include_3d_view: Whether to include 3D view alongside network view (only used if plot_graph=True)
    """
    print("🔧 Testing Torus Segmentation with Surface Area Validation")
    print("=" * 70)

    # Create torus mesh using demos function (axis of symmetry along x-axis for z-slicing)
    major_radius = 1.0  # Distance from center to tube center
    minor_radius = 0.3  # Tube radius
    torus = create_torus_mesh(
        major_radius=major_radius, minor_radius=minor_radius, major_segments=64, minor_segments=32, axis="x"
    )

    print(f"Created torus:")
    print(f"  Major radius: {major_radius}, Minor radius: {minor_radius}")
    print(f"  Vertices: {len(torus.vertices)}, Faces: {len(torus.faces)}")
    print(f"  Volume: {torus.volume:.6f}")
    print(f"  Surface area: {torus.area:.6f}")
    print(f"  Z-bounds: [{torus.bounds[0,2]:.2f}, {torus.bounds[1,2]:.2f}]")
    print(f"  Watertight: {torus.is_watertight}")

    # Theoretical calculations for torus
    theoretical_volume = 2 * np.pi**2 * major_radius * minor_radius**2
    theoretical_surface_area = 4 * np.pi**2 * major_radius * minor_radius

    print(f"\nTheoretical properties:")
    print(f"  Volume: {theoretical_volume:.6f}")
    print(f"  Surface area: {theoretical_surface_area:.6f}")

    # Segment the torus
    slice_height = 0.3
    print(f"\nSegmenting with slice_height = {slice_height}...")

    segmenter = MeshSegmenter()
    segments = segmenter.segment_mesh(torus, slice_height=slice_height)

    print(f"\nSegmentation results:")
    print(f"  Number of segments: {len(segments)}")

    # Validate volume conservation
    print(f"\n📊 Volume Conservation:")
    print(f"  Original volume: {torus.volume:.6f}")

    total_segment_volume = sum(seg.volume for seg in segments)
    print(f"  Total segment volume: {total_segment_volume:.6f}")

    volume_conservation = (total_segment_volume / torus.volume) * 100
    volume_error = abs(torus.volume - total_segment_volume)
    print(f"  Conservation: {volume_conservation:.1f}%")
    print(f"  Error: {volume_error:.6f}")

    # Validate surface areas for each segment
    print(f"\n📋 Surface Area Analysis:")
    total_ext_area = 0
    total_int_area = 0
    surface_area_valid = True

    for i, seg in enumerate(segments):
        ext_area = seg.external_surface_area
        int_area = seg.internal_surface_area

        total_ext_area += ext_area
        total_int_area += int_area

        z_min, z_max = seg.z_min, seg.z_max
        segment_type = f"Segment {i}"

        print(f"  {segment_type}:")
        print(f"    Volume: {seg.volume:.6f}, Z: [{z_min:.2f}, {z_max:.2f}]")
        print(f"    External area: {ext_area:.6f}")
        print(f"    Internal area: {int_area:.6f}")

    # Check surface area conservation
    print(f"\n🔬 Surface Area Conservation:")
    print(f"  Original mesh area: {torus.area:.6f}")
    print(f"  Total external area: {total_ext_area:.6f}")
    ext_area_error = abs(total_ext_area - torus.area)
    print(f"  External area error: {ext_area_error:.6f}")

    print(f"  Total internal area: {total_int_area:.6f}")

    # For torus, internal area is complex due to topology
    # The torus can have varying cross-sectional shapes when sliced
    # Simple estimate: internal area should be roughly proportional to number of internal faces
    # But we'll use a more relaxed validation since exact calculation is topology-dependent

    # Count number of slices that generated internal faces
    num_slices_with_internal = sum(1 for seg in segments if seg.internal_surface_area > 0)
    print(f"  Slices with internal faces: {num_slices_with_internal}")

    # For torus, internal area varies significantly based on cut topology
    # Use the actual total as a sanity check rather than exact prediction
    print(f"  Internal area per segment (avg): {total_int_area/len(segments):.6f}")

    # Overall validation
    volume_valid = volume_error < 1e-6
    ext_area_valid = ext_area_error < 0.2  # More tolerance for torus complexity

    # For torus internal area, we'll check that:
    # 1. We have reasonable internal area (not zero, not excessive)
    # 2. Internal area scales reasonably with number of cuts
    reasonable_internal = 0.1 < total_int_area < 50.0  # Sanity bounds
    int_area_valid = reasonable_internal

    # Validate connectivity graph
    connectivity_valid = validate_connectivity_graph(segmenter, test_name="Torus Segmentation")

    print(f"\n🎯 Validation Results:")
    print(f"  Volume conservation: {'✅ PASS' if volume_valid else '❌ FAIL'}")
    print(f"  External area conservation: {'✅ PASS' if ext_area_valid else '❌ FAIL'}")
    print(f"  Internal area calculation: {'✅ PASS' if int_area_valid else '❌ FAIL'}")
    print(f"  Connectivity graph: {'✅ PASS' if connectivity_valid else '❌ FAIL'}")

    overall_pass = volume_valid and ext_area_valid and int_area_valid and connectivity_valid
    print(f"\n🏆 Overall Result: {'✅ ALL TESTS PASS' if overall_pass else '❌ SOME TESTS FAILED'}")

    # Plot connectivity graph if requested
    if plot_graph:
        print(f"\n📊 Generating connectivity graph visualization...")
        graph_path = segmenter.visualize_connectivity_graph(
            save_path="tests/data/torus_connectivity_graph.png", show_plot=show_plot, include_3d_view=include_3d_view
        )
        if graph_path:
            print(f"  Connectivity graph saved to: {graph_path}")
        else:
            print(f"  Failed to save connectivity graph")

    return overall_pass and connectivity_valid


def test_neuron_segmentation(plot_graph: bool = False, show_plot: bool = True, include_3d_view: bool = False):
    """Test segmentation on a simplified neuron mesh with complex topology validation.

    Args:
        plot_graph: If True, generate and save connectivity graph visualization
        show_plot: Whether to display the plot (only used if plot_graph=True)
        include_3d_view: Whether to include 3D view alongside network view (only used if plot_graph=True)
    """
    print("🔧 Testing Neuron Segmentation with Complex Topology Validation")
    print("=" * 70)

    # Create demo neuron mesh using demos function
    soma_radius = 8.0
    dendrite_length = 30.0
    dendrite_radius = 1.5
    axon_length = 60.0
    axon_radius = 1.0
    num_dendrites = 3
    dendrite_angle = 25.0

    neuron = create_demo_neuron_mesh(
        soma_radius=soma_radius,
        dendrite_length=dendrite_length,
        dendrite_radius=dendrite_radius,
        axon_length=axon_length,
        axon_radius=axon_radius,
        num_dendrites=num_dendrites,
        dendrite_angle=dendrite_angle,
    )

    print(f"Created demo neuron:")
    print(f"  Soma radius: {soma_radius} μm")
    print(f"  Dendrites: {num_dendrites} × {dendrite_length} μm (radius: {dendrite_radius} μm)")
    print(f"  Axon: {axon_length} μm (radius: {axon_radius} μm)")
    print(f"  Vertices: {len(neuron.vertices)}, Faces: {len(neuron.faces)}")
    print(f"  Volume: {neuron.volume:.6f}")
    print(f"  Surface area: {neuron.area:.6f}")
    print(f"  Z-bounds: [{neuron.bounds[0,2]:.2f}, {neuron.bounds[1,2]:.2f}]")
    print(f"  Watertight: {neuron.is_watertight}")

    # Theoretical calculations for neuron components
    soma_volume = (4 / 3) * np.pi * soma_radius**3
    dendrite_volume = np.pi * dendrite_radius**2 * dendrite_length
    axon_volume = np.pi * axon_radius**2 * axon_length
    total_theoretical_volume = soma_volume + num_dendrites * dendrite_volume + axon_volume

    print(f"\nTheoretical component volumes:")
    print(f"  Soma: {soma_volume:.6f}")
    print(f"  All dendrites: {num_dendrites * dendrite_volume:.6f}")
    print(f"  Axon: {axon_volume:.6f}")
    print(f"  Total theoretical: {total_theoretical_volume:.6f}")

    # Segment the neuron with smaller slice height due to complex geometry
    slice_height = 8.0  # Smaller slices for complex topology
    print(f"\nSegmenting with slice_height = {slice_height}...")

    segmenter = MeshSegmenter()
    segments = segmenter.segment_mesh(neuron, slice_height=slice_height)

    print(f"\nSegmentation results:")
    print(f"  Number of segments: {len(segments)}")

    if len(segments) == 0:
        print("❌ No segments created! Segmentation failed.")
        return False

    # Validate volume conservation
    total_segment_volume = sum(seg.volume for seg in segments)
    volume_error = abs(total_segment_volume - neuron.volume)
    volume_conservation = (total_segment_volume / neuron.volume) * 100

    print(f"\n📊 Volume Conservation:")
    print(f"  Original volume: {neuron.volume:.6f}")
    print(f"  Total segment volume: {total_segment_volume:.6f}")
    print(f"  Conservation: {volume_conservation:.1f}%")
    print(f"  Error: {volume_error:.6f}")

    # Analyze segments by their topological complexity
    print(f"\n📋 Segment Analysis:")
    total_ext_area = 0
    total_int_area = 0

    # Count segments in different regions
    soma_segments = []
    dendrite_segments = []
    axon_segments = []

    for i, seg in enumerate(segments):
        z_mid = (seg.z_min + seg.z_max) / 2
        ext_area = seg.external_surface_area
        int_area = seg.internal_surface_area

        total_ext_area += ext_area
        total_int_area += int_area

        # Classify segment by Z position (rough estimate)
        if z_mid > soma_radius * 0.3:  # Upper region (dendrites)
            segment_type = "Dendrite region"
            dendrite_segments.append(seg)
        elif z_mid > -soma_radius * 0.8:  # Middle region (soma)
            segment_type = "Soma region"
            soma_segments.append(seg)
        else:  # Lower region (axon)
            segment_type = "Axon region"
            axon_segments.append(seg)

        print(f"  Segment {i} ({segment_type}):")
        print(f"    Volume: {seg.volume:.6f}, Z: [{seg.z_min:.2f}, {seg.z_max:.2f}]")
        print(f"    External area: {ext_area:.6f}")
        print(f"    Internal area: {int_area:.6f}")

    print(f"\n🧠 Regional Analysis:")
    print(f"  Dendrite region segments: {len(dendrite_segments)}")
    print(f"  Soma region segments: {len(soma_segments)}")
    print(f"  Axon region segments: {len(axon_segments)}")

    if dendrite_segments:
        dendrite_vol = sum(s.volume for s in dendrite_segments)
        print(f"  Dendrite region volume: {dendrite_vol:.6f}")
    if soma_segments:
        soma_vol = sum(s.volume for s in soma_segments)
        print(f"  Soma region volume: {soma_vol:.6f}")
    if axon_segments:
        axon_vol = sum(s.volume for s in axon_segments)
        print(f"  Axon region volume: {axon_vol:.6f}")

    # Check surface area conservation
    print(f"\n🔬 Surface Area Conservation:")
    print(f"  Original mesh area: {neuron.area:.6f}")
    print(f"  Total external area: {total_ext_area:.6f}")
    ext_area_error = abs(total_ext_area - neuron.area)
    print(f"  External area error: {ext_area_error:.6f}")
    print(f"  Total internal area: {total_int_area:.6f}")

    # Overall validation - more relaxed for complex neuron topology
    volume_valid = volume_error < 1e-5  # Slightly more tolerance due to boolean operations
    ext_area_valid = ext_area_error < 200.0  # Much more tolerance for complex geometry and boolean operations

    # For neuron internal area, check reasonableness - neuron has complex branching topology
    reasonable_internal = 10.0 < total_int_area < 2000.0  # Wider sanity bounds for neuron with dendrites
    has_segments = len(segments) > 0
    int_area_valid = reasonable_internal and has_segments

    print(f"\n🎯 Validation Results:")
    print(f"  Volume conservation: {'✅ PASS' if volume_valid else '❌ FAIL'}")
    print(f"  External area conservation: {'✅ PASS' if ext_area_valid else '❌ FAIL'}")
    print(f"  Internal area reasonableness: {'✅ PASS' if int_area_valid else '❌ FAIL'}")
    print(f"  Segments generated: {'✅ PASS' if has_segments else '❌ FAIL'}")

    # Validate connectivity graph
    connectivity_valid = validate_connectivity_graph(segmenter, test_name="Neuron Segmentation")

    all_tests_pass = volume_valid and ext_area_valid and int_area_valid and has_segments and connectivity_valid
    print(f"\n🏆 Overall Result: {'✅ ALL TESTS PASS' if all_tests_pass else '❌ SOME TESTS FAILED'}")

    # Plot connectivity graph if requested
    if plot_graph:
        print(f"\n📊 Generating connectivity graph visualization...")
        graph_path = segmenter.visualize_connectivity_graph(
            save_path="tests/data/neuron_connectivity_graph.png", show_plot=show_plot, include_3d_view=include_3d_view
        )
        if graph_path:
            print(f"  Connectivity graph saved to: {graph_path}")
        else:
            print(f"  Failed to save connectivity graph")

    return all_tests_pass and connectivity_valid


def run_all_tests():
    """Run all segmentation tests."""
    print("🧪 Running All Segmentation Tests")
    print("=" * 50)

    # Test: Cylinder segmentation
    print("\n" + "=" * 50)
    cylinder_pass = test_cylinder_segmentation()

    # Test: Torus segmentation
    print("\n" + "=" * 50)
    torus_pass = test_torus_segmentation()

    # Test: Neuron segmentation
    print("\n" + "=" * 50)
    neuron_pass = test_neuron_segmentation()

    # Summary
    print("\n" + "=" * 50)
    print("🏆 TEST SUMMARY")
    print("=" * 50)
    print(f"Cylinder test: {'✅ PASS' if cylinder_pass else '❌ FAIL'}")
    print(f"Torus test: {'✅ PASS' if torus_pass else '❌ FAIL'}")
    print(f"Neuron test: {'✅ PASS' if neuron_pass else '❌ FAIL'}")

    if cylinder_pass and torus_pass and neuron_pass:
        print("🎉 ALL TESTS PASSED!")
        return True
    else:
        print("💥 TESTS FAILED!")
        return False


if __name__ == "__main__":
    # Run individual test or all tests
    import sys
    import argparse

    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description="Run mesh segmentation tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_segmentation.py                      # Run all tests
  python test_segmentation.py cylinder             # Run cylinder test only
  python test_segmentation.py torus                # Run torus test only
  python test_segmentation.py neuron               # Run neuron test only
  python test_segmentation.py --plot               # Run all tests with network view plots
  python test_segmentation.py cylinder --plot      # Run cylinder test with network view plot
  python test_segmentation.py neuron --save-only   # Run neuron test, save plot but don't show
  python test_segmentation.py --plot --include-3d  # Run all tests with both network and 3D views
        """,
    )

    parser.add_argument(
        "test_type",
        nargs="?",
        choices=["cylinder", "torus", "neuron"],
        help="Type of test to run (default: run all tests)",
    )

    parser.add_argument("--plot", action="store_true", help="Generate and display connectivity graph plots")

    parser.add_argument(
        "--save-only", action="store_true", help="Save connectivity graph plots to file without displaying"
    )

    parser.add_argument(
        "--include-3d", action="store_true", help="Include 3D view alongside network view in connectivity graphs"
    )

    args = parser.parse_args()

    # Determine plot_graph setting and show_plot setting
    plot_graph = args.plot or args.save_only
    show_plot = args.plot  # Only show if --plot is used, not if --save-only
    include_3d_view = args.include_3d

    # Run the appropriate test(s)
    if args.test_type == "cylinder":
        test_cylinder_segmentation(plot_graph=plot_graph, show_plot=show_plot, include_3d_view=include_3d_view)
    elif args.test_type == "torus":
        test_torus_segmentation(plot_graph=plot_graph, show_plot=show_plot, include_3d_view=include_3d_view)
    elif args.test_type == "neuron":
        test_neuron_segmentation(plot_graph=plot_graph, show_plot=show_plot, include_3d_view=include_3d_view)
    else:
        # Run all tests - for now, don't pass plot_graph to run_all_tests
        # since it doesn't support the parameter yet
        if plot_graph:
            print("🧪 Running All Segmentation Tests with Connectivity Graphs")
            print("=" * 60)

            # Test: Cylinder segmentation
            print("\n" + "=" * 50)
            cylinder_pass = test_cylinder_segmentation(
                plot_graph=plot_graph, show_plot=show_plot, include_3d_view=include_3d_view
            )

            # Test: Torus segmentation
            print("\n" + "=" * 50)
            torus_pass = test_torus_segmentation(
                plot_graph=plot_graph, show_plot=show_plot, include_3d_view=include_3d_view
            )

            # Test: Neuron segmentation
            print("\n" + "=" * 50)
            neuron_pass = test_neuron_segmentation(
                plot_graph=plot_graph, show_plot=show_plot, include_3d_view=include_3d_view
            )

            # Summary
            print("\n" + "=" * 50)
            print("🏆 TEST SUMMARY")
            print("=" * 50)
            print(f"Cylinder test: {'✅ PASS' if cylinder_pass else '❌ FAIL'}")
            print(f"Torus test: {'✅ PASS' if torus_pass else '❌ FAIL'}")
            print(f"Neuron test: {'✅ PASS' if neuron_pass else '❌ FAIL'}")

            if cylinder_pass and torus_pass and neuron_pass:
                print("🎉 ALL TESTS PASSED!")
            else:
                print("💥 TESTS FAILED!")
        else:
            run_all_tests()

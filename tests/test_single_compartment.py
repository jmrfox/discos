"""
Test script for single compartment segmentation functionality.

This test creates a 1x1 cylinder and segments it with a slice_height of 2.
Since the cylinder is shorter than the slice, it should create a single segment
and a connectivity graph with one node.
"""

import sys
import os

# Add the parent directory to path for importing gencomo
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import trimesh
import networkx as nx

from gencomo import MeshSegmenter
from gencomo.demos import create_cylinder_mesh



validate_single_compartment_connectivity(segmenter, test_name="Single Compartment Test"):
    """Validate that the connectivity graph has exactly one node (single compartment).

    Args:
        segmenter: MeshSegmenter instance with connectivity graph
        test_name: Name of the test for reporting

    Returns:
        bool: True if graph has exactly one node, False otherwise
    """
    print(f"\n🔗 {test_name} - Single Compartment Validation:")

    if segmenter.connectivity_graph is None:
        print("❌ No connectivity graph found")
        return False

    # Check number of nodes (should be 1)
    num_nodes = len(segmenter.connectivity_graph.nodes)
    print(f"  Number of nodes: {num_nodes}")

    if num_nodes != 1:
        print(f"❌ Expected 1 node, got {num_nodes}")
        return False

    # Check number of edges (should be 0 for single node)
    num_edges = len(segmenter.connectivity_graph.edges)
    print(f"  Number of edges: {num_edges}")

    if num_edges != 0:
        print(f"❌ Expected 0 edges for single node, got {num_edges}")
        return False

    # Check that the graph is trivially connected (single node)
    is_connected = nx.is_connected(segmenter.connectivity_graph)
    print(f"  Is connected: {is_connected}")

    if not is_connected:
        print("❌ Single node graph should be trivially connected")
        return False

    # Get the single node
    node_id = list(segmenter.connectivity_graph.nodes)[0]
    print(f"  Single node ID: {node_id}")

    # Validate connected components
    components = list(nx.connected_components(segmenter.connectivity_graph))
    print(f"  Connected components: {len(components)}")

    if len(components) != 1:
        print(f"❌ Expected 1 connected component, got {len(components)}")
        return False

    component = components[0]
    if len(component) != 1:
        print(f"❌ Expected single component with 1 node, got {len(component)}")
        return False

    print("✅ Single compartment connectivity validation passed!")
    return True



test_single_compartment_cylinder(plot_graph: bool = True, show_plot: bool = True):
    """
    Test single compartment segmentation using a 1x1 cylinder with slice_height=2.

    Args:
        plot_graph: Whether to generate and save connectivity graph plot
        show_plot: Whether to display the plot interactively

    Returns:
        bool: True if test passes, False otherwise
    """
    print("=" * 80)
    print("🧪 SINGLE COMPARTMENT CYLINDER SEGMENTATION TEST")
    print("=" * 80)

    # Create a 1x1 cylinder (radius=0.5, length=1.0)
    print("Creating 1x1 cylinder mesh...")
    cylinder = create_cylinder_mesh(radius=0.5, length=1.0, resolution=20)

    print(f"Cylinder properties:")
    print(f"  Volume: {cylinder.volume:.6f}")
    print(f"  Surface area: {cylinder.area:.6f}")
    print(f"  Bounds: {cylinder.bounds}")
    print(f"  Height: {cylinder.bounds[1, 2] - cylinder.bounds[0, 2]:.6f}")

    # Segment the mesh with slice_height=2 (larger than cylinder length of 1)
    slice_height = 2.0
    print(f"\nSegmenting with slice_height = {slice_height} (larger than cylinder length)...")

    segmenter = MeshSegmenter()
    segments = segmenter.segment_mesh(cylinder, slice_height=slice_height)

    print(f"Segmentation results:")
    print(f"  Number of segments: {len(segments)}")

    # Validate that we get exactly one segment
    if len(segments) != 1:
        print(f"❌ Expected 1 segment, got {len(segments)}! Test failed.")
        return False

    segment = segments[0]
    print(f"  Single segment ID: {segment.id}")
    print(f"  Segment volume: {segment.volume:.6f}")
    print(f"  Segment slice index: {segment.slice_index}")

    # Validate volume conservation
    volume_error = abs(segment.volume - cylinder.volume)
    volume_conservation = (segment.volume / cylinder.volume) * 100

    print(f"\n📊 Volume Conservation:")
    print(f"  Original volume: {cylinder.volume:.6f}")
    print(f"  Segment volume: {segment.volume:.6f}")
    print(f"  Volume error: {volume_error:.8f}")
    print(f"  Conservation: {volume_conservation:.2f}%")

    # Volume conservation check
    if volume_error > 1e-6:
        print(f"❌ Volume conservation failed! Error: {volume_error}")
        return False

    print("✅ Volume conservation passed!")

    # Validate surface areas
    print(f"\n📏 Surface Area Analysis:")
    print(f"  External surface area: {segment.external_surface_area:.6f}")
    print(f"  Internal surface area: {segment.internal_surface_area:.6f}")
    print(f"  Total surface area: {segment.external_surface_area + segment.internal_surface_area:.6f}")

    # For a single compartment, we should have external surface area but minimal internal area
    # (internal area might be small due to end caps being treated as cuts)
    if segment.external_surface_area <= 0:
        print(f"❌ External surface area should be > 0, got {segment.external_surface_area}")
        return False

    print("✅ Surface area analysis passed!")

    # Validate connectivity graph for single compartment
    connectivity_valid = validate_single_compartment_connectivity(segmenter, "Single Compartment Test")
    if not connectivity_valid:
        return False

    # Get segmentation statistics
    print(f"\n📈 Segmentation Statistics:")
    stats = segmenter.compute_segmentation_statistics()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key}: {sub_value}")
        else:
            print(f"  {key}: {value}")

    # Validate slicing behavior
    print(f"\n🔪 Slicing Analysis:")
    print(f"  Number of cross-sections: {len(segmenter.cross_sections)}")
    print(f"  Number of slices: {len(segmenter.slices)}")

    # Since slice_height=2 > cylinder_height=1, there should be no cross-sections
    if len(segmenter.cross_sections) != 0:
        print(f"❌ Expected 0 cross-sections, got {len(segmenter.cross_sections)}")
        return False

    # There should be exactly 1 slice (the entire cylinder)
    if len(segmenter.slices) != 1:
        print(f"❌ Expected 1 slice, got {len(segmenter.slices)}")
        return False

    slice_info = segmenter.slices[0]
    print(f"  Single slice: z=[{slice_info['z_min']:.3f}, {slice_info['z_max']:.3f}], index={slice_info['index']}")

    print("✅ Slicing analysis passed!")

    # Plot connectivity graph if requested
    if plot_graph:
        print(f"\n📊 Generating connectivity graph plot...")

        try:
            output_dir = os.path.join(os.path.dirname(__file__), "output")
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, "single_compartment_connectivity_graph.png")

            segmenter.visualize_connectivity_graph(
                save_path=save_path, show_plot=show_plot, include_3d_view=False, figsize=(8, 6)
            )

            print(f"✅ Connectivity graph saved to: {save_path}")

        except Exception as e:
            print(f"⚠️ Could not generate connectivity graph plot: {e}")

    print(f"\n🎉 Single compartment test completed successfully!")
    print("=" * 80)

    return True



test_single_compartment_edge_cases():
    """Test edge cases for single compartment scenarios."""
    print("\n" + "=" * 60)
    print("🧪 SINGLE COMPARTMENT EDGE CASES")
    print("=" * 60)

    test_cases = [
        {"radius": 0.5, "length": 1.0, "slice_height": 1.5, "name": "slice_height > length"},
        {"radius": 0.5, "length": 1.0, "slice_height": 1.0, "name": "slice_height = length"},
        {"radius": 0.5, "length": 1.0, "slice_height": 10.0, "name": "slice_height >> length"},
        {"radius": 1.0, "length": 0.5, "slice_height": 1.0, "name": "short wide cylinder"},
    ]

    all_passed = True

    for i, case in enumerate(test_cases):
        print(f"\n--- Edge Case {i+1}: {case['name']} ---")

        try:
            # Create cylinder
            cylinder = create_cylinder_mesh(radius=case["radius"], length=case["length"], resolution=16)

            print(f"Cylinder: r={case['radius']}, l={case['length']}")
            print(f"Slice height: {case['slice_height']}")

            # Segment
            segmenter = MeshSegmenter()
            segments = segmenter.segment_mesh(cylinder, slice_height=case["slice_height"])

            # Should always get exactly 1 segment
            if len(segments) != 1:
                print(f"❌ Expected 1 segment, got {len(segments)}")
                all_passed = False
                continue

            # Validate connectivity
            if not validate_single_compartment_connectivity(segmenter, f"Edge Case {i+1}"):
                all_passed = False
                continue

            print(f"✅ Edge case {i+1} passed!")

        except Exception as e:
            print(f"❌ Edge case {i+1} failed with error: {e}")
            all_passed = False

    if all_passed:
        print("\n✅ All edge cases passed!")
    else:
        print("\n❌ Some edge cases failed!")

    return all_passed



run_all_single_compartment_tests():
    """Run all single compartment tests."""
    print("🚀 Starting single compartment test suite...")

    # Test 1: Basic single compartment test
    test1_passed = test_single_compartment_cylinder(plot_graph=True, show_plot=False)

    # Test 2: Edge cases
    test2_passed = test_single_compartment_edge_cases()

    # Summary
    print("\n" + "=" * 80)
    print("📋 SINGLE COMPARTMENT TEST SUITE SUMMARY")
    print("=" * 80)

    tests = [
        ("Single Compartment Cylinder", test1_passed),
        ("Edge Cases", test2_passed),
    ]

    all_passed = True
    for test_name, passed in tests:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n🎉 All single compartment tests PASSED!")
    else:
        print("\n💥 Some single compartment tests FAILED!")

    return all_passed


if __name__ == "__main__":
    success = run_all_single_compartment_tests()
    sys.exit(0 if success else 1)

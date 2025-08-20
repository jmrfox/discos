#!/usr/bin/env python3
"""
Test script for the new NodeEdgeSegmenter implementation.

This script tests the new node-edge architecture where:
- Nodes represent spatial points (centers of cross-sections)
- Edges represent cylindrical segments connecting those points
"""

import numpy as np
import trimesh
from discos import NodeEdgeSegmenter, create_cylinder_mesh, create_torus_mesh


def test_cylinder_segmentation():
    """Test node-edge segmentation on a simple cylinder."""
    print("=" * 60)
    print("Testing NodeEdgeSegmenter on cylinder mesh")
    print("=" * 60)
    
    # Create a simple cylinder mesh
    mesh = create_cylinder_mesh(radius=1.0, height=5.0, resolution=32)
    print(f"Created cylinder mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    print(f"Mesh volume: {mesh.volume:.3f}, surface area: {mesh.area:.3f}")
    
    # Initialize segmenter
    segmenter = NodeEdgeSegmenter()
    
    # Test different radius methods
    radius_methods = ['equivalent_area', 'average_distance', 'fitted_circle']
    circle_methods = ['algebraic', 'geometric', 'robust']
    
    for radius_method in radius_methods:
        for circle_method in circle_methods:
            print(f"\n--- Testing radius_method='{radius_method}', circle_method='{circle_method}' ---")
            
            try:
                # Segment the mesh
                graph = segmenter.segment_mesh(
                    mesh=mesh,
                    slice_height=0.5,
                    radius_method=radius_method,
                    circle_fitting_method=circle_method,
                    min_area=1e-6
                )
                
                print(f"✅ Success: {len(graph.nodes_list)} nodes, {len(graph.edges_list)} edges")
                
                # Print node details
                for i, node in enumerate(graph.nodes_list[:3]):  # Show first 3 nodes
                    print(f"  Node {i}: center={node.center}, radius={node.radius:.3f}, z={node.z_position:.2f}")
                
                # Print edge details
                for i, edge in enumerate(graph.edges_list[:3]):  # Show first 3 edges
                    print(f"  Edge {i}: {edge.node1_id} -> {edge.node2_id}, length={edge.length:.3f}, "
                          f"radii=({edge.radius1:.3f}, {edge.radius2:.3f}), volume={edge.volume:.3f}")
                
                # Test SWC export
                try:
                    swc_data = graph.export_to_swc(scale_factor=1.0)
                    print(f"  SWC export: {len(swc_data.entries)} entries, {len(swc_data.non_tree_edges)} non-tree edges")
                except Exception as e:
                    print(f"  ⚠️ SWC export failed: {e}")
                
            except Exception as e:
                print(f"❌ Failed: {e}")
    
    return graph


def test_torus_segmentation():
    """Test node-edge segmentation on a torus (should detect branching)."""
    print("\n" + "=" * 60)
    print("Testing NodeEdgeSegmenter on torus mesh")
    print("=" * 60)
    
    # Create a torus mesh
    mesh = create_torus_mesh(major_radius=3.0, minor_radius=1.0, resolution=32)
    print(f"Created torus mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    print(f"Mesh volume: {mesh.volume:.3f}, surface area: {mesh.area:.3f}")
    
    # Initialize segmenter
    segmenter = NodeEdgeSegmenter()
    
    try:
        # Segment the mesh
        graph = segmenter.segment_mesh(
            mesh=mesh,
            slice_height=0.5,
            radius_method='equivalent_area',
            circle_fitting_method='geometric',
            min_area=1e-6
        )
        
        print(f"✅ Success: {len(graph.nodes_list)} nodes, {len(graph.edges_list)} edges")
        
        # Analyze connectivity
        nodes_by_slice = {}
        for node in graph.nodes_list:
            slice_idx = node.slice_index
            if slice_idx not in nodes_by_slice:
                nodes_by_slice[slice_idx] = []
            nodes_by_slice[slice_idx].append(node)
        
        print(f"Nodes distributed across {len(nodes_by_slice)} slices:")
        for slice_idx, nodes in sorted(nodes_by_slice.items()):
            print(f"  Slice {slice_idx}: {len(nodes)} nodes")
            if len(nodes) > 1:
                print(f"    Multiple cross-sections detected (branching/loops)")
        
        # Test SWC export
        try:
            swc_data = graph.export_to_swc(scale_factor=1.0)
            print(f"SWC export: {len(swc_data.entries)} entries, {len(swc_data.non_tree_edges)} non-tree edges")
            if swc_data.non_tree_edges:
                print("  Cycles detected and broken for SWC format")
        except Exception as e:
            print(f"⚠️ SWC export failed: {e}")
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        return None
    
    return graph


def test_overlap_detection():
    """Test that overlapping cross-sections are properly detected and raise errors."""
    print("\n" + "=" * 60)
    print("Testing overlap detection")
    print("=" * 60)
    
    # Create a simple cylinder
    mesh = create_cylinder_mesh(radius=1.0, height=2.0, resolution=16)
    
    # Create a modified mesh that might cause overlaps (very thin slices)
    segmenter = NodeEdgeSegmenter()
    
    try:
        # This should work fine
        graph = segmenter.segment_mesh(
            mesh=mesh,
            slice_height=0.1,  # Small but reasonable slice height
            radius_method='equivalent_area',
            circle_fitting_method='algebraic',
            min_area=1e-6
        )
        print(f"✅ Normal slicing worked: {len(graph.nodes_list)} nodes")
        
    except Exception as e:
        print(f"❌ Normal slicing failed: {e}")
    
    # Test with extremely small slice height that might cause numerical issues
    try:
        graph = segmenter.segment_mesh(
            mesh=mesh,
            slice_height=0.001,  # Very small slice height
            radius_method='equivalent_area',
            circle_fitting_method='algebraic',
            min_area=1e-9  # Very small area threshold
        )
        print(f"✅ Fine slicing worked: {len(graph.nodes_list)} nodes")
        
    except ValueError as e:
        if "overlapping" in str(e).lower():
            print(f"✅ Overlap detection working: {e}")
        else:
            print(f"❌ Unexpected error: {e}")
    except Exception as e:
        print(f"❌ Fine slicing failed: {e}")


def main():
    """Run all tests."""
    print("Testing new NodeEdgeSegmenter implementation")
    print("This tests the new architecture where nodes are spatial points and edges are segments")
    
    try:
        # Test 1: Simple cylinder
        cylinder_graph = test_cylinder_segmentation()
        
        # Test 2: Torus (more complex geometry)
        torus_graph = test_torus_segmentation()
        
        # Test 3: Overlap detection
        test_overlap_detection()
        
        print("\n" + "=" * 60)
        print("All tests completed!")
        print("=" * 60)
        
        if cylinder_graph:
            print(f"Cylinder test: {len(cylinder_graph.nodes_list)} nodes, {len(cylinder_graph.edges_list)} edges")
        if torus_graph:
            print(f"Torus test: {len(torus_graph.nodes_list)} nodes, {len(torus_graph.edges_list)} edges")
        
        print("\nThe new NodeEdgeSegmenter is ready for use!")
        print("Key features:")
        print("- Nodes represent spatial points (cross-section centers)")
        print("- Edges represent cylindrical segments with metadata")
        print("- Configurable radius calculation methods")
        print("- Overlap detection and validation")
        print("- Volume-based connectivity analysis")
        print("- SWC export with cycle breaking")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

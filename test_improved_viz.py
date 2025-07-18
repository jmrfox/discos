import trimesh
import numpy as np
import matplotlib.pyplot as plt
import os
from gencomo.mesh import MeshSegmenter
from gencomo.model import SegmentGraph

# Create output directory
output_dir = os.path.join(os.path.dirname(__file__), "viz_output")
os.makedirs(output_dir, exist_ok=True)

def test_cylinder_visualization():
    """Test visualization with a simple cylinder"""
    print("\n=== Testing with cylinder ===")
    
    # Create a cylinder for testing
    cylinder = trimesh.primitives.Cylinder(radius=1.0, height=5.0, sections=16)
    
    # Segment the cylinder
    segmenter = MeshSegmenter()
    segments = segmenter.segment_mesh(cylinder, slice_height=1.0, min_volume=0.01)
    segment_graph = segmenter.get_segment_graph()
    
    print(f"Created {len(segments)} segments")
    print(f"Graph has {len(segment_graph.graph.nodes())} nodes and {len(segment_graph.graph.edges())} edges")
    
    # Print segment properties
    print("\nSegment properties:")
    for i, segment in enumerate(segments):
        print(f"Segment {i}: ID={segment.id}, Volume={segment.volume:.4f}, Z-range: {segment.z_min:.2f} to {segment.z_max:.2f}")
        if hasattr(segment, 'centroid') and segment.centroid is not None:
            print(f"  Centroid: ({segment.centroid[0]:.2f}, {segment.centroid[1]:.2f}, {segment.centroid[2]:.2f})")
    
    # Test visualization with different parameters
    
    # 1. Default parameters
    print("\nTesting with default parameters...")
    fig1 = segment_graph.visualize(
        color_by='slice_index',
        show_plot=False,
        repulsion_strength=0.1,
        iterations=50
    )
    save_path1 = os.path.join(output_dir, "cylinder_default.png")
    fig1.savefig(save_path1)
    print(f"Default visualization saved to {save_path1}")
    
    # 2. Higher repulsion strength
    print("\nTesting with higher repulsion strength...")
    fig2 = segment_graph.visualize(
        color_by='volume',
        show_plot=False,
        repulsion_strength=0.3,
        iterations=100
    )
    save_path2 = os.path.join(output_dir, "cylinder_high_repulsion.png")
    fig2.savefig(save_path2)
    print(f"High repulsion visualization saved to {save_path2}")
    
    # 3. Using y-axis for horizontal positioning
    print("\nTesting with y-axis for horizontal positioning...")
    fig3 = segment_graph.visualize(
        color_by='slice_index',
        show_plot=False,
        repulsion_strength=0.2,
        iterations=75,
        horizontal_axis='y'
    )
    save_path3 = os.path.join(output_dir, "cylinder_y_horizontal.png")
    fig3.savefig(save_path3)
    print(f"Y-axis horizontal visualization saved to {save_path3}")
    
    return segment_graph

def test_with_multiple_segments_at_same_z():
    """Test visualization with multiple segments at the same z-level"""
    print("\n=== Testing with multiple segments at same z-level ===")
    
    # Create a custom graph with multiple segments at the same z-level
    segment_graph = SegmentGraph()
    
    # Add nodes with same z-level but different x,y
    for i in range(5):
        segment_id = f"seg_0_{i}"
        # Add node directly to the underlying networkx graph
        segment_graph.graph.add_node(
            segment_id,
            volume=1.0 + i*0.5,
            centroid=[i*0.5, i*0.3, 0.0],  # Same z, different x,y
            slice_index=0
        )
        # Also add to segments list
        segment_graph.segments.append(segment_id)
    
    # Add nodes at different z-levels
    for z in range(1, 4):
        for i in range(3):
            segment_id = f"seg_{z}_{i}"
            # Add node directly to the underlying networkx graph
            segment_graph.graph.add_node(
                segment_id,
                volume=1.0 + i*0.5,
                centroid=[i*0.5, i*0.3, z*1.0],  # Different z
                slice_index=z
            )
            # Also add to segments list
            segment_graph.segments.append(segment_id)
    
    # Add some edges
    for z in range(3):
        for i in range(3):
            segment_graph.graph.add_edge(f"seg_{z}_{i}", f"seg_{z+1}_{i}")
    
    print(f"Created custom graph with {len(segment_graph.graph.nodes())} nodes")
    
    # Test visualization
    fig = segment_graph.visualize(
        color_by='slice_index',
        show_plot=False,
        repulsion_strength=0.2,
        iterations=100
    )
    save_path = os.path.join(output_dir, "custom_graph.png")
    fig.savefig(save_path)
    print(f"Custom graph visualization saved to {save_path}")
    
    return segment_graph

if __name__ == "__main__":
    # Test with cylinder
    cylinder_graph = test_cylinder_visualization()
    
    # Test with custom graph having multiple segments at same z-level
    custom_graph = test_with_multiple_segments_at_same_z()
    
    print(f"\nAll visualizations saved to: {output_dir}")
    
    # Show the plots
    plt.show()

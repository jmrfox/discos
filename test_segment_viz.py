import trimesh
import numpy as np
import matplotlib.pyplot as plt
import os
from gencomo.mesh import MeshSegmenter
from gencomo.model import SegmentGraph
from gencomo.utils import data_path

# Create a test directory if it doesn't exist
output_dir = os.path.join(os.path.dirname(__file__), "test_output")
os.makedirs(output_dir, exist_ok=True)

def test_with_cylinder():
    """Test visualization with a simple cylinder"""
    print("\n=== Testing with cylinder ===")
    cylinder = trimesh.primitives.Cylinder(radius=1.0, height=5.0, sections=16)
    print(f"Cylinder created with {len(cylinder.vertices)} vertices and {len(cylinder.faces)} faces")
    
    # Segment the cylinder
    segmenter = MeshSegmenter()
    segments = segmenter.segment_mesh(cylinder, slice_height=1.0, min_volume=0.01)
    print(f"Created {len(segments)} segments")
    
    # Get segment graph
    segment_graph = segmenter.get_segment_graph()
    print(f"Graph has {len(segment_graph.graph.nodes())} nodes and {len(segment_graph.graph.edges())} edges")
    
    # Print node properties
    print("\nNode properties:")
    for node, data in segment_graph.graph.nodes(data=True):
        print(f"Node {node}:")
        for key, value in data.items():
            if isinstance(value, (int, float, str)):
                print(f"  {key}: {value}")
            elif isinstance(value, np.ndarray) and len(value) <= 3:
                print(f"  {key}: {value}")
    
    # Test visualization with default parameters
    print("\nVisualizing with default parameters...")
    fig1 = segment_graph.visualize(
        color_by='slice_index',
        show_plot=False
    )
    save_path1 = os.path.join(output_dir, "cylinder_default.png")
    fig1.savefig(save_path1)
    print(f"Default visualization saved to {save_path1}")
    
    # Test visualization with custom parameters
    print("\nVisualizing with custom parameters...")
    fig2 = segment_graph.visualize(
        color_by='volume',
        node_scale=500.0,
        repulsion_strength=0.2,
        iterations=100,
        show_plot=False
    )
    save_path2 = os.path.join(output_dir, "cylinder_custom.png")
    fig2.savefig(save_path2)
    print(f"Custom visualization saved to {save_path2}")
    
    return segment_graph

def test_with_real_mesh():
    """Test visualization with a real mesh if available"""
    try:
        # Try to load a real mesh from the data directory
        mesh_path = data_path("meshes/test_neuron.obj")
        if not os.path.exists(mesh_path):
            print("\n=== No real mesh found for testing ===")
            return None
            
        print(f"\n=== Testing with real mesh: {mesh_path} ===")
        mesh = trimesh.load(mesh_path)
        print(f"Mesh loaded with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")
        
        # Segment the mesh
        segmenter = MeshSegmenter()
        segments = segmenter.segment_mesh(mesh, slice_height=1.0, min_volume=0.01)
        print(f"Created {len(segments)} segments")
        
        # Get segment graph
        segment_graph = segmenter.get_segment_graph()
        print(f"Graph has {len(segment_graph.graph.nodes())} nodes and {len(segment_graph.graph.edges())} edges")
        
        # Visualize with default parameters
        fig = segment_graph.visualize(
            color_by='slice_index',
            show_plot=False
        )
        save_path = os.path.join(output_dir, "real_mesh_viz.png")
        fig.savefig(save_path)
        print(f"Real mesh visualization saved to {save_path}")
        
        return segment_graph
    except Exception as e:
        print(f"Error testing with real mesh: {e}")
        return None

if __name__ == "__main__":
    # Test with cylinder
    cylinder_graph = test_with_cylinder()
    
    # Test with real mesh if available
    real_mesh_graph = test_with_real_mesh()
    
    print("\n=== Visualization tests completed ===")
    print(f"Output files saved to: {output_dir}")
    
    # Show the plots at the end
    plt.show()

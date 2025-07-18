import trimesh
import numpy as np
from gencomo.mesh import MeshSegmenter
from gencomo.model import SegmentGraph

# Create a simple cylinder for testing
cylinder = trimesh.primitives.Cylinder(radius=1.0, height=5.0, sections=16)

# Segment the cylinder
segmenter = MeshSegmenter()
segments = segmenter.segment_mesh(cylinder, slice_height=1.0, min_volume=0.01)
segment_graph = segmenter.get_segment_graph()

# Test different visualization parameters
print(f"Number of segments: {len(segment_graph.segments)}")
print("Testing visualization with default parameters...")
segment_graph.visualize(show_plot=True)

print("\nTesting visualization with custom parameters...")
segment_graph.visualize(
    color_by='volume',
    node_scale=500.0,  # Smaller nodes
    repulsion_strength=0.2,  # Stronger repulsion
    iterations=200,  # More iterations for better positioning
    show_plot=True
)

print("\nDone!")

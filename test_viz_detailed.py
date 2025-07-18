import trimesh
import numpy as np
import matplotlib.pyplot as plt
from gencomo.mesh import MeshSegmenter
from gencomo.model import SegmentGraph

# Create a simple cylinder for testing
print("Creating test cylinder...")
cylinder = trimesh.primitives.Cylinder(radius=1.0, height=5.0, sections=16)
print(f"Cylinder created with {len(cylinder.vertices)} vertices and {len(cylinder.faces)} faces")

# Segment the cylinder
print("\nSegmenting cylinder...")
segmenter = MeshSegmenter()
segments = segmenter.segment_mesh(cylinder, slice_height=1.0, min_volume=0.01)
print(f"Created {len(segments)} segments")

# Get segment graph
segment_graph = segmenter.get_segment_graph()
print(f"Graph has {len(segment_graph.graph.nodes())} nodes and {len(segment_graph.graph.edges())} edges")

# Print segment properties
print("\nSegment properties:")
for i, segment in enumerate(segments):
    print(f"Segment {i}: ID={segment.id}, Volume={segment.volume:.4f}, Z-range: {segment.z_min:.2f} to {segment.z_max:.2f}")
    if hasattr(segment, 'centroid') and segment.centroid is not None:
        print(f"  Centroid: ({segment.centroid[0]:.2f}, {segment.centroid[1]:.2f}, {segment.centroid[2]:.2f})")

# Test visualization with custom parameters
print("\nVisualizing segment graph...")
fig = segment_graph.visualize(
    color_by='slice_index',
    node_scale=500.0,
    repulsion_strength=0.2,
    iterations=100,
    show_plot=False  # Don't show yet, save first
)

# Save the visualization
save_path = "segment_graph_visualization.png"
plt.savefig(save_path)
print(f"Visualization saved to {save_path}")

# Now show the plot
plt.show()

print("\nDone!")

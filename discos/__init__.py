"""
DISCOS: DISrete COllinear Skeletonization

A Python package for 3D mesh manipulation and skeletonization of complex neuronal morphologies.
Provides tools for mesh processing, segmentation, and conversion to SWC format using discrete
collinear skeletonization algorithms.
"""

__version__ = "0.1.0"
__author__ = "Jordan M. R. Fox"
__email__ = "jordanmrfox@gmail.com"

# Mesh functions (primary 3D representation)
from .mesh import MeshManager

# Segmentation functions
from .segmentation import MeshSegmenter, Segment, SegmentGraph, NodeEdgeSegmenter, NodeEdgeGraph, GraphNode, GraphEdge

# Demo mesh functions from demos module
from .demos import (
    create_cylinder_mesh,
    create_torus_mesh,
    create_branching_mesh,
    create_demo_neuron_mesh,
    save_demo_meshes,
)


# Utility functions
from .utils import data_path

__all__ = [
    # Mesh processing (primary 3D representation)
    "MeshManager",
    "MeshSegmenter",
    "Segment",
    "SegmentGraph",
    # New node-edge architecture
    "NodeEdgeSegmenter",
    "NodeEdgeGraph",
    "GraphNode",
    "GraphEdge",
    # Demo mesh functions
    "create_cylinder_mesh",
    "create_torus_mesh",
    "create_branching_mesh",
    "create_demo_neuron_mesh",
    "save_demo_meshes",
    # Utility functions
    "data_path",
]

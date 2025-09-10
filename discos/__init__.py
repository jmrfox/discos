"""
DISCOS: DISrete COllinear Skeletonization

A Python package for 3D mesh manipulation and skeletonization of complex neuronal morphologies.
Provides tools for mesh processing, segmentation, and conversion to SWC format using discrete
collinear skeletonization algorithms.
"""

__version__ = "0.1.0"
__author__ = "Jordan M. R. Fox"
__email__ = "jordanmrfox@gmail.com"

# Demo mesh functions from demos module
from .demo import (
    create_branching_mesh,
    create_cylinder_mesh,
    create_demo_neuron_mesh,
    create_torus_mesh,
    save_demo_meshes,
)

# Mesh functions (primary 3D representation)
from .mesh import MeshManager

# Utility functions
from .path import data_path
from .polylines import PolylinesSkeleton

# Skeletonization functions
from .skeleton import CrossSection, Junction, Segment, SkeletonGraph
from .trace import TraceOptions, build_traced_skeleton_graph, trace_polylines_to_swc

__all__ = [
    # Mesh processing (primary 3D representation)
    "MeshManager",
    # Skeletonization functions
    "CrossSection",
    "Segment",
    "SkeletonGraph",
    "Junction",
    # Polylines skeleton handler
    "PolylinesSkeleton",
    # Tracing utilities (polyline-guided SWC)
    "TraceOptions",
    "build_traced_skeleton_graph",
    "trace_polylines_to_swc",
    # Demo mesh functions
    "create_cylinder_mesh",
    "create_torus_mesh",
    "create_branching_mesh",
    "create_demo_neuron_mesh",
    "save_demo_meshes",
    # Path functions
    "data_path",
]

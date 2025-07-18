"""
GenCoMo: GENeral-morphology COmpartmental MOdeling

A Python package for compartmental simulation of neurons with complex morphologies
using mesh-based geometric approaches rather than traditional SWC cylinder models.
"""

__version__ = "0.1.0"
__author__ = "Jordan M. R. Fox"
__email__ = "jordanmrfox@gmail.com"

# Mesh functions from mesh submodule (primary 3D representation)
from .mesh import (
    MeshProcessor,
    MeshSegmenter,
    visualize_mesh_3d,
    analyze_mesh,
    print_mesh_analysis,
    repair_mesh,
    preprocess_mesh,
    visualize_mesh_slice_interactive,
    visualize_mesh_slice_grid,
)

# Demo mesh functions from demos module
from .demos import (
    create_cylinder_mesh,
    create_torus_mesh,
    create_branching_mesh,
    create_demo_neuron_mesh,
    save_demo_meshes,
)

# Core modules and functionality
# Note: slicer, regions, and graph modules moved to dev_storage/old_modules
# The current approach uses MeshSegmenter for 3D mesh segmentation
from .ode import ODESystem
from .simulation import Simulator

# Utility functions
from .utils import data_path

__all__ = [
    # Mesh processing (primary 3D representation)
    "MeshProcessor",
    "MeshSegmenter",
    "visualize_mesh_3d",
    "analyze_mesh",
    "repair_mesh",
    "preprocess_mesh",
    "visualize_mesh_slice_interactive",
    "visualize_mesh_slice_grid",
    # Demo mesh functions
    "create_cylinder_mesh",
    "create_torus_mesh",
    "create_branching_mesh",
    "create_demo_neuron_mesh",
    "save_demo_meshes",
    # Core modules
    # Note: ZAxisSlicer, RegionDetector, GraphBuilder moved to dev_storage/old_modules
    "MeshSegmenter",
    "Segment",
    "ODESystem",
    "Simulator",
    # Utility functions
    "data_path",
]

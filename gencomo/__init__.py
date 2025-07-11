"""
GenCoMo: GENeral-morphology COmpartmental MOdeling

A Python package for compartmental simulation of neurons with complex morphologies
using mesh-based geometric approaches rather than traditional SWC cylinder models.
"""

__version__ = "0.1.0"
__author__ = "Jordan Fox"
__email__ = "jmrfox@example.com"

from .core import Neuron, Compartment, CompartmentGraph

# Mesh functions from mesh module
from .mesh import (
    MeshProcessor,
    visualize_mesh_3d,
    analyze_mesh_properties,
)

# Demo mesh functions from demos module
from .demos import (
    create_cylinder_mesh,
    create_y_shaped_mesh,
    create_mesh_with_hole,
    save_test_meshes,
    create_cylinder_zstack,
    create_y_shaped_zstack,
    create_hole_zstack,
    save_test_zstacks,
)
from .slicer import ZAxisSlicer
from .regions import RegionDetector
from .graph import GraphBuilder
from .ode import ODESystem
from .simulation import Simulator

# Z-stack functions (core functionality) from dedicated module
from .z_stack import (
    visualize_zstack_3d,
    visualize_zstack_slices,
    compare_zstack_slices,
    save_zstack_data,
    load_zstack_data,
    analyze_zstack_properties,
    mesh_to_zstack,
    load_mesh_file_to_zstack,
)


__all__ = [
    "Neuron",
    "Compartment",
    "CompartmentGraph",
    "MeshProcessor",
    "ZAxisSlicer",
    "RegionDetector",
    "GraphBuilder",
    "ODESystem",
    "Simulator",
    # Z-stack functions (core functionality)
    "visualize_zstack_3d",
    "visualize_zstack_slices",
    "compare_zstack_slices",
    "save_zstack_data",
    "load_zstack_data",
    "analyze_zstack_properties",
    # Mesh conversion functions
    "mesh_to_zstack",
    "load_mesh_file_to_zstack",
    # Demo mesh functions
    "create_cylinder_mesh",
    "create_y_shaped_mesh",
    "create_mesh_with_hole",
    "save_test_meshes",
    # Demo z-stack functions
    "create_cylinder_zstack",
    "create_y_shaped_zstack",
    "create_hole_zstack",
    "save_test_zstacks",
    # Mesh processing functions
    "visualize_mesh_3d",
    "analyze_mesh_properties",
]

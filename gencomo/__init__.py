"""
GenCoMo: GENeral-morphology COmpartmental MOdeling

A Python package for compartmental simulation of neurons with complex morphologies
using mesh-based geometric approaches rather than traditional SWC cylinder models.
"""

__version__ = "0.1.0"
__author__ = "Jordan Fox"
__email__ = "jmrfox@example.com"

from .core import Neuron, Compartment, CompartmentGraph
from .mesh import MeshProcessor
from .slicer import ZAxisSlicer
from .regions import RegionDetector
from .graph import GraphBuilder
from .ode import ODESystem
from .simulation import Simulator
from .visualization import (
    # Z-stack functions (primary format)
    create_cylinder_zstack,
    create_y_shaped_zstack,
    create_hole_zstack,
    visualize_zstack_3d,
    save_zstack_data,
    load_zstack_data,
    analyze_zstack_properties,
    # Mesh conversion functions
    mesh_to_zstack,
    load_mesh_file_to_zstack,
    # Legacy mesh functions (for backward compatibility)
    create_cylinder_mesh,
    create_y_shaped_mesh,
    create_mesh_with_hole,
    visualize_mesh_3d,
    save_test_meshes,
    analyze_mesh_properties,
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
    # Z-stack functions (primary format)
    "create_cylinder_zstack",
    "create_y_shaped_zstack",
    "create_hole_zstack",
    "visualize_zstack_3d",
    "save_zstack_data",
    "load_zstack_data",
    "analyze_zstack_properties",
    # Mesh conversion functions
    "mesh_to_zstack",
    "load_mesh_file_to_zstack",
    # Legacy mesh functions
    "create_cylinder_mesh",
    "create_y_shaped_mesh",
    "create_mesh_with_hole",
    "visualize_mesh_3d",
    "save_test_meshes",
    "analyze_mesh_properties",
]

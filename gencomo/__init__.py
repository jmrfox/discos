"""
GenCoMo: GENeral-morphology COmpartmental MOdeling

A Python package for compartmental simulation of neurons with complex morphologies
using mesh-based geometric approaches rather than traditional SWC cylinder models.
"""

__version__ = "0.1.0"
__author__ = "Jordan Fox"
__email__ = "jmrfox@example.com"

from .core import Neuron, Compartment, CompartmentGraph

# Mesh functions from mesh module (primary 3D representation)
from .mesh import (
    MeshProcessor,
    visualize_mesh_3d,
    analyze_mesh,
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
from .slicer import ZAxisSlicer
from .regions import RegionDetector
from .graph import GraphBuilder
from .ode import ODESystem
from .simulation import Simulator

__all__ = [
    # Core classes
    "Neuron",
    "Compartment",
    "CompartmentGraph",
    # Mesh processing (primary 3D representation)
    "MeshProcessor",
    "visualize_mesh_3d",
    "analyze_mesh",
    "visualize_mesh_slice_interactive",
    "visualize_mesh_slice_grid",
    # Demo mesh functions
    "create_cylinder_mesh",
    "create_torus_mesh",
    "create_branching_mesh",
    "create_demo_neuron_mesh",
    "save_demo_meshes",
    # Core modules
    "ZAxisSlicer",
    "RegionDetector",
    "GraphBuilder",
    "ODESystem",
    "Simulator",
]

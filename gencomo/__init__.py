"""
GenCoMo: GENeral-morphology COmpartmental MOdeling

A Python package for compartmental simulation of neurons with complex morphologies
using mesh-based geometric approaches rather than traditional SWC cylinder models.
"""

__version__ = "0.1.0"
__author__ = "Jordan M. R. Fox"
__email__ = "jordanmrfox@gmail.com"

# Mesh functions (primary 3D representation)
from .mesh import MeshManager

# Segmentation functions
from .segmentation import MeshSegmenter, Segment, SegmentGraph

# Demo mesh functions from demos module
from .demos import (
    create_cylinder_mesh,
    create_torus_mesh,
    create_branching_mesh,
    create_demo_neuron_mesh,
    save_demo_meshes,
)

# Parameter management system
from .parameters import IndependentParameter, ParameterSet, DerivedParameter, ParameterBank

# Core modules and functionality
from .ode import ODESystem
from .simulation import Simulator

# Utility functions
from .utils import data_path

__all__ = [
    # Mesh processing (primary 3D representation)
    "MeshManager",
    "MeshSegmenter",
    "Segment",
    "SegmentGraph",
    # Demo mesh functions
    "create_cylinder_mesh",
    "create_torus_mesh",
    "create_branching_mesh",
    "create_demo_neuron_mesh",
    "save_demo_meshes",
    # Parameter management
    "IndependentParameter",
    "ParameterSet",
    "DerivedParameter",
    "ParameterBank",
    # Core modules
    "ODESystem",
    "Simulator",
    # Utility functions
    "data_path",
]

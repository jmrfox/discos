"""
Mesh processing submodule for GenCoMo.

This submodule contains all mesh-related functionality including:
- Mesh loading, processing, and analysis
- Mesh preprocessing and repair
- Mesh visualization
- Mesh segmentation
"""

from .processor import MeshProcessor
from .segmentation import MeshSegmenter
from .utils import (
    analyze_mesh,
    print_mesh_analysis,
    repair_mesh,
    preprocess_mesh,
    visualize_mesh_3d,
    visualize_mesh_slice_interactive,
    visualize_mesh_slice_grid,
)

__all__ = [
    "MeshProcessor",
    "MeshSegmenter",
    "analyze_mesh",
    "repair_mesh",
    "preprocess_mesh",
    "visualize_mesh_3d",
    "visualize_mesh_slice_interactive",
    "visualize_mesh_slice_grid",
]

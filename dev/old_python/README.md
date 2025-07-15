# Old Modules Archive

This directory contains old modules that were moved from the main gencomo package on July 13, 2025.

## Moved Files:
- `regions.py` - Region detection from 2D cross-sections
- `graph.py` - Graph builder for compartment connectivity 
- `slicer.py` - Z-axis slicing utilities
- `cli.py` - Command line interface (depended on above modules)

## Reason for Move:
These modules represented an older approach to mesh compartmentalization based on:
1. 2D cross-sectional slicing
2. Region detection in cross-sections
3. Graph building from 2D regions

The current approach uses **3D mesh segmentation** (`MeshSegmenter`) which:
- Works directly with 3D mesh geometry
- Provides better volume and surface area conservation
- Has been thoroughly tested with cylinder and torus geometries
- Achieves perfect volume conservation and excellent surface area calculations

## Status:
- All old modules have been successfully moved
- Package imports work correctly without these modules
- All tests pass (cylinder and torus segmentation)
- CLI entry point disabled in setup.py

## Current Active Modules:
- `segmentation.py` - Main 3D mesh segmentation (MeshSegmenter)
- `mesh.py` - Mesh processing and visualization
- `demos.py` - Demo mesh generation functions
- `core.py` - Core data structures
- `ode.py` - ODE system definitions
- `simulation.py` - Simulation framework
- `voxels.py` - Voxel-based operations
- `z_stack.py` - Z-stack processing utilities

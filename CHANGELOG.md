# Changelog

All notable changes to DISCOS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of DISCOS
- Mesh-based compartmental modeling for complex neuronal morphologies
- Z-axis slicing with contour extraction
- Region detection with hole handling
- Graph-based compartment connectivity
- Hodgkin-Huxley biophysical simulation
- Command-line interface
- Basic and advanced examples
- Comprehensive test suite

### Features
- `MeshProcessor` - Load, preprocess, and repair neuronal meshes
- `ZAxisSlicer` - Slice meshes along z-axis with configurable resolution
- `RegionDetector` - Detect closed regions in slices with area filtering
- `GraphBuilder` - Build compartment connectivity using overlap/distance methods
- `ODESystem` - Hodgkin-Huxley dynamics with temperature correction
- `Simulator` - High-level simulation interface with stimulus protocols
- Support for STL, PLY, OBJ mesh formats via trimesh
- Multiple integration methods (RK45, DOP853, Radau)
- Results export to NPZ, HDF5, CSV formats
- Spike detection and propagation velocity analysis

## [0.1.0] - 2025-01-XX

### Added
- Initial package structure and core modules
- Basic functionality for mesh-based compartmental modeling
- Documentation and examples

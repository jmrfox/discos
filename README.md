# DISCOS: Discrete Collinear Skeletonization

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

DISCOS is a Python package for 3D mesh manipulation and skeletonization. It provides tools for processing meshes, performing segmentation using our "discrete collinear skeletonization" algorithm, and converting the results to SWC format for use in simulation software.
Development is currently focused on handling complex neuronal morphologies, but the skeletonization algorithm is generalizable to other types of meshes.

## Key Features

- **3D Mesh Processing**: Load, manipulate, and analyze mesh geometries
- **Collinear Skeletonization**: Advanced skeletonization algorithm for complex geometries  
- **Graph-based Representation**: Build connectivity graphs from skeletonized meshes
- **SWC Export**: Convert processed meshes to SWC format for simulation software
- **Interactive Visualization**: Explore mesh data and skeletonization results
- **Demo Functions**: Pre-built functions for creating test geometries and workflows

## Installation

### From source

```bash
git clone https://github.com/jmrfox/discos.git
cd discos
pip install -e .
```

## Documentation

- Live site: [https://jmrfox.github.io/discos/](https://jmrfox.github.io/discos/)
- Build locally with pdoc:

```bash
uv run python -m pdoc -o docs discos
```

### Publish to GitHub Pages

1. Commit and push the generated `docs/` folder to the `main` branch.
2. In your repository on GitHub, go to `Settings` → `Pages`.
3. Under "Build and deployment", set:
   - Source: "Deploy from a branch"
   - Branch: `main` and folder `/docs`.
4. Save. Wait 1–2 minutes for deployment, then visit the site URL shown (should be `https://jmrfox.github.io/discos/`).

### Dependencies

DISCOS requires several scientific computing packages:

```bash
pip install -r requirements.txt
```

Key dependencies include:

- `numpy`, `scipy` - numerical computing
- `trimesh` - mesh processing
- `networkx` - graph algorithms for connectivity
- `matplotlib`, `plotly` - visualization (optional)

## Quick Start

### Basic Workflow

```python
import logging
from discos.mesh import MeshManager
from discos.skeleton import skeletonize

# Configure logging (optional)
logging.basicConfig(level=logging.INFO)

# 1. Load a mesh
mm = MeshManager()
mesh = mm.load_mesh("path/to/mesh.stl")

# 2. Build the skeleton graph
skel = skeletonize(mm, n_slices=31, verbosity=1)

# 3. Export to SWC
skel.to_swc("mesh.skel.swc", type_index=5, annotate_cycles=True, cycle_mode="remove_edge")
```

### Working with Demo Functions

```python
from discos.demo import (
    create_cylinder_mesh,
    create_torus_mesh,
    create_branching_mesh,
)

# Create demo meshes
cylinder = create_cylinder_mesh(radius=5.0, length=20.0)
torus = create_torus_mesh(major_radius=10.0, minor_radius=3.0)
branching = create_branching_mesh(trunk_radius=4.0, branch_angle=45.0)
```

## Core Concepts

**Mesh Processing**: DISCOS starts with a 3D mesh representing the neuronal membrane. The mesh can be preprocessed (centered, aligned, validated) before analysis.

**Skeletonization**: DISCOS uses a specialized discrete collinear skeletonization algorithm designed for neuronal morphologies, different from traditional neural skeletonization approaches.

**Graph Construction**: Connections between segments are established based on geometric relationships, creating a graph representation of the morphology.

**SWC Export**: The final graph structure is converted to SWC format, providing compatibility with simulation software like NEURON and Arbor.

### Segmentation and Skeletonization (Detailed)

This section describes the current algorithm implemented in `discos/skeleton.py` in precise terms.

#### Inputs and Parameters

- `mesh_or_manager`: a `trimesh.Trimesh` or `MeshManager`.
- `n_slices` (int): number of vertical bands. Internally yields `n_slices-1` interior cuts plus two bounding planes.
- `radius_mode` (str): how the cross-section radius is derived. Default `equivalent_area` computes r = sqrt(A / π).
- `validate_volume` (bool, default True): verify that the sum of band volumes matches the mesh volume within `volume_tol`.
- `volume_tol` (float, default 0.05): relative tolerance for volume validation.
- `enforce_connected` (bool, default True): ensure the final graph is a single connected component.
- `connect_isolated_terminals` (bool, default True): safety net to connect isolated terminal nodes at the extremal slices to a nearest neighbor in the adjacent slice.
- `verbosity` (0/1/2) or legacy `verbose`: controls logging output; see section “Logging and Verbosity”.

#### Step 1 — Validation

- Require watertightness (`mesh.is_watertight`).
- Require a single connected component (`mesh.split(only_watertight=False)` yields 1); if the split itself fails, proceed with a warning at verbosity≥1.
- Require a finite, strictly increasing z-range.

Edge case warning: If `n_slices` is a power of two, a `RuntimeWarning` is emitted and a warning is logged due to known connectivity edge cases on certain shapes (e.g., tori).

#### Step 2 — Uniform Slicing

- Compute equally spaced z coordinates from zmin to zmax. The open bands are [z0,z1], [z1,z2], …, [z_{n-1}, z_n].
- The interior cuts are at z1…z_{n-1}. The bounding planes at z0 and z_n are handled with a small probe offset to find terminal sections.

#### Step 3 — Cross-sections and Junction Fitting

- For each cut plane z=c, intersect the mesh with the plane normal to +Z to obtain a set of closed 2D polygons (via `trimesh.Trimesh.section` and conversion to `shapely` polygons).
- Enforce non-overlap within the same cut:
  - For any polygon pair (Pi,Pj), compute intersection area; if area > tolerance, raise `ValueError` with optional diagnostic plotting (guarded) and log concise diagnostics at DEBUG.
- For each polygon, compute attributes:
  - Area A (2D polygon area) and centroid (x,y) in the plane; the z of the cut is attached to make a 3D center.
  - Radius r by `radius_mode`:
    - `equivalent_area` (default): r = sqrt(A/π).
    - Other modes can be added; the README reflects the default here.
- Create a `Junction` per polygon with fields: `id`, `slice_index`, `cross_section_index`, `z`, `center=(x,y,z)`, `radius=r`, `area=A`.

#### Step 4 — Band Extraction and Component Identification

- For each band [z_low, z_high], extract the submesh bounded by these planes (two-plane slice). Split the band submesh into connected components (one or more per band).
- For each component, identify local cross-sections near the two faces of the band:
  - Probe slightly inside the band near z_low to compute “lower” centroids and near z_high for “upper” centroids. If the immediate probe yields none, incrementally step slightly further inside within a small budget until some centroids are found or the budget is exhausted.
- Match those local centroids to the previously created junctions located at exactly z_low and z_high for the respective slice indices. Matching is proximity-based in the (x,y) plane.

Matching details (lower side; symmetric for upper):

- Start by greedy nearest-centroid matching from local centroids to candidate junctions in the same cut; deduplicate preserving order.
- Robust minimal fallbacks for the first/last bands:
  - If no lower junctions matched in band 0 but there are junctions in slice 0, choose the single closest one to the mean of upper centroids.
  - If no upper junctions matched in band 0 but there are junctions in slice 1, choose the single closest one to the mean of lower centroids.
  - Similarly, when needed on the top band to avoid stranded nodes.

#### Step 5 — Edge Construction (Connectivity)

- For each component, construct a fully bipartite set of edges between its matched lower junctions and matched upper junctions.
- Edges are only between adjacent slices and never within the same slice.
- Each edge is annotated with a `segment_id` and the band bounds (`z_lower`, `z_upper`). Where available, component `volume` contributes to accumulated band volume used in validation.

#### Step 6 — Volume Validation (Optional)

- Sum the absolute volumes of all band components. Compare to the absolute mesh volume.
- If the relative error exceeds `volume_tol`, raise `ValueError` (logged); otherwise log an INFO summary with the relative error.

#### Step 7 — Safety Net (Optional)

- If enabled, connect isolated terminal junctions in the bottom slice (0) to their nearest neighbor in slice 1, and isolated terminal junctions in the top slice (n-1) to their nearest neighbor in the preceding slice. These edges are marked with a reserved `segment_id` so they can be identified.

#### Output Graph and Export

- The resulting `SkeletonGraph` stores the `junctions`, `cross_sections`, and `segments`, and maintains an undirected `networkx.Graph` `G` with node and edge attributes.
- If `enforce_connected` is True, validate that the graph is a single connected component or raise.
- Export to SWC via `SkeletonGraph.to_swc(path, ...)`. If cycles exist and a tree is required, choose a cycle-breaking strategy such as removing a single edge per cycle. See the docstring for available options.

Example: building a skeleton and writing SWC

```python
import logging
import numpy as np
from discos.mesh import MeshManager
from discos.skeleton import skeletonize
from discos.demo import create_torus_mesh

# Configure logging
logging.basicConfig(level=logging.INFO)  # set DEBUG for detailed traces

mesh = create_torus_mesh(major_radius=np.pi, minor_radius=np.pi/3)
mm = MeshManager(mesh)

skel = skeletonize(mm, n_slices=31, validate_volume=True, verbosity=1)
skel.to_swc("torus.skel.swc", type_index=5, annotate_cycles=True, cycle_mode="remove_edge")
```

---

## Logging and Verbosity

DISCOS uses Python's `logging` infrastructure in its modules (`discos.skeleton`, `discos.mesh`, `discos.demo`, `discos.path`). You control output globally or per-module.

- Verbosity levels in `skeletonize()`:
  - `verbosity=0`: silent (default unless `verbose=True` is used)
  - `verbosity=1`: basic run info (high-level progress)
  - `verbosity=2`: detailed diagnostics of each step
  - `verbose=True` (legacy) maps to `verbosity=2` for backward compatibility

Quick usage:

```python
import logging
from discos.skeleton import skeletonize

logging.basicConfig(level=logging.INFO)  # show INFO and above

# Narrow to a single module (optional):
logging.getLogger("discos.skeleton").setLevel(logging.DEBUG)  # detailed

skel = skeletonize(mesh, n_slices=31, verbosity=1)
```

Modules with loggers:

- `discos.skeleton` — algorithm progress and diagnostics
- `discos.mesh` — mesh load/repair summaries and analysis output
- `discos.demo` — demo mesh generation progress; warnings mirrored to logs
- `discos.path` — quiet by default; debug-level path resolutions

## API Reference

### Core Classes

- `MeshManager` — Mesh loading and preprocessing utilities.
- `SkeletonGraph` — Graph of segment connectivity, with SWC export and plotting helpers.
- `Junction` — Fitted disk representing a cross-section.
- `CrossSection` — A set of polygons and associated junction ids at a given cut.
- `Segment` — Band component metadata and matched lower/upper junctions.

### Core Functions

- `discos.skeleton.skeletonize(mesh_or_manager, n_slices, ...) -> SkeletonGraph`
- Demo creators: `discos.demo.create_cylinder_mesh`, `create_torus_mesh`, `create_branching_mesh`

### Key Parameters (skeletonize)

- `n_slices` (int): number of vertical bands.
- `radius_mode` (str): radius computation for cross-sections (default: `equivalent_area`).
- `validate_volume` (bool): enable volume consistency check.
- `volume_tol` (float): relative tolerance for volume check (default 0.05).
- `enforce_connected` (bool): require the output graph be a single connected component.
- `connect_isolated_terminals` (bool): safety-net connections at extremal slices.
- `verbosity` (int 0/1/2) or `verbose` (legacy): logging level.

## Testing

Run the test suite with uv:

```bash
uv run pytest
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use DISCOS in your research, please cite:

```bibtex
@software{discos,
  title={DISCOS: DISrete COllinear Skeletonization},
  author={Fox, Jordan},
  year={2025},
  url={https://github.com/jmrfox/discos}
}
```

## Support

- **Documentation**: [GitHub Wiki](https://github.com/jmrfox/discos/wiki)
- **Issues**: [GitHub Issues](https://github.com/jmrfox/discos/issues)
- **Discussions**: [GitHub Discussions](https://github.com/jmrfox/discos/discussions)

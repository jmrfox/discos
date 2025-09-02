"""
Skeletonization of a 3D mesh.

The main objective of skeletonization is to represent a 3D mesh in a compact form that
captures its essential shape and connectivity. The DISCOS algorithm achieves this by
slicing up the original mesh along the z-axis into a series of slices. Within each slice,
 the algorithm identifies contiguous volumes (segments) and creates a graph connecting them.
 The graph is a skeleton of the original mesh, where each edge represents a segment and each node
  represents a connection between two segments at the z-value of each cutting plane.

Important terminology:
- "Original Mesh": A mesh is a 3D object defined by a set of vertices, edges, and faces. DISCOS does not destroy the mesh you give it.
    For use in DISCOS, the original mesh should be a single watertight volume with no self-intersections.
- "Bounding planes": The two planes of constant z-value that bound the original mesh: one at the minimum z-value and one at the maximum z-value.
- "Terminal points": The points where the bounding planes intersect the mesh. 
    In other words, they represent the minimum and maximum z-value of the mesh.
- "Cut": A plane of constant z-value that intersects the mesh. A single cut always has at least one cross-section associated with it.
- "Slice": A 3D partition of the mesh with its volume either between two adjacent cuts, or between one cut and one bounding plane.
    A single slice might contain multiple segments (closed volumes). We use the method MeshManager.slice_mesh_by_z to create the slices.
- "Segment": A contiguous 3D volume of the mesh with its volume between two cuts or between one cut and one bounding plane. 
    Each segment has a single contiguous external surface and at least one contiguous internal surface.
- "Cross-section": A contiguous 2D curve (and its area) resulting from intersecting the mesh with a cut. 
    It has a surface area (internal to the original mesh volume) in the xy-plane.
    Every cross-section is shared by two segments. Every cross-section has one Junction fitted to it.
- "Junction": A disk with radius and center position fitted to the cross-section. Junctions are fit from cross-sections 
    and become the nodes of the skeleton graph. 
    (Several algorithms exist for fitting the disk, e.g. least squares, least absolute deviations, etc.)
- "Internal surface area": The surface area of a segment that overlaps a cross-section.
- "External surface area": The surface area of a segment that is shared with the original mesh.
- "Skeleton": A graph representation where segments are edges and nodes are junctions and terminal points.
    Edges only connect nodes on adjacent planes. No two nodes at the same cut (z-value) may be connected by an edge.

The algorithm follows these steps:

1. Validate input mesh: Check if mesh is watertight, has no self-intersections, and is a single hull.
    If not, raise an error. Mesh can be passed as a trimesh object or as an instance of discos.mesh.MeshManager.

2. Create bounding planes at the minimum and maximum z-value of the mesh.
    Extract terminal points.
    Create cuts at regular intervals along the z-axis using MeshManager.slice_mesh_by_z. 
    This results in the set of slices. 
    The algorithm takes a parameter n_slices which is how many 3D slices (partitions) should be created;
        equivalently, cuts = n_slices - 1. 
    
3. For each cut, extract the cross-section. Note that one cross section may include multiple closed areas.
    Fit a junction (radius and center position) to each closed area in the cross-section.
    
4. Build the SkeletonGraph. Each junction becomes a node and each edge connects two nodes on adjacent cuts.
    A node is connected to another if and only if there is a series of mesh edges connecting their two junctions (in adjacent cuts only), 
    meaning they belong to the same segment at different z-values.

5. Validate volume and surface area conservation. 
    The sum of the volumes of the segments should agree with the volume of the original mesh within a tolerance.



EXAMPLE 1:
    Consider a cylinder with radius 1 m and height 4 m, aligned with the z-axis.
    The cylinder is centered at the origin (0, 0, 0) and extends from z = -2 to z = 2.
    We will slice the cylinder with n_slices = 4. Each segment should have the same volume.
    There are 3 cuts, each with one cross-section. Each cross section has one junction.
    The skeleton graph should have 5 nodes and 4 edges. 
    One node at each of the 3 junctions and one at each point where the mesh intersects the bounding planes,
        and 4 edges (one between each pair of adjacent planes, corresponding to the 4 slices.
    



"""

import math
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import trimesh

# ---- Data Structures -------------------------------------------------------


@dataclass
class Junction:
    """Fitted disk (center and radius) which exists at a cross-section.

    A Junction is fundamentally different from a CrossSection: it is only the
    fitted disk summarized by a point (x, y, z) and a radius, living on a cut plane.
    """

    center: np.ndarray
    radius: float
    z: float
    cut_index: Optional[int] = None
    index_within_cut: Optional[int] = None
    cross_section_id: Optional[str] = None


@dataclass
class CrossSection:
    """Planar cross-section curve at a specific z-position (on a cut or bounding plane).

    Holds the 2D boundary and its associated `Junction` (fitted disk). For backward
    compatibility, mirrors `center` and `radius` from the Junction and keeps
    legacy slice_* naming synchronized with cut_*.
    """

    z: float
    area: float
    boundary_2d: Optional[np.ndarray]
    cut_index: Optional[int] = None
    index_within_cut: Optional[int] = None
    junction: Optional[Junction] = None
    # Back-compat mirrors
    center: Optional[np.ndarray] = None
    radius: Optional[float] = None
    # Legacy nomenclature (mirrored)
    slice_index: Optional[int] = None
    index_within_slice: Optional[int] = None

    def __post_init__(self) -> None:
        # If legacy fields were provided without new ones, map them forward
        if self.slice_index is not None and getattr(self, "cut_index", None) is None:
            self.cut_index = int(self.slice_index)
        if (
            self.index_within_slice is not None
            and getattr(self, "index_within_cut", None) is None
        ):
            self.index_within_cut = int(self.index_within_slice)
        # Keep legacy mirrors synchronized for external consumers/tests
        if getattr(self, "cut_index", None) is not None:
            self.slice_index = int(self.cut_index)
        if getattr(self, "index_within_cut", None) is not None:
            self.index_within_slice = int(self.index_within_cut)
        # Mirror center/radius from Junction for back-compat if not explicitly set
        if self.junction is not None:
            if self.center is None:
                self.center = np.array(self.junction.center, dtype=float)
            if self.radius is None:
                self.radius = float(self.junction.radius)


@dataclass
class Segment:
    """Segment representing the 3D slice between two adjacent cuts.

    Attributes:
        u_id: Node ID of the first cross-section (e.g., "cut{i}_cs{j}").
        v_id: Node ID of the second cross-section (on the adjacent cut).
        length: Center-line length between cross-section centers.
        r1: Radius at node u.
        r2: Radius at node v.
        volume: Approximate volume as a circular frustum between radii r1 and r2.
        center_line: 2x3 array of endpoints of the segment center-line.
    """

    u_id: str
    v_id: str
    length: float
    r1: float
    r2: float
    volume: float
    center_line: np.ndarray


# ---- Helpers ---------------------------------------------------------------


def _ensure_trimesh(mesh_or_manager: Union[trimesh.Trimesh, Any]) -> trimesh.Trimesh:
    """Return a `trimesh.Trimesh` from either a Trimesh or MeshManager-like object.

    If `mesh_or_manager` is a wrapper (e.g., a `MeshManager`) that exposes a
    `.mesh` attribute containing a `trimesh.Trimesh`, that underlying mesh is
    returned. Otherwise, the input must already be a `trimesh.Trimesh`.
    """
    if isinstance(mesh_or_manager, trimesh.Trimesh):
        return mesh_or_manager
    # Lazy import/interface to avoid circular import
    mesh_attr = getattr(mesh_or_manager, "mesh", None)
    if isinstance(mesh_attr, trimesh.Trimesh):
        return mesh_attr
    raise TypeError("Expected trimesh.Trimesh or an object with .mesh (Trimesh)")


def _fit_circle_algebraic(points_2d: np.ndarray) -> Tuple[np.ndarray, float]:
    """Algebraic circle fit: returns (center_xy, radius).

    Requires at least 3 points. Uses linear least-squares on the algebraic form.
    """
    if points_2d is None or len(points_2d) < 3:
        raise ValueError("Need at least 3 points to fit a circle")

    x = points_2d[:, 0]
    y = points_2d[:, 1]
    A = np.column_stack([2 * x, 2 * y, np.ones(len(x))])
    b = x**2 + y**2
    try:
        params = np.linalg.lstsq(A, b, rcond=None)[0]
        cx, cy, c = params
        r = float(np.sqrt(max(c + cx**2 + cy**2, 0.0)))
        return np.array([cx, cy], dtype=float), r
    except np.linalg.LinAlgError:
        ctr = np.mean(points_2d, axis=0)
        r = float(np.mean(np.linalg.norm(points_2d - ctr, axis=1)))
        return ctr.astype(float), r


def _polygon_area_and_centroid(points_2d: np.ndarray) -> Tuple[float, np.ndarray]:
    """Compute signed area and centroid of a 2D polygon using the shoelace formula.

    Points must form a closed polygon (first point equals last) or will be closed.
    Returns (area_abs, centroid_xy).
    """
    if len(points_2d) < 3:
        return 0.0, np.array([np.nan, np.nan], dtype=float)
    pts = np.asarray(points_2d, dtype=float)
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])
    x = pts[:, 0]
    y = pts[:, 1]
    cross = x[:-1] * y[1:] - x[1:] * y[:-1]
    A = 0.5 * np.sum(cross)
    if abs(A) < 1e-15:
        return 0.0, np.array([np.nan, np.nan], dtype=float)
    Cx = (1.0 / (6.0 * A)) * np.sum((x[:-1] + x[1:]) * cross)
    Cy = (1.0 / (6.0 * A)) * np.sum((y[:-1] + y[1:]) * cross)
    return abs(A), np.array([Cx, Cy], dtype=float)


def _extract_cross_sections_for_plane(
    mesh: trimesh.Trimesh,
    z_value: float,
    cut_index: int,
    radius_method: str = "area",
) -> List[CrossSection]:
    """Extract cross-sections at a given z-plane and fit disks.

    Uses `trimesh.Trimesh.section` -> `Path3D` -> discrete polylines.
    For each discrete closed loop, computes area and a center/radius.

    Args:
        mesh: Watertight input mesh.
        z_value: World z of the cutting plane.
        cut_index: Index of the plane in the ordered stack of planes.
        radius_method: 'area' uses equivalent-area circle; 'algebraic' fits
            a circle to boundary points via linear least squares.

    Returns:
        List of `CrossSection` objects ordered by discovery; `index_within_cut`
        reflects that order within the given `cut_index`.
    """
    path = mesh.section(
        plane_origin=[0.0, 0.0, float(z_value)], plane_normal=[0.0, 0.0, 1.0]
    )
    if path is None:
        return []

    sections: List[CrossSection] = []
    # For robustness: use 3D discrete loops returned by trimesh, then drop z
    # to obtain 2D boundaries for area/center/radius computations.
    try:
        loops_3d = path.discrete
    except Exception:
        loops_3d = []

    idx = 0
    for loop in loops_3d:
        loop = np.asarray(loop, dtype=float)
        if loop.shape[0] < 3:
            continue
        boundary_2d = loop[:, :2]
        area, centroid_2d = _polygon_area_and_centroid(boundary_2d)
        if area <= 0:
            continue

        if radius_method == "area":
            # Equivalent area circle; center is polygon centroid
            r = float(np.sqrt(area / math.pi))
            cx, cy = float(centroid_2d[0]), float(centroid_2d[1])
        elif radius_method == "algebraic":
            # Fit radius algebraically but center at polygon centroid to avoid drift
            _ctr2d, r = _fit_circle_algebraic(boundary_2d)
            cx, cy = float(centroid_2d[0]), float(centroid_2d[1])
        else:
            raise ValueError(f"Unknown radius_method: {radius_method}")

        center_3d = np.array([cx, cy, float(z_value)], dtype=float)
        sections.append(
            CrossSection(
                z=float(z_value),
                area=float(area),
                center=center_3d,
                radius=float(r),
                boundary_2d=boundary_2d,
                cut_index=cut_index,
                index_within_cut=idx,
            )
        )
        idx += 1

    return sections


def _check_overlap_in_cut(cross_sections: List[CrossSection], tolerance: float) -> None:
    """Raise ValueError if any two disks on the same cut overlap beyond tolerance.

    Enforces the rule that cross-sections in the same plane must not overlap.
    This prevents edges from being created between nodes at the same z-plane.
    """
    n = len(cross_sections)
    for i in range(n):
        for j in range(i + 1, n):
            ci = cross_sections[i]
            cj = cross_sections[j]
            if abs(ci.z - cj.z) <= 1e-9:  # same plane
                d = float(np.linalg.norm(ci.center[:2] - cj.center[:2]))
                if d + tolerance < (ci.radius + cj.radius):
                    raise ValueError(
                        f"Overlapping cross-sections at z={ci.z:.6g}: "
                        f"cut {ci.cut_index} cs {ci.index_within_cut} and "
                        f"cut {cj.cut_index} cs {cj.index_within_cut}"
                    )


def _check_overlap_in_slice(
    cross_sections: List[CrossSection], tolerance: float
) -> None:
    """Backward-compat wrapper. Use `_check_overlap_in_cut`.

    Accepts cross-sections (possibly constructed with legacy slice_* fields)
    and enforces non-overlap within the same plane.
    """
    return _check_overlap_in_cut(cross_sections, tolerance)


def _line_of_sight_inside(
    mesh: trimesh.Trimesh, p0: np.ndarray, p1: np.ndarray, n_samples: int
) -> bool:
    """Check if the straight segment from p0 to p1 stays inside the mesh.

    Samples interior points along the segment (excluding endpoints) and uses
    mesh.contains to verify containment.
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    if n_samples <= 0:
        n_samples = 1
    # Exclude exact endpoints to avoid sitting exactly on surface
    ts = np.linspace(0.0, 1.0, n_samples + 2)[1:-1]
    pts = p0[None, :] * (1.0 - ts[:, None]) + p1[None, :] * ts[:, None]
    try:
        inside = mesh.contains(pts)
        return bool(np.all(inside))
    except Exception:
        # Fallback: if `contains` isn't available or fails, assume connectivity
        # if a ray along the center-line doesn't immediately intersect the surface.
        # This is a best-effort approximation and may be conservative.
        direction = p1 - p0
        length = np.linalg.norm(direction)
        if length <= 0:
            return False
        direction = direction / length
        # Cast a single ray from slightly inside p0 toward p1
        eps = 1e-6 * max(1.0, np.linalg.norm(mesh.extents))
        origins = (p0 + eps * direction).reshape(1, 3)
        vectors = direction.reshape(1, 3)
        locations, index_ray, index_tri = mesh.ray.intersects_location(origins, vectors)
        # If we don't hit the surface immediately, assume inside
        return len(locations) == 0


def _tube_inside(
    mesh: trimesh.Trimesh,
    p0: np.ndarray,
    p1: np.ndarray,
    r0: float,
    r1: float,
    axial_steps: int = 5,
    radial_dirs: int = 8,
    radius_fraction: float = 0.25,
) -> bool:
    """Check if a small-radius tube along the center-line stays inside the mesh.

    We sample rings of points around the line from p0 to p1. At each axial step,
    we take a radius = radius_fraction * lerp(r0, r1) and verify points are inside.
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    dir_vec = p1 - p0
    L = float(np.linalg.norm(dir_vec))
    if L <= 0:
        return False
    dir_unit = dir_vec / L

    # Build an orthonormal frame (u, v) perpendicular to dir_unit
    # Pick an arbitrary vector not parallel to dir_unit
    a = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(a, dir_unit)) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    u = np.cross(dir_unit, a)
    nu = np.linalg.norm(u)
    if nu == 0:
        return _line_of_sight_inside(mesh, p0, p1, max(1, axial_steps))
    u /= nu
    v = np.cross(dir_unit, u)

    ts = np.linspace(0.0, 1.0, max(2, axial_steps))
    thetas = np.linspace(0.0, 2.0 * math.pi, max(4, radial_dirs), endpoint=False)

    # Prefer contains if available
    use_contains = True
    try:
        _ = mesh.contains(np.zeros((1, 3)))
    except Exception:
        use_contains = False

    for t in ts:
        c = p0 * (1.0 - t) + p1 * t
        r = radius_fraction * (r0 * (1.0 - t) + r1 * t)
        if r <= 0:
            # Degenerate ring; skip
            continue
        ring_pts = []
        for th in thetas:
            offset = r * (math.cos(th) * u + math.sin(th) * v)
            ring_pts.append(c + offset)
        P = np.asarray(ring_pts)
        if use_contains:
            try:
                inside = mesh.contains(P)
                if not bool(np.all(inside)):
                    return False
            except Exception:
                # Fallback: ray heuristic from each point outward along +dir_unit
                origins = P + (1e-6 * max(1.0, np.linalg.norm(mesh.extents))) * dir_unit
                vectors = np.tile(dir_unit, (len(P), 1))
                locations, _, _ = mesh.ray.intersects_location(origins, vectors)
                if len(locations) > 0:
                    return False
        else:
            origins = P + (1e-6 * max(1.0, np.linalg.norm(mesh.extents))) * dir_unit
            vectors = np.tile(dir_unit, (len(P), 1))
            locations, _, _ = mesh.ray.intersects_location(origins, vectors)
            if len(locations) > 0:
                return False
    return True


def _voxel_connectivity(
    mesh: trimesh.Trimesh,
    p0: np.ndarray,
    p1: np.ndarray,
    r0: float,
    r1: float,
    nx: int = 20,
    ny: int = 20,
    nz: int = 20,
    margin_frac: float = 0.6,
) -> bool:
    """Coarse voxel BFS inside the mesh to test connectivity between neighborhoods.

    Builds an axis-aligned box around the endpoints expanded by max(r0, r1)*margin,
    samples a regular grid, keeps voxels whose centers are inside the mesh, and
    runs a 6-connected BFS from the start neighborhood (near p0) to the end (near p1).

    Returns True if a path exists. This is a robust fallback for branch points.
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    r_max = float(max(r0, r1))
    # Bounding box
    mins = np.minimum(p0, p1) - margin_frac * r_max
    maxs = np.maximum(p0, p1) + margin_frac * r_max
    # Ensure non-zero extents
    ext = np.maximum(maxs - mins, 1e-9)
    nx = max(8, int(nx))
    ny = max(8, int(ny))
    nz = max(8, int(nz))

    xs = np.linspace(mins[0], maxs[0], nx)
    ys = np.linspace(mins[1], maxs[1], ny)
    zs = np.linspace(mins[2], maxs[2], nz)
    # Create grid of points (centers)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    P = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    # Try contains in batches (to avoid failures on large arrays)
    inside_flat = np.zeros(P.shape[0], dtype=bool)
    batch = 50000
    try:
        for start in range(0, P.shape[0], batch):
            end = min(start + batch, P.shape[0])
            inside_flat[start:end] = mesh.contains(P[start:end])
    except Exception:
        # If contains not available, degrade to line-of-sight which we already tried
        return False

    inside = inside_flat.reshape((nx, ny, nz))

    # Start/goal masks: within a fraction of r0/r1 from endpoints
    def near_mask(c: np.ndarray, r: float, frac: float = 0.35) -> np.ndarray:
        d2 = (X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2
        return d2 <= (max(r * frac, 1e-9) ** 2)

    start_mask = near_mask(p0, r0)
    goal_mask = near_mask(p1, r1)
    start = np.logical_and(inside, start_mask)
    goal = np.logical_and(inside, goal_mask)
    if not start.any() or not goal.any():
        return False

    # BFS over 6-neighborhood
    from collections import deque

    visited = np.zeros_like(inside, dtype=bool)
    q = deque()
    sx, sy, sz = np.where(start)
    for a, b, c in zip(sx, sy, sz):
        visited[a, b, c] = True
        q.append((a, b, c))

    neigh = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

    while q:
        a, b, c = q.popleft()
        if goal[a, b, c]:
            return True
        for da, db, dc in neigh:
            ia, ib, ic = a + da, b + db, c + dc
            if 0 <= ia < nx and 0 <= ib < ny and 0 <= ic < nz:
                if not visited[ia, ib, ic] and inside[ia, ib, ic]:
                    visited[ia, ib, ic] = True
                    q.append((ia, ib, ic))
    return False


def _frustum_volume(h: float, r1: float, r2: float) -> float:
    """Volume of a circular frustum with height h and radii r1, r2."""
    return math.pi * h * (r1 * r1 + r1 * r2 + r2 * r2) / 3.0


class SkeletonGraph(nx.Graph):
    """Skeleton graph where nodes are cross-sections and edges are segments.

    Node attributes:
        - center: np.ndarray(3,)
        - radius: float
        - z: float
        - cut_index: int
        - index_within_cut: int
        - area: float

    Edge attributes:
        - length: float
        - r1: float
        - r2: float
        - volume: float (frustum approximation)
        - center_line: np.ndarray shape (2, 3)
    """

    def draw(
        self,
        axis: str = "x",
        ax: Any = None,
        figsize: Optional[Tuple[float, float]] = None,
        with_labels: bool = False,
        node_size: int = 50,
        node_color: str = "C0",
        edge_color: str = "0.6",
        min_hv_ratio: float = 0.3,
        pad_frac: float = 0.05,
        **kwargs: Any,
    ) -> Any:
        """Draw the skeleton graph in 2D using (x,z) or (y,z) coordinates.

        Args:
            axis: Horizontal axis to use ("x" or "y"). Vertical axis is always z.
            ax: Optional matplotlib Axes to draw into. If None, a new figure/axes is created.
            figsize: Optional (width, height) in inches when creating a new figure.
            with_labels: Whether to render node labels.
            node_size: Node marker size passed to networkx.draw.
            node_color: Node color.
            edge_color: Edge color.
            min_hv_ratio: Minimum desired horizontal-to-vertical data range ratio.
                If the horizontal data span is less than this ratio times the
                vertical (z) span, x/y limits are expanded around the mean to
                meet this minimum. Helps avoid an overly thin vertical line.
            pad_frac: Fractional padding added to both axes limits for readability.
            **kwargs: Additional kwargs forwarded to networkx.draw.

        Returns:
            The matplotlib Axes used for drawing.
        """
        axis = axis.lower()
        if axis not in ("x", "y"):
            raise ValueError("axis must be 'x' or 'y'")

        # Build 2D positions from node 3D centers
        idx = 0 if axis == "x" else 1
        pos: Dict[str, Tuple[float, float]] = {}
        for n, attrs in self.nodes(data=True):
            c = attrs.get("center")
            if c is None or len(c) != 3:
                continue
            pos[n] = (float(c[idx]), float(c[2]))  # (x|y, z)

        # Compute data bounds
        if len(pos) == 0:
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)
            ax.set_xlabel(f"{axis} (horizontal)")
            ax.set_ylabel("z (vertical)")
            return ax

        xs = np.array([p[0] for p in pos.values()], dtype=float)
        zs = np.array([p[1] for p in pos.values()], dtype=float)
        xmin, xmax = float(xs.min()), float(xs.max())
        zmin, zmax = float(zs.min()), float(zs.max())
        xmid = 0.5 * (xmin + xmax)
        zmid = 0.5 * (zmin + zmax)
        xr = max(xmax - xmin, 1e-12)
        zr = max(zmax - zmin, 1e-12)

        # Enforce a minimum horizontal span relative to vertical span
        min_xr = max(min_hv_ratio * zr, xr)
        if min_xr > xr:
            xmin = xmid - 0.5 * min_xr
            xmax = xmid + 0.5 * min_xr
            xr = min_xr

        # Apply padding
        xmin -= pad_frac * xr
        xmax += pad_frac * xr
        zmin -= pad_frac * zr
        zmax += pad_frac * zr

        # Prepare axes
        if ax is None:
            # Auto figsize that respects data aspect for clarity
            if figsize is None:
                # Base height and width with reasonable minimums
                base_h = 4.0
                # width scaled by data aspect (horizontal over vertical)
                aspect = xr / zr if zr > 0 else 1.0
                base_w = max(4.0, base_h * max(aspect, min_hv_ratio))
                figsize = (base_w, base_h)
            fig, ax = plt.subplots(figsize=figsize)

        # Draw using networkx helper
        nx.draw(
            self,
            pos=pos,
            ax=ax,
            with_labels=with_labels,
            node_size=node_size,
            node_color=node_color,
            edge_color=edge_color,
            **kwargs,
        )

        ax.set_xlabel(f"{axis} (horizontal)")
        ax.set_ylabel("z (vertical)")
        # Set limits and aspect for a clear, readable plot
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(zmin, zmax)
        try:
            ax.set_aspect("equal", adjustable="box")
        except Exception:
            # Fallback to auto if backend cannot honor equal aspect
            ax.set_aspect("auto")

        # Return provided or newly-created axes
        return ax

    def plot_cross_section(
        self,
        node_id: str,
        ax: Any = None,
        boundary_color: str = "k",
        circle_color: str = "C1",
        center_color: str = "C2",
        linewidth: float = 1.5,
        alpha_boundary: float = 0.9,
        alpha_circle: float = 0.8,
        show_center: bool = True,
        title: Optional[str] = None,
    ) -> Any:
        """Plot the cross-section boundary curve with its fitted disk.

        Expects node attributes `boundary_2d`, `center`, and `radius`.

        Args:
            node_id: Graph node identifier (e.g., "cut3_cs0").
            ax: Optional matplotlib Axes; if None, a new one is created.
            boundary_color: Color for the boundary polyline.
            circle_color: Color for the fitted circle.
            center_color: Color for the center marker.
            linewidth: Line width for boundary and circle.
            alpha_boundary: Alpha for the boundary.
            alpha_circle: Alpha for the circle.
            show_center: Whether to draw the center point.
            title: Optional title; defaults to node id with cut/index.

        Returns:
            The matplotlib Axes used for drawing.
        """
        if node_id not in self.nodes:
            raise KeyError(f"Node '{node_id}' not in SkeletonGraph")

        attrs = self.nodes[node_id]
        boundary_2d = attrs.get("boundary_2d")
        center = attrs.get("center")
        radius = attrs.get("radius")

        if boundary_2d is None or len(boundary_2d) < 2:
            warnings.warn(f"Node '{node_id}' has no boundary_2d; cannot plot boundary.")
        if center is None or radius is None:
            raise ValueError(
                f"Node '{node_id}' missing center/radius required for plotting"
            )

        # Prepare axes
        if ax is None:
            fig, ax = plt.subplots()

        # Plot boundary polyline
        if boundary_2d is not None and len(boundary_2d) >= 2:
            bx = boundary_2d[:, 0]
            by = boundary_2d[:, 1]
            ax.plot(bx, by, color=boundary_color, lw=linewidth, alpha=alpha_boundary)

        # Plot fitted circle
        cx, cy = float(center[0]), float(center[1])
        r = float(radius)
        theta = np.linspace(0.0, 2.0 * math.pi, 256)
        ax.plot(
            cx + r * np.cos(theta),
            cy + r * np.sin(theta),
            color=circle_color,
            lw=linewidth,
            alpha=alpha_circle,
        )
        if show_center:
            ax.plot([cx], [cy], marker="o", color=center_color, ms=4)

        # Cosmetics
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        default_title = title
        if default_title is None:
            ci = attrs.get("cut_index")
            ii = attrs.get("index_within_cut")
            default_title = f"{node_id} (cut {ci}, cs {ii})"
        ax.set_title(default_title)

        return ax

    def plot_all_cross_sections(
        self,
        sort_by: Tuple[str, str] = ("cut_index", "index_within_cut"),
        max_cols: int = 4,
        figsize: Optional[Tuple[float, float]] = None,
        share_axes: bool = False,
        boundary_color: str = "k",
        circle_color: str = "C1",
    ) -> Any:
        """Plot every cross-section boundary with its fitted disk in a grid.

        Args:
            sort_by: Tuple of node attribute names to sort nodes (primary, secondary).
            max_cols: Max number of subplot columns.
            figsize: Figure size; auto if None.
            share_axes: Whether to share x/y among subplots.
            boundary_color: Color for boundaries.
            circle_color: Color for circles.

        Returns:
            The matplotlib Figure created.
        """
        # Gather nodes that have boundary_2d and required attributes
        nodes = []
        for nid, attrs in self.nodes(data=True):
            if attrs.get("center") is None or attrs.get("radius") is None:
                continue
            nodes.append((nid, attrs))

        # Sort nodes by provided attributes if present
        def sort_key(item: Tuple[str, Dict[str, Any]]):
            _, a = item
            return tuple(a.get(k, 0) for k in sort_by)

        nodes.sort(key=sort_key)
        n = len(nodes)
        if n == 0:
            warnings.warn("No nodes available to plot")
            return None

        ncols = max(1, int(max_cols))
        nrows = int(math.ceil(n / ncols))
        if figsize is None:
            figsize = (4 * ncols, 4 * nrows)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=figsize,
            squeeze=False,
            sharex=share_axes,
            sharey=share_axes,
        )

        for idx, (nid, _) in enumerate(nodes):
            r = idx // ncols
            c = idx % ncols
            ax = axes[r][c]
            self.plot_cross_section(
                nid,
                ax=ax,
                boundary_color=boundary_color,
                circle_color=circle_color,
                show_center=True,
            )

        # Hide any unused subplots
        for k in range(n, nrows * ncols):
            r = k // ncols
            c = k % ncols
            axes[r][c].axis("off")

        fig.tight_layout()
        return fig

    @classmethod
    def from_mesh(
        cls,
        mesh_or_manager: Union[trimesh.Trimesh, Any],
        n_slices: int,
        radius_method: str = "area",
        samples_per_edge: int = 25,
        overlap_tolerance: float = 1e-6,
        validate: bool = True,
        volume_tol: float = 0.05,
        verbose: bool = False,
    ) -> "SkeletonGraph":
        """Construct a skeleton graph from a mesh.

        Args:
            mesh_or_manager: trimesh.Trimesh or object with .mesh (Trimesh)
            n_slices: Number of 3D slices (thus internal cuts = n_slices - 1). Must be >= 1.
            radius_method: 'area' (equivalent area circle) or 'algebraic'.
            samples_per_edge: Samples for inside-volume line-of-sight.
            overlap_tolerance: Tolerance for overlap detection on the same cut.
            validate: If True, validate approximate volume conservation.
            volume_tol: Relative tolerance for volume conservation (e.g., 0.05 = 5%).

        Returns:
            SkeletonGraph instance.

        Notes:
            - Nodes are cross-sections at each z-plane, located at fitted circle
              centers with an associated radius and area.
            - Edges only connect nodes from adjacent planes when a straight
              center-line between node centers stays inside the mesh (no same-plane
              connections).
            - Cross-sections within the same plane must not overlap; an error is
              raised if they do (with a small `overlap_tolerance`).
        """
        mesh = _ensure_trimesh(mesh_or_manager)
        # Optional diagnostics collector
        logs: List[str] = []

        def vlog(msg: str) -> None:
            if verbose:
                logs.append(str(msg))

        # 1. Validate mesh
        if not mesh.is_watertight:
            raise ValueError("Input mesh must be watertight")

        # 2. Build z-positions for nodes (bounding planes + cuts)
        if n_slices < 1:
            raise ValueError("n_slices must be >= 1")
        zmin, zmax = float(mesh.bounds[0, 2]), float(mesh.bounds[1, 2])
        # n_slices implies cuts = n_slices - 1, total node planes = n_slices + 1
        # These z-positions include the bounding planes and internal cuts.
        z_positions = np.linspace(zmin, zmax, n_slices + 1)

        # Slightly nudge the first/last planes inward to avoid grazing the
        # exterior surface exactly at extremes (helps robust sectioning).
        eps = 1e-6 * max(1.0, float(np.linalg.norm(mesh.extents)))
        if len(z_positions) >= 2:
            z_positions[0] = min(z_positions[0] + eps, z_positions[1] - eps * 0.5)
            z_positions[-1] = max(z_positions[-1] - eps, z_positions[-2] + eps * 0.5)

        # 3. Extract cross-sections at each node plane (each cut or bounding plane)
        cs_by_cut: List[List[CrossSection]] = []
        vlog(
            f"Building {len(z_positions)} planes (including bounds) for n_slices={n_slices}"
        )
        for ci, z in enumerate(z_positions):
            sections = _extract_cross_sections_for_plane(
                mesh, float(z), cut_index=ci, radius_method=radius_method
            )
            if len(sections) == 0:
                warnings.warn(f"No cross-sections found at z={float(z):.6g}")
                vlog(f"cut {ci}: no cross-sections at z={float(z):.6g}")
            # Enforce non-overlap among same-plane cross-sections.
            _check_overlap_in_cut(sections, tolerance=overlap_tolerance)
            if len(sections) > 0:
                vlog(f"cut {ci}: {len(sections)} cross-section(s) at z={float(z):.6g}")
            cs_by_cut.append(sections)

        # 4. Build graph nodes
        G = cls()
        node_ids_by_cut: List[List[str]] = []
        for ci, sections in enumerate(cs_by_cut):
            ids_this_cut: List[str] = []
            for cs in sections:
                node_id = f"cut{ci}_cs{cs.index_within_cut}"
                ids_this_cut.append(node_id)
                G.add_node(
                    node_id,
                    center=cs.center,
                    radius=cs.radius,
                    z=cs.z,
                    cut_index=cs.cut_index,
                    index_within_cut=cs.index_within_cut,
                    # Legacy mirrors for compatibility with existing code/tests
                    slice_index=cs.slice_index,
                    index_within_slice=cs.index_within_slice,
                    area=cs.area,
                    boundary_2d=cs.boundary_2d,
                )
            node_ids_by_cut.append(ids_this_cut)

        # 5. Connect adjacent cuts based on inside-volume connectivity tests.
        #    Only adjacent cuts may connect; same-cut connections are disallowed.
        segments: List[Segment] = []
        for ci in range(len(cs_by_cut) - 1):
            lower = cs_by_cut[ci]
            upper = cs_by_cut[ci + 1]
            if verbose:
                vlog(
                    f"slice {ci}: lower cut {ci} has {len(lower)} node(s), upper cut {ci+1} has {len(upper)} node(s)"
                )

            # Compute per-cut centroids and angular positions for gating
            def cut_stats(sections: List[CrossSection]):
                if not sections:
                    return np.array([0.0, 0.0]), np.array([]), np.array([])
                C = np.vstack([s.center[:2] for s in sections])
                cxy = C.mean(axis=0)
                V = C - cxy
                R = np.linalg.norm(V, axis=1)
                theta = np.arctan2(V[:, 1], V[:, 0])
                return cxy, R, theta

            lower_cxy, lower_R, lower_theta = cut_stats(lower)
            upper_cxy, upper_R, upper_theta = cut_stats(upper)

            # Define near-axis thresholds (skip angle gating there to allow branching)
            def near_axis(R: np.ndarray) -> float:
                if R.size == 0:
                    return 0.0
                med = float(np.median(R))
                return 0.25 * med

            lower_axis_eps = near_axis(lower_R)
            upper_axis_eps = near_axis(upper_R)
            # Start with a permissive angular gate; will be bypassed if voxel connectivity succeeds
            angle_gate = math.pi / 2.0  # 90 degrees
            made_any_edge_this_slice = 0
            pruned_no_connectivity = 0
            pruned_xy_gate = 0
            pruned_angle_gate = 0
            for i, c0 in enumerate(lower):
                # Gather all valid candidates to upper with XY distance for pruning
                candidates: List[Tuple[int, float, float]] = []  # (j, d_xy, length)
                length_cache: Dict[int, float] = {}
                for j, c1 in enumerate(upper):
                    # Adaptive sampling based on geometric separation
                    length_3d = float(np.linalg.norm(c1.center - c0.center))
                    # Increase line-of-sight samples for longer spans
                    n_line_samples = max(samples_per_edge, int(5 + 0.5 * length_3d))
                    axial_steps = max(3, int(max(1, n_line_samples) // 4))
                    # Connectivity checks (any can pass). Try LOS -> tube -> voxel
                    los_ok = _line_of_sight_inside(
                        mesh, c0.center, c1.center, n_line_samples
                    )
                    tube_ok = _tube_inside(
                        mesh,
                        c0.center,
                        c1.center,
                        float(c0.radius),
                        float(c1.radius),
                        axial_steps=axial_steps,
                        radial_dirs=8,
                        radius_fraction=0.25,
                    )
                    voxel_ok = False
                    if not tube_ok and not los_ok:
                        voxel_ok = _voxel_connectivity(
                            mesh,
                            c0.center,
                            c1.center,
                            float(c0.radius),
                            float(c1.radius),
                            # Slightly denser grid for robustness on tricky branch geometry
                            nx=28,
                            ny=28,
                            nz=28,
                        )
                    if not (los_ok or tube_ok or voxel_ok):
                        pruned_no_connectivity += 1
                        continue
                    # XY gating to avoid cross connections across a void
                    d_xy = float(np.linalg.norm(c1.center[:2] - c0.center[:2]))
                    # Require XY proximity relative to disk sizes
                    # Be more permissive, and if voxel connectivity was found, bypass this gate
                    overlap_gate = d_xy <= 1.75 * (float(c0.radius) + float(c1.radius))
                    if not (overlap_gate or voxel_ok):
                        pruned_xy_gate += 1
                        continue
                    # Angular gating around per-cut centroid to keep side-consistent matches
                    # Skip if either endpoint is near the cut centroid (branching / axial nodes)
                    i_near_axis = (
                        (lower_R[i] <= max(1e-9, lower_axis_eps))
                        if i < lower_R.size
                        else True
                    )
                    j_near_axis = (
                        (upper_R[j] <= max(1e-9, upper_axis_eps))
                        if j < upper_R.size
                        else True
                    )
                    # If voxel connectivity is positive, we trust it and bypass angular gating as well
                    if not (i_near_axis or j_near_axis) and not voxel_ok:
                        th0 = float(lower_theta[i]) if i < lower_theta.size else 0.0
                        th1 = float(upper_theta[j]) if j < upper_theta.size else 0.0
                        dth = abs((th1 - th0 + math.pi) % (2.0 * math.pi) - math.pi)
                        if dth > angle_gate:
                            pruned_angle_gate += 1
                            continue
                    length = float(np.linalg.norm(c1.center - c0.center))
                    length_cache[j] = length
                    candidates.append((j, d_xy, length))

                if not candidates:
                    continue
                # Nearest-neighbor pruning in XY
                candidates.sort(key=lambda t: t[1])
                dmin = candidates[0][1]
                # Allow those within 1.25x of nearest
                keep = [c for c in candidates if c[1] <= 1.25 * dmin]
                # Cap connections per node to avoid crisscross; if branching (one-to-many), allow up to 2
                max_conn = 2 if (len(lower) == 1 and len(upper) > 1) else 1
                keep = keep[:max_conn]

                for j, d_xy, length in keep:
                    c1 = upper[j]
                    u_id = f"cut{ci}_cs{i}"
                    v_id = f"cut{ci + 1}_cs{j}"
                    vol = _frustum_volume(length, c0.radius, c1.radius)
                    seg = Segment(
                        u_id=u_id,
                        v_id=v_id,
                        length=length,
                        r1=float(c0.radius),
                        r2=float(c1.radius),
                        volume=vol,
                        center_line=np.vstack([c0.center, c1.center]),
                    )
                    segments.append(seg)
                    G.add_edge(
                        u_id,
                        v_id,
                        length=length,
                        r1=float(c0.radius),
                        r2=float(c1.radius),
                        volume=vol,
                        center_line=np.vstack([c0.center, c1.center]),
                    )
                    made_any_edge_this_slice += 1

            if verbose:
                vlog(
                    f"slice {ci}: edges made={made_any_edge_this_slice}, pruned (no_conn={pruned_no_connectivity}, xy_gate={pruned_xy_gate}, angle_gate={pruned_angle_gate})"
                )

        # 6. Validate volume conservation (approximate).
        #    Compare the sum of frustum volumes against the mesh volume.
        if validate and len(segments) > 0:
            approx_vol = float(sum(s.volume for s in segments))
            mesh_vol = float(mesh.volume) if hasattr(mesh, "volume") else np.nan
            if mesh_vol and mesh_vol > 0:
                rel_err = abs(approx_vol - mesh_vol) / mesh_vol
                if rel_err > volume_tol:
                    warnings.warn(
                        f"Approximate segment volume ({approx_vol:.6g}) "
                        f"differs from mesh volume ({mesh_vol:.6g}) by {rel_err:.2%}"
                    )

        # Attach verbose logs to the graph for user introspection
        if verbose:
            G.graph["log"] = logs
            # Also emit to stdout once for convenience
            for line in logs:
                print(f"[SkeletonGraph] {line}")
        return G


def skeletonize(
    mesh_or_manager: Union[trimesh.Trimesh, Any],
    n_slices: int,
    radius_method: str = "area",
    samples_per_edge: int = 25,
    overlap_tolerance: float = 1e-6,
    validate: bool = True,
    volume_tol: float = 0.05,
    verbose: bool = False,
) -> SkeletonGraph:
    """Top-level convenience function to build a skeleton graph.

    See `SkeletonGraph.from_mesh` for parameter details.
    """
    return SkeletonGraph.from_mesh(
        mesh_or_manager=mesh_or_manager,
        n_slices=n_slices,
        radius_method=radius_method,
        samples_per_edge=samples_per_edge,
        overlap_tolerance=overlap_tolerance,
        validate=validate,
        volume_tol=volume_tol,
        verbose=verbose,
    )

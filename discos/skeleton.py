"""
Skeletonization of a 3D mesh.

The main objective of skeletonization is to represent a 3D mesh in a compact form that
captures its essential shape and connectivity. The DISCOS algorithm achieves this by
slicing up the original mesh along the z-axis into a series of slices.

Important terminology:
- "Original Mesh": A mesh is a 3D object defined by a set of vertices, edges, and faces. Original means this is the mesh that is segmented.
    For use in DISCOS, the original mesh should be a single watertight volume with no self-intersections.
- "Bounding planes": The two planes of constant z-value that bound the original mesh: one at the minimum z-value and one at the maximum z-value.
- "Cut": A plane of constant z-value that intersects the mesh. A single cut always has at least one cross-section associated with it.
- "Slice": A 3D partition of the mesh with its volume either between two cuts, or between one cut and a bounding plane.
    A single slice might contain multiple closed volumes.
- "Segment": A *contiguous* 3D volume of the mesh with its volume between two cuts. Each segment has a single
    contiguous external surface and at least one contiguous internal surface.
- "Cross-section": A contiguous 2D area resulting from intersecting the mesh with a cut. It has a x,y,z position and a surface area (internal to the original mesh volume) in the xy-plane.
    Cross-sections occur at the nodes of the skeleton graph. Every cross-section is shared by two segments.
    Every cross-section has a fitted disk with radius and center position.
    (Several algorithms exist for fitting the disk, e.g. least squares, least absolute deviations, etc.)
- "Internal surface area": The surface area of a segment that overlaps a cross-section.
- "External surface area": The surface area of a segment that is shared with the original mesh.
- "Skeleton": A graph representation where segments are edges and nodes are cross sections (shared by two segments in neighboring slices)
    which share a cross section. No two nodes at the same cut (z-value) may be connected by an edge.

The algorithm follows these steps:

1. Validate input mesh: Check if mesh is watertight, has no self-intersections, and is a single hull.
    If not, raise an error. Mesh can be passed as a trimesh object or as an instance of discos.mesh.MeshManager.

2. Create bounding planes and cuts along z-axis using trimesh functionality. The cuts are spaced at regular intervals along the z-axis. The algorithm takes a parameter n_slices which is how many slices should be created. This is one greater than the number of cuts.
    Where a cut intersects the mesh, create new vertices and faces to represent the 2D cross-section. Fit a disk to the cross-section (by least squares or some other algorithm) and store the radius and center position.
    Create nodes where each bounding plane intersects the mesh, making one node at the minimum z-value and one node at the maximum z-value.

3. Identify segments within each slice. Segments are identified as the contiguous volume components of the mesh within each slice. Keeping track of surface areas and volumes is secondary. What's most important is that the overall connectivity stucture of the volume is preserved, which is handled next.

4. Build the Skeleton graph based on shared cross sections. First, place a node at each cross section. Then, connect two nodes with an edge if there is a path of contiguous volume between the two points *and* the two points are strictly in neighboring slices.

5. Validate volume and surface area conservation. The sum of the volumes of the segments should agree with the volume of the original mesh within a tolerance.


EXAMPLE:
    Consider a cylinder with radius 1 m and height 4 m, aligned with the z-axis.
    It's volume is 4*pi*1^2 = 4pi m^3, and it's surface area is 4*2pi*1 (side) + 2*pi*1^2 (caps) = 10pi m^2.
    We will slice the cylinder with n_slices = 4. Each segment should have the same volume = pi m^3.
    The two segments on the ends will have the same external surface area = 2*pi*1*1 (side) + pi*1^2 (cap) = 3pi m^2
        and the same internal surface area = pi*1^2 (cross section) = pi m^2.
    The two segments in the middle will have the same external surface area = 2*pi*1*1 (side) = 2pi m^2
        and the same internal surface area = 2*pi*1^2 (two cross sections) = 2pi m^2.
    The skeleton graph should have 5 nodes (one at each cut and one at each bounding plane) and 4 edges (one between each pair of cuts).



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
class CrossSection:
    """Planar cross-section at a specific z-position.

    Attributes:
        z: Z position of the cross-section plane in world coordinates.
        area: 2D area (inside the mesh) of the cross-section in the xy-plane.
        center: 3D center position [x, y, z] of the fitted disk (node position).
        radius: Radius of the fitted disk.
        boundary_2d: Optional Nx2 array of 2D boundary points (xy) used for fitting.
        slice_index: Index of the slice this cross-section belongs to (0..n_slices).
        index_within_slice: Index among cross-sections at the same z level.
    """

    z: float
    area: float
    center: np.ndarray
    radius: float
    boundary_2d: Optional[np.ndarray]
    slice_index: int
    index_within_slice: int


@dataclass
class Segment:
    """Segment connecting two cross-sections in adjacent slices.

    Attributes:
        u_id: Node ID of the first cross-section (e.g., "slice{i}_cs{j}").
        v_id: Node ID of the second cross-section (adjacent slice).
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
    slice_index: int,
    radius_method: str = "area",
) -> List[CrossSection]:
    """Extract cross-sections at a given z-plane and fit disks.

    Uses `trimesh.Trimesh.section` -> `Path3D` -> discrete polylines.
    For each discrete closed loop, computes area and a center/radius.

    Args:
        mesh: Watertight input mesh.
        z_value: World z of the cutting plane.
        slice_index: Index of the plane in the slice stack.
        radius_method: 'area' uses equivalent-area circle; 'algebraic' fits
            a circle to boundary points via linear least squares.

    Returns:
        List of `CrossSection` objects ordered by discovery; `index_within_slice`
        reflects that order within the given `slice_index`.
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
            # Equivalent area circle
            r = float(np.sqrt(area / math.pi))
            cx, cy = centroid_2d
        elif radius_method == "algebraic":
            ctr2d, r = _fit_circle_algebraic(boundary_2d)
            cx, cy = float(ctr2d[0]), float(ctr2d[1])
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
                slice_index=slice_index,
                index_within_slice=idx,
            )
        )
        idx += 1

    return sections


def _check_overlap_in_slice(
    cross_sections: List[CrossSection], tolerance: float
) -> None:
    """Raise ValueError if any two disks in the same slice overlap beyond tolerance.

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
                        f"slice {ci.slice_index} cs {ci.index_within_slice} and "
                        f"slice {cj.slice_index} cs {cj.index_within_slice}"
                    )


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


def _frustum_volume(h: float, r1: float, r2: float) -> float:
    """Volume of a circular frustum with height h and radii r1, r2."""
    return math.pi * h * (r1 * r1 + r1 * r2 + r2 * r2) / 3.0


class SkeletonGraph(nx.Graph):
    """Skeleton graph where nodes are cross-sections and edges are segments.

    Node attributes:
        - center: np.ndarray(3,)
        - radius: float
        - z: float
        - slice_index: int
        - index_within_slice: int
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
        with_labels: bool = False,
        node_size: int = 50,
        node_color: str = "C0",
        edge_color: str = "0.6",
        **kwargs: Any,
    ) -> Any:
        """Draw the skeleton graph in 2D using (x,z) or (y,z) coordinates.

        Args:
            axis: Horizontal axis to use ("x" or "y"). Vertical axis is always z.
            ax: Optional matplotlib Axes to draw into. If None, a new figure/axes is created.
            with_labels: Whether to render node labels.
            node_size: Node marker size passed to networkx.draw.
            node_color: Node color.
            edge_color: Edge color.
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

        # Prepare axes
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots()
            created_fig = True

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
        # Aim for equal aspect in data coordinates when possible
        try:
            ax.set_aspect("equal", adjustable="box")
        except Exception:
            pass

        # If we created the figure, return its axes; otherwise return provided ax
        return ax

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
    ) -> "SkeletonGraph":
        """Construct a skeleton graph from a mesh.

        Args:
            mesh_or_manager: trimesh.Trimesh or object with .mesh (Trimesh)
            n_slices: Number of slices (cuts = n_slices - 1). Must be >= 1.
            radius_method: 'area' (equivalent area circle) or 'algebraic'.
            samples_per_edge: Samples for inside-volume line-of-sight.
            overlap_tolerance: Tolerance for overlap detection in same slice.
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

        # 3. Extract cross-sections at each node plane
        cs_by_slice: List[List[CrossSection]] = []
        for si, z in enumerate(z_positions):
            sections = _extract_cross_sections_for_plane(
                mesh, float(z), slice_index=si, radius_method=radius_method
            )
            if len(sections) == 0:
                warnings.warn(f"No cross-sections found at z={float(z):.6g}")
            # Enforce non-overlap among same-plane cross-sections.
            _check_overlap_in_slice(sections, tolerance=overlap_tolerance)
            cs_by_slice.append(sections)

        # 4. Build graph nodes
        G = cls()
        node_ids_by_slice: List[List[str]] = []
        for si, sections in enumerate(cs_by_slice):
            ids_this_slice: List[str] = []
            for cs in sections:
                node_id = f"slice{si}_cs{cs.index_within_slice}"
                ids_this_slice.append(node_id)
                G.add_node(
                    node_id,
                    center=cs.center,
                    radius=cs.radius,
                    z=cs.z,
                    slice_index=cs.slice_index,
                    index_within_slice=cs.index_within_slice,
                    area=cs.area,
                )
            node_ids_by_slice.append(ids_this_slice)

        # 5. Connect adjacent slices based on inside-volume line-of-sight.
        #    Only adjacent slices may connect; same-slice connections are disallowed.
        segments: List[Segment] = []
        for si in range(len(cs_by_slice) - 1):
            lower = cs_by_slice[si]
            upper = cs_by_slice[si + 1]
            for i, c0 in enumerate(lower):
                for j, c1 in enumerate(upper):
                    # Rule: only adjacent slices may connect (we are in adjacent slices)
                    # Check inside-volume path via straight line between centers.
                    # If all sampled points lie inside, add an edge representing a
                    # circular frustum between radii r1 and r2.
                    if _line_of_sight_inside(
                        mesh, c0.center, c1.center, samples_per_edge
                    ):
                        u_id = f"slice{si}_cs{i}"
                        v_id = f"slice{si + 1}_cs{j}"
                        length = float(np.linalg.norm(c1.center - c0.center))
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

        return G


def skeletonize(
    mesh_or_manager: Union[trimesh.Trimesh, Any],
    n_slices: int,
    radius_method: str = "area",
    samples_per_edge: int = 25,
    overlap_tolerance: float = 1e-6,
    validate: bool = True,
    volume_tol: float = 0.05,
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
    )

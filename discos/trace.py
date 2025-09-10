"""
Polyline-guided tracing to SWC.

This module builds an SWC skeleton by tracing along user-provided polylines and
estimating radii from local mesh cross-sections perpendicular to polyline
Tangents.

High level:
- Resample each input polyline at approximately fixed arc-length spacing.
- For each resampled point P with local tangent t, intersect the mesh with the
  plane passing through P with normal = t. Compose the resulting closed curves
  into polygons-with-holes and select the polygon that contains P (in the local
  plane) or, failing that, the one whose boundary is closest to P.
- Fit an equivalent circle radius from that polygon using one of:
    * equivalent_area:    r = sqrt(A / pi)
    * equivalent_perimeter: r = L / (2*pi), using the exterior boundary length
- Create a SkeletonGraph with:
    - one node per sample with center at P (exact polyline coordinate) and the
      fitted radius; area stored for diagnostics
    - edges connecting consecutive samples along each polyline
- Export SWC using SkeletonGraph.to_swc. By default, cycles are broken using the
  "duplicate_junction" strategy (duplicate a branching node and rewire one
  incident cycle edge to the duplicate) so the SWC remains a single tree while
  preserving all modeled connections.

Notes:
- This module intentionally does not change connectivity beyond linking samples
  along each polyline. If polylines contain cycles, SWC export will break cycles
  according to the selected mode (remove one edge or duplicate a junction).
- Coordinates are taken directly from the polylines (no snapping by default).
  An optional snap-to-mesh pass is provided via PolylinesSkeleton if desired.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import shapely.geometry as sgeom
import trimesh

from .mesh import MeshManager
from .polylines import PolylinesSkeleton
from .skeleton import SkeletonGraph

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class TraceOptions:
    spacing: float = 1.0  # sampling step along polylines (mesh units)
    radius_mode: str = "equivalent_area"  # or "equivalent_perimeter"
    annotate_cycles: bool = True
    cycle_mode: str = "duplicate_junction"  # or "remove_edge"
    # When the exact plane P,t yields an empty section, try small offsets
    # along the normal by +/- k * eps until found or max_tries.
    section_probe_eps: float = 1e-4
    section_probe_tries: int = 3
    # Optional: snap polylines to mesh surface prior to tracing
    snap_polylines_to_mesh: bool = False
    max_snap_distance: Optional[float] = None
    # Undo selected transforms at SWC export (names from MeshManager.transform_stack)
    undo_transforms: Optional[List[str]] = None
    type_index: int = 3


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_traced_skeleton_graph(
    mesh_or_manager: Union[trimesh.Trimesh, MeshManager],
    polylines: PolylinesSkeleton,
    *,
    options: Optional[TraceOptions] = None,
) -> SkeletonGraph:
    """
    Trace polylines over a mesh and build a SkeletonGraph with nodes at sampled
    polyline points and radii fit from local mesh cross-sections.

    Args:
        mesh_or_manager: The mesh to intersect against (trimesh.Trimesh) or a
            MeshManager instance.
        polylines: Set of polylines (already in the same coordinate frame as the mesh).
        options: TraceOptions controlling sampling, radius mode, and cycle handling.

    Returns:
        A populated SkeletonGraph. Use `to_swc` to export with cycle-breaking.
    """
    if options is None:
        options = TraceOptions()

    # Resolve mesh
    if isinstance(mesh_or_manager, MeshManager):
        mm = mesh_or_manager
        mesh = mm.to_trimesh()
        transform_stack = getattr(mm, "transform_stack", [])
    elif isinstance(mesh_or_manager, trimesh.Trimesh):
        mesh = mesh_or_manager
        mm = None
        transform_stack = []
    else:
        raise TypeError("mesh_or_manager must be a trimesh.Trimesh or MeshManager")

    if mesh is None or len(getattr(mesh, "vertices", [])) == 0:
        raise ValueError("Mesh is empty or not provided")

    # Optionally snap polylines to the mesh surface
    pls = polylines.copy()
    if options.snap_polylines_to_mesh:
        try:
            moved, mean = pls.snap_to_mesh_surface(
                mesh,
                project_outside_only=True,
                max_distance=options.max_snap_distance,
            )
            logger.info(
                "Polylines snapped to mesh surface: moved=%d, mean=%.4g", moved, mean
            )
        except Exception as e:
            logger.warning("Failed snapping polylines to mesh: %s", e)

    # Pre-compute mesh scale for section probing epsilon
    V = np.asarray(mesh.vertices, dtype=float)
    bbox_size = float(np.linalg.norm(V.max(axis=0) - V.min(axis=0))) if V.size else 1.0
    eps = max(
        1e-12, float(options.section_probe_eps) * (bbox_size if bbox_size > 0 else 1.0)
    )

    # Build a KDTree over mesh vertices for robust radius fallback (if SciPy available)
    v_kdtree = None
    try:
        if V.size > 0:
            from scipy.spatial import cKDTree  # type: ignore

            v_kdtree = cKDTree(V)
    except Exception:
        v_kdtree = None

    # Build graph
    skel = SkeletonGraph()

    # Snapshot transforms from the mesh manager for selective SWC undo, if any
    try:
        _applied_transforms: List[Dict[str, Any]] = []
        for t in transform_stack:
            try:
                _applied_transforms.append(
                    {
                        "name": getattr(t, "name", None),
                        "M": np.array(getattr(t, "M", np.eye(4)), dtype=float),
                        "is_uniform_scale": bool(getattr(t, "is_uniform_scale", False)),
                        "uniform_scale": (
                            float(getattr(t, "uniform_scale", 1.0))
                            if getattr(t, "is_uniform_scale", False)
                            else None
                        ),
                    }
                )
            except Exception:
                continue
        skel.transforms_applied = _applied_transforms
    except Exception:
        skel.transforms_applied = []

    # Helper to allocate node ids
    next_id = 0

    def alloc_id() -> int:
        nonlocal next_id
        nid = next_id
        next_id += 1
        return nid

    # Process each polyline
    for pl_index, pl in enumerate(pls.as_arrays()):
        if pl is None or pl.size == 0 or pl.shape[0] < 2:
            continue
        # Resample
        samples = _resample_polyline(pl, float(options.spacing))
        if samples.shape[0] == 0:
            continue
        # Precompute tangents on the resampled curve
        tangents = _estimate_tangents(samples)

        prev_node: Optional[int] = None
        for i in range(samples.shape[0]):
            P = samples[i]
            n = tangents[i]
            if not np.all(np.isfinite(n)) or float(np.linalg.norm(n)) <= 1e-12:
                # Fallback tangent from original segment if degenerate
                if i + 1 < samples.shape[0]:
                    n = samples[i + 1] - samples[i]
                elif i > 0:
                    n = samples[i] - samples[i - 1]
                else:
                    n = np.array([0.0, 0.0, 1.0], dtype=float)
                n = n / (np.linalg.norm(n) + 1e-12)

            # Fit local cross-section radius
            poly2d = _cross_section_polygon_near_point(
                mesh=mesh,
                origin=P,
                normal=n,
                eps=eps,
                max_tries=int(options.section_probe_tries),
            )
            if poly2d is not None:
                area = float(poly2d.area)
                if options.radius_mode == "equivalent_perimeter":
                    perim = float(poly2d.exterior.length)
                    radius = perim / (2.0 * math.pi) if perim > 0 else 0.0
                else:
                    radius = math.sqrt(area / math.pi) if area > 0 else 0.0
            else:
                # No section found; robust fallback radius = distance to nearest mesh vertex
                area = 0.0
                try:
                    if v_kdtree is not None and V.size > 0:
                        dd, _ = v_kdtree.query(P, k=1)
                        radius = float(dd)
                    else:
                        try:
                            from trimesh.proximity import closest_point  # type: ignore

                            CP, dist, _tri = closest_point(mesh, P.reshape(1, 3))
                            _ = CP  # unused except for sanity
                            radius = float(dist[0])
                        except Exception:
                            if V.size > 0:
                                radius = float(np.min(np.linalg.norm(V - P, axis=1)))
                            else:
                                radius = 0.0
                except Exception:
                    radius = 0.0

            # Add node
            nid = alloc_id()
            j = {
                "id": nid,
                "z": float(P[2]),
                "center": np.array(P, dtype=float),
                "radius": float(radius),
                "area": float(area),
                "slice_index": int(pl_index),
                "cross_section_index": int(i),
            }
            skel.add_junction(_junction_from_dict(j))

            # Edge from previous sample
            if prev_node is not None and prev_node != nid:
                try:
                    skel.G.add_edge(
                        prev_node, nid, kind="trace", polyline_index=int(pl_index)
                    )
                except Exception:
                    pass
            prev_node = nid

    return skel


def trace_polylines_to_swc(
    mesh_or_manager: Union[trimesh.Trimesh, MeshManager],
    polylines: PolylinesSkeleton,
    swc_path: str,
    *,
    options: Optional[TraceOptions] = None,
) -> None:
    """
    Convenience wrapper: build a traced SkeletonGraph from polylines+mesh and
    export directly to an SWC file using the same cycle-breaking logic as
    SkeletonGraph.to_swc.

    Args:
        mesh_or_manager: trimesh or MeshManager
        polylines: polylines in the same coordinate frame
        swc_path: destination filepath (.swc)
        options: TraceOptions
    """
    if options is None:
        options = TraceOptions()
    skel = build_traced_skeleton_graph(mesh_or_manager, polylines, options=options)
    skel.to_swc(
        swc_path,
        type_index=options.type_index,
        annotate_cycles=bool(options.annotate_cycles),
        cycle_mode=str(options.cycle_mode),
        undo_transforms=options.undo_transforms,
        force_single_tree=True,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _junction_from_dict(d: Dict[str, Any]):
    """Create a skeleton.Junction dataclass instance from a dict of fields."""
    from .skeleton import Junction  # local import to avoid circulars at import time

    return Junction(
        id=int(d["id"]),
        z=float(d["z"]),
        center=np.asarray(d["center"], dtype=float),
        radius=float(d["radius"]),
        area=float(d["area"]),
        slice_index=int(d["slice_index"]),
        cross_section_index=int(d["cross_section_index"]),
    )


def _resample_polyline(pl: np.ndarray, spacing: float) -> np.ndarray:
    """Resample a polyline at approximately constant arc-length spacing.

    Includes the first and last vertex; inserts intermediate points every
    multiple of `spacing` along cumulative arclength.
    """
    P = np.asarray(pl, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3 or P.shape[0] == 0:
        return np.zeros((0, 3), dtype=float)
    if P.shape[0] == 1:
        return P.copy()

    seg = np.linalg.norm(P[1:] - P[:-1], axis=1)
    L = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(L[-1])

    if total <= 0.0:
        return P[[0], :].copy()

    step = float(max(spacing, 1e-12))
    # Always include start and end
    targets = list(np.arange(0.0, total, step))
    if targets[-1] != total:
        targets.append(total)

    out: List[np.ndarray] = []
    si = 0  # segment index
    for t in targets:
        # advance si until L[si] <= t <= L[si+1]
        while si < len(seg) and L[si + 1] < t:
            si += 1
        if si >= len(seg):
            out.append(P[-1])
            continue
        t0 = L[si]
        t1 = L[si + 1]
        if t1 <= t0:
            out.append(P[si])
            continue
        alpha = (t - t0) / (t1 - t0)
        Q = (1.0 - alpha) * P[si] + alpha * P[si + 1]
        out.append(Q)

    return np.vstack(out) if out else P[[0], :].copy()


def _estimate_tangents(P: np.ndarray) -> np.ndarray:
    """Estimate unit tangents for a polyline represented by points P."""
    n = P.shape[0]
    T = np.zeros_like(P)
    if n == 1:
        T[0] = np.array([0.0, 0.0, 1.0], dtype=float)
        return T
    for i in range(n):
        if i == 0:
            v = P[1] - P[0]
        elif i == n - 1:
            v = P[n - 1] - P[n - 2]
        else:
            v = P[i + 1] - P[i - 1]
        norm = float(np.linalg.norm(v))
        if norm <= 1e-12 or not np.isfinite(norm):
            T[i] = np.array([0.0, 0.0, 1.0], dtype=float)
        else:
            T[i] = v / norm
    return T


def _plane_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return orthonormal (u, v, n) basis for a plane with normal n.

    u and v span the plane; n is the normalized input normal.
    """
    n = np.asarray(normal, dtype=float)
    n = n / (np.linalg.norm(n) + 1e-12)
    # pick a not-near-parallel axis
    ax = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, ax)
    u /= np.linalg.norm(u) + 1e-12
    v = np.cross(n, u)
    v /= np.linalg.norm(v) + 1e-12
    return u, v, n


def _world_to_local_plane(P: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Return 4x4 transform that maps world -> local plane coords centered at P.

    In local coordinates, the plane is z=0; x,y span the plane.
    """
    u, v, n = _plane_basis(normal)
    R = np.column_stack([u, v, n])  # world axes -> local
    # We want X_local = R^T * (X_world - P)
    M = np.eye(4, dtype=float)
    M[:3, :3] = R.T
    M[:3, 3] = -R.T @ np.asarray(P, dtype=float)
    return M


def _compose_polygons_with_holes(polys: List[sgeom.Polygon]) -> List[sgeom.Polygon]:
    """Compose simple polygons into polygons-with-holes by containment parity.

    Mirrors discos.skeleton._compose_polygons_with_holes but kept local to avoid
    tight coupling.
    """
    if not polys:
        return []
    # Precompute bounds and pairwise containment depth
    n = len(polys)
    contains = [[False] * n for _ in range(n)]
    depths = [0] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            try:
                contains[i][j] = polys[i].contains(polys[j])
            except Exception:
                contains[i][j] = False

    def depth_of(idx: int) -> int:
        d = 0
        for k in range(n):
            if contains[k][idx]:
                d += 1
        return d

    depths = [depth_of(i) for i in range(n)]
    # Group children under each even-depth parent
    result: List[sgeom.Polygon] = []
    for i in range(n):
        if depths[i] % 2 != 0:
            continue  # skip holes at odd depth
        exterior_poly = polys[i]
        holes_coords: List[List[Tuple[float, float]]] = []
        for j in range(n):
            if i == j:
                continue
            if contains[i][j] and depths[j] == depths[i] + 1:
                try:
                    ring = list(polys[j].exterior.coords)
                    holes_coords.append([(float(x), float(y)) for x, y in ring])
                except Exception:
                    continue
        try:
            composed = sgeom.Polygon(exterior_poly.exterior.coords, holes=holes_coords)
            if composed.is_valid and composed.area > 0:
                result.append(composed)
        except Exception:
            # Fallback to original exterior only
            if exterior_poly.is_valid and exterior_poly.area > 0:
                result.append(exterior_poly)
    return result


def _cross_section_polygon_near_point(
    mesh: trimesh.Trimesh,
    origin: np.ndarray,
    normal: np.ndarray,
    *,
    eps: float,
    max_tries: int,
) -> Optional[sgeom.Polygon]:
    """
    Intersect `mesh` with the plane through `origin` and normal `normal`, returning
    a single 2D polygon in the local plane most consistent with `origin`.

    Selection preference:
    1) A polygon that contains the local origin (0,0)
    2) Otherwise, polygon with minimal distance from its boundary to the origin

    If no section is found exactly at the plane, try small offsets along the
    normal by +/- k*eps for k=1..max_tries. Returns None if no polygons found.
    """
    P = np.asarray(origin, dtype=float)
    n = np.asarray(normal, dtype=float)
    n = n / (np.linalg.norm(n) + 1e-12)

    # Try origin plane and probe offsets
    offsets = [0.0]
    for k in range(1, int(max_tries) + 1):
        offsets.extend([+k * float(eps), -k * float(eps)])

    for off in offsets:
        o = P + off * n
        try:
            path = mesh.section(plane_origin=o, plane_normal=n)
        except Exception:
            path = None
        if path is None or not hasattr(path, "entities") or len(path.entities) == 0:
            continue

        # Map 3D curve points to local plane 2D
        M = _world_to_local_plane(P, n)
        polys_2d: List[sgeom.Polygon] = []
        try:
            for ent in getattr(path, "entities", []):
                # entity.points is (m,3) in world coords; transform to local
                pts3 = np.asarray(getattr(ent, "points", None), dtype=float)
                if pts3.ndim != 2 or pts3.shape[1] != 3 or pts3.shape[0] < 2:
                    continue
                # homogeneous transform
                ones = np.ones((pts3.shape[0], 1), dtype=float)
                v2 = (M @ np.hstack([pts3, ones]).T).T[:, :3]
                # ensure closure and take XY
                XY = v2[:, :2]
                if not np.allclose(XY[0], XY[-1]):
                    XY = np.vstack([XY, XY[0]])
                poly = sgeom.Polygon(XY)
                if poly.is_valid and poly.area > 0:
                    polys_2d.append(poly)
        except Exception:
            polys_2d = []

        if not polys_2d:
            continue

        # Compose holes by containment parity
        try:
            composed = _compose_polygons_with_holes(polys_2d)
        except Exception:
            composed = polys_2d
        if not composed:
            continue

        origin_pt = sgeom.Point(0.0, 0.0)
        # Prefer polygons that contain the origin
        containing = [poly for poly in composed if poly.contains(origin_pt)]
        if containing:
            # If multiple, pick the one with smallest area to be conservative
            containing.sort(key=lambda p: float(p.area))
            return containing[0]
        # Else pick polygon with minimum distance to origin
        composed.sort(key=lambda p: float(p.exterior.distance(origin_pt)))
        return composed[0]

    return None

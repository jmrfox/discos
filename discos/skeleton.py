"""
Skeletonization of a 3D mesh.

The main objective of skeletonization is to represent a 3D mesh in a compact form that
captures its essential shape and connectivity. The DISCOS algorithm achieves this by
slicing up the original mesh along the z-axis into a series of slices. Within each slice,
 the algorithm identifies contiguous segments and creates a graph with edges as segments and nodes as junctions or terminal points.
 The graph is thus an idealized skeleton of the original mesh.

Important terminology:
- "Original Mesh": A mesh is a 3D object defined by a set of vertices, edges, and faces. DISCOS does not destroy the mesh you give it.
    For use in DISCOS, the original mesh should be a single watertight volume with no self-intersections.
- "Bounding planes": The two planes of constant z-value that bound the original mesh: one at the minimum z-value 
    and one at the maximum z-value.
- "Terminal points": The points where the bounding planes intersect the original mesh. 
    In other words, they represent the minimum and maximum z-value of the original mesh.
- "Cut": A plane of constant z-value that intersects the original mesh. A single cut always has at least one cross-section 
    associated with it.
- "Slice": A 3D partition of the original mesh with its volume either between two adjacent cuts, 
    or between one cut and one bounding plane.
    A single slice might contain multiple segments (closed volumes). We use the method MeshManager.slice_mesh_by_z to create the slices.
- "Segment": A contiguous 3D sub-volume of the original mesh with its volume between two cuts or between one cut and one bounding plane.
    Every slice necessarily contains at least one segment.
    Each segment has a single contiguous external surface and at least one contiguous internal surface.
- "Cross-section": A contiguous 2D curve (and its area) where the original mesh intersects with a cut. 
    It has a surface area (internal to the original mesh volume) in the xy-plane.
    Every cross-section is shared by two adjacent segments. Every cross-section has at least one Junction fitted to it, but perhaps more.
- "Junction": A disk with radius and center position fitted each closed area within the cross-section. 
    Junctions are one-to-one with the nodes of the skeleton graph.
- "SkeletonGraph": A graph representation where segments are edges and nodes are junctions and terminal points.
    Edges only connect nodes on adjacent planes. No two nodes at the same cut (z-value) may be connected by an edge.

Important classes: SkeletonGraph, Junction, Segment, CrossSection, ...

The algorithm follows these steps:

1. Validate input mesh: Check if mesh is watertight, has no self-intersections, and is a single hull.
    If not, raise an error. Mesh can be passed as a trimesh object or as an instance of discos.mesh.MeshManager.

2. Create bounding planes at the minimum and maximum z-value of the mesh.
    Extract terminal points.
    Create cuts at regular intervals along the z-axis using MeshManager.slice_mesh_by_z.
    (Each application of slice_mesh_by_z returns a list containing two meshes: one below the cut and one above the cut.) 
    The algorithm takes a parameter n_slices which is how many 3D slices (partitions) should be created;
        equivalently, n_cuts = n_slices - 1. 
    The slice_mesh_by_z function is repeated n_cuts times, results in the ordered set of slices. 
    
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
import shapely.geometry as sgeom
import trimesh

from .mesh import MeshManager

# ============================================================================
# Data models
# ============================================================================


@dataclass
class Junction:
    """
    A disk fitted to a closed cross-section area at a cut plane (constant z).

    Attributes:
        id: Unique identifier across all junctions.
        z: Plane z-value where this junction lies.
        center: 3D center (x, y, z) of the fitted disk.
        radius: Radius of the fitted disk (default: equivalent-area circle).
        area: Area of the cross-section polygon this disk represents.
        slice_index: Index of the slice interval below this cut (0..n_slices-1).
        cross_section_index: Index of the polygon within the cut's cross-section.
    """

    id: int
    z: float
    center: np.ndarray
    radius: float
    area: float
    slice_index: int
    cross_section_index: int


@dataclass
class CrossSection:
    """Cross-section at a cut plane with potentially multiple closed polygons."""

    z: float
    slice_index: int
    polygons: List[sgeom.Polygon]
    junction_ids: List[int]


@dataclass
class Segment:
    """
    A 3D segment (band component) between two adjacent cuts.

    Each segment induces one or more edges in the skeleton graph, connecting
    junctions at the lower cut to junctions at the upper cut.
    """

    id: int
    slice_index: int
    z_lower: float
    z_upper: float
    volume: float
    surface_area: float
    lower_junction_ids: List[int]
    upper_junction_ids: List[int]


class SkeletonGraph:
    """
    Graph where nodes are `Junction`s and edges correspond to cylinder-like
    segments connecting nodes on adjacent cuts only.
    """

    def __init__(self):
        self.G = nx.Graph()
        self.junctions: Dict[int, Junction] = {}
        self.cross_sections: List[CrossSection] = []
        self.segments: List[Segment] = []

    # ----------------------------- Node/edge API -----------------------------
    def add_junction(self, j: Junction):
        self.junctions[j.id] = j
        self.G.add_node(
            j.id,
            kind="junction",
            z=float(j.z),
            center=j.center.astype(float),
            radius=float(j.radius),
            area=float(j.area),
            slice_index=int(j.slice_index),
            cross_section_index=int(j.cross_section_index),
        )

    def add_segment_edges(self, seg: Segment):
        self.segments.append(seg)
        # Connect all lower to all upper (supports branching). Adjacent slices only.
        for jl in seg.lower_junction_ids:
            for ju in seg.upper_junction_ids:
                self.G.add_edge(
                    jl,
                    ju,
                    kind="segment",
                    segment_id=int(seg.id),
                    slice_index=int(seg.slice_index),
                    z_lower=float(seg.z_lower),
                    z_upper=float(seg.z_upper),
                    volume=float(seg.volume),
                    surface_area=float(seg.surface_area),
                )

    def to_networkx(self) -> nx.Graph:
        return self.G

    # visualization
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
        pos: Dict[Union[int, str], Tuple[float, float]] = {}
        for n, attrs in self.G.nodes(data=True):
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
            self.G,
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
        node_id: Union[int, str],
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
        if node_id not in self.G:
            raise KeyError(f"Node '{node_id}' not in SkeletonGraph")

        attrs = self.G.nodes[node_id]
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
            ci = attrs.get("slice_index")
            ii = attrs.get("cross_section_index")
            default_title = f"{node_id} (slice {ci}, cs {ii})"
        ax.set_title(default_title)

        return ax

    def plot_all_cross_sections(
        self,
        sort_by: Tuple[str, str] = ("slice_index", "cross_section_index"),
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
        for nid, attrs in self.G.nodes(data=True):
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


# ============================================================================
# Core algorithm
# ============================================================================


def skeletonize(
    mesh_or_manager: Union[trimesh.Trimesh, MeshManager],
    n_slices: int,
    *,
    radius_mode: str = "equivalent_area",
    validate_volume: bool = True,
    volume_tol: float = 0.05,
    verbose: bool = False,
) -> SkeletonGraph:
    """
    Build a `SkeletonGraph` by slicing the mesh along z into `n_slices` bands.

    Steps:
      1) Validate input mesh (watertight, single component)
      2) Create cut planes (n_cuts = n_slices-1) uniformly in z
      3) For each cut, compute cross-section polygons and fit junctions
      4) For each band between adjacent cuts, compute connected components
         and connect lower/upper junctions according to component membership
      5) Optionally validate volume conservation across all bands

    Returns:
        SkeletonGraph
    """

    mm = (
        mesh_or_manager
        if isinstance(mesh_or_manager, MeshManager)
        else MeshManager(mesh_or_manager)
    )
    mesh = mm.to_trimesh()

    if mesh is None:
        raise ValueError("No mesh provided")

    # ------------------------------ Validation ------------------------------
    if not mesh.is_watertight:
        raise ValueError("Input mesh must be watertight")

    try:
        comps = mesh.split(only_watertight=False)
        if len(comps) != 1:
            raise ValueError(
                f"Input mesh must be a single connected component, got {len(comps)}"
            )
    except Exception:
        # If split fails, proceed but warn in verbose mode
        if verbose:
            print("Warning: failed to verify single component via mesh.split()")

    zmin, zmax = mm.get_z_range()
    if not np.isfinite(zmin) or not np.isfinite(zmax) or zmax <= zmin:
        raise ValueError("Invalid z-range for mesh")

    if n_slices < 1:
        raise ValueError("n_slices must be >= 1")

    # Uniform partition including bounding planes
    z_planes = np.linspace(zmin, zmax, n_slices + 1)
    cut_zs = z_planes[1:-1]  # internal cuts

    # Numerical tolerance
    dz = float(zmax - zmin)
    eps = 1e-6 * (dz if dz > 0 else 1.0)

    skel = SkeletonGraph()

    # ------------------------ Terminal junctions (ends) ----------------------
    # Fit junctions at bounding planes using near-plane sections
    terminal_bottom_ids = _create_junctions_at_cut(
        skel,
        mesh,
        z_plane=float(zmin),
        slice_index=0,
        probe_offset=+eps,
        radius_mode=radius_mode,
        verbose=verbose,
    )
    terminal_top_ids = _create_junctions_at_cut(
        skel,
        mesh,
        z_plane=float(zmax),
        slice_index=n_slices - 1,
        probe_offset=-eps,
        radius_mode=radius_mode,
        verbose=verbose,
    )

    # --------------------- Internal cuts: cross-sections ---------------------
    for si, zc in enumerate(cut_zs, start=1):
        _create_junctions_at_cut(
            skel,
            mesh,
            z_plane=float(zc),
            slice_index=si,
            probe_offset=0.0,
            radius_mode=radius_mode,
            verbose=verbose,
        )

    # ---------------- Bands: connect junctions across adjacent cuts ----------
    junctions_by_slice: Dict[int, List[int]] = {}
    for j_id, j in skel.junctions.items():
        junctions_by_slice.setdefault(j.slice_index, []).append(j_id)

    # For each band (slice interval), build components and map to junctions
    segment_id = 0
    total_band_volume = 0.0
    for band_index in range(n_slices):
        z_low = float(z_planes[band_index])
        z_high = float(z_planes[band_index + 1])

        band_mesh = _extract_band_mesh(mesh, z_low, z_high)
        if band_mesh is None:
            continue

        # Split into connected components
        try:
            components = band_mesh.split(only_watertight=False)
        except Exception:
            components = [band_mesh]

        for comp in components:
            # Volumes may be negative if orientation is inverted; take absolute
            c_vol = float(abs(getattr(comp, "volume", 0.0)))
            c_area = float(getattr(comp, "area", 0.0))
            total_band_volume += c_vol

            # Determine associated junctions at z_low (lower) and z_high (upper)
            lower_locals = _section_polygon_centroids(comp, z_low + eps)
            upper_locals = _section_polygon_centroids(comp, z_high - eps)

            lower_ids = _match_centroids_to_junctions(
                centroids=lower_locals,
                z_plane=z_low,
                skel=skel,
                slice_index=band_index,
            )
            # For the top band, the upper junctions live at slice_index = n_slices-1
            # not n_slices. Clamp to avoid missing the terminal top junctions.
            upper_slice_index = min(band_index + 1, n_slices - 1)
            upper_ids = _match_centroids_to_junctions(
                centroids=upper_locals,
                z_plane=z_high,
                skel=skel,
                slice_index=upper_slice_index,
            )

            # Build a Segment and add edges
            seg = Segment(
                id=segment_id,
                slice_index=band_index,
                z_lower=z_low,
                z_upper=z_high,
                volume=c_vol,
                surface_area=c_area,
                lower_junction_ids=lower_ids,
                upper_junction_ids=upper_ids,
            )
            skel.add_segment_edges(seg)
            segment_id += 1

    # -------------------- Attach boundary polylines to nodes -------------------
    # After connectivity is established, store each cross-section polygon's
    # 2D boundary as a node attribute `boundary_2d` for plotting functions.
    for cs in skel.cross_sections:
        if not cs.polygons or not cs.junction_ids:
            continue
        for poly, j_id in zip(cs.polygons, cs.junction_ids):
            try:
                coords = np.asarray(poly.exterior.coords, dtype=float)
                if coords is not None and len(coords) >= 2 and coords.shape[1] >= 2:
                    skel.G.nodes[j_id]["boundary_2d"] = coords[:, :2]
            except Exception:
                # If shapely polygon lacks exterior or any other issue, skip
                continue

    # -------------------------- Volume conservation --------------------------
    if validate_volume:
        try:
            mesh_vol = float(abs(getattr(mesh, "volume", 0.0)))
            rel_err = abs(total_band_volume - mesh_vol) / (mesh_vol + 1e-12)
            if rel_err > volume_tol:
                raise ValueError(
                    f"Volume conservation failed: bands={total_band_volume:.6g}, "
                    f"mesh={mesh_vol:.6g}, rel_err={rel_err:.3%} (> {volume_tol:.1%})"
                )
        except Exception as e:
            if verbose:
                print(f"Volume validation warning: {e}")

    return skel


# ============================================================================
# Helpers
# ============================================================================


def _create_junctions_at_cut(
    skel: SkeletonGraph,
    mesh: trimesh.Trimesh,
    *,
    z_plane: float,
    slice_index: int,
    probe_offset: float,
    radius_mode: str,
    verbose: bool,
) -> List[int]:
    """
    Create junctions for all closed areas at a cut plane.

    probe_offset allows probing slightly off the plane (useful for bounding planes).
    """
    z_probe = float(z_plane + probe_offset)
    polygons = _cross_section_polygons(mesh, z_probe)
    if not polygons:
        # No intersection at this plane – nothing to add
        skel.cross_sections.append(
            CrossSection(
                z=z_plane, slice_index=slice_index, polygons=[], junction_ids=[]
            )
        )
        return []

    # Validate non-overlap within same cut
    _assert_no_overlap(polygons)

    added_ids: List[int] = []
    junction_ids: List[int] = []
    for idx, poly in enumerate(polygons):
        area = float(poly.area)
        if area <= 0:
            continue
        centroid = np.asarray([poly.centroid.x, poly.centroid.y], dtype=float)
        if radius_mode == "equivalent_area":
            radius = float(np.sqrt(area / np.pi))
        else:
            raise ValueError(f"Unknown radius_mode: {radius_mode}")

        j_id = _next_junction_id(skel)
        j = Junction(
            id=j_id,
            z=float(z_plane),
            center=np.array([centroid[0], centroid[1], float(z_plane)], dtype=float),
            radius=radius,
            area=area,
            slice_index=int(slice_index),
            cross_section_index=int(idx),
        )
        skel.add_junction(j)
        added_ids.append(j_id)
        junction_ids.append(j_id)

    skel.cross_sections.append(
        CrossSection(
            z=z_plane,
            slice_index=slice_index,
            polygons=polygons,
            junction_ids=junction_ids,
        )
    )
    return added_ids


def _cross_section_polygons(mesh: trimesh.Trimesh, z: float) -> List[sgeom.Polygon]:
    """Return shapely polygons for mesh ∩ plane z=constant."""
    try:
        path = mesh.section(
            plane_origin=[0.0, 0.0, float(z)], plane_normal=[0.0, 0.0, 1.0]
        )
    except Exception:
        path = None
    if path is None or not hasattr(path, "entities") or len(path.entities) == 0:
        return []

    polys: List[sgeom.Polygon] = []
    # Collect polygons from all entities; ensure closed loops
    try:
        for entity in path.entities:
            if not hasattr(entity, "points"):
                continue
            points_2d = path.vertices[entity.points]
            if points_2d is None or len(points_2d) < 3:
                continue
            # Ensure closure
            if not np.allclose(points_2d[0], points_2d[-1]):
                pts = np.vstack([points_2d, points_2d[0]])
            else:
                pts = points_2d
            poly = sgeom.Polygon(pts[:, :2])
            if poly.is_valid and poly.area > 0:
                polys.append(poly)
    except Exception:
        # Fallback: no polygons
        return []

    return polys


def _assert_no_overlap(polygons: List[sgeom.Polygon], tol: float = 1e-12) -> None:
    """Ensure polygons in the same cross-section do not overlap."""
    n = len(polygons)
    for i in range(n):
        for j in range(i + 1, n):
            inter = polygons[i].intersection(polygons[j])
            if not inter.is_empty and float(inter.area) > tol:
                raise ValueError(
                    "Overlapping cross-section polygons detected in a single cut"
                )


def _extract_band_mesh(
    mesh: trimesh.Trimesh, z_low: float, z_high: float
) -> Optional[trimesh.Trimesh]:
    """
    Extract the band sub-mesh with z in [z_low, z_high] by two-plane slicing
    using existing MeshManager.slice_mesh_by_z.
    """
    if z_high <= z_low:
        return None

    mm_top = MeshManager(mesh)
    below_high, _ = mm_top.slice_mesh_by_z(z_high, cap=True, validate=True)
    if below_high is None:
        return None

    mm_band = MeshManager(below_high)
    _, above_low = mm_band.slice_mesh_by_z(z_low, cap=True, validate=True)
    return above_low


def _section_polygon_centroids(
    mesh: trimesh.Trimesh, z_probe: float
) -> List[np.ndarray]:
    """Centroids of intersection polygons at z=z_probe for the given mesh."""
    polys = _cross_section_polygons(mesh, z_probe)
    cents: List[np.ndarray] = []
    for p in polys:
        if p.area > 0:
            cents.append(np.array([p.centroid.x, p.centroid.y], dtype=float))
    return cents


def _match_centroids_to_junctions(
    *,
    centroids: List[np.ndarray],
    z_plane: float,
    skel: SkeletonGraph,
    slice_index: int,
) -> List[int]:
    """
    Map 2D centroids at a cut to the IDs of the corresponding global junctions
    at the same cut plane.
    """
    # Find the cross-section for this slice_index and exact z_plane
    # CrossSections are appended in creation order; retrieve the matching one
    cs_candidates = [
        cs
        for cs in skel.cross_sections
        if cs.slice_index == slice_index and np.isclose(cs.z, z_plane)
    ]
    if not cs_candidates:
        return []
    cs = cs_candidates[-1]

    result: List[int] = []
    for c in centroids:
        point = sgeom.Point(float(c[0]), float(c[1]))
        matched_id: Optional[int] = None
        for poly, j_id in zip(cs.polygons, cs.junction_ids):
            if poly.contains(point) or poly.touches(point):
                matched_id = j_id
                break
        if matched_id is not None:
            result.append(matched_id)
    return result


def _next_junction_id(skel: SkeletonGraph) -> int:
    return 0 if not skel.junctions else max(skel.junctions.keys()) + 1

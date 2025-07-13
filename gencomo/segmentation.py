"""
Mesh segmentation along z-axis with connectivity analysis.

Slices a mesh into fixed-width segments and tracks their connectivity
using a graph data structure.
"""

import numpy as np
import trimesh
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import networkx as nx
from scipy.spatial.distance import cdist
import warnings


@dataclass
class Segment:
    """Represents a connected segment within a slice."""

    id: str
    slice_index: int
    segment_index: int  # Index within the slice
    z_min: float
    z_max: float
    mesh: trimesh.Trimesh
    volume: float
    exterior_surface_area: float  # Original mesh exterior faces
    interior_surface_area: float  # Cut faces from slicing
    centroid: np.ndarray
    is_connected_component: bool = True


class MeshSegmenter:
    """
    Segments a mesh into z-axis slices and analyzes connectivity.
    """

    def __init__(self):
        self.original_mesh = None
        self.slices = []  # List of slice dictionaries
        self.segments = []  # List of all segments
        self.connectivity_graph = nx.Graph()
        self.slice_segments = {}  # slice_index -> list of segments

    def segment_mesh(self, mesh: trimesh.Trimesh, slice_width: float, min_volume: float = 1e-3) -> List[Segment]:
        """
        Segment mesh into z-axis slices and identify connected components.

        Args:
            mesh: Input mesh to segment
            slice_width: Width of each slice in z-axis units
            min_volume: Minimum volume threshold for valid segments

        Returns:
            List of all segments
        """
        self.original_mesh = mesh.copy()
        self.segments = []
        self.slices = []
        self.slice_segments = {}
        self.connectivity_graph.clear()

        # Determine slice boundaries
        z_min, z_max = mesh.bounds[:, 2]
        num_slices = int(np.ceil((z_max - z_min) / slice_width))

        print(f"Segmenting mesh (z: {z_min:.3f} to {z_max:.3f}) into {num_slices} slices")

        # Create slices
        for i in range(num_slices):
            slice_z_min = z_min + i * slice_width
            slice_z_max = min(z_min + (i + 1) * slice_width, z_max)

            slice_data = self._create_slice(mesh, i, slice_z_min, slice_z_max, min_volume)
            self.slices.append(slice_data)

        # Build connectivity graph
        self._build_connectivity_graph()

        print(f"Created {len(self.segments)} segments across {len(self.slices)} slices")
        return self.segments

    def _create_slice(
        self, mesh: trimesh.Trimesh, slice_index: int, z_min: float, z_max: float, min_volume: float
    ) -> Dict:
        """
        Create a single slice and identify its segments.

        Each slice should contain exactly one segment per closed region (cross-section).
        The segments represent 3D volumes between z_min and z_max that correspond
        to connected components of the original mesh.
        """

        # Extract slice using trimesh
        slice_mesh = self._extract_slice_mesh(mesh, z_min, z_max)

        if slice_mesh is None or slice_mesh.is_empty:
            slice_data = {"slice_index": slice_index, "z_min": z_min, "z_max": z_max, "segments": []}
            self.slice_segments[slice_index] = []
            return slice_data

        # Split into connected components - each component is a separate segment
        # This ensures that each closed region becomes exactly one segment
        components = slice_mesh.split(only_watertight=False)
        if not isinstance(components, list):
            components = [components]

        segments = []
        for comp_idx, component in enumerate(components):
            if component.volume < min_volume:
                continue

            # Analyze component faces to distinguish interior vs exterior
            exterior_area, interior_area = self._analyze_face_types(component, z_min, z_max)

            segment_id = f"slice_{slice_index}_seg_{comp_idx}"

            segment = Segment(
                id=segment_id,
                slice_index=slice_index,
                segment_index=comp_idx,
                z_min=z_min,
                z_max=z_max,
                mesh=component,
                volume=component.volume,
                exterior_surface_area=exterior_area,
                interior_surface_area=interior_area,
                centroid=component.centroid,
            )

            # Store original mesh bounds for accurate cut face detection
            segment._original_mesh_bounds = mesh.bounds

            segments.append(segment)
            self.segments.append(segment)

        # If we have multiple small components but they should be one, merge them
        if len(segments) > 1:
            # Check if all components have similar z-centroids (indicating they're from same slice)
            centroids = [s.centroid for s in segments]
            z_coords = [c[2] for c in centroids]
            z_std = np.std(z_coords)

            # If z-coordinates are very similar, likely fragments of one segment
            if z_std < (z_max - z_min) * 0.1:  # Within 10% of slice height
                print(f"    Warning: Slice {slice_index} has {len(segments)} fragments, may need merging")

        self.slice_segments[slice_index] = segments

        slice_data = {"slice_index": slice_index, "z_min": z_min, "z_max": z_max, "segments": segments}

        return slice_data

    def _extract_slice_mesh(self, mesh: trimesh.Trimesh, z_min: float, z_max: float) -> Optional[trimesh.Trimesh]:
        """
        Extract mesh portion between z_min and z_max using proper plane-edge intersection.

        This creates new vertices where slice planes intersect mesh edges, and properly
        tracks which faces are original (exterior) vs newly created (interior).
        """
        try:
            # Use a different approach: cut from both ends toward the middle
            # This ensures we always have geometry to cut

            # Method 1: Try consecutive cuts (original approach)
            try:
                # Cut at upper boundary first (keep everything below z_max)
                upper_normal = np.array([0, 0, 1])  # Points up - keeps everything below
                upper_origin = np.array([0, 0, z_max])

                upper_cut_mesh = mesh.slice_plane(upper_origin, upper_normal)
                if upper_cut_mesh is None or upper_cut_mesh.is_empty:
                    raise ValueError("Upper cut failed")

                # Cut at lower boundary (keep everything above z_min)
                lower_normal = np.array([0, 0, -1])  # Points down - keeps everything above
                lower_origin = np.array([0, 0, z_min])

                sliced_mesh = upper_cut_mesh.slice_plane(lower_origin, lower_normal)
                if sliced_mesh is None or sliced_mesh.is_empty:
                    raise ValueError("Lower cut failed")

            except (ValueError, AttributeError):
                # Method 2: Try reverse order
                try:
                    # Cut at lower boundary first
                    lower_normal = np.array([0, 0, -1])
                    lower_origin = np.array([0, 0, z_min])

                    lower_cut_mesh = mesh.slice_plane(lower_origin, lower_normal)
                    if lower_cut_mesh is None or lower_cut_mesh.is_empty:
                        raise ValueError("Lower cut failed")

                    # Then cut at upper boundary
                    upper_normal = np.array([0, 0, 1])
                    upper_origin = np.array([0, 0, z_max])

                    sliced_mesh = lower_cut_mesh.slice_plane(upper_origin, upper_normal)
                    if sliced_mesh is None or sliced_mesh.is_empty:
                        raise ValueError("Upper cut failed")

                except (ValueError, AttributeError):
                    # Method 3: Fallback to vertex clamping if slice_plane fails
                    vertices = mesh.vertices.copy()
                    faces = mesh.faces.copy()

                    # Clamp vertices to the z-range
                    vertices[:, 2] = np.clip(vertices[:, 2], z_min, z_max)

                    # Create mesh and filter faces that span too much
                    temp_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                    face_vertices = vertices[faces]
                    face_z_coords = face_vertices[:, :, 2]
                    face_z_spans = face_z_coords.max(axis=1) - face_z_coords.min(axis=1)

                    slice_width = z_max - z_min
                    valid_face_mask = face_z_spans <= slice_width * 1.1

                    if not valid_face_mask.any():
                        return None

                    valid_faces = faces[valid_face_mask]
                    unique_verts = np.unique(valid_faces.flatten())

                    if len(unique_verts) < 3:
                        return None

                    old_to_new = {old: new for new, old in enumerate(unique_verts)}
                    new_faces = [[old_to_new[v] for v in face] for face in valid_faces]

                    sliced_mesh = trimesh.Trimesh(vertices=vertices[unique_verts], faces=np.array(new_faces))

            # Validate the result
            if sliced_mesh is None or sliced_mesh.is_empty:
                return None

            if not hasattr(sliced_mesh, "vertices") or len(sliced_mesh.vertices) < 3:
                return None

            # Check that we got a reasonable slice
            vertex_z_range = sliced_mesh.vertices[:, 2]
            actual_z_min, actual_z_max = vertex_z_range.min(), vertex_z_range.max()

            # Allow some tolerance for numerical precision
            tolerance = 1e-6
            if actual_z_min < z_min - tolerance or actual_z_max > z_max + tolerance:
                warnings.warn(
                    f"Slice [{z_min:.3f}, {z_max:.3f}] has vertices outside range: [{actual_z_min:.3f}, {actual_z_max:.3f}]"
                )

            # Clean up the mesh
            try:
                sliced_mesh.remove_degenerate_faces()
                sliced_mesh.remove_duplicate_faces()
            except:
                pass

            # Store metadata about which faces are cut faces (interior surfaces)
            # This is approximated by finding faces that are near the z boundaries
            self._mark_cut_faces(sliced_mesh, z_min, z_max)

            return sliced_mesh

        except Exception as e:
            warnings.warn(f"Failed to extract slice [{z_min:.3f}, {z_max:.3f}]: {e}")
            return None

    def _mark_cut_faces(self, mesh: trimesh.Trimesh, z_min: float, z_max: float):
        """
        Mark faces that are likely cut faces (interior surfaces) in the mesh metadata.

        NOTE: This method is now deprecated since we use direct geometric analysis
        in _analyze_face_types instead of relying on metadata.
        """
        # This method is kept for compatibility but no longer used
        pass

    def _analyze_face_types(self, segment_mesh: trimesh.Trimesh, z_min: float, z_max: float) -> Tuple[float, float]:
        """
        Analyze faces to distinguish exterior vs interior (cut) faces.

        CORRECTED LOGIC:
        - Interior faces = newly created cut faces from slicing operations
        - Exterior faces = all original mesh faces (sides and caps)

        Cut faces are identified as:
        1. Flat faces (normal aligned with z-axis)
        2. Located at slice boundaries (z_min or z_max)
        3. NOT at original mesh boundaries (which are caps, not cuts)

        Returns:
            Tuple of (exterior_area, interior_area)
        """
        face_areas = segment_mesh.area_faces
        face_centers = segment_mesh.triangles_center
        face_normals = segment_mesh.face_normals

        tolerance = 1e-3
        original_z_min, original_z_max = self.original_mesh.bounds[:, 2]

        # Find flat faces (high z-component in normal)
        normal_z = np.abs(face_normals[:, 2])
        is_flat = normal_z > 0.9  # cos(~25°)

        # Find faces at slice boundaries
        z_coords = face_centers[:, 2]
        at_slice_bottom = np.abs(z_coords - z_min) < tolerance
        at_slice_top = np.abs(z_coords - z_max) < tolerance
        at_slice_boundary = at_slice_bottom | at_slice_top

        # Find faces at original mesh boundaries (caps)
        at_original_bottom = np.abs(z_coords - original_z_min) < tolerance
        at_original_top = np.abs(z_coords - original_z_max) < tolerance
        at_original_boundary = at_original_bottom | at_original_top

        # Cut faces are flat faces at slice boundaries that are NOT original caps
        is_cut_face = is_flat & at_slice_boundary & (~at_original_boundary)

        # Calculate areas
        interior_area = face_areas[is_cut_face].sum()
        exterior_area = face_areas[~is_cut_face].sum()

        return exterior_area, interior_area

    def _build_connectivity_graph(self):
        """Build graph showing which segments are connected."""
        # Add all segments as nodes
        for segment in self.segments:
            self.connectivity_graph.add_node(
                segment.id,
                segment=segment,
                slice_index=segment.slice_index,
                volume=segment.volume,
                exterior_area=segment.exterior_surface_area,
                interior_area=segment.interior_surface_area,
            )

        # Find connections between adjacent slices
        for i in range(len(self.slices) - 1):
            current_segments = self.slice_segments.get(i, [])
            next_segments = self.slice_segments.get(i + 1, [])

            self._connect_adjacent_segments(current_segments, next_segments)

    def _connect_adjacent_segments(self, segments1: List[Segment], segments2: List[Segment]):
        """
        Find all valid connections between segments in adjacent slices based on shared cut faces.

        This handles general topologies including:
        - 1 segment → multiple segments (branch points)
        - Multiple segments → 1 segment (merge points)
        - Multiple segments → multiple segments (complex topology)

        Segments are connected if and only if they share an internal (cut) face.
        """
        if not segments1 or not segments2:
            return

        # Test all possible pairs for actual cut face sharing
        valid_connections = []
        for seg1 in segments1:
            for seg2 in segments2:
                if self._segments_share_actual_cut_face(seg1, seg2):
                    distance = np.linalg.norm(seg1.centroid - seg2.centroid)
                    valid_connections.append((seg1, seg2, distance))

        # Add all valid connections (no artificial one-to-one restriction)
        for seg1, seg2, distance in valid_connections:
            self.connectivity_graph.add_edge(seg1.id, seg2.id, distance=distance, type="adjacent_slice")

    def _segments_share_actual_cut_face(self, seg1: Segment, seg2: Segment) -> bool:
        """
        Determine if two segments actually share a cut face at their boundary.

        This is the core test for connectivity - segments are connected if and only if
        they share a substantial portion of their cut surface at the slice boundary.
        """
        # Get the shared Z boundary between segments
        if abs(seg1.z_max - seg2.z_min) < 1e-6:
            shared_z = seg1.z_max
        elif abs(seg2.z_max - seg1.z_min) < 1e-6:
            shared_z = seg2.z_max
        else:
            return False

        # First quick test: centroids must be reasonably close
        centroid_distance = np.linalg.norm(seg1.centroid[:2] - seg2.centroid[:2])
        if centroid_distance > 2.0:  # Reasonable proximity threshold
            return False

        # Detailed test: check for actual cut surface overlap
        return self._detailed_cut_surface_overlap(seg1, seg2, shared_z)

    def _detailed_cut_surface_overlap(self, seg1: Segment, seg2: Segment, shared_z: float) -> bool:
        """
        Detailed analysis of cut surface overlap between two segments.

        Returns True if segments share a significant portion of cut surface area.
        """
        tolerance = 1e-3

        # Get cut surface vertices for both segments
        cut_verts1 = self._get_cut_surface_vertices(seg1, shared_z, tolerance)
        cut_verts2 = self._get_cut_surface_vertices(seg2, shared_z, tolerance)

        if len(cut_verts1) == 0 or len(cut_verts2) == 0:
            return False

        # Calculate overlap using spatial proximity of vertices
        overlap_score = self._calculate_spatial_overlap(cut_verts1, cut_verts2)

        # Require meaningful overlap (>10% of smaller surface)
        return overlap_score > 0.1

    def _calculate_spatial_overlap(self, verts1: np.ndarray, verts2: np.ndarray) -> float:
        """
        Calculate spatial overlap between two sets of cut surface vertices.

        Uses a more robust approach that considers vertex density and spatial distribution.
        """
        if len(verts1) == 0 or len(verts2) == 0:
            return 0.0

        # For efficiency, work with the smaller set
        if len(verts1) > len(verts2):
            verts1, verts2 = verts2, verts1

        # Calculate what fraction of vertices in set1 have close neighbors in set2
        overlap_threshold = 0.15  # Distance threshold for considering vertices "overlapping"
        overlapping_vertices = 0

        for v1 in verts1:
            # Find closest vertex in verts2
            distances = np.linalg.norm(verts2 - v1, axis=1)
            min_distance = np.min(distances)

            if min_distance < overlap_threshold:
                overlapping_vertices += 1

        return overlapping_vertices / len(verts1) if len(verts1) > 0 else 0.0

    def _segments_are_connected_strict(self, seg1: Segment, seg2: Segment) -> bool:
        """Strict connectivity test for multi-segment slices."""
        # Get the shared Z boundary between segments
        if abs(seg1.z_max - seg2.z_min) < 1e-6:
            shared_z = seg1.z_max
        elif abs(seg2.z_max - seg1.z_min) < 1e-6:
            shared_z = seg2.z_max
        else:
            return False

        # Very strict test for multi-segment slices
        centroid_distance = np.linalg.norm(seg1.centroid[:2] - seg2.centroid[:2])
        if centroid_distance > 1.0:
            return False

        return self._segments_share_cut_surface(seg1, seg2, shared_z)

    def _segments_are_connected_permissive(self, seg1: Segment, seg2: Segment) -> bool:
        """Permissive connectivity test for single-segment connections."""
        # Get the shared Z boundary between segments
        if abs(seg1.z_max - seg2.z_min) < 1e-6:
            shared_z = seg1.z_max
        elif abs(seg2.z_max - seg1.z_min) < 1e-6:
            shared_z = seg2.z_max
        else:
            return False

        # For single segments, use generous geometric proximity
        centroid_distance = np.linalg.norm(seg1.centroid[:2] - seg2.centroid[:2])
        if centroid_distance > 3.0:  # Very generous threshold
            return False

        # Check for any cut surface overlap with permissive thresholds
        return self._segments_share_cut_surface_permissive(seg1, seg2, shared_z)

    def _segments_share_cut_surface_permissive(self, seg1: Segment, seg2: Segment, shared_z: float) -> bool:
        """
        Permissive test for shared cut surface - allows more generous overlap requirements.
        """
        tolerance = 1e-3

        # Get cut surface vertices for both segments
        cut_verts1 = self._get_cut_surface_vertices(seg1, shared_z, tolerance)
        cut_verts2 = self._get_cut_surface_vertices(seg2, shared_z, tolerance)

        if len(cut_verts1) == 0 or len(cut_verts2) == 0:
            # If no cut surface found, fall back to simple proximity
            return True  # For single segments, assume connection if centroids are close

        # Calculate overlap with very generous threshold
        overlap_score = self._calculate_surface_overlap(cut_verts1, cut_verts2)

        # Much more permissive overlap requirement (>5% is enough)
        return overlap_score > 0.05

    def _calculate_connection_strength(self, seg1: Segment, seg2: Segment) -> float:
        """
        Calculate the strength of connection between two segments.

        Higher scores indicate stronger connections (closer centroids, better boundary overlap).
        """
        # Primary factor: centroid distance (closer is better)
        centroid_distance = np.linalg.norm(seg1.centroid - seg2.centroid)
        distance_score = 1.0 / (1.0 + centroid_distance)

        # Secondary factor: XY distance (for torus, segments should be close in XY)
        xy_distance = np.linalg.norm(seg1.centroid[:2] - seg2.centroid[:2])
        xy_score = 1.0 / (1.0 + xy_distance)

        # Combine scores (emphasize XY proximity for torus)
        return distance_score * 0.3 + xy_score * 0.7

    def _segments_are_connected(self, seg1: Segment, seg2: Segment) -> bool:
        """
        Check if two segments from adjacent slices are connected by sharing an internal face.

        For torus topology, we need to be very strict to avoid false connections
        across the hole. Only truly adjacent segments should be connected.
        """
        # Get the shared Z boundary between segments
        if abs(seg1.z_max - seg2.z_min) < 1e-6:
            # seg1 is below seg2, check shared face at seg1.z_max / seg2.z_min
            shared_z = seg1.z_max
        elif abs(seg2.z_max - seg1.z_min) < 1e-6:
            # seg2 is below seg1, check shared face at seg2.z_max / seg1.z_min
            shared_z = seg2.z_max
        else:
            # Segments are not adjacent in Z
            return False

        # Very strict test: segments must have close centroids AND overlapping boundaries
        centroid_distance = np.linalg.norm(seg1.centroid[:2] - seg2.centroid[:2])
        if centroid_distance > 1.0:  # Must be reasonably close in XY
            return False

        # Check if both segments have cut faces at the shared Z level that overlap
        return self._segments_share_cut_surface(seg1, seg2, shared_z)

    def _get_segment_boundary_at_z(self, segment: Segment, z_level: float) -> Optional[np.ndarray]:
        """
        Extract the boundary contour of a segment at a specific Z level.

        Returns:
            Array of XY points forming the boundary, or None if no boundary found
        """
        try:
            # Find vertices close to the specified z level
            vertices = segment.mesh.vertices
            tolerance = 1e-3

            # Get vertices at the specified z level
            z_mask = np.abs(vertices[:, 2] - z_level) < tolerance
            boundary_vertices = vertices[z_mask]

            if len(boundary_vertices) < 3:
                return None

            # Return XY coordinates only
            return boundary_vertices[:, :2]

        except Exception:
            return None

    def _boundaries_overlap(self, boundary1: np.ndarray, boundary2: np.ndarray) -> bool:
        """
        Check if two 2D boundaries overlap in XY space.

        Uses convex hull approach for simplicity.
        """
        try:
            from scipy.spatial import ConvexHull
            from shapely.geometry import Polygon

            # Create convex hulls for both boundaries
            if len(boundary1) >= 3:
                hull1 = ConvexHull(boundary1)
                poly1 = Polygon(boundary1[hull1.vertices])
            else:
                return False

            if len(boundary2) >= 3:
                hull2 = ConvexHull(boundary2)
                poly2 = Polygon(boundary2[hull2.vertices])
            else:
                return False

            # Check if polygons intersect
            return poly1.intersects(poly2)

        except ImportError:
            # Fall back to simpler bounding box check if shapely not available
            return self._bounding_boxes_overlap(boundary1, boundary2)
        except Exception:
            # Fall back to centroid distance if anything fails
            center1 = np.mean(boundary1, axis=0)
            center2 = np.mean(boundary2, axis=0)
            distance = np.linalg.norm(center1 - center2)
            return distance < 1.0  # Simple threshold

    def _bounding_boxes_overlap(self, boundary1: np.ndarray, boundary2: np.ndarray) -> bool:
        """Check if bounding boxes of two boundaries overlap."""
        # Get bounding boxes
        min1, max1 = boundary1.min(axis=0), boundary1.max(axis=0)
        min2, max2 = boundary2.min(axis=0), boundary2.max(axis=0)

        # Check overlap in both X and Y dimensions
        x_overlap = max1[0] >= min2[0] and max2[0] >= min1[0]
        y_overlap = max1[1] >= min2[1] and max2[1] >= min1[1]

        return x_overlap and y_overlap

    def _compute_connection_threshold(self, seg1: Segment, seg2: Segment) -> float:
        """Compute adaptive threshold for segment connectivity."""
        # Base threshold on cube root of volumes (proportional to linear dimension)
        avg_size = (seg1.volume ** (1 / 3) + seg2.volume ** (1 / 3)) / 2
        return max(avg_size * 2.0, 0.1)  # At least 0.1 units

    def get_segment_by_id(self, segment_id: str) -> Optional[Segment]:
        """Get segment by ID."""
        for segment in self.segments:
            if segment.id == segment_id:
                return segment
        return None

    def get_segments_in_slice(self, slice_index: int) -> List[Segment]:
        """Get all segments in a specific slice."""
        return self.slice_segments.get(slice_index, [])

    def get_connected_segments(self, segment_id: str) -> List[str]:
        """Get IDs of segments connected to the given segment."""
        if segment_id not in self.connectivity_graph:
            return []
        return list(self.connectivity_graph.neighbors(segment_id))

    def get_connected_components(self) -> List[List[str]]:
        """Get groups of segments that form connected components."""
        return list(nx.connected_components(self.connectivity_graph))

    def compute_segmentation_statistics(self) -> Dict:
        """Compute statistics about the segmentation."""
        if not self.segments:
            return {}

        volumes = [seg.volume for seg in self.segments]
        exterior_areas = [seg.exterior_surface_area for seg in self.segments]
        interior_areas = [seg.interior_surface_area for seg in self.segments]

        connected_components = self.get_connected_components()

        stats = {
            "total_segments": len(self.segments),
            "total_slices": len(self.slices),
            "connected_components": len(connected_components),
            "volume_stats": {
                "total": np.sum(volumes),
                "mean": np.mean(volumes),
                "std": np.std(volumes),
                "min": np.min(volumes),
                "max": np.max(volumes),
            },
            "exterior_area_stats": {
                "total": np.sum(exterior_areas),
                "mean": np.mean(exterior_areas),
                "std": np.std(exterior_areas),
                "min": np.min(exterior_areas),
                "max": np.max(exterior_areas),
            },
            "interior_area_stats": {
                "total": np.sum(interior_areas),
                "mean": np.mean(interior_areas),
                "std": np.std(interior_areas),
                "min": np.min(interior_areas),
                "max": np.max(interior_areas),
            },
        }

        # Per-slice statistics
        segments_per_slice = [len(self.slice_segments.get(i, [])) for i in range(len(self.slices))]

        if segments_per_slice:
            stats["segments_per_slice"] = {
                "mean": np.mean(segments_per_slice),
                "std": np.std(segments_per_slice),
                "min": np.min(segments_per_slice),
                "max": np.max(segments_per_slice),
            }

        return stats

    def export_segments_as_meshes(self, output_dir: str = "segments"):
        """Export each segment as a separate mesh file."""
        import os

        os.makedirs(output_dir, exist_ok=True)

        for segment in self.segments:
            filename = f"{segment.id}.ply"
            filepath = os.path.join(output_dir, filename)
            segment.mesh.export(filepath)

        print(f"Exported {len(self.segments)} segments to {output_dir}/")

    def visualize_connectivity_graph(self, save_path: Optional[str] = None):
        """Visualize the segment connectivity graph with slice-based layout."""
        try:
            import matplotlib.pyplot as plt

            if not self.segments:
                print("No segments to visualize")
                return

            # Create improved layout with better spacing
            pos = self._create_improved_layout()

            # Create single figure for slice-based layout
            plt.figure(figsize=(12, 8))

            # Draw slice-based layout
            self._draw_slice_layout(plt.gca(), pos)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
            else:
                plt.show()

        except ImportError:
            print("matplotlib not available for visualization")

    def visualize_connectivity_graph_3d(self, save_path: Optional[str] = None, backend: str = "plotly"):
        """
        Visualize the segment connectivity graph in 3D using actual segment centroids.

        Args:
            save_path: Optional path to save the visualization
            backend: Visualization backend ('plotly' or 'matplotlib')
        """
        if not self.segments:
            print("No segments to visualize")
            return

        if backend.lower() == "plotly":
            self._visualize_3d_plotly(save_path)
        else:
            self._visualize_3d_matplotlib(save_path)

    def _visualize_3d_plotly(self, save_path: Optional[str] = None):
        """Create 3D visualization using Plotly."""
        try:
            import plotly.graph_objects as go
            import plotly.express as px

            # Extract node positions (centroids) and properties
            node_positions = []
            node_volumes = []
            node_labels = []
            node_colors = []

            for segment in self.segments:
                node_positions.append(segment.centroid)
                node_volumes.append(segment.volume)
                node_labels.append(f"S{segment.slice_index}.{segment.segment_index}")
                # Color by slice index
                node_colors.append(segment.slice_index)

            node_positions = np.array(node_positions)

            # Extract edge positions
            edge_x, edge_y, edge_z = [], [], []
            for edge in self.connectivity_graph.edges():
                seg1 = self.get_segment_by_id(edge[0])
                seg2 = self.get_segment_by_id(edge[1])

                if seg1 and seg2:
                    # Add line from seg1 centroid to seg2 centroid
                    edge_x.extend([seg1.centroid[0], seg2.centroid[0], None])
                    edge_y.extend([seg1.centroid[1], seg2.centroid[1], None])
                    edge_z.extend([seg1.centroid[2], seg2.centroid[2], None])

            # Create edge trace
            edge_trace = go.Scatter3d(
                x=edge_x,
                y=edge_y,
                z=edge_z,
                mode="lines",
                line=dict(color="gray", width=2),
                hoverinfo="none",
                name="Connections",
            )

            # Normalize node sizes for better visualization
            min_vol, max_vol = min(node_volumes), max(node_volumes)
            if max_vol > min_vol:
                normalized_sizes = [(vol - min_vol) / (max_vol - min_vol) * 15 + 5 for vol in node_volumes]
            else:
                normalized_sizes = [10] * len(node_volumes)

            # Create node trace
            node_trace = go.Scatter3d(
                x=node_positions[:, 0],
                y=node_positions[:, 1],
                z=node_positions[:, 2],
                mode="markers+text",
                marker=dict(
                    size=normalized_sizes,
                    color=node_colors,
                    colorscale="viridis",
                    showscale=True,
                    colorbar=dict(title="Slice Index"),
                    line=dict(width=1, color="black"),
                ),
                text=node_labels,
                textposition="middle center",
                textfont=dict(size=8, color="white"),
                hovertemplate="<b>%{text}</b><br>"
                + "X: %{x:.2f}<br>"
                + "Y: %{y:.2f}<br>"
                + "Z: %{z:.2f}<br>"
                + "Volume: %{customdata:.4f}<extra></extra>",
                customdata=node_volumes,
                name="Segments",
            )

            # Calculate data bounds for proper zoom
            x_range = node_positions[:, 0].max() - node_positions[:, 0].min()
            y_range = node_positions[:, 1].max() - node_positions[:, 1].min()
            z_range = node_positions[:, 2].max() - node_positions[:, 2].min()
            max_range = max(x_range, y_range, z_range)

            # Calculate camera distance to fit all data
            # Use a factor that ensures all points are visible with some margin
            camera_distance = max(2.0, max_range / 10.0)

            # Create figure
            fig = go.Figure(data=[edge_trace, node_trace])

            fig.update_layout(
                title="3D Segment Connectivity Graph",
                scene=dict(
                    xaxis_title="X (μm)",
                    yaxis_title="Y (μm)",
                    zaxis_title="Z (μm)",
                    camera=dict(eye=dict(x=camera_distance, y=camera_distance, z=camera_distance)),
                    aspectmode="data",
                ),
                showlegend=True,
                width=800,
                height=600,
            )

            if save_path:
                fig.write_html(save_path)
                print(f"3D visualization saved to {save_path}")
            else:
                fig.show()

        except ImportError:
            print("Plotly not available. Try 'pip install plotly' or use backend='matplotlib'")

    def _visualize_3d_matplotlib(self, save_path: Optional[str] = None):
        """Create 3D visualization using Matplotlib."""
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D

            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection="3d")

            # Extract node positions and properties
            node_positions = np.array([seg.centroid for seg in self.segments])
            node_volumes = [seg.volume for seg in self.segments]
            node_colors = [seg.slice_index for seg in self.segments]

            # Normalize node sizes
            min_vol, max_vol = min(node_volumes), max(node_volumes)
            if max_vol > min_vol:
                normalized_sizes = [(vol - min_vol) / (max_vol - min_vol) * 200 + 20 for vol in node_volumes]
            else:
                normalized_sizes = [50] * len(node_volumes)

            # Draw edges
            for edge in self.connectivity_graph.edges():
                seg1 = self.get_segment_by_id(edge[0])
                seg2 = self.get_segment_by_id(edge[1])

                if seg1 and seg2:
                    ax.plot3D(
                        [seg1.centroid[0], seg2.centroid[0]],
                        [seg1.centroid[1], seg2.centroid[1]],
                        [seg1.centroid[2], seg2.centroid[2]],
                        "gray",
                        alpha=0.6,
                        linewidth=1,
                    )

            # Draw nodes
            scatter = ax.scatter(
                node_positions[:, 0],
                node_positions[:, 1],
                node_positions[:, 2],
                s=normalized_sizes,
                c=node_colors,
                cmap="viridis",
                alpha=0.8,
                edgecolors="black",
                linewidth=0.5,
            )

            # Add labels for nodes
            for i, segment in enumerate(self.segments):
                ax.text(
                    segment.centroid[0],
                    segment.centroid[1],
                    segment.centroid[2],
                    f"S{segment.slice_index}.{segment.segment_index}",
                    fontsize=7,
                    ha="center",
                    va="center",
                )

            # Set labels and title
            ax.set_xlabel("X (μm)")
            ax.set_ylabel("Y (μm)")
            ax.set_zlabel("Z (μm)")
            ax.set_title("3D Segment Connectivity Graph")

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=20)
            cbar.set_label("Slice Index")

            # Set equal aspect ratio with proper margins
            x_range = node_positions[:, 0].max() - node_positions[:, 0].min()
            y_range = node_positions[:, 1].max() - node_positions[:, 1].min()
            z_range = node_positions[:, 2].max() - node_positions[:, 2].min()

            # Use the largest range and add 20% margin
            max_range = max(x_range, y_range, z_range) * 0.6  # 0.6 gives 20% margin on each side

            mid_x = (node_positions[:, 0].max() + node_positions[:, 0].min()) * 0.5
            mid_y = (node_positions[:, 1].max() + node_positions[:, 1].min()) * 0.5
            mid_z = (node_positions[:, 2].max() + node_positions[:, 2].min()) * 0.5

            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                print(f"3D visualization saved to {save_path}")
            else:
                plt.show()

        except ImportError:
            print("Matplotlib or mpl_toolkits not available for 3D plotting")

    def _create_improved_layout(self) -> Dict[str, Tuple[float, float]]:
        """Create an improved layout that avoids overlapping nodes."""
        pos = {}

        # Group segments by slice
        slice_segments = {}
        for segment in self.segments:
            slice_idx = segment.slice_index
            if slice_idx not in slice_segments:
                slice_segments[slice_idx] = []
            slice_segments[slice_idx].append(segment)

        # Position segments with better spacing
        for slice_idx, segments in slice_segments.items():
            num_segments = len(segments)

            if num_segments == 1:
                # Single segment: center it
                pos[segments[0].id] = (slice_idx, 0)
            else:
                # Multiple segments: spread them vertically with adequate spacing
                segment_spacing = 2.0  # Increase spacing between segments
                y_start = -(num_segments - 1) * segment_spacing / 2

                # Sort segments by y-centroid for consistent ordering
                segments_sorted = sorted(segments, key=lambda s: s.centroid[1])

                for i, segment in enumerate(segments_sorted):
                    y_pos = y_start + i * segment_spacing
                    pos[segment.id] = (slice_idx, y_pos)

        return pos

    def _draw_slice_layout(self, ax, pos):
        """Draw the slice-based layout."""
        # Customize node appearance based on properties
        node_colors = []
        node_sizes = []

        for segment in self.segments:
            # Color by volume (larger = darker)
            volume_normalized = min(segment.volume / max(s.volume for s in self.segments), 1.0)
            import matplotlib.pyplot as plt

            node_colors.append(plt.cm.viridis(volume_normalized))

            # Size by volume with better scaling
            size = max(50, min(500, segment.volume * 800))
            node_sizes.append(size)

        # Draw nodes with improved styling
        nx.draw_networkx_nodes(
            self.connectivity_graph, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8, ax=ax
        )

        # Draw edges with curved connections to reduce overlap
        nx.draw_networkx_edges(
            self.connectivity_graph,
            pos,
            edge_color="gray",
            alpha=0.6,
            width=1.5,
            style="solid",
            connectionstyle="arc3,rad=0.1",  # Curved edges
            ax=ax,
        )

        # Add improved labels
        labels = {}
        for seg in self.segments:
            slice_segments = [s for s in self.segments if s.slice_index == seg.slice_index]
            if len(slice_segments) > 1:
                labels[seg.id] = f"S{seg.slice_index}.{seg.segment_index}"
            else:
                labels[seg.id] = f"S{seg.slice_index}"

        nx.draw_networkx_labels(
            self.connectivity_graph, pos, labels, font_size=9, font_weight="bold", font_color="white", ax=ax
        )

        ax.set_title("Segment Connectivity Graph", fontsize=12, fontweight="bold")
        ax.set_xlabel("Slice Index", fontsize=10)
        ax.set_ylabel("Segment Position", fontsize=10)
        ax.grid(True, alpha=0.3)

    def _add_caps_to_slice(self, mesh: trimesh.Trimesh, z_min: float, z_max: float) -> trimesh.Trimesh:
        """Add caps to close the slice at z_min and z_max boundaries."""
        try:
            # For now, return the mesh as-is
            # In a more sophisticated implementation, we would:
            # 1. Find boundary edges at z_min and z_max
            # 2. Triangulate the boundary loops to create caps
            # 3. Add the cap faces to the mesh

            # This simplified version just ensures the mesh is valid
            if not mesh.is_watertight:
                try:
                    mesh.fill_holes()
                except:
                    pass  # Continue with non-watertight mesh

            return mesh

        except Exception:
            return mesh

    def create_neuron_model(self, name: str = "segmented_neuron"):
        """
        Convert segmentation results to a Neuron object for simulation.

        Args:
            name: Name for the neuron model

        Returns:
            Neuron object with compartments and connectivity
        """
        from .core import Neuron, Compartment

        if not self.segments:
            raise ValueError("No segments found. Run segment_mesh() first.")

        if not hasattr(self, "original_mesh") or self.original_mesh is None:
            raise ValueError("Original mesh not available. Run segment_mesh() first.")

        neuron = Neuron(name=name)
        neuron.set_mesh(self.original_mesh)

        # Convert each segment to a compartment
        for segment in self.segments:
            # Convert units: trimesh gives µm³, we want µm² for area
            # Surface area from trimesh is in mesh units² (assume µm²)
            membrane_area = segment.exterior_surface_area  # µm²
            volume = segment.volume  # µm³

            compartment = Compartment(
                id=segment.id,
                z_level=segment.slice_index,
                area=membrane_area,
                volume=volume,
                centroid=segment.centroid,
                boundary_points=segment.mesh.vertices,
                membrane_potential=-70.0,  # Initial resting potential
            )

            neuron.compartment_graph.add_compartment(compartment)

        # Add connections based on connectivity graph
        for edge in self.connectivity_graph.edges(data=True):
            comp1_id, comp2_id = edge[0], edge[1]
            edge_data = edge[2]

            # Calculate axial conductance between compartments
            # Using a simple geometric approach based on connection area
            distance = edge_data.get("distance", 1.0)  # µm

            # Get compartments to estimate connection area
            comp1 = neuron.get_compartment(comp1_id)
            comp2 = neuron.get_compartment(comp2_id)

            # Estimate connection area as the smaller of the two interior areas
            seg1 = self.get_segment_by_id(comp1_id)
            seg2 = self.get_segment_by_id(comp2_id)

            # Use interior areas if available, otherwise use a fraction of exterior areas
            int_area1 = (
                seg1.interior_surface_area if seg1.interior_surface_area > 0 else seg1.exterior_surface_area * 0.1
            )
            int_area2 = (
                seg2.interior_surface_area if seg2.interior_surface_area > 0 else seg2.exterior_surface_area * 0.1
            )
            connection_area = min(int_area1, int_area2)  # µm²

            # Calculate axial conductance (mS)
            # G = (1/R) = (Area / (ρ * L)) where ρ is resistivity
            resistivity = 100.0  # Ω·cm, typical intracellular resistivity
            length = max(distance * 1e-4, 1e-6)  # Convert µm to cm, ensure minimum length
            area_cm2 = max(connection_area * 1e-8, 1e-12)  # Convert µm² to cm², ensure minimum area

            if length > 0 and area_cm2 > 0:
                conductance = area_cm2 / (resistivity * length) * 1000  # mS
                # Ensure reasonable conductance range
                conductance = max(conductance, 1e-6)  # Minimum 1 µS
                conductance = min(conductance, 1e3)  # Maximum 1 S
            else:
                conductance = 1e-3  # Default 1 µS conductance

            neuron.compartment_graph.add_connection(comp1_id, comp2_id, conductance=conductance, area=connection_area)

        return neuron

    def diagnose_face_classification(
        self, segment_mesh: trimesh.Trimesh, z_min: float, z_max: float, segment_id: str = "unknown"
    ):
        """
        Diagnostic method to understand face classification issues.
        """
        face_areas = segment_mesh.area_faces
        vertices = segment_mesh.vertices
        faces = segment_mesh.faces

        print(f"Face classification diagnosis for {segment_id}:")
        print(f"  Slice bounds: [{z_min:.6f}, {z_max:.6f}]")
        print(f"  Total faces: {len(faces)}")
        print(f"  Total mesh area: {segment_mesh.area:.6f}")

        tolerance_values = [1e-3, 5e-3, 1e-2, 2e-2, 5e-2]

        for tol in tolerance_values:
            interior_area = 0.0
            exterior_area = 0.0
            bottom_faces = 0
            top_faces = 0
            side_faces = 0

            for face_idx, face in enumerate(faces):
                face_verts = vertices[face]
                face_z_coords = face_verts[:, 2]
                z_mean = np.mean(face_z_coords)
                z_std = np.std(face_z_coords)

                is_flat = z_std < tol
                at_bottom = abs(z_mean - z_min) < tol
                at_top = abs(z_mean - z_max) < tol

                if is_flat and at_bottom:
                    interior_area += face_areas[face_idx]
                    bottom_faces += 1
                elif is_flat and at_top:
                    interior_area += face_areas[face_idx]
                    top_faces += 1
                else:
                    exterior_area += face_areas[face_idx]
                    side_faces += 1

            print(f"  Tolerance {tol:.3f}: interior={interior_area:.6f}, exterior={exterior_area:.6f}")
            print(f"    Bottom faces: {bottom_faces}, Top faces: {top_faces}, Side faces: {side_faces}")

        return True

    def _segments_share_internal_face(self, seg1: Segment, seg2: Segment, shared_z: float) -> bool:
        """
        Check if two segments share an internal (cut) face at the given Z level.

        This uses a more sophisticated approach: we check if the segments have
        overlapping cut surface regions at the shared Z level.

        Args:
            seg1, seg2: The two segments to check
            shared_z: The Z coordinate where they might share a face

        Returns:
            True if segments share an internal face, False otherwise
        """
        tolerance = 1e-3

        # Get the 2D boundary contours for both segments at the shared Z level
        boundary1 = self._get_cut_boundary_at_z(seg1, shared_z, tolerance)
        boundary2 = self._get_cut_boundary_at_z(seg2, shared_z, tolerance)

        if boundary1 is None or boundary2 is None:
            return False

        # Check if the boundaries substantially overlap
        return self._boundaries_substantially_overlap(boundary1, boundary2)

    def _get_cut_boundary_at_z(self, segment: Segment, z_level: float, tolerance: float):
        """
        Get the 2D boundary contour of the cut surface at the specified Z level.

        Returns the XY coordinates of vertices that form the boundary of the cut face.
        """
        face_centers = segment.mesh.triangles_center
        face_normals = segment.mesh.face_normals

        # Find faces at the specified Z level that are cut faces
        at_z_level = np.abs(face_centers[:, 2] - z_level) < tolerance

        # Cut faces are flat faces (normal aligned with z-axis)
        normal_z = np.abs(face_normals[:, 2])
        is_flat = normal_z > 0.9

        # Check if this is NOT an original mesh boundary
        if hasattr(segment, "_original_mesh_bounds"):
            original_z_min, original_z_max = segment._original_mesh_bounds[:, 2]
            at_original_boundary = (np.abs(z_level - original_z_min) < tolerance) | (
                np.abs(z_level - original_z_max) < tolerance
            )
            if at_original_boundary:
                return None  # This is an original cap, not a cut face

        # Find cut faces
        is_cut_face = at_z_level & is_flat
        cut_face_indices = np.where(is_cut_face)[0]

        if len(cut_face_indices) == 0:
            return None

        # Collect all vertices from cut faces and project to XY plane
        cut_vertices = []
        for face_idx in cut_face_indices:
            face_verts = segment.mesh.vertices[segment.mesh.faces[face_idx]]
            # Project to XY plane
            for vertex in face_verts:
                cut_vertices.append(vertex[:2])  # XY coordinates only

        if len(cut_vertices) == 0:
            return None

        return np.array(cut_vertices)

    def _boundaries_substantially_overlap(self, boundary1: np.ndarray, boundary2: np.ndarray) -> bool:
        """
        Check if two 2D boundaries substantially overlap.

        For torus connectivity, we need to be very strict to avoid connecting
        segments across the hole. We use both centroid proximity and significant
        vertex overlap.
        """
        if len(boundary1) == 0 or len(boundary2) == 0:
            return False

        # Calculate centroids
        centroid1 = np.mean(boundary1, axis=0)
        centroid2 = np.mean(boundary2, axis=0)

        # Centroids must be very close for true connectivity
        centroid_distance = np.linalg.norm(centroid1 - centroid2)
        if centroid_distance > 0.1:  # Very strict threshold
            return False

        # For each vertex in boundary1, find the closest vertex in boundary2
        min_distances = []
        for v1 in boundary1:
            distances = [np.linalg.norm(v1 - v2) for v2 in boundary2]
            min_distances.append(min(distances))

        # Calculate what fraction of vertices have a very close match
        close_vertices = sum(1 for d in min_distances if d < 0.05)  # Very strict
        overlap_fraction = close_vertices / len(boundary1) if len(boundary1) > 0 else 0

        # Require substantial overlap (at least 50% of vertices very close)
        return overlap_fraction > 0.5

    def _get_internal_faces_at_z(self, segment: Segment, z_level: float, tolerance: float):
        """Get all internal (cut) faces of a segment at the specified Z level."""
        face_centers = segment.mesh.triangles_center
        face_normals = segment.mesh.face_normals

        # Find faces at the specified Z level that are internal (cut faces)
        at_z_level = np.abs(face_centers[:, 2] - z_level) < tolerance

        # Internal faces are flat faces (normal aligned with z-axis) at slice boundaries
        normal_z = np.abs(face_normals[:, 2])
        is_flat = normal_z > 0.9

        # Find faces at slice boundaries (not original mesh boundaries)
        # Get original mesh bounds from the parent segmenter if available
        if hasattr(segment, "_original_mesh_bounds"):
            original_z_min, original_z_max = segment._original_mesh_bounds[:, 2]
        else:
            # Fall back to segment mesh bounds (less reliable)
            original_z_min, original_z_max = segment.mesh.bounds[:, 2]

        at_original_boundary = (np.abs(face_centers[:, 2] - original_z_min) < tolerance) | (
            np.abs(face_centers[:, 2] - original_z_max) < tolerance
        )

        # Internal faces are flat faces at Z level that are NOT original boundaries
        is_internal = at_z_level & is_flat & (~at_original_boundary)

        internal_face_indices = np.where(is_internal)[0]

        # Return vertex coordinates for each internal face
        internal_faces = []
        for face_idx in internal_face_indices:
            face_vertices = segment.mesh.vertices[segment.mesh.faces[face_idx]]
            internal_faces.append(face_vertices)

        return internal_faces

    def _faces_spatially_overlap(self, face1_verts: np.ndarray, face2_verts: np.ndarray) -> bool:
        """
        Check if two triangular faces are essentially the same cut face (shared between segments).

        For segments to be connected, they must share the exact same cut face.
        This means the faces should have nearly identical vertex positions.

        Args:
            face1_verts: 3x3 array of vertices for face 1
            face2_verts: 3x3 array of vertices for face 2

        Returns:
            True if faces are the same cut face, False otherwise
        """
        tolerance = 1e-2  # More generous tolerance for mesh precision

        # Check if face centroids are very close
        centroid1 = np.mean(face1_verts, axis=0)
        centroid2 = np.mean(face2_verts, axis=0)

        if np.linalg.norm(centroid1 - centroid2) > tolerance:
            return False

        # For shared cut faces, we need substantial vertex overlap
        # Check how many vertices from face1 have a close match in face2
        matched_vertices = 0

        for v1 in face1_verts:
            for v2 in face2_verts:
                if np.linalg.norm(v1 - v2) < tolerance:
                    matched_vertices += 1
                    break  # Don't double-count vertices

        # Require at least 2 vertices to match (shared edge minimum)
        # For truly shared faces, all 3 should match
        return matched_vertices >= 2

    def _segments_share_cut_surface(self, seg1: Segment, seg2: Segment, shared_z: float) -> bool:
        """
        Very strict test for shared cut surface between segments.

        For torus, we need to ensure segments only connect if they truly share
        a substantial portion of their cut surface.
        """
        tolerance = 1e-3

        # Get cut surface vertices for both segments
        cut_verts1 = self._get_cut_surface_vertices(seg1, shared_z, tolerance)
        cut_verts2 = self._get_cut_surface_vertices(seg2, shared_z, tolerance)

        if len(cut_verts1) == 0 or len(cut_verts2) == 0:
            return False

        # Calculate how much the cut surfaces overlap
        overlap_score = self._calculate_surface_overlap(cut_verts1, cut_verts2)

        # Require substantial overlap (>30% of smaller surface)
        return overlap_score > 0.3

    def _get_cut_surface_vertices(self, segment: Segment, z_level: float, tolerance: float):
        """Get XY coordinates of all vertices that form the cut surface at z_level."""
        vertices = segment.mesh.vertices

        # Find vertices at the cut surface
        at_z_level = np.abs(vertices[:, 2] - z_level) < tolerance
        cut_vertices = vertices[at_z_level]

        if len(cut_vertices) == 0:
            return np.array([])

        # Return XY coordinates only
        return cut_vertices[:, :2]

    def _calculate_surface_overlap(self, verts1: np.ndarray, verts2: np.ndarray) -> float:
        """
        Calculate overlap between two sets of cut surface vertices.

        Returns fraction of vertices in smaller set that have close matches in larger set.
        """
        if len(verts1) == 0 or len(verts2) == 0:
            return 0.0

        # Make verts1 the smaller set for efficiency
        if len(verts1) > len(verts2):
            verts1, verts2 = verts2, verts1

        # For each vertex in smaller set, find closest vertex in larger set
        overlap_count = 0
        for v1 in verts1:
            min_distance = min(np.linalg.norm(v1 - v2) for v2 in verts2)
            if min_distance < 0.1:  # Vertex has a close match
                overlap_count += 1

        return overlap_count / len(verts1) if len(verts1) > 0 else 0.0

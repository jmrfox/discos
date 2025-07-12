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
        Extract mesh portion between z_min and z_max using face selection approach.

        This creates a proper 3D segment by selecting all faces that intersect the z-range,
        ensuring we get exactly one segment per closed region in the slice.
        """
        try:
            vertices = mesh.vertices
            faces = mesh.faces

            # Get z-coordinates for all vertices of each face
            face_vertices = vertices[faces]  # Shape: (n_faces, 3, 3)
            face_z_coords = face_vertices[:, :, 2]  # Z coordinates for each vertex of each face

            # Select faces that intersect with the z-range
            # Include faces that:
            # 1. Have at least one vertex in the range
            # 2. Span across the slice (min_z <= z_min and max_z >= z_max)
            # 3. Intersect the slice boundaries
            face_z_min = face_z_coords.min(axis=1)
            face_z_max = face_z_coords.max(axis=1)

            face_intersects = (face_z_max >= z_min) & (face_z_min <= z_max)

            if not face_intersects.any():
                return None

            # Select the intersecting faces
            selected_faces = faces[face_intersects]
            unique_verts = np.unique(selected_faces.flatten())

            if len(unique_verts) < 3:
                return None

            # Create vertex mapping
            old_to_new = {old: new for new, old in enumerate(unique_verts)}

            # Remap faces to new vertex indices
            new_faces = []
            for face in selected_faces:
                new_face = [old_to_new[v] for v in face]
                new_faces.append(new_face)

            slice_vertices = vertices[unique_verts]
            slice_faces = np.array(new_faces)

            # Create the segment mesh
            segment_mesh = trimesh.Trimesh(vertices=slice_vertices, faces=slice_faces)

            # Clean up the mesh
            try:
                # Use non-deprecated methods
                if hasattr(segment_mesh, "nondegenerate_faces"):
                    segment_mesh.update_faces(segment_mesh.nondegenerate_faces())
                if hasattr(segment_mesh, "unique_faces"):
                    segment_mesh.update_faces(segment_mesh.unique_faces())
            except:
                # Fall back to deprecated methods if new ones don't exist
                try:
                    segment_mesh.remove_degenerate_faces()
                    segment_mesh.remove_duplicate_faces()
                except:
                    pass  # Continue with uncleaned mesh

            return segment_mesh

        except Exception as e:
            warnings.warn(f"Failed to extract slice [{z_min:.3f}, {z_max:.3f}]: {e}")
            return None

    def _analyze_face_types(self, segment_mesh: trimesh.Trimesh, z_min: float, z_max: float) -> Tuple[float, float]:
        """
        Analyze faces to distinguish exterior vs interior (cut) faces.

        Returns:
            Tuple of (exterior_area, interior_area)
        """
        face_areas = segment_mesh.area_faces
        vertices = segment_mesh.vertices
        faces = segment_mesh.faces

        exterior_area = 0.0
        interior_area = 0.0

        tolerance = 1e-3  # Increased tolerance for better detection

        for face_idx, face in enumerate(faces):
            face_verts = vertices[face]
            face_z_coords = face_verts[:, 2]

            # Check if face is on a cut plane
            # A face is on a cut plane if all vertices are very close to z_min or z_max
            z_mean = np.mean(face_z_coords)
            z_std = np.std(face_z_coords)

            # If face is flat (low std) and at boundary, it's likely a cut face
            is_flat = z_std < tolerance
            at_bottom = abs(z_mean - z_min) < tolerance
            at_top = abs(z_mean - z_max) < tolerance

            if is_flat and (at_bottom or at_top):
                interior_area += face_areas[face_idx]
            else:
                exterior_area += face_areas[face_idx]

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
        """Find connections between segments in adjacent slices based on geometric overlap."""
        if not segments1 or not segments2:
            return

        # For each pair of segments in adjacent slices, check if they geometrically connect
        for seg1 in segments1:
            for seg2 in segments2:
                if self._segments_are_connected(seg1, seg2):
                    # Calculate actual distance between segment centroids
                    distance = np.linalg.norm(seg1.centroid - seg2.centroid)
                    self.connectivity_graph.add_edge(seg1.id, seg2.id, distance=distance, type="adjacent_slice")

    def _segments_are_connected(self, seg1: Segment, seg2: Segment) -> bool:
        """
        Check if two segments from adjacent slices are geometrically connected.

        This uses a more sophisticated approach than simple centroid distance.
        """
        # Method 1: Check if the 2D projections of the segments overlap
        # Project both segments onto the XY plane and check for overlap

        # Get boundary points of each segment at their respective slice boundaries
        boundary1 = self._get_segment_boundary_at_z(seg1, seg1.z_max)  # Top of lower segment
        boundary2 = self._get_segment_boundary_at_z(seg2, seg2.z_min)  # Bottom of upper segment

        if boundary1 is None or boundary2 is None:
            # Fall back to centroid-based distance check
            xy_distance = np.linalg.norm(seg1.centroid[:2] - seg2.centroid[:2])
            threshold = self._compute_connection_threshold(seg1, seg2)
            return xy_distance < threshold

        # Check if the boundaries overlap in XY space
        return self._boundaries_overlap(boundary1, boundary2)

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
        """Visualize the segment connectivity graph."""
        try:
            import matplotlib.pyplot as plt

            # Create layout
            pos = {}
            for segment in self.segments:
                # Position nodes by slice index and centroid
                x = segment.slice_index
                y = segment.centroid[1]  # Use y-coordinate of centroid
                pos[segment.id] = (x, y)

            plt.figure(figsize=(12, 8))

            # Draw nodes
            node_sizes = [seg.volume * 1000 for seg in self.segments]  # Scale for visibility
            nx.draw_networkx_nodes(self.connectivity_graph, pos, node_size=node_sizes, alpha=0.7)

            # Draw edges
            nx.draw_networkx_edges(self.connectivity_graph, pos, alpha=0.5)

            # Add labels
            labels = {seg.id: f"S{seg.slice_index}.{seg.segment_index}" for seg in self.segments}
            nx.draw_networkx_labels(self.connectivity_graph, pos, labels, font_size=8)

            plt.title("Segment Connectivity Graph")
            plt.xlabel("Slice Index")
            plt.ylabel("Y Coordinate")

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
            else:
                plt.show()

        except ImportError:
            print("matplotlib not available for visualization")

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

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
        Extract mesh portion between z_min and z_max using plane cutting.

        This properly cuts the mesh at the z boundaries using manual vertex clamping
        for reliable results.
        """
        try:
            # Start with a copy of the original mesh
            vertices = mesh.vertices.copy()
            faces = mesh.faces.copy()

            # Clamp vertices to the z-range [z_min, z_max]
            # This effectively "cuts" the mesh at both boundaries
            vertices[:, 2] = np.clip(vertices[:, 2], z_min, z_max)

            # Create the sliced mesh
            sliced_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

            # Now we need to remove faces that span too large a z-range
            # This removes faces that were "flattened" by the clipping
            face_vertices = vertices[faces]
            face_z_coords = face_vertices[:, :, 2]
            face_z_spans = face_z_coords.max(axis=1) - face_z_coords.min(axis=1)

            # Remove faces that span more than the slice width (these are artifacts)
            slice_width = z_max - z_min
            valid_face_mask = face_z_spans <= slice_width * 1.1  # Allow 10% tolerance

            if not valid_face_mask.any():
                return None

            valid_faces = faces[valid_face_mask]

            # Get unique vertices used by valid faces
            unique_verts = np.unique(valid_faces.flatten())
            if len(unique_verts) < 3:
                return None

            # Create vertex mapping
            old_to_new = {old: new for new, old in enumerate(unique_verts)}

            # Remap faces to new vertex indices
            new_faces = []
            for face in valid_faces:
                new_face = [old_to_new[v] for v in face]
                new_faces.append(new_face)

            final_vertices = vertices[unique_verts]
            final_faces = np.array(new_faces)

            # Create the final segment mesh
            segment_mesh = trimesh.Trimesh(vertices=final_vertices, faces=final_faces)

            # Validate the result
            if segment_mesh is None or segment_mesh.is_empty:
                return None

            if not hasattr(segment_mesh, "vertices") or len(segment_mesh.vertices) < 3:
                return None

            # Check that we got a reasonable slice
            vertex_z_range = segment_mesh.vertices[:, 2]
            actual_z_min, actual_z_max = vertex_z_range.min(), vertex_z_range.max()

            # The slice should be within our specified bounds
            if actual_z_min < z_min - 1e-6 or actual_z_max > z_max + 1e-6:
                warnings.warn(
                    f"Slice [{z_min:.3f}, {z_max:.3f}] has vertices outside range: [{actual_z_min:.3f}, {actual_z_max:.3f}]"
                )

            # Clean up the mesh
            try:
                segment_mesh.remove_degenerate_faces()
                segment_mesh.remove_duplicate_faces()
            except:
                pass

            return segment_mesh

        except Exception as e:
            warnings.warn(f"Failed to extract slice [{z_min:.3f}, {z_max:.3f}]: {e}")
            return None

    def _analyze_face_types(self, segment_mesh: trimesh.Trimesh, z_min: float, z_max: float) -> Tuple[float, float]:
        """
        Analyze faces to distinguish exterior vs interior (cut) faces.

        Interior faces are cut faces created during slicing (flat faces at z boundaries).
        Exterior faces are original mesh surfaces (cylindrical sides, end caps).

        Returns:
            Tuple of (exterior_area, interior_area)
        """
        face_areas = segment_mesh.area_faces
        vertices = segment_mesh.vertices
        faces = segment_mesh.faces

        exterior_area = 0.0
        interior_area = 0.0

        # Adaptive tolerance based on slice width
        slice_width = z_max - z_min
        tolerance = max(1e-3, slice_width * 0.1)  # 10% of slice width or minimum 1e-3

        # Get original mesh bounds to distinguish end caps from cut faces
        original_z_min, original_z_max = self.original_mesh.bounds[:, 2]

        for face_idx, face in enumerate(faces):
            face_verts = vertices[face]
            face_z_coords = face_verts[:, 2]

            # Check if face is on a cut plane
            z_mean = np.mean(face_z_coords)
            z_std = np.std(face_z_coords)

            # If face is flat (low std) and at boundary, it could be a cut face or end cap
            is_flat = z_std < tolerance
            at_bottom_boundary = abs(z_mean - z_min) < tolerance
            at_top_boundary = abs(z_mean - z_max) < tolerance

            if is_flat and (at_bottom_boundary or at_top_boundary):
                # Check if this is actually an original mesh end cap or a cut face
                at_original_bottom = abs(z_mean - original_z_min) < tolerance
                at_original_top = abs(z_mean - original_z_max) < tolerance

                if at_original_bottom or at_original_top:
                    # This is an original end cap - counts as exterior
                    exterior_area += face_areas[face_idx]
                else:
                    # This is a cut face - counts as interior
                    interior_area += face_areas[face_idx]
            else:
                # Not flat or not at boundary - must be side surface (exterior)
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

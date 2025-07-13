"""
Systematic mesh segmentation implementation using trimesh.

This module implements a robust mesh segmentation algorithm that:
0. Validates single-hull input meshes
1. Creates cross-sectional cuts along z-axis using trimesh.intersections
2. Identifies segments within each slice
3. Builds connectivity graph based on shared internal faces
4. Validates volume and surface area conservation
"""

import numpy as np
import trimesh
import networkx as nx
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import warnings


@dataclass
class CrossSection:
    """Represents a planar cross-section through the mesh."""

    z_position: float
    intersection_lines: np.ndarray  # 3D line segments from mesh intersection
    intersection_2d: Optional[object] = None  # 2D planar projection if available
    area: float = 0.0


@dataclass
class Segment:
    """Represents a single closed volume segment within a slice."""

    id: str
    slice_index: int
    segment_index: int  # Index within the slice
    mesh: trimesh.Trimesh
    volume: float
    external_surface_area: float  # Original mesh faces
    internal_surface_area: float  # Cut faces from slicing
    centroid: np.ndarray
    z_min: float
    z_max: float


class MeshSegmenter:
    """Systematic mesh segmentation using cross-sectional cuts."""

    def __init__(self):
        self.original_mesh = None
        self.slice_height = None
        self.cross_sections: List[CrossSection] = []
        self.slices: List[Dict] = []  # List of slice boundary info
        self.segments: List[Segment] = []
        self.connectivity_graph = None

    def segment_mesh(self, mesh: trimesh.Trimesh, slice_height: float, min_volume: float = 1e-6) -> List[Segment]:
        """
        Segment mesh into volumetric segments.

        Args:
            mesh: Input mesh (must be single closed volume)
            slice_height: Height of each slice
            min_volume: Minimum segment volume threshold

        Returns:
            List of segments
        """
        self.original_mesh = mesh.copy()
        self.slice_height = slice_height
        self.segments = []
        self.cross_sections = []

        # Step 0: Validate input mesh is single hull
        self._validate_single_hull_mesh(mesh)

        # Step 1: Compute cross-sectional cuts using trimesh.intersections
        self._compute_cross_sections(mesh)

        # Step 2: Extract slices and identify segments
        self._extract_slices_and_segments(mesh, min_volume)

        # Step 3: Build connectivity graph
        self._build_connectivity_graph()

        # Step 4: Validate conservation
        self._validate_conservation()

        return self.segments

    def _validate_single_hull_mesh(self, mesh: trimesh.Trimesh):
        """Step 0: Validate that mesh is a single closed volume."""
        if not mesh.is_watertight:
            raise ValueError("Input mesh must be watertight (closed volume)")

        # Check for multiple disconnected components
        components = mesh.split(only_watertight=False)
        if len(components) > 1:
            raise ValueError(
                f"Input mesh has {len(components)} disconnected components. " "Mesh must be a single connected volume."
            )

        # Store original bounds for later face classification
        self.original_bounds = mesh.bounds.copy()

        # Annotate all original faces as "external"
        if not hasattr(mesh, "face_attributes"):
            mesh.face_attributes = {}
        mesh.face_attributes["face_type"] = ["external"] * len(mesh.faces)

        print(f"✅ Validated single-hull mesh: {len(mesh.faces)} external faces, " f"volume={mesh.volume:.3f}")

    def _compute_cross_sections(self, mesh: trimesh.Trimesh):
        """Step 1: Compute cross-sectional cuts using trimesh.intersections."""
        z_min, z_max = mesh.bounds[:, 2]

        # Create cutting planes every slice_height units
        z_positions = np.arange(z_min + self.slice_height, z_max, self.slice_height)

        print(f"Computing {len(z_positions)} cross-sections from z={z_min:.2f} to z={z_max:.2f}")

        for i, z_pos in enumerate(z_positions):
            # Create cutting plane (normal pointing up in z-direction)
            plane_origin = np.array([0, 0, z_pos])
            plane_normal = np.array([0, 0, 1])

            try:
                # Use trimesh.intersections to find intersection lines
                lines = trimesh.intersections.mesh_plane(mesh, plane_normal, plane_origin, return_faces=False)

                if lines is not None and len(lines) > 0:
                    # Create cross-section object
                    cs = CrossSection(z_position=z_pos, intersection_lines=lines)

                    # Try to create 2D projection for area calculation
                    try:
                        section_2d = mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)
                        if section_2d is not None:
                            cs.intersection_2d = section_2d
                            if hasattr(section_2d, "area"):
                                cs.area = section_2d.area
                    except:
                        pass  # 2D projection failed, continue with lines only

                    self.cross_sections.append(cs)
                    print(f"  Cross-section {i}: z={z_pos:.2f}, {len(lines)} line segments, area={cs.area:.3f}")
                else:
                    print(f"  Cross-section {i}: z={z_pos:.2f}, no intersection")

            except Exception as e:
                print(f"  Cross-section {i}: z={z_pos:.2f}, error: {e}")

        # Store slice boundaries based on cross-sections
        z_all = [z_min] + [cs.z_position for cs in self.cross_sections] + [z_max]
        self.slices = [{"z_min": z_all[i], "z_max": z_all[i + 1], "index": i} for i in range(len(z_all) - 1)]

        print(f"✅ Computed {len(self.cross_sections)} cross-sections, creating {len(self.slices)} slices")

    def _extract_slices_and_segments(self, mesh: trimesh.Trimesh, min_volume: float):
        """Step 2: Extract slices and identify segments within each."""
        for slice_info in self.slices:
            slice_idx = slice_info["index"]
            z_min = slice_info["z_min"]
            z_max = slice_info["z_max"]

            print(f"\nProcessing slice {slice_idx}: z=[{z_min:.2f}, {z_max:.2f}]")

            # Extract slice mesh using two cutting planes
            slice_mesh = self._extract_slice_mesh(mesh, z_min, z_max)

            if slice_mesh is None or slice_mesh.volume < min_volume:
                print(f"  Slice {slice_idx}: No valid mesh extracted (volume too small)")
                continue

            # Find connected components (closed volumes) within the slice
            components = slice_mesh.split(only_watertight=False)
            valid_components = [c for c in components if c.volume >= min_volume]

            if len(valid_components) == 0:
                raise ValueError(f"Slice {slice_idx} has zero segments - this is erroneous")

            print(f"  Slice {slice_idx}: {len(valid_components)} closed volumes (segments)")

            # Transfer face attributes to each component
            self._transfer_face_attributes(slice_mesh, valid_components)

            # Create segment objects for each closed volume
            for seg_idx, component in enumerate(valid_components):
                segment_id = f"seg_{slice_idx}_{seg_idx}"

                # Analyze face types (external vs internal)
                ext_area, int_area = self._analyze_face_types(component, z_min, z_max)

                segment = Segment(
                    id=segment_id,
                    slice_index=slice_idx,
                    segment_index=seg_idx,
                    mesh=component,
                    volume=component.volume,
                    external_surface_area=ext_area,
                    internal_surface_area=int_area,
                    centroid=component.centroid,
                    z_min=z_min,
                    z_max=z_max,
                )

                self.segments.append(segment)
                print(
                    f"    Segment {segment_id}: vol={segment.volume:.3f}, "
                    f"ext_area={ext_area:.3f}, int_area={int_area:.3f}"
                )

                # Validate that all segments have both external and internal areas
                if ext_area <= 0:
                    warnings.warn(f"Segment {segment_id} has zero external surface area")
                if int_area <= 0:
                    warnings.warn(f"Segment {segment_id} has zero internal surface area")

        print(f"✅ Extracted {len(self.segments)} total segments across {len(self.slices)} slices")

    def _extract_slice_mesh(self, mesh: trimesh.Trimesh, z_min: float, z_max: float) -> Optional[trimesh.Trimesh]:
        """Extract mesh slice between two z-planes by cutting and capping."""
        try:
            # Start with the original mesh
            working_mesh = mesh.copy()

            # Cut at the lower plane (z_min) - keep the upper part
            if z_min > mesh.bounds[0, 2]:  # Only cut if not at the bottom
                plane_origin = np.array([0, 0, z_min])
                plane_normal = np.array([0, 0, 1])  # Points up (keep upper part)
                working_mesh = working_mesh.slice_plane(plane_origin, plane_normal, cap=True)
                if working_mesh is None:
                    return None

            # Cut at the upper plane (z_max) - keep the lower part
            if z_max < mesh.bounds[1, 2]:  # Only cut if not at the top
                plane_origin = np.array([0, 0, z_max])
                plane_normal = np.array([0, 0, -1])  # Points down (keep lower part)
                working_mesh = working_mesh.slice_plane(plane_origin, plane_normal, cap=True)
                if working_mesh is None:
                    return None

            # Reconstruct face annotations after slicing
            self._reconstruct_face_annotations(working_mesh, z_min, z_max)

            # Ensure the result is watertight
            if not working_mesh.is_watertight:
                # Try to fix holes
                try:
                    working_mesh.fill_holes()
                except:
                    pass

            # Final check - must have reasonable volume
            if working_mesh.volume < 1e-9:
                return None

            return working_mesh

        except Exception as e:
            print(f"    Error extracting slice: {e}")
            return None

    def _analyze_face_types(self, segment_mesh: trimesh.Trimesh, z_min: float, z_max: float) -> Tuple[float, float]:
        """Analyze face types and compute surface areas using face annotations."""
        external_area = 0.0
        internal_area = 0.0

        # Get face areas
        face_areas = segment_mesh.area_faces

        # Use face annotations to classify faces
        if hasattr(segment_mesh, "face_attributes") and "face_type" in segment_mesh.face_attributes:
            face_types = segment_mesh.face_attributes["face_type"]

            for i, area in enumerate(face_areas):
                if i < len(face_types):
                    if face_types[i] == "external":
                        external_area += area
                    elif face_types[i] == "internal":
                        internal_area += area
        else:
            # Fallback to geometric classification if annotations missing
            print("⚠️ Face annotations missing, using geometric fallback")
            face_centers = segment_mesh.triangles_center
            face_normals = segment_mesh.face_normals
            z_tolerance = self.slice_height * 0.01

            for i, (center, area, normal) in enumerate(zip(face_centers, face_areas, face_normals)):
                z_pos = center[2]
                is_at_lower = abs(z_pos - z_min) < z_tolerance
                is_at_upper = abs(z_pos - z_max) < z_tolerance
                is_horizontal = abs(normal[2]) > 0.8

                if (is_at_lower or is_at_upper) and is_horizontal:
                    internal_area += area
                else:
                    external_area += area

        return external_area, internal_area

    def _build_connectivity_graph(self):
        """Step 3: Build connectivity graph based on shared internal faces."""
        self.connectivity_graph = nx.Graph()

        # Add all segments as nodes
        for segment in self.segments:
            self.connectivity_graph.add_node(segment.id, segment=segment)

        connections_found = 0

        # Find connections between segments in adjacent slices
        for i in range(len(self.slices) - 1):
            current_slice_segments = [s for s in self.segments if s.slice_index == i]
            next_slice_segments = [s for s in self.segments if s.slice_index == i + 1]

            # Check all pairs for shared internal faces
            for seg1 in current_slice_segments:
                for seg2 in next_slice_segments:
                    if self._segments_share_internal_face(seg1, seg2):
                        self.connectivity_graph.add_edge(seg1.id, seg2.id)
                        connections_found += 1
                        print(f"  Connected {seg1.id} ↔ {seg2.id}")

        print(f"✅ Built connectivity graph: {len(self.connectivity_graph.nodes)} nodes, " f"{connections_found} edges")

    def _segments_share_internal_face(self, seg1: Segment, seg2: Segment) -> bool:
        """Check if two segments share an internal face."""
        # They must be in adjacent slices
        if abs(seg1.slice_index - seg2.slice_index) != 1:
            return False

        # Get the boundary z-position between the slices
        if seg1.slice_index < seg2.slice_index:
            boundary_z = seg1.z_max  # Should equal seg2.z_min
        else:
            boundary_z = seg2.z_max  # Should equal seg1.z_min

        # Find faces at the boundary for both segments
        faces1 = self._get_boundary_faces(seg1, boundary_z)
        faces2 = self._get_boundary_faces(seg2, boundary_z)

        # Check for geometric overlap
        return self._faces_overlap(faces1, faces2)

    def _get_boundary_faces(self, segment: Segment, z_pos: float) -> List[np.ndarray]:
        """Get face triangles at a specific z-boundary."""
        boundary_faces = []
        z_tolerance = self.slice_height * 0.01

        face_centers = segment.mesh.triangles_center
        triangles = segment.mesh.triangles

        for i, center in enumerate(face_centers):
            if abs(center[2] - z_pos) < z_tolerance:
                # This face is at the boundary
                boundary_faces.append(triangles[i])

        return boundary_faces

    def _faces_overlap(self, faces1: List[np.ndarray], faces2: List[np.ndarray]) -> bool:
        """Check if faces from two segments overlap in XY plane."""
        if not faces1 or not faces2:
            return False

        # Check for overlap using triangle centroids in XY plane
        for face1 in faces1:
            centroid1 = np.mean(face1[:, :2], axis=0)  # XY only

            for face2 in faces2:
                centroid2 = np.mean(face2[:, :2], axis=0)  # XY only

                distance = np.linalg.norm(centroid1 - centroid2)
                if distance < 0.1:  # Threshold for overlap
                    return True

        return False

    def _validate_conservation(self):
        """Step 4: Validate volume and surface area conservation."""
        # Volume conservation
        total_segment_volume = sum(seg.volume for seg in self.segments)
        original_volume = self.original_mesh.volume
        volume_error = abs(total_segment_volume - original_volume) / original_volume * 100

        # Surface area conservation (external faces only)
        total_external_area = sum(seg.external_surface_area for seg in self.segments)
        original_area = self.original_mesh.area
        area_error = abs(total_external_area - original_area) / original_area * 100

        print(f"\n📊 CONSERVATION VALIDATION:")
        print(
            f"Volume: segments={total_segment_volume:.4f} vs original={original_volume:.4f} "
            f"(error: {volume_error:.2f}%)"
        )
        print(
            f"Surface area: segments={total_external_area:.4f} vs original={original_area:.4f} "
            f"(error: {area_error:.2f}%)"
        )

        # Check tolerances
        if volume_error > 5.0:
            warnings.warn(f"Volume conservation error too high: {volume_error:.2f}%")
        if area_error > 5.0:
            warnings.warn(f"Surface area conservation error too high: {area_error:.2f}%")

        if volume_error <= 5.0 and area_error <= 5.0:
            print("✅ Conservation validation passed")
        else:
            print("❌ Conservation validation failed")

    # Utility methods for analysis
    def get_segments_in_slice(self, slice_index: int) -> List[Segment]:
        """Get all segments in a specific slice."""
        return [seg for seg in self.segments if seg.slice_index == slice_index]

    def get_connected_segments(self, segment_id: str) -> List[str]:
        """Get segments connected to the given segment."""
        if self.connectivity_graph is None:
            return []
        return list(self.connectivity_graph.neighbors(segment_id))

    def get_connected_components(self) -> List[List[str]]:
        """Get connected components in the graph."""
        if self.connectivity_graph is None:
            return []
        return [list(component) for component in nx.connected_components(self.connectivity_graph)]

    def compute_segmentation_statistics(self) -> Dict:
        """Compute statistics about the segmentation."""
        volumes = [seg.volume for seg in self.segments]

        return {
            "num_segments": len(self.segments),
            "num_slices": len(self.slices),
            "volume_stats": {
                "total": sum(volumes),
                "mean": np.mean(volumes),
                "std": np.std(volumes),
                "min": min(volumes) if volumes else 0,
                "max": max(volumes) if volumes else 0,
            },
            "connectivity_stats": {
                "num_components": len(self.get_connected_components()) if self.connectivity_graph else 0,
                "num_edges": len(self.connectivity_graph.edges) if self.connectivity_graph else 0,
            },
        }

    def _reconstruct_face_annotations(self, mesh: trimesh.Trimesh, z_min: float, z_max: float):
        """Reconstruct face annotations after slicing by analyzing geometry."""
        face_centers = mesh.triangles_center
        face_normals = mesh.face_normals
        face_count = len(mesh.faces)

        # Initialize face_attributes
        if not hasattr(mesh, "face_attributes"):
            mesh.face_attributes = {}

        face_types = []
        z_tolerance = self.slice_height * 0.01

        # Get original mesh bounds to distinguish caps from cuts
        original_z_min = self.original_bounds[0, 2]  # Store original bounds
        original_z_max = self.original_bounds[1, 2]

        for i, (center, normal) in enumerate(zip(face_centers, face_normals)):
            z_pos = center[2]

            # Check if face is horizontal (cap or cut)
            is_horizontal = abs(normal[2]) > 0.8

            if is_horizontal:
                # Check if this is at slice boundary (internal cut) or original boundary (external cap)
                is_at_slice_lower = abs(z_pos - z_min) < z_tolerance
                is_at_slice_upper = abs(z_pos - z_max) < z_tolerance
                is_at_original_lower = abs(z_pos - original_z_min) < z_tolerance
                is_at_original_upper = abs(z_pos - original_z_max) < z_tolerance

                # Original caps are external, slice cuts are internal
                if is_at_original_lower or is_at_original_upper:
                    face_types.append("external")  # Original cap
                elif is_at_slice_lower or is_at_slice_upper:
                    face_types.append("internal")  # Cut face
                else:
                    face_types.append("external")  # Default for horizontal faces
            else:
                # Side faces are always external
                face_types.append("external")

        mesh.face_attributes["face_type"] = face_types

        # Debug output
        external_count = face_types.count("external")
        internal_count = face_types.count("internal")
        print(f"  🏷️  Annotated {external_count} external + {internal_count} internal faces")

    def _transfer_face_attributes(self, source_mesh: trimesh.Trimesh, components: list):
        """Transfer face attributes from source mesh to split components."""
        if not hasattr(source_mesh, "face_attributes") or "face_type" not in source_mesh.face_attributes:
            return

        # For a single connected component (typical case), the component should have
        # the same faces as the source mesh, just potentially reordered
        if len(components) == 1:
            component = components[0]
            if not hasattr(component, "face_attributes"):
                component.face_attributes = {}

            # If face counts match, directly copy the attributes
            if len(component.faces) == len(source_mesh.faces):
                component.face_attributes["face_type"] = source_mesh.face_attributes["face_type"].copy()
                print(f"    🔄 Transferred face attributes: {len(component.faces)} faces")
            else:
                print(f"    ⚠️ Face count mismatch: source={len(source_mesh.faces)}, component={len(component.faces)}")
        else:
            print(f"    ⚠️ Multiple components ({len(components)}), face transfer not implemented")

    def visualize_connectivity_graph(
        self, save_path: str = None, show_plot: bool = True, include_3d_view: bool = False, figsize: tuple = None
    ):
        """
        Visualize the connectivity graph of segments.

        Args:
            save_path: Path to save the plot (optional)
            show_plot: Whether to display the plot
            include_3d_view: Whether to include 3D view alongside network view (default: False)
            figsize: Figure size as (width, height). Auto-determined if None.

        Returns:
            matplotlib figure object or save path
        """
        if self.connectivity_graph is None:
            raise ValueError("No connectivity graph available. Run segment_mesh() first.")

        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
        except ImportError:
            print("Matplotlib not available for visualization")
            return None

        # Determine figure layout and size
        if include_3d_view:
            if figsize is None:
                figsize = (12, 8)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        else:
            if figsize is None:
                figsize = (8, 6)
            fig, ax_main = plt.subplots(1, 1, figsize=figsize)

        # Group segments by slice/z-level
        z_levels = {}
        for segment in self.segments:
            z_level = segment.slice_index
            if z_level not in z_levels:
                z_levels[z_level] = []
            z_levels[z_level].append(segment)

        # Position nodes by z-level and arrange within each level
        pos = {}
        colors = []
        node_sizes = []

        # Color map for different z-levels
        cmap = plt.cm.viridis
        max_z = max(z_levels.keys()) if z_levels else 0

        for z_level, segments in z_levels.items():
            y_pos = z_level * 2  # Vertical spacing between levels

            # Arrange segments horizontally within each level
            if len(segments) == 1:
                x_positions = [0]
            else:
                x_positions = [(i - (len(segments) - 1) / 2) * 1.5 for i in range(len(segments))]

            for i, segment in enumerate(segments):
                pos[segment.id] = (x_positions[i], y_pos)
                # Color by z-level
                colors.append(cmap(z_level / max_z if max_z > 0 else 0))
                # Size by volume
                node_sizes.append(max(100, min(1000, segment.volume * 1000)))

        # Helper function to apply aspect ratio constraints
        def apply_aspect_ratio_fix(ax, positions, min_aspect_ratio=0.5):
            if positions:
                x_coords = [p[0] for p in positions.values()]
                y_coords = [p[1] for p in positions.values()]
                x_range = max(x_coords) - min(x_coords) if len(set(x_coords)) > 1 else 2.0
                y_range = max(y_coords) - min(y_coords) if len(set(y_coords)) > 1 else 2.0

                current_aspect = x_range / y_range if y_range > 0 else 1.0

                if current_aspect < min_aspect_ratio:
                    # Expand x_range to meet minimum aspect ratio
                    target_x_range = y_range * min_aspect_ratio
                    x_center = (max(x_coords) + min(x_coords)) / 2
                    x_margin = (target_x_range - x_range) / 2
                    ax.set_xlim(min(x_coords) - x_margin, max(x_coords) + x_margin)
                elif current_aspect > (1 / min_aspect_ratio):
                    # Expand y_range to prevent overly wide plots
                    target_y_range = x_range * min_aspect_ratio
                    y_center = (max(y_coords) + min(y_coords)) / 2
                    y_margin = (target_y_range - y_range) / 2
                    ax.set_ylim(min(y_coords) - y_margin, max(y_coords) + y_margin)

        if include_3d_view:
            # Left plot: 3D-like representation showing z-levels
            ax1.set_title("Connectivity Graph (3D View)", fontsize=14, fontweight="bold")

            # Draw the graph
            nx.draw_networkx_nodes(
                self.connectivity_graph, pos, ax=ax1, node_color=colors, node_size=node_sizes, alpha=0.8
            )
            nx.draw_networkx_edges(self.connectivity_graph, pos, ax=ax1, edge_color="gray", width=2, alpha=0.6)

            nx.draw_networkx_labels(self.connectivity_graph, pos, ax=ax1, font_size=8, font_weight="bold")

            ax1.set_xlabel("Horizontal Position", fontsize=12)
            ax1.set_ylabel("Z-Level (Slice Index)", fontsize=12)
            ax1.grid(True, alpha=0.3)

            # Apply aspect ratio fix for 3D view
            apply_aspect_ratio_fix(ax1, pos)
            ax1.set_aspect("equal")

        # Network view (main plot or right plot)
        ax_network = ax2 if include_3d_view else ax_main
        ax_network.set_title(
            "Connectivity Graph" + (" (Network View)" if include_3d_view else ""), fontsize=14, fontweight="bold"
        )

        # Use the same positioning as the 3D view but with z-level for network visualization
        pos_network = {}
        for segment in self.segments:
            z_level = segment.slice_index
            # Find the segment's position in the original pos dict
            if segment.id in pos:
                x_pos, y_pos = pos[segment.id]
                # Use horizontal position directly and z-level for vertical
                pos_network[segment.id] = (x_pos, z_level)

        nx.draw_networkx_nodes(
            self.connectivity_graph, pos_network, ax=ax_network, node_color=colors, node_size=node_sizes, alpha=0.8
        )

        nx.draw_networkx_edges(
            self.connectivity_graph, pos_network, ax=ax_network, edge_color="gray", width=2, alpha=0.6
        )

        nx.draw_networkx_labels(self.connectivity_graph, pos_network, ax=ax_network, font_size=8, font_weight="bold")

        ax_network.set_xlabel("Horizontal Position", fontsize=12)
        ax_network.set_ylabel("Z-Level (Slice Index)", fontsize=12)
        ax_network.grid(True, alpha=0.3)

        # Apply aspect ratio fix for network view
        apply_aspect_ratio_fix(ax_network, pos_network)
        ax_network.set_aspect("equal")

        # Add statistics text
        stats_text = f"""Graph Statistics:
Segments: {len(self.segments)}
Connections: {len(self.connectivity_graph.edges)}
Z-levels: {len(z_levels)}
Total Volume: {sum(s.volume for s in self.segments):.3f}"""

        fig.text(
            0.02, 0.02, stats_text, fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8)
        )

        plt.tight_layout()

        # Save plot if path provided
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"📊 Connectivity graph saved to: {save_path}")

        if show_plot:
            plt.show()

        return save_path if save_path else fig

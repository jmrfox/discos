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
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
import warnings
import math
import json
from collections import deque
from scipy.optimize import minimize


# Utility functions for circle fitting and cross-section analysis
def fit_circle_to_points(
    points_2d: np.ndarray, method: str = "algebraic"
) -> Tuple[np.ndarray, float]:
    """Fit a circle to 2D points using various methods.

    Args:
        points_2d: Nx2 array of 2D points
        method: 'algebraic', 'geometric', or 'robust'

    Returns:
        Tuple of (center, radius)
    """
    if len(points_2d) < 3:
        raise ValueError("Need at least 3 points to fit a circle")

    if method == "algebraic":
        return _fit_circle_algebraic(points_2d)
    elif method == "geometric":
        return _fit_circle_geometric(points_2d)
    elif method == "robust":
        return _fit_circle_robust(points_2d)
    else:
        raise ValueError(f"Unknown circle fitting method: {method}")


def _fit_circle_algebraic(points: np.ndarray) -> Tuple[np.ndarray, float]:
    """Algebraic circle fitting (fastest, least accurate)."""
    x, y = points[:, 0], points[:, 1]

    # Set up the system Ax = b
    A = np.column_stack([2 * x, 2 * y, np.ones(len(x))])
    b = x**2 + y**2

    # Solve using least squares
    try:
        params = np.linalg.lstsq(A, b, rcond=None)[0]
        center = np.array([params[0], params[1]])
        radius = np.sqrt(params[2] + params[0] ** 2 + params[1] ** 2)
        return center, radius
    except np.linalg.LinAlgError:
        # Fallback to centroid and average distance
        center = np.mean(points, axis=0)
        radius = np.mean(np.linalg.norm(points - center, axis=1))
        return center, radius


def _fit_circle_geometric(points: np.ndarray) -> Tuple[np.ndarray, float]:
    """Geometric circle fitting (more accurate)."""

    def objective(params):
        center = params[:2]
        radius = params[2]
        distances = np.linalg.norm(points - center, axis=1)
        return np.sum((distances - radius) ** 2)

    # Initial guess from algebraic method
    center_init, radius_init = _fit_circle_algebraic(points)
    x0 = np.array([center_init[0], center_init[1], radius_init])

    # Optimize
    result = minimize(objective, x0, method="BFGS")

    if result.success:
        center = result.x[:2]
        radius = abs(result.x[2])
        return center, radius
    else:
        # Fallback to algebraic method
        return _fit_circle_algebraic(points)


def _fit_circle_robust(points: np.ndarray) -> Tuple[np.ndarray, float]:
    """Robust circle fitting using RANSAC-like approach."""
    best_center, best_radius = None, None
    best_inliers = 0

    n_iterations = min(100, len(points) * 2)
    inlier_threshold = 0.1  # Adjust based on your data scale

    for _ in range(n_iterations):
        # Sample 3 random points
        if len(points) >= 3:
            sample_idx = np.random.choice(len(points), 3, replace=False)
            sample_points = points[sample_idx]

            try:
                center, radius = _fit_circle_algebraic(sample_points)

                # Count inliers
                distances = np.linalg.norm(points - center, axis=1)
                inliers = np.sum(np.abs(distances - radius) < inlier_threshold)

                if inliers > best_inliers:
                    best_inliers = inliers
                    best_center = center
                    best_radius = radius
            except:
                continue

    if best_center is not None:
        return best_center, best_radius
    else:
        # Fallback to geometric method
        return _fit_circle_geometric(points)


def calculate_radius_from_area(area: float) -> float:
    """Calculate radius from area assuming circular cross-section."""
    return np.sqrt(area / np.pi)


def calculate_radius_from_boundary(
    boundary_points: np.ndarray, center: np.ndarray
) -> float:
    """Calculate average radius from center to boundary points."""
    distances = np.linalg.norm(boundary_points - center, axis=1)
    return np.mean(distances)


def detect_cross_section_overlap(
    cross_sections: List["CrossSection"], tolerance: float = 0.1
) -> List[Tuple[int, int]]:
    """Detect overlapping cross-sections within the same slice.

    Args:
        cross_sections: List of cross-sections to check
        tolerance: Spatial tolerance for overlap detection

    Returns:
        List of (i, j) indices of overlapping cross-sections
    """
    overlaps = []

    for i in range(len(cross_sections)):
        for j in range(i + 1, len(cross_sections)):
            cs1, cs2 = cross_sections[i], cross_sections[j]

            # Check if they're at the same z-level
            if abs(cs1.z_position - cs2.z_position) < 1e-6:
                # Check if centers are too close (overlap)
                if cs1.center is not None and cs2.center is not None:
                    center_distance = np.linalg.norm(cs1.center[:2] - cs2.center[:2])
                    min_separation = cs1.radius + cs2.radius + tolerance

                    if center_distance < min_separation:
                        overlaps.append((i, j))

    return overlaps


@dataclass
class CrossSection:
    """Represents a planar cross-section through the mesh."""

    z_position: float
    intersection_lines: np.ndarray  # 3D line segments from mesh intersection
    intersection_2d: Optional[object] = None  # 2D planar projection if available
    area: float = 0.0
    center: Optional[np.ndarray] = None  # Center of best-fit circle
    radius: float = 0.0  # Radius of approximating circle
    boundary_points: Optional[np.ndarray] = (
        None  # 2D boundary points for circle fitting
    )


@dataclass
class Point:
    """Represents a spatial point (center of cross-section) in the new graph architecture."""

    id: str
    z_position: float
    center: np.ndarray  # 3D center point (x, y, z)
    radius: float  # Radius of approximating circle
    cross_section: "CrossSection"  # Reference to the cross-section
    slice_index: int
    cross_section_index: int  # Index within the slice (for multiple cross-sections)


@dataclass
class Segment:
    """Represents a cylindrical segment connecting two points."""

    id: str
    point1_id: str
    point2_id: str
    length: float
    radius1: float  # Radius at point1
    radius2: float  # Radius at point2
    center_line: np.ndarray  # 3D line from point1 to point2
    volume: float  # Approximate cylinder volume
    surface_area: float  # Approximate cylinder surface area

@dataclass
class SWCData:
    """Represents SWC format data with cycle-breaking annotations."""

    entries: List[str]  # List of SWC entry strings
    metadata: Dict[str, Any]  # Metadata about the conversion
    non_tree_edges: List[Dict[str, Any]]  # Removed edges for cycle breaking
    root_segment: str  # Root segment ID
    scale_factor: float  # Scaling factor used

    def write_to_file(self, filename: str) -> None:
        """Write SWC data to file with annotations."""
        with open(filename, "w") as f:
            # Write header with metadata
            f.write("# SWC file generated from SegmentGraph for Arbor simulator\n")
            f.write("# Format: SampleID TypeID x y z radius ParentID\n")
            f.write("# TypeID: 5=segment (all segments use type 5)\n")
            f.write(f"# Total segments: {self.metadata.get('total_segments', 0)}\n")
            f.write(f"# Tree connections: {len(self.entries) - 1}\n")  # -1 for root
            # No soma segments in simplified approach
            f.write(f"# Scale factor: {self.scale_factor}\n")
            f.write("# Generated for Arbor compatibility with cycle breaking\n")
            f.write("#\n")

            # Write cycle-breaking information
            if self.non_tree_edges:
                f.write("# CYCLE-BREAKING ANNOTATIONS:\n")
                f.write(
                    "# The following connections were removed to create a tree structure.\n"
                )
                f.write(
                    "# These should be restored as additional connections in Arbor after import.\n"
                )
                f.write("#\n")
                f.write(
                    "# Format: # CONNECT sample_id_1 <-> sample_id_2 (original_node_1 <-> original_node_2)\n"
                )

                for edge in self.non_tree_edges:
                    sample_id_1 = edge.get("sample_id_1", "UNKNOWN")
                    sample_id_2 = edge.get("sample_id_2", "UNKNOWN")
                    node_1 = edge.get("original_node_1", "UNKNOWN")
                    node_2 = edge.get("original_node_2", "UNKNOWN")
                    f.write(
                        f"# CONNECT {sample_id_1} <-> {sample_id_2} ({node_1} <-> {node_2})\n"
                    )

                f.write("#\n")
                f.write("# To restore these connections in Arbor:\n")
                f.write("# 1. Load the SWC file into Arbor\n")
                f.write("# 2. Parse the CONNECT annotations above\n")
                f.write(
                    "# 3. Add gap junctions or custom connections between the specified sample IDs\n"
                )
                f.write("#\n")
            else:
                f.write("# No cycles detected - original graph was already a tree\n")
                f.write("#\n")

            # Write data entries
            for entry in self.entries:
                f.write(entry + "\n")

        # Also create a companion JSON file with connection information
        if self.non_tree_edges:
            connections_file = filename.replace(".swc", "_connections.json")
            # Use top-level import

            connections_data = {
                "metadata": self.metadata,
                "non_tree_connections": self.non_tree_edges,
            }

            with open(connections_file, "w") as f:
                json.dump(connections_data, f, indent=2)

            print(f"📋 Non-tree connections saved to: {connections_file}")

        print(f"✅ SWC data written to: {filename}")

    def get_summary(self) -> str:
        """Get a summary of the SWC data."""
        summary = f"SWC Data Summary:\n"
        summary += f"  - Total segments: {self.metadata.get('total_segments', 0)}\n"
        # No soma segments in simplified approach
        summary += f"  - Root segment: {self.root_segment}\n"
        summary += f"  - Tree edges: {len(self.entries) - 1}\n"
        summary += f"  - Non-tree edges: {len(self.non_tree_edges)}\n"
        summary += f"  - Scale factor: {self.scale_factor}\n"
        if self.non_tree_edges:
            summary += f"  ⚠️  {len(self.non_tree_edges)} cycle-breaking connections need post-processing\n"
        return summary



class SegmentGraph(nx.Graph):
    """
    Graph representation of segmented mesh using graph-based architecture.

    This class wraps a NetworkX graph to represent the connectivity
    between segments in a compartmental model using the graph-based approach:
    
    - Nodes represent spatial points (Point objects at cross-section centers)
    - Edges represent cylindrical segments (Segment objects) connecting those points

    Note that this graph is allowed to have cycles, and thus it is not a tree.
    """

    def __init__(self):
        """
        Initialize a SegmentGraph by super()
        """
        super().__init__()
        # Graph-based architecture support
        self.points_list: List[Point] = []
        self.segments_list: List[Segment] = []
        self.point_dict: Dict[str, Point] = {}
        self.segment_dict: Dict[str, Segment] = {}

    def add_point(self, point: Point) -> None:
        """Add a Point to the graph."""
        self.points_list.append(point)
        self.point_dict[point.id] = point

        # Add to NetworkX graph with node attributes
        self.add_node(
            point.id,
            center=point.center,
            radius=point.radius,
            z_position=point.z_position,
            slice_index=point.slice_index,
            cross_section_index=point.cross_section_index,
        )

    def add_segment(self, segment: Segment) -> None:
        """Add a Segment to the graph."""
        self.segments_list.append(segment)
        self.segment_dict[segment.id] = segment

        # Add to NetworkX graph with edge attributes
        self.add_edge(
            segment.point1_id,
            segment.point2_id,
            edge_id=segment.id,
            length=segment.length,
            radius1=segment.radius1,
            radius2=segment.radius2,
            volume=segment.volume,
            center_line=segment.center_line,
        )

    def get_point_by_id(self, point_id: str) -> Optional[Point]:
        """Get Point by ID."""
        return self.point_dict.get(point_id)

    def get_segment_by_id(self, segment_id: str) -> Optional[Segment]:
        """Get Segment by ID."""
        return self.segment_dict.get(segment_id)

    def set_segment_properties(
        self, segment_id: str, properties: Dict[str, Any]
    ) -> None:
        """
        Set properties for a specific segment.

        Args:
            segment_id: ID of the segment to update
            properties: Dictionary of properties to set
        """
        if segment_id not in self.nodes:
            raise ValueError(f"Segment {segment_id} not found in graph")

        for key, value in properties.items():
            self.nodes[segment_id][key] = value

    def get_segment_properties(self, segment_id: str) -> Dict[str, Any]:
        """
        Get all properties of a specific segment.

        Args:
            segment_id: ID of the segment

        Returns:
            Dictionary of segment properties
        """
        if segment_id not in self.nodes:
            raise ValueError(f"Segment {segment_id} not found in graph")

        return dict(self.nodes[segment_id])

    def _build_tree_adjacency_list(self, tree_edges: set) -> dict:
        """
        Build adjacency list from tree edges.

        Args:
            tree_edges: Set of edges forming the spanning tree

        Returns:
            Dictionary mapping each node to its neighbors in the tree
        """
        tree_adj = {}
        for u, v in tree_edges:
            if u not in tree_adj:
                tree_adj[u] = []
            if v not in tree_adj:
                tree_adj[v] = []
            tree_adj[u].append(v)
            tree_adj[v].append(u)
        return tree_adj

    def export_to_swc(
        self,
        scale_factor: float = 1.0,
        cycle_breaking_strategy: str = "minimum_spanning_tree",
    ) -> "SWCData":
        """
        Export the segment graph to SWC format data compatible with Arbor simulator.

        SWC format represents neuronal morphology as a tree structure where each node
        has: SampleID, TypeID, x, y, z, radius, ParentID

        Since SWC requires a tree (no cycles), this method breaks cycles by creating
        a spanning tree and annotates removed edges for post-processing in Arbor.

        Args:
            scale_factor: Scaling factor to convert units (default: 1.0 for micrometers)
            cycle_breaking_strategy: Strategy for breaking cycles ('minimum_spanning_tree', 'bfs_tree')

        Returns:
            SWCData object containing the SWC entries and metadata

        Note:
            - All segments use type ID 5 (custom segment type)
            - Tree structure ensures proper parent-child relationships for Arbor compatibility
            - Cycles are broken and non-tree edges are annotated for post-processing
        """
        return self._export_to_swc(scale_factor)

    def _export_to_swc(self, scale_factor: float = 1.0) -> "SWCData":
        """Export graph-based architecture to SWC format."""
        if len(self.points_list) == 0:
            raise ValueError("Cannot export empty graph to SWC format")

        # Find root point (lowest z-position)
        root_point = min(self.points_list, key=lambda p: p.z_position)

        # Build spanning tree to break cycles
        tree_edges, non_tree_edges = self._break_cycles_for_swc(root_point.id)

        # Build parent-child relationships
        parent_map = self._build_parent_map_from_tree(tree_edges, root_point.id)

        # Assign sample IDs
        point_to_sample_id = self._assign_sample_ids_bfs(root_point.id, tree_edges)

        # Generate SWC entries
        swc_entries = []
        for point in sorted(self.points_list, key=lambda p: point_to_sample_id[p.id]):
            x = point.center[0] * scale_factor
            y = point.center[1] * scale_factor
            z = point.center[2] * scale_factor
            r = point.radius * scale_factor

            parent_point_id = parent_map.get(point.id, -1)
            parent_sample_id = (
                point_to_sample_id.get(parent_point_id, -1)
                if parent_point_id != -1
                else -1
            )

            sample_id = point_to_sample_id[point.id]
            type_id = 5  # All points use type 5

            swc_entries.append(
                f"{sample_id} {type_id} {x:.6f} {y:.6f} {z:.6f} {r:.6f} {parent_sample_id}"
            )

        # Create metadata
        metadata = {
            "total_segments": len(self.segments_list),
            "total_nodes": len(self.points_list),
            "tree_edges": len(tree_edges),
            "non_tree_edges": len(non_tree_edges),
        }

        # Format non-tree edges
        formatted_non_tree_edges = [
            {
                "sample_id_1": point_to_sample_id.get(u, None),
                "sample_id_2": point_to_sample_id.get(v, None),
                "original_node_1": u,
                "original_node_2": v,
            }
            for u, v in sorted(non_tree_edges)
        ]

        return SWCData(
            entries=swc_entries,
            metadata=metadata,
            non_tree_edges=formatted_non_tree_edges,
            root_segment=root_point.id,
            scale_factor=scale_factor,
        )


    def _break_cycles_and_create_tree(
        self, root_node: str, strategy: str = "minimum_spanning_tree"
    ) -> tuple:
        """
        Break cycles in the graph by creating a spanning tree.

        Args:
            root_node: Root node for the spanning tree
            strategy: Strategy for creating spanning tree

        Returns:
            Tuple of (tree_edges, non_tree_edges) where:
            - tree_edges: Set of edges forming the spanning tree
            - non_tree_edges: Set of edges removed to break cycles
        """
        # Use top-level imports

        if strategy == "minimum_spanning_tree":
            # Use minimum spanning tree based on edge weights (distance between centroids)
            # Add edge weights based on euclidean distance between segment centroids
            weighted_graph = self.copy()
            for u, v in weighted_graph.edges():
                centroid_u = np.array(
                    weighted_graph.nodes[u].get("centroid", [0, 0, 0])
                )
                centroid_v = np.array(
                    weighted_graph.nodes[v].get("centroid", [0, 0, 0])
                )
                distance = np.linalg.norm(centroid_u - centroid_v)
                weighted_graph[u][v]["weight"] = distance

            # Create minimum spanning tree
            mst = nx.minimum_spanning_tree(weighted_graph)
            tree_edges = set(mst.edges())

        elif strategy == "bfs_tree":
            # Use BFS tree from root node
            tree_edges = set()
            visited = set([root_node])
            queue = deque([root_node])

            while queue:
                current = queue.popleft()
                for neighbor in self.neighbors(current):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        tree_edges.add((min(current, neighbor), max(current, neighbor)))
                        queue.append(neighbor)
        else:
            raise ValueError(f"Unknown cycle breaking strategy: {strategy}")

        # Identify non-tree edges (edges that were removed to break cycles)
        all_edges = set((min(u, v), max(u, v)) for u, v in self.edges())
        tree_edges_normalized = set((min(u, v), max(u, v)) for u, v in tree_edges)
        non_tree_edges = all_edges - tree_edges_normalized

        return tree_edges_normalized, non_tree_edges

    def _build_parent_map_from_tree(self, tree_edges: set, root_node: str) -> dict:
        """
        Build parent-child mapping from tree edges using BFS from root.

        Args:
            tree_edges: Set of edges forming the spanning tree
            root_node: Root node of the tree

        Returns:
            Dictionary mapping each node to its parent (root maps to -1)
        """
        # Build adjacency list from tree edges
        tree_adj = self._build_tree_adjacency_list(tree_edges)

        # BFS to establish parent-child relationships
        parent_map = {root_node: -1}
        visited = set([root_node])
        queue = deque([root_node])

        while queue:
            current = queue.popleft()
            for neighbor in tree_adj.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent_map[neighbor] = current
                    queue.append(neighbor)

        return parent_map

    def _assign_sample_ids_bfs(self, root_node: str, tree_edges: set) -> dict:
        """
        Assign sample IDs using BFS order to ensure parent IDs < child IDs.

        Args:
            root_node: Root node of the tree
            tree_edges: Set of edges forming the spanning tree

        Returns:
            Dictionary mapping node IDs to sample IDs
        """
        # Build adjacency list from tree edges
        tree_adj = self._build_tree_adjacency_list(tree_edges)

        # BFS assignment ensures parent IDs are assigned before child IDs
        node_to_sample_id = {}
        sample_id = 1
        queue = deque([root_node])
        visited = set([root_node])
        node_to_sample_id[root_node] = sample_id
        sample_id += 1

        while queue:
            current = queue.popleft()
            for neighbor in tree_adj.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    node_to_sample_id[neighbor] = sample_id
                    sample_id += 1
                    queue.append(neighbor)

        return node_to_sample_id


    def _break_cycles_for_swc(self, root_point_id: str) -> Tuple[set, set]:
        """Break cycles using minimum spanning tree for graph-based architecture."""
        # Create weighted graph based on segment lengths
        weighted_graph = self.copy()
        for segment in self.segments_list:
            if weighted_graph.has_edge(segment.point1_id, segment.point2_id):
                weighted_graph[segment.point1_id][segment.point2_id]["weight"] = segment.length

        # Create minimum spanning tree
        mst = nx.minimum_spanning_tree(weighted_graph)
        tree_edges = set(mst.edges())

        # Identify non-tree edges
        all_edges = set((min(u, v), max(u, v)) for u, v in self.edges())
        tree_edges_normalized = set((min(u, v), max(u, v)) for u, v in tree_edges)
        non_tree_edges = all_edges - tree_edges_normalized

        return tree_edges_normalized, non_tree_edges

    def _build_parent_map_from_tree(self, tree_edges: set, root_point_id: str) -> dict:
        """Build parent-child mapping from tree edges for graph-based architecture."""
        # Build adjacency list
        tree_adj = {}
        for u, v in tree_edges:
            if u not in tree_adj:
                tree_adj[u] = []
            if v not in tree_adj:
                tree_adj[v] = []
            tree_adj[u].append(v)
            tree_adj[v].append(u)

        # BFS to establish parent-child relationships
        parent_map = {root_point_id: -1}
        visited = set([root_point_id])
        queue = deque([root_point_id])

        while queue:
            current = queue.popleft()
            for neighbor in tree_adj.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent_map[neighbor] = current
                    queue.append(neighbor)

        return parent_map

    def _assign_sample_ids_bfs(self, root_point_id: str, tree_edges: set) -> dict:
        """Assign sample IDs using BFS order for graph-based architecture."""
        # Build adjacency list
        tree_adj = {}
        for u, v in tree_edges:
            if u not in tree_adj:
                tree_adj[u] = []
            if v not in tree_adj:
                tree_adj[v] = []
            tree_adj[u].append(v)
            tree_adj[v].append(u)

        # BFS assignment
        point_to_sample_id = {}
        sample_id = 1
        queue = deque([root_point_id])
        visited = set([root_point_id])
        point_to_sample_id[root_point_id] = sample_id
        sample_id += 1

        while queue:
            current = queue.popleft()
            for neighbor in tree_adj.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    point_to_sample_id[neighbor] = sample_id
                    sample_id += 1
                    queue.append(neighbor)

        return point_to_sample_id

    def visualize(
        self,
        color_by: str = "slice_index",
        show_plot: bool = True,
        save_path: str = None,
        figsize: tuple = (12, 10),
        node_scale: float = 1000.0,
        repulsion_strength: float = 0.0,
        iterations: int = 100,
        x_weight: float = 0.5,
        y_weight: float = 0.5,
    ) -> Any:
        """
        Visualize the segment graph.

        Args:
            color_by: Property to color nodes by ('slice_index', 'volume', or any other node attribute)
            show_plot: Whether to display the plot
            save_path: Path to save the plot
            figsize: Figure size
            node_scale: Scaling factor for node sizes
            repulsion_strength: Strength of node repulsion (higher = more spread)
            iterations: Number of iterations for position optimization
            x_weight: Weight for x-coordinate in horizontal positioning (0.0 to 1.0)
            y_weight: Weight for y-coordinate in horizontal positioning (0.0 to 1.0)

        Returns:
            Figure object
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            import numpy as np

            # Create figure
            fig, ax = plt.subplots(figsize=figsize)

            # Normalize weights for horizontal positioning
            total_weight = x_weight + y_weight
            if total_weight == 0:
                # Default to equal weighting if both weights are zero
                x_weight = y_weight = 0.5
            else:
                # Normalize weights to sum to 1
                x_weight = x_weight / total_weight
                y_weight = y_weight / total_weight

            # Initialize positions based on segment centroids
            pos = {}
            z_levels = {}  # Track z-levels for each node

            # First pass: collect z-levels and initial positions
            for node, data in self.nodes(data=True):
                # Get centroid with better error handling
                centroid = data.get("centroid", None)
                if centroid is None:
                    # Generate a random position if no centroid is available
                    # This ensures nodes don't stack on top of each other
                    raise ValueError(f"Node {node} has no centroid")
                    # import random
                    # centroid = [random.uniform(-1, 1), random.uniform(-1, 1), data.get('slice_index', 0) or 0]

                # Store z-level for vertical positioning
                z_levels[node] = centroid[2]

                # Use weighted combination of x and y for horizontal positioning
                h_pos = x_weight * centroid[0] + y_weight * centroid[1]

                # Initialize position
                pos[node] = np.array([h_pos, centroid[2]])  # [horizontal, vertical]

            # Group nodes by similar z-levels
            z_groups = {}
            z_tolerance = 0.1  # Tolerance for considering nodes at same z-level

            for node, z in z_levels.items():
                # Find if this node belongs to an existing z-group
                assigned = False
                for group_z, group_nodes in z_groups.items():
                    if abs(z - group_z) < z_tolerance:
                        group_nodes.append(node)
                        assigned = True
                        break

                if not assigned:
                    # Create new group
                    z_groups[z] = [node]

            # For nodes at the same z-level that have identical horizontal positions,
            # spread them out horizontally
            for group_z, group_nodes in z_groups.items():
                if len(group_nodes) > 1:
                    # Check if nodes have identical horizontal positions
                    h_positions = [pos[node][0] for node in group_nodes]
                    if len(set(h_positions)) < len(h_positions):
                        # Spread nodes horizontally
                        spread = 0.5  # Base spread distance
                        for i, node in enumerate(group_nodes):
                            # Offset from center, alternating left and right
                            offset = spread * (i - (len(group_nodes) - 1) / 2)
                            pos[node][0] = pos[node][0] + offset

            # Apply repulsion to prevent overlapping nodes
            if repulsion_strength > 0 and iterations > 0:
                # First normalize all positions to [0,1] range for stable repulsion
                pos_array = np.array(list(pos.values()))
                if len(pos_array) > 0:  # Check if there are any nodes
                    min_pos = pos_array.min(axis=0)
                    max_pos = pos_array.max(axis=0)
                    range_pos = max_pos - min_pos
                    # Avoid division by zero
                    range_pos = np.where(range_pos == 0, 1, range_pos)

                    # Normalize positions
                    for node in pos:
                        pos[node] = (pos[node] - min_pos) / range_pos

                    # Apply repulsion iterations
                    for _ in range(iterations):
                        # Calculate repulsive forces for each node
                        forces = {}
                        for node1 in self.nodes():
                            force = np.zeros(2)
                            for node2 in self.nodes():
                                if node1 != node2:
                                    diff = pos[node1] - pos[node2]
                                    dist = np.linalg.norm(diff)

                                    # Apply stronger repulsion for nodes at similar z-levels
                                    z_similarity = 1.0
                                    if abs(diff[1]) < 0.1:  # Similar z-level
                                        z_similarity = (
                                            5.0  # Stronger repulsion horizontally
                                        )

                                    # Avoid division by zero
                                    if dist < 0.01:
                                        dist = 0.01

                                    # Repulsive force inversely proportional to distance
                                    # Stronger in horizontal direction for nodes at same z-level
                                    force_magnitude = repulsion_strength / (dist**2)
                                    force_vector = diff / dist * force_magnitude

                                    # Apply stronger horizontal force for nodes at similar heights
                                    if abs(diff[1]) < 0.1:  # Similar heights
                                        force_vector[
                                            0
                                        ] *= z_similarity  # Boost horizontal component

                                    force += force_vector

                            forces[node1] = force

                        # Apply forces with limited vertical movement
                        for node, force in forces.items():
                            # Limit vertical movement to preserve z-level ordering
                            force[1] *= 0.1  # Reduce vertical component
                            pos[node] += force * 0.05  # Small step size for stability

                    # Scale positions back to original range and offset
                    for node in pos:
                        pos[node] = pos[node] * range_pos + min_pos

            # Node colors based on specified property
            if color_by in nx.get_node_attributes(self, color_by):
                property_values = list(nx.get_node_attributes(self, color_by).values())
                node_colors = property_values
                cmap = cm.viridis
            else:
                # Default: color by slice_index
                slice_indices = [
                    data.get("slice_index", 0) for _, data in self.nodes(data=True)
                ]
                node_colors = slice_indices
                cmap = cm.viridis

            # Node sizes based on radius (nodes represent spatial points with radii)
            node_sizes = []
            for _, data in self.nodes(data=True):
                radius = data.get("radius", 1.0)  # Default radius if not available
                # Scale radius for visualization (convert to marker size)
                marker_size = max(radius * node_scale, 20)  # Minimum size of 20
                node_sizes.append(marker_size)

            # Draw nodes and edges separately for better control
            nx.draw_networkx_nodes(
                self,
                pos=pos,
                node_color=node_colors,
                cmap=cmap,
                node_size=node_sizes,
                alpha=0.8,
                ax=ax,
            )

            # Draw edges (these represent the segments)
            nx.draw_networkx_edges(
                self,
                pos=pos,
                edge_color="gray",
                width=2,
                alpha=0.6,
                ax=ax,
            )

            # Draw node labels (spatial points)
            nx.draw_networkx_labels(
                self,
                pos=pos,
                font_size=8,
                font_weight="bold",
                ax=ax,
            )

            # Draw edge labels (segments) - show edge IDs or segment info
            edge_labels = {}
            for i, (u, v, data) in enumerate(self.edges(data=True)):
                # Use edge index as segment identifier
                edge_labels[(u, v)] = f"S{i+1}"

            nx.draw_networkx_edge_labels(
                self,
                pos=pos,
                edge_labels=edge_labels,
                font_size=6,
                font_color="red",
                ax=ax,
            )

            # Add title and labels
            ax.set_title(
                "Segment Graph: Nodes=Spatial Points, Edges=Segments",
                fontsize=14,
                fontweight="bold",
            )
            ax.set_xlabel(
                f"Horizontal Position (X*{x_weight:.2f} + Y*{y_weight:.2f})",
                fontsize=12,
            )
            ax.set_ylabel("Z Position (Height)", fontsize=12)

            # Add colorbar if coloring by property
            if color_by:
                sm = plt.cm.ScalarMappable(cmap=cmap)
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax)
                cbar.set_label(color_by.replace("_", " ").title())

            # Save if path provided
            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
                print(f"Graph visualization saved to: {save_path}")

            if show_plot:
                plt.show()

            return fig

        except ImportError:
            warnings.warn("Matplotlib is required for visualization")
            return None




class MeshSegmenter:
    """
    Mesh segmentation using graph-based approach.

    Creates points at cross-section centers and segments connecting them.

    ALGORITHM:
    1. Slice the mesh at regular z-intervals to create cross-sections
    2. Create points at cross-section centers with calculated radii
    3. Build connectivity between points in adjacent slices
    4. Create segments connecting the points

    REQUIREMENTS:
    - Input mesh must be watertight and represent a single connected volume
    - Cross-sections within the same slice never overlap
    - Points in the same slice never connect to each other
    - Connectivity only exists between adjacent slices
    """

    def __init__(self):
        self.original_mesh = None
        self.slice_height = None
        self.cross_sections: List[CrossSection] = []
        self.graph = SegmentGraph()
        self.radius_method = "equivalent_area"
        self.circle_fitting_method = "geometric"

    def segment_mesh(
        self,
        mesh: trimesh.Trimesh,
        slice_height: float,
        radius_method: str = "equivalent_area",
        circle_fitting_method: str = "geometric",
        min_area: float = 1e-6,
    ) -> SegmentGraph:
        """
        Segment mesh using graph-based approach.

        Creates points at cross-section centers and segments connecting them.

        Args:
            mesh: Input mesh (must be single closed volume)
            slice_height: Height of each slice
            radius_method: Method for calculating radius ('equivalent_area', 'average_distance', 'fitted_circle')
            circle_fitting_method: Method for circle fitting ('algebraic', 'geometric', 'robust')
            min_area: Minimum cross-section area threshold

        Returns:
            SegmentGraph: Graph with points and segments
        """
        # Validate slice_height
        if slice_height <= 0:
            raise ValueError(f"slice_height must be positive, got {slice_height}")

        self.original_mesh = mesh.copy()
        self.slice_height = slice_height
        self.radius_method = radius_method
        self.circle_fitting_method = circle_fitting_method
        
        return self._segment_mesh(mesh, slice_height, min_area)

    def _segment_mesh(
        self,
        mesh: trimesh.Trimesh,
        slice_height: float,
        min_area: float = 1e-6,
    ) -> SegmentGraph:
        """Segment mesh using graph-based approach."""
        self.cross_sections = []
        self.graph = SegmentGraph()

        # Step 1: Validate input mesh
        self._validate_single_hull_mesh(mesh)

        # Step 2: Compute cross-sections and create points
        self._compute_cross_sections_and_points(mesh, min_area)

        # Step 3: Detect overlapping cross-sections (error if found)
        self._validate_no_overlaps()

        # Step 4: Build connectivity between points
        self._build_point_connectivity(mesh)

        # Step 5: Create segments with cylinder metadata
        self._create_cylinder_segments()

        print(
            f"✅ Created graph: {len(self.graph.points_list)} points, {len(self.graph.segments_list)} segments"
        )
        return self.graph


    def _validate_single_hull_mesh(self, mesh: trimesh.Trimesh):
        """Step 0: Validate that mesh is a single closed volume."""
        if not mesh.is_watertight:
            raise ValueError("Input mesh must be watertight (closed volume)")

        # Check for multiple disconnected components
        components = mesh.split(only_watertight=False)
        if len(components) > 1:
            raise ValueError(
                f"Input mesh has {len(components)} disconnected components. "
                "Mesh must be a single connected volume."
            )

        # Store original bounds for later face classification
        self.original_bounds = mesh.bounds.copy()

        # Annotate all original faces as "external"
        if not hasattr(mesh, "face_attributes"):
            mesh.face_attributes = {}
        mesh.face_attributes["face_type"] = ["external"] * len(mesh.faces)

        print(
            f"✅ Validated single-hull mesh: {len(mesh.faces)} external faces, "
            f"volume={mesh.volume:.3f}"
        )


    # Helper methods
    def _compute_cross_sections_and_points(self, mesh: trimesh.Trimesh, min_area: float):
        """Compute cross-sections and create points at their centers."""
        z_min, z_max = mesh.bounds[:, 2]
        z_positions = np.arange(z_min + self.slice_height, z_max, self.slice_height)

        print(f"Computing cross-sections at {len(z_positions)} z-positions")

        for slice_idx, z_pos in enumerate(z_positions):
            plane_origin = np.array([0, 0, z_pos])
            plane_normal = np.array([0, 0, 1])

            try:
                # Get 2D cross-section
                section_2d = mesh.section(
                    plane_origin=plane_origin, plane_normal=plane_normal
                )

                if (
                    section_2d is not None
                    and hasattr(section_2d, "area")
                    and section_2d.area > min_area
                ):
                    # Handle multiple disconnected cross-sections (branching)
                    if hasattr(section_2d, "split"):
                        sections = section_2d.split()
                    else:
                        sections = [section_2d]

                    for cs_idx, section in enumerate(sections):
                        if section.area > min_area:
                            cross_section = self._create_cross_section_and_point(
                                section, z_pos, slice_idx, cs_idx
                            )
                            self.cross_sections.append(cross_section)

                            print(
                                f"  Created point at z={z_pos:.2f}, area={section.area:.3f}, "
                                f"center={cross_section.center}, radius={cross_section.radius:.3f}"
                            )

            except Exception as e:
                print(f"  Error at z={z_pos:.2f}: {e}")

        print(f"✅ Created {len(self.cross_sections)} cross-sections and points")

    def _create_cross_section_and_point(
        self, section_2d, z_pos: float, slice_idx: int, cs_idx: int
    ) -> CrossSection:
        """Create a cross-section and corresponding graph point."""
        # Get boundary points in 2D
        if hasattr(section_2d, "vertices"):
            boundary_2d = section_2d.vertices
        else:
            # Fallback: sample points from the boundary
            boundary_2d = self._sample_boundary_points(section_2d)

        # Fit circle to boundary points
        center_2d, fitted_radius = fit_circle_to_points(
            boundary_2d, self.circle_fitting_method
        )

        # Calculate radius using specified method
        if self.radius_method == "equivalent_area":
            radius = calculate_radius_from_area(section_2d.area)
        elif self.radius_method == "average_distance":
            radius = calculate_radius_from_boundary(boundary_2d, center_2d)
        elif self.radius_method == "fitted_circle":
            radius = fitted_radius
        else:
            raise ValueError(f"Unknown radius method: {self.radius_method}")

        # Create 3D center point
        center_3d = np.array([center_2d[0], center_2d[1], z_pos])

        # Create CrossSection
        cross_section = CrossSection(
            z_position=z_pos,
            intersection_lines=np.array([]),  # Not used in new architecture
            intersection_2d=section_2d,
            area=section_2d.area,
            center=center_3d,
            radius=radius,
            boundary_points=boundary_2d,
        )

        # Create Point
        point_id = f"point_{slice_idx}_{cs_idx}"
        point = Point(
            id=point_id,
            z_position=z_pos,
            center=center_3d,
            radius=radius,
            cross_section=cross_section,
            slice_index=slice_idx,
            cross_section_index=cs_idx,
        )

        # Add point to graph
        self.graph.add_point(point)

        return cross_section

    def _sample_boundary_points(self, section_2d, n_points: int = 50) -> np.ndarray:
        """Sample points from the boundary of a 2D section."""
        # This is a fallback method if vertices are not directly available
        try:
            # Try to get boundary points from the section
            if hasattr(section_2d, "boundary"):
                boundary = section_2d.boundary
                if hasattr(boundary, "vertices"):
                    return boundary.vertices

            # Fallback: create a rough circular approximation
            center = np.array([0, 0])  # Will be refined by circle fitting
            radius_est = np.sqrt(section_2d.area / np.pi)
            angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
            points = center + radius_est * np.column_stack(
                [np.cos(angles), np.sin(angles)]
            )
            return points

        except:
            # Ultimate fallback
            center = np.array([0, 0])
            radius_est = np.sqrt(section_2d.area / np.pi)
            angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
            points = center + radius_est * np.column_stack(
                [np.cos(angles), np.sin(angles)]
            )
            return points

    def _validate_no_overlaps(self):
        """Validate that cross-sections don't overlap within same slice."""
        overlaps = detect_cross_section_overlap(self.cross_sections)

        if overlaps:
            overlap_info = []
            for i, j in overlaps:
                cs1, cs2 = self.cross_sections[i], self.cross_sections[j]
                overlap_info.append(
                    f"Cross-sections {i} and {j} at z={cs1.z_position:.3f}: "
                    f"centers {cs1.center[:2]} and {cs2.center[:2]}, "
                    f"radii {cs1.radius:.3f} and {cs2.radius:.3f}"
                )

            raise ValueError(
                f"Found {len(overlaps)} overlapping cross-sections:\n"
                + "\n".join(overlap_info)
            )

        print("✅ No overlapping cross-sections detected")

    def _build_point_connectivity(self, mesh: trimesh.Trimesh):
        """Build connectivity between points in adjacent slices."""
        print("Building point connectivity through volume analysis...")

        # Group points by slice
        points_by_slice = {}
        for point in self.graph.points_list:
            slice_idx = point.slice_index
            if slice_idx not in points_by_slice:
                points_by_slice[slice_idx] = []
            points_by_slice[slice_idx].append(point)

        connections_found = 0

        # Check connectivity between adjacent slices
        for slice_idx in sorted(points_by_slice.keys())[:-1]:
            next_slice_idx = slice_idx + 1

            if next_slice_idx in points_by_slice:
                current_points = points_by_slice[slice_idx]
                next_points = points_by_slice[next_slice_idx]

                # Check each pair of points from adjacent slices
                for point1 in current_points:
                    for point2 in next_points:
                        if self._points_connected_through_volume(point1, point2, mesh):
                            # We'll create the actual segment later in _create_cylinder_segments
                            # For now, just mark them as connected in NetworkX graph
                            if not self.graph.has_edge(point1.id, point2.id):
                                self.graph.add_edge(
                                    point1.id, point2.id, temp_connection=True
                                )
                                connections_found += 1
                                print(f"  Connected {point1.id} ↔ {point2.id}")

        print(f"✅ Found {connections_found} point connections")

    def _points_connected_through_volume(
        self, point1: Point, point2: Point, mesh: trimesh.Trimesh
    ) -> bool:
        """Determine if two points are connected through contiguous volume."""
        # Basic geometric check: are the cross-sections close enough?
        center1_2d = point1.center[:2]
        center2_2d = point2.center[:2]
        distance_2d = np.linalg.norm(center1_2d - center2_2d)

        # If centers are too far apart, they're probably not connected
        max_distance = (point1.radius + point2.radius) * 2.0  # Generous threshold
        if distance_2d > max_distance:
            return False

        # Volume-based connectivity check
        # Sample points between the two cross-sections and check if they're inside the mesh
        z1, z2 = point1.z_position, point2.z_position

        # Create interpolated points along the potential connection
        n_samples = 10
        z_samples = np.linspace(z1, z2, n_samples)

        # Interpolate centers and radii
        center_samples = np.array(
            [
                np.array([center1_2d[0], center1_2d[1], z])
                + (z - z1)
                / (z2 - z1)
                * np.array(
                    [center2_2d[0] - center1_2d[0], center2_2d[1] - center1_2d[1], 0]
                )
                for z in z_samples
            ]
        )

        # Check if sample points are inside the mesh
        inside_count = 0
        for center_sample in center_samples:
            if mesh.contains([center_sample])[0]:
                inside_count += 1

        # Require most sample points to be inside the mesh
        connectivity_threshold = 0.7  # 70% of samples must be inside
        return (inside_count / n_samples) >= connectivity_threshold

    def _create_cylinder_segments(self):
        """Create Segment objects for all connected point pairs."""
        print("Creating cylinder segments with metadata...")

        segment_count = 0
        for point1_id, point2_id in self.graph.edges():
            point1 = self.graph.get_point_by_id(point1_id)
            point2 = self.graph.get_point_by_id(point2_id)

            if point1 and point2:
                # Calculate segment properties
                center_line = np.array([point1.center, point2.center])
                length = np.linalg.norm(point2.center - point1.center)

                # Approximate cylinder volume (truncated cone)
                r1, r2 = point1.radius, point2.radius
                volume = (np.pi * length / 3) * (r1**2 + r1 * r2 + r2**2)

                # Create Segment
                segment_id = f"segment_{point1_id}_{point2_id}"
                segment = Segment(
                    id=segment_id,
                    point1_id=point1_id,
                    point2_id=point2_id,
                    length=length,
                    radius1=r1,
                    radius2=r2,
                    center_line=center_line,
                    volume=volume,
                )

                # Add to graph
                self.graph.add_segment(segment)
                segment_count += 1

        print(f"✅ Created {segment_count} cylinder segments")

    def get_connected_components(self) -> List[List[str]]:
        """Get connected components in the graph."""
        if not self.graph:
            return []
        return self.graph.get_connected_components()

    def get_segment_graph(self):
        """Return the SegmentGraph instance maintained by this segmenter.

        Returns:
            SegmentGraph: A graph representation of the segmented structure
        """
        if not self.segments or self.graph.number_of_nodes() == 0:
            raise ValueError("Must run segment_mesh before getting segment graph")

        return self.graph

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
                "num_components": (
                    len(self.get_connected_components()) if self.graph else 0
                ),
                "num_edges": len(self.graph.edges) if self.graph else 0,
            },
        }

    def _reconstruct_face_annotations(
        self, mesh: trimesh.Trimesh, z_min: float, z_max: float
    ):
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
        print(
            f"  🏷️  Annotated {external_count} external + {internal_count} internal faces"
        )

    def _transfer_face_attributes(self, source_mesh: trimesh.Trimesh, components: list):
        """Transfer face attributes from source mesh to split components."""
        if (
            not hasattr(source_mesh, "face_attributes")
            or "face_type" not in source_mesh.face_attributes
        ):
            return

        source_face_types = source_mesh.face_attributes["face_type"]

        if len(components) == 1:
            # Single component case - direct transfer
            component = components[0]
            if not hasattr(component, "face_attributes"):
                component.face_attributes = {}

            if len(component.faces) == len(source_mesh.faces):
                component.face_attributes["face_type"] = source_face_types.copy()
                print(
                    f"    🔄 Transferred face attributes: {len(component.faces)} faces"
                )
            else:
                print(
                    f"    ⚠️ Face count mismatch: source={len(source_mesh.faces)}, component={len(component.faces)}"
                )
        else:
            # Multiple components case - map faces based on geometric matching
            print(
                f"    🔄 Transferring face attributes to {len(components)} components..."
            )

            for comp_idx, component in enumerate(components):
                if not hasattr(component, "face_attributes"):
                    component.face_attributes = {}

                # Map component faces to source faces by geometric similarity
                comp_face_types = self._map_faces_to_source(
                    component, source_mesh, source_face_types
                )
                component.face_attributes["face_type"] = comp_face_types

                external_count = np.sum(comp_face_types == "external")
                internal_count = np.sum(comp_face_types == "internal")
                print(
                    f"      Component {comp_idx}: {external_count} external + {internal_count} internal faces"
                )

    def _map_faces_to_source(
        self,
        component: trimesh.Trimesh,
        source_mesh: trimesh.Trimesh,
        source_face_types: np.ndarray,
    ) -> np.ndarray:
        """Map component faces to source mesh faces based on geometric proximity."""
        comp_centroids = component.triangles_center
        source_centroids = source_mesh.triangles_center

        face_types = np.array(["external"] * len(comp_centroids), dtype=object)

        # For each component face, find the closest source face
        for i, comp_centroid in enumerate(comp_centroids):
            distances = np.linalg.norm(source_centroids - comp_centroid, axis=1)
            closest_idx = np.argmin(distances)

            # If the closest source face is very close, use its type
            if distances[closest_idx] < 0.1:  # Tolerance for geometric matching
                face_types[i] = source_face_types[closest_idx]
            else:
                # If no close match, classify based on geometry
                # Faces at slice boundaries (z_min/z_max) are likely internal
                z_coord = comp_centroid[2]
                if (
                    abs(z_coord - source_mesh.bounds[0, 2]) < 0.01
                    or abs(z_coord - source_mesh.bounds[1, 2]) < 0.01
                ):
                    face_types[i] = "internal"
                else:
                    face_types[i] = "external"

        return face_types

    def visualize_connectivity_graph(
        self,
        save_path: str = None,
        show_plot: bool = True,
        include_3d_view: bool = False,
        figsize: tuple = None,
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
        if self.graph is None:
            raise ValueError(
                "No connectivity graph available. Run segment_mesh() first."
            )

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
                x_positions = [
                    (i - (len(segments) - 1) / 2) * 1.5 for i in range(len(segments))
                ]

            for i, segment in enumerate(segments):
                pos[segment.id] = (x_positions[i], y_pos)
                # Color by z-level
                colors.append(cmap(z_level / max_z if max_z > 0 else 0))
                # Smaller, more reasonable node sizes based on volume
                base_size = 50  # Smaller base size
                volume_factor = min(3.0, max(0.5, segment.volume / 100))  # Scale factor
                node_sizes.append(int(base_size * volume_factor))

        # Helper function to apply aspect ratio constraints
        def apply_aspect_ratio_fix(ax, positions, min_aspect_ratio=0.5):
            if positions:
                x_coords = [p[0] for p in positions.values()]
                y_coords = [p[1] for p in positions.values()]
                x_range = (
                    max(x_coords) - min(x_coords) if len(set(x_coords)) > 1 else 2.0
                )
                y_range = (
                    max(y_coords) - min(y_coords) if len(set(y_coords)) > 1 else 2.0
                )

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
            ax1.set_title(
                "Connectivity Graph (3D View)", fontsize=14, fontweight="bold"
            )

            # Draw the graph with improved styling
            # Draw edges first so they appear behind nodes
            nx.draw_networkx_edges(
                self.graph,
                pos,
                ax=ax1,
                edge_color="darkblue",
                width=2.5,
                alpha=0.7,
                style="-",
            )

            # Draw nodes with smaller size and better visibility
            nx.draw_networkx_nodes(
                self.graph,
                pos,
                ax=ax1,
                node_color=colors,
                node_size=node_sizes,
                alpha=0.9,
                edgecolors="black",
                linewidths=1.0,
            )

            # Draw labels with better contrast
            nx.draw_networkx_labels(
                self.graph,
                pos,
                ax=ax1,
                font_size=7,
                font_weight="bold",
                font_color="white",
            )

            ax1.set_xlabel("Horizontal Position", fontsize=12)
            ax1.set_ylabel("Z-Level (Slice Index)", fontsize=12)
            ax1.grid(True, alpha=0.3)

            # Apply aspect ratio fix for 3D view
            apply_aspect_ratio_fix(ax1, pos)
            ax1.set_aspect("equal")

        # Network view (main plot or right plot)
        ax_network = ax2 if include_3d_view else ax_main
        ax_network.set_title(
            "Connectivity Graph" + (" (Network View)" if include_3d_view else ""),
            fontsize=14,
            fontweight="bold",
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
            self.graph,
            pos_network,
            ax=ax_network,
            node_color=colors,
            node_size=node_sizes,
            alpha=0.8,
        )

        nx.draw_networkx_edges(
            self.graph,
            pos_network,
            ax=ax_network,
            edge_color="gray",
            width=2,
            alpha=0.6,
        )

        nx.draw_networkx_labels(
            self.graph, pos_network, ax=ax_network, font_size=8, font_weight="bold"
        )

        ax_network.set_xlabel("Horizontal Position", fontsize=12)
        ax_network.set_ylabel("Z-Level (Slice Index)", fontsize=12)
        ax_network.grid(True, alpha=0.3)

        # Apply aspect ratio fix for network view
        apply_aspect_ratio_fix(ax_network, pos_network)
        ax_network.set_aspect("equal")

        # Add statistics text
        stats_text = f"""Graph Statistics:
Segments: {len(self.segments)}
Connections: {len(self.graph.edges)}
Z-levels: {len(z_levels)}
Total Volume: {sum(s.volume for s in self.segments):.3f}"""

        fig.text(
            0.02,
            0.02,
            stats_text,
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
        )

        plt.tight_layout()

        # Save plot if path provided
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"📊 Connectivity graph saved to: {save_path}")

        if show_plot:
            plt.show()

        return save_path if save_path else fig

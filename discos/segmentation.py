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
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN


# Utility functions for circle fitting and cross-section analysis
def fit_circle_to_points(points_2d: np.ndarray, method: str = 'algebraic') -> Tuple[np.ndarray, float]:
    """Fit a circle to 2D points using various methods.
    
    Args:
        points_2d: Nx2 array of 2D points
        method: 'algebraic', 'geometric', or 'robust'
        
    Returns:
        Tuple of (center, radius)
    """
    if len(points_2d) < 3:
        raise ValueError("Need at least 3 points to fit a circle")
    
    if method == 'algebraic':
        return _fit_circle_algebraic(points_2d)
    elif method == 'geometric':
        return _fit_circle_geometric(points_2d)
    elif method == 'robust':
        return _fit_circle_robust(points_2d)
    else:
        raise ValueError(f"Unknown circle fitting method: {method}")


def _fit_circle_algebraic(points: np.ndarray) -> Tuple[np.ndarray, float]:
    """Algebraic circle fitting (fastest, least accurate)."""
    x, y = points[:, 0], points[:, 1]
    
    # Set up the system Ax = b
    A = np.column_stack([2*x, 2*y, np.ones(len(x))])
    b = x**2 + y**2
    
    # Solve using least squares
    try:
        params = np.linalg.lstsq(A, b, rcond=None)[0]
        center = np.array([params[0], params[1]])
        radius = np.sqrt(params[2] + params[0]**2 + params[1]**2)
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
        return np.sum((distances - radius)**2)
    
    # Initial guess from algebraic method
    center_init, radius_init = _fit_circle_algebraic(points)
    x0 = np.array([center_init[0], center_init[1], radius_init])
    
    # Optimize
    result = minimize(objective, x0, method='BFGS')
    
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


def calculate_radius_from_boundary(boundary_points: np.ndarray, center: np.ndarray) -> float:
    """Calculate average radius from center to boundary points."""
    distances = np.linalg.norm(boundary_points - center, axis=1)
    return np.mean(distances)


def detect_cross_section_overlap(cross_sections: List['CrossSection'], tolerance: float = 0.1) -> List[Tuple[int, int]]:
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
    boundary_points: Optional[np.ndarray] = None  # 2D boundary points for circle fitting


@dataclass
class GraphNode:
    """Represents a spatial point (center of cross-section) in the new graph architecture."""
    
    id: str
    z_position: float
    center: np.ndarray  # 3D center point (x, y, z)
    radius: float  # Radius of approximating circle
    cross_section: 'CrossSection'  # Reference to the cross-section
    slice_index: int
    cross_section_index: int  # Index within the slice (for multiple cross-sections)


@dataclass
class GraphEdge:
    """Represents a cylindrical segment connecting two nodes."""
    
    id: str
    node1_id: str
    node2_id: str
    length: float
    radius1: float  # Radius at node1
    radius2: float  # Radius at node2
    center_line: np.ndarray  # 3D line from node1 to node2
    volume: float  # Approximate cylinder volume


@dataclass
class Segment:
    """Legacy segment class - kept for compatibility during transition."""

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
        with open(filename, 'w') as f:
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
                f.write("# The following connections were removed to create a tree structure.\n")
                f.write("# These should be restored as additional connections in Arbor after import.\n")
                f.write("#\n")
                f.write("# Format: # CONNECT sample_id_1 <-> sample_id_2 (original_node_1 <-> original_node_2)\n")
                
                for edge in self.non_tree_edges:
                    sample_id_1 = edge.get('sample_id_1', 'UNKNOWN')
                    sample_id_2 = edge.get('sample_id_2', 'UNKNOWN')
                    node_1 = edge.get('original_node_1', 'UNKNOWN')
                    node_2 = edge.get('original_node_2', 'UNKNOWN')
                    f.write(f"# CONNECT {sample_id_1} <-> {sample_id_2} ({node_1} <-> {node_2})\n")
                
                f.write("#\n")
                f.write("# To restore these connections in Arbor:\n")
                f.write("# 1. Load the SWC file into Arbor\n")
                f.write("# 2. Parse the CONNECT annotations above\n")
                f.write("# 3. Add gap junctions or custom connections between the specified sample IDs\n")
                f.write("#\n")
            else:
                f.write("# No cycles detected - original graph was already a tree\n")
                f.write("#\n")
            
            # Write data entries
            for entry in self.entries:
                f.write(entry + "\n")
        
        # Also create a companion JSON file with connection information
        if self.non_tree_edges:
            connections_file = filename.replace('.swc', '_connections.json')
            # Use top-level import
            
            connections_data = {
                'metadata': self.metadata,
                'non_tree_connections': self.non_tree_edges
            }
            
            with open(connections_file, 'w') as f:
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


class NodeEdgeGraph(nx.Graph):
    """Graph representation using the new node-edge architecture.
    
    In this architecture:
    - Nodes represent spatial points (centers of cross-sections)
    - Edges represent cylindrical segments connecting those points
    - Edge attributes contain cylinder metadata (length, radii, volume)
    """
    
    def __init__(self):
        super().__init__()
        self.nodes_list: List[GraphNode] = []
        self.edges_list: List[GraphEdge] = []
        self.node_dict: Dict[str, GraphNode] = {}
        self.edge_dict: Dict[str, GraphEdge] = {}
    
    def add_graph_node(self, node: GraphNode) -> None:
        """Add a GraphNode to the graph."""
        self.nodes_list.append(node)
        self.node_dict[node.id] = node
        
        # Add to NetworkX graph with node attributes
        self.add_node(node.id, 
                     center=node.center,
                     radius=node.radius,
                     z_position=node.z_position,
                     slice_index=node.slice_index,
                     cross_section_index=node.cross_section_index)
    
    def add_graph_edge(self, edge: GraphEdge) -> None:
        """Add a GraphEdge to the graph."""
        self.edges_list.append(edge)
        self.edge_dict[edge.id] = edge
        
        # Add to NetworkX graph with edge attributes
        self.add_edge(edge.node1_id, edge.node2_id,
                     edge_id=edge.id,
                     length=edge.length,
                     radius1=edge.radius1,
                     radius2=edge.radius2,
                     volume=edge.volume,
                     center_line=edge.center_line)
    
    def get_node_by_id(self, node_id: str) -> Optional[GraphNode]:
        """Get GraphNode by ID."""
        return self.node_dict.get(node_id)
    
    def get_edge_by_id(self, edge_id: str) -> Optional[GraphEdge]:
        """Get GraphEdge by ID."""
        return self.edge_dict.get(edge_id)
    
    def export_to_swc(self, scale_factor: float = 1.0, 
                      radius_method: str = 'equivalent_area') -> 'SWCData':
        """Export to SWC format using the new node-edge structure."""
        if len(self.nodes_list) == 0:
            raise ValueError("Cannot export empty graph to SWC format")
        
        # Find root node (lowest z-position)
        root_node = min(self.nodes_list, key=lambda n: n.z_position)
        
        # Build spanning tree to break cycles
        tree_edges, non_tree_edges = self._break_cycles_for_swc(root_node.id)
        
        # Build parent-child relationships
        parent_map = self._build_parent_map_from_tree(tree_edges, root_node.id)
        
        # Assign sample IDs
        node_to_sample_id = self._assign_sample_ids_bfs(root_node.id, tree_edges)
        
        # Generate SWC entries
        swc_entries = []
        for node in sorted(self.nodes_list, key=lambda n: node_to_sample_id[n.id]):
            x = node.center[0] * scale_factor
            y = node.center[1] * scale_factor
            z = node.center[2] * scale_factor
            r = node.radius * scale_factor
            
            parent_node_id = parent_map.get(node.id, -1)
            parent_sample_id = node_to_sample_id.get(parent_node_id, -1) if parent_node_id != -1 else -1
            
            sample_id = node_to_sample_id[node.id]
            type_id = 5  # All nodes use type 5
            
            swc_entries.append(f"{sample_id} {type_id} {x:.6f} {y:.6f} {z:.6f} {r:.6f} {parent_sample_id}")
        
        # Create metadata
        metadata = {
            'total_segments': len(self.edges_list),
            'total_nodes': len(self.nodes_list),
            'tree_edges': len(tree_edges),
            'non_tree_edges': len(non_tree_edges)
        }
        
        # Format non-tree edges
        formatted_non_tree_edges = [
            {
                'sample_id_1': node_to_sample_id.get(u, None),
                'sample_id_2': node_to_sample_id.get(v, None),
                'original_node_1': u,
                'original_node_2': v
            }
            for u, v in sorted(non_tree_edges)
        ]
        
        return SWCData(
            entries=swc_entries,
            metadata=metadata,
            non_tree_edges=formatted_non_tree_edges,
            root_segment=root_node.id,
            scale_factor=scale_factor
        )
    
    def _break_cycles_for_swc(self, root_node_id: str) -> Tuple[set, set]:
        """Break cycles using minimum spanning tree."""
        # Create weighted graph based on edge lengths
        weighted_graph = self.copy()
        for edge in self.edges_list:
            if weighted_graph.has_edge(edge.node1_id, edge.node2_id):
                weighted_graph[edge.node1_id][edge.node2_id]['weight'] = edge.length
        
        # Create minimum spanning tree
        mst = nx.minimum_spanning_tree(weighted_graph)
        tree_edges = set(mst.edges())
        
        # Identify non-tree edges
        all_edges = set((min(u, v), max(u, v)) for u, v in self.edges())
        tree_edges_normalized = set((min(u, v), max(u, v)) for u, v in tree_edges)
        non_tree_edges = all_edges - tree_edges_normalized
        
        return tree_edges_normalized, non_tree_edges
    
    def _build_parent_map_from_tree(self, tree_edges: set, root_node_id: str) -> dict:
        """Build parent-child mapping from tree edges."""
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
        parent_map = {root_node_id: -1}
        visited = set([root_node_id])
        queue = deque([root_node_id])
        
        while queue:
            current = queue.popleft()
            for neighbor in tree_adj.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent_map[neighbor] = current
                    queue.append(neighbor)
        
        return parent_map
    
    def _assign_sample_ids_bfs(self, root_node_id: str, tree_edges: set) -> dict:
        """Assign sample IDs using BFS order."""
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
        node_to_sample_id = {}
        sample_id = 1
        queue = deque([root_node_id])
        visited = set([root_node_id])
        node_to_sample_id[root_node_id] = sample_id
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


class SegmentGraph(nx.Graph):
    """
    Graph representation of segmented mesh.
    
    This class wraps a NetworkX graph to represent the connectivity
    between segments in a compartmental model. It provides methods
    for analyzing the graph structure and accessing properties of
    segments based on their position and connectivity.

    Each node represents a segment, and each edge is a connectivity between segments.

    Note that this graph is allowed to have cycles, and thus it is not a tree.
    """
    
    def __init__(self):
        """
        Initialize a SegmentGraph by super()
        """
        super().__init__()
    
    def set_segment_properties(self, segment_id: str, properties: Dict[str, Any]) -> None:
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
    
    def export_to_swc(self, scale_factor: float = 1.0, 
                      cycle_breaking_strategy: str = 'minimum_spanning_tree') -> 'SWCData':
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
        if len(self.nodes) == 0:
            raise ValueError("Cannot export empty graph to SWC format")
        
        # Step 1: Find root node (segment with smallest z-coordinate)
        root_node = min(self.nodes, key=lambda n: self.nodes[n].get('centroid', [0, 0, float('inf')])[2])
        
        # Step 2: Break cycles by creating spanning tree and identify non-tree edges
        tree_edges, non_tree_edges = self._break_cycles_and_create_tree(root_node, cycle_breaking_strategy)
        
        # Step 3: Build parent-child relationships from spanning tree
        parent_map = self._build_parent_map_from_tree(tree_edges, root_node)
        
        # Step 4: Assign sample IDs ensuring parent IDs < child IDs (Arbor requirement)
        node_to_sample_id = self._assign_sample_ids_bfs(root_node, tree_edges)
        
        # Step 5: Generate SWC entries with type 5 for all segments
        swc_entries = []
        
        for node_id in sorted(node_to_sample_id.keys(), key=lambda x: node_to_sample_id[x]):
            node_data = self.nodes[node_id]
            
            # Validate that node has centroid data
            if 'centroid' not in node_data:
                raise ValueError(f"Node {node_id} has no centroid data")
            
            centroid = node_data.get('centroid', [0, 0, 0])
            volume = node_data.get('volume', 0)
            
            # Calculate radius from volume assuming spherical approximation
            # V = (4/3) * π * r³, so r = (3V / 4π)^(1/3)
            radius = (3 * volume / (4 * math.pi)) ** (1/3) if volume > 0 else 1.0
            
            # Scale coordinates and radius
            x = centroid[0] * scale_factor
            y = centroid[1] * scale_factor
            z = centroid[2] * scale_factor
            r = radius * scale_factor
            
            # Determine parent sample ID
            parent_node = parent_map.get(node_id, -1)
            parent_sample_id = node_to_sample_id.get(parent_node, -1) if parent_node != -1 else -1
            
            # Use type 5 for all segments
            type_id = 5
            
            # SWC entry: SampleID TypeID x y z radius ParentID
            sample_id_val = node_to_sample_id[node_id]
            
            swc_entries.append(f"{sample_id_val} {type_id} {x:.6f} {y:.6f} {z:.6f} {r:.6f} {parent_sample_id}")
        
        # Step 6: Create SWCData object with streamlined metadata
        metadata = {
            'total_segments': len(self.nodes),
            'total_edges': len(self.edges()),
            'tree_edges': len(tree_edges),
            'non_tree_edges': len(non_tree_edges),
            'cycle_breaking_strategy': cycle_breaking_strategy
        }
        
        # Format non-tree edges for SWCData
        formatted_non_tree_edges = [
            {
                'sample_id_1': node_to_sample_id.get(u, None),
                'sample_id_2': node_to_sample_id.get(v, None),
                'original_node_1': u,
                'original_node_2': v,
                'centroid_1': self.nodes[u].get('centroid', [0, 0, 0]),
                'centroid_2': self.nodes[v].get('centroid', [0, 0, 0])
            }
            for u, v in sorted(non_tree_edges)
        ]
        
        swc_data = SWCData(
            entries=swc_entries,
            metadata=metadata,
            non_tree_edges=formatted_non_tree_edges,
            root_segment=root_node,
            scale_factor=scale_factor
        )
        
        print(f"✅ Generated SWC data for {len(self.nodes)} segments")
        print(f"   - All segments use type ID 5")
        print(f"   - Root segment: {root_node} (sample ID: {node_to_sample_id[root_node]})")
        print(f"   - Tree edges: {len(tree_edges)}")
        print(f"   - Non-tree edges (annotated): {len(non_tree_edges)}")
        if non_tree_edges:
            print(f"   ⚠️  {len(non_tree_edges)} cycle-breaking connections need post-processing in Arbor")
        
        return swc_data
    
    def _break_cycles_and_create_tree(self, root_node: str, strategy: str = 'minimum_spanning_tree') -> tuple:
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
        
        if strategy == 'minimum_spanning_tree':
            # Use minimum spanning tree based on edge weights (distance between centroids)
            # Add edge weights based on euclidean distance between segment centroids
            weighted_graph = self.copy()
            for u, v in weighted_graph.edges():
                centroid_u = np.array(weighted_graph.nodes[u].get('centroid', [0, 0, 0]))
                centroid_v = np.array(weighted_graph.nodes[v].get('centroid', [0, 0, 0]))
                distance = np.linalg.norm(centroid_u - centroid_v)
                weighted_graph[u][v]['weight'] = distance
            
            # Create minimum spanning tree
            mst = nx.minimum_spanning_tree(weighted_graph)
            tree_edges = set(mst.edges())
            
        elif strategy == 'bfs_tree':
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
    
    def visualize(self, 
                 color_by: str = 'slice_index', 
                 show_plot: bool = True,
                 save_path: str = None,
                 figsize: tuple = (12, 10),
                 node_scale: float = 1000.0,
                 repulsion_strength: float = 0.0,
                 iterations: int = 100,
                 x_weight: float = 0.5,
                 y_weight: float = 0.5) -> Any:
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
                centroid = data.get('centroid', None)
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
                                        z_similarity = 5.0  # Stronger repulsion horizontally
                                        
                                    # Avoid division by zero
                                    if dist < 0.01:
                                        dist = 0.01
                                        
                                    # Repulsive force inversely proportional to distance
                                    # Stronger in horizontal direction for nodes at same z-level
                                    force_magnitude = repulsion_strength / (dist ** 2)
                                    force_vector = diff / dist * force_magnitude
                                    
                                    # Apply stronger horizontal force for nodes at similar heights
                                    if abs(diff[1]) < 0.1:  # Similar heights
                                        force_vector[0] *= z_similarity  # Boost horizontal component
                                        
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
                slice_indices = [data.get('slice_index', 0) for _, data in self.nodes(data=True)]
                node_colors = slice_indices
                cmap = cm.viridis
            
            # Node sizes based on volume with better scaling
            volumes = [data.get('volume', 1.0) for _, data in self.nodes(data=True)]
            if volumes:
                # Scale volumes for better visualization
                min_vol = min(volumes) if min(volumes) > 0 else 1e-6
                max_vol = max(volumes)
                # Logarithmic scaling for better size distribution
                log_volumes = [np.log10(max(v, min_vol)) for v in volumes]
                # Scale to reasonable node sizes
                min_size = 50  # Minimum node size
                max_size = 1000  # Maximum node size
                if max(log_volumes) > min(log_volumes):
                    node_sizes = [min_size + (max_size - min_size) * 
                                 (v - min(log_volumes)) / (max(log_volumes) - min(log_volumes)) 
                                 for v in log_volumes]
                else:
                    node_sizes = [min_size + (max_size - min_size) * 0.5 for _ in log_volumes]
                
                # Apply user scaling factor
                node_sizes = [s * node_scale / 1000.0 for s in node_sizes]
            else:
                node_sizes = 100
            
            # Draw the graph
            nx.draw_networkx(
                self, 
                pos=pos,
                node_color=node_colors,
                cmap=cmap,
                node_size=node_sizes,
                with_labels=True,
                font_size=8,
                font_weight='bold',
                edge_color='gray',
                width=2,
                alpha=0.8,
                ax=ax
            )
            
            # Add title and labels
            ax.set_title('Segment Graph Visualization', fontsize=14, fontweight='bold')
            ax.set_xlabel(f'Horizontal Position (X*{x_weight:.2f} + Y*{y_weight:.2f})', fontsize=12)
            ax.set_ylabel('Z Position (Height)', fontsize=12)
            
            # Add colorbar if coloring by property
            if color_by:
                sm = plt.cm.ScalarMappable(cmap=cmap)
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax)
                cbar.set_label(color_by.replace('_', ' ').title())
            
            # Save if path provided
            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Graph visualization saved to: {save_path}")
                
            if show_plot:
                plt.show()
                
            return fig
            
        except ImportError:
            warnings.warn("Matplotlib is required for visualization")
            return None



class NodeEdgeSegmenter:
    """New segmentation algorithm using node-edge architecture.
    
    This segmenter creates nodes at cross-section centers and edges as 
    cylindrical segments connecting them.
    """
    
    def __init__(self):
        self.original_mesh = None
        self.slice_height = None
        self.cross_sections: List[CrossSection] = []
        self.graph = NodeEdgeGraph()
        self.radius_method = 'equivalent_area'
        self.circle_fitting_method = 'geometric'
    
    def segment_mesh(self, mesh: trimesh.Trimesh, slice_height: float, 
                    radius_method: str = 'equivalent_area',
                    circle_fitting_method: str = 'geometric',
                    min_area: float = 1e-6) -> NodeEdgeGraph:
        """Segment mesh using the new node-edge architecture.
        
        Args:
            mesh: Input mesh (must be single closed volume)
            slice_height: Height of each slice
            radius_method: Method for calculating radius ('equivalent_area', 'average_distance', 'fitted_circle')
            circle_fitting_method: Method for circle fitting ('algebraic', 'geometric', 'robust')
            min_area: Minimum cross-section area threshold
            
        Returns:
            NodeEdgeGraph: Graph with nodes as spatial points and edges as segments
        """
        if slice_height <= 0:
            raise ValueError(f"slice_height must be positive, got {slice_height}")
        
        self.original_mesh = mesh.copy()
        self.slice_height = slice_height
        self.radius_method = radius_method
        self.circle_fitting_method = circle_fitting_method
        self.cross_sections = []
        self.graph = NodeEdgeGraph()
        
        # Step 1: Validate input mesh
        self._validate_single_hull_mesh(mesh)
        
        # Step 2: Compute cross-sections and create nodes
        self._compute_cross_sections_and_nodes(mesh, min_area)
        
        # Step 3: Detect overlapping cross-sections (error if found)
        self._validate_no_overlaps()
        
        # Step 4: Build connectivity between nodes (the tricky part!)
        self._build_node_connectivity(mesh)
        
        # Step 5: Create edges with cylinder metadata
        self._create_cylinder_edges()
        
        print(f"✅ Created node-edge graph: {len(self.graph.nodes_list)} nodes, {len(self.graph.edges_list)} edges")
        return self.graph
    
    def _validate_single_hull_mesh(self, mesh: trimesh.Trimesh):
        """Validate that mesh is a single closed volume."""
        if not mesh.is_watertight:
            raise ValueError("Input mesh must be watertight (closed volume)")
        
        components = mesh.split(only_watertight=False)
        if len(components) > 1:
            raise ValueError(
                f"Input mesh has {len(components)} disconnected components. "
                "Mesh must be a single connected volume."
            )
        
        print(f"✅ Validated single-hull mesh: volume={mesh.volume:.3f}")
    
    def _compute_cross_sections_and_nodes(self, mesh: trimesh.Trimesh, min_area: float):
        """Compute cross-sections and create nodes at their centers."""
        z_min, z_max = mesh.bounds[:, 2]
        z_positions = np.arange(z_min + self.slice_height, z_max, self.slice_height)
        
        print(f"Computing cross-sections at {len(z_positions)} z-positions")
        
        for slice_idx, z_pos in enumerate(z_positions):
            plane_origin = np.array([0, 0, z_pos])
            plane_normal = np.array([0, 0, 1])
            
            try:
                # Get 2D cross-section
                section_2d = mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)
                
                if section_2d is not None and hasattr(section_2d, 'area') and section_2d.area > min_area:
                    # Handle multiple disconnected cross-sections (branching)
                    if hasattr(section_2d, 'split'):
                        sections = section_2d.split()
                    else:
                        sections = [section_2d]
                    
                    for cs_idx, section in enumerate(sections):
                        if section.area > min_area:
                            cross_section = self._create_cross_section_and_node(
                                section, z_pos, slice_idx, cs_idx
                            )
                            self.cross_sections.append(cross_section)
                            
                            print(f"  Created node at z={z_pos:.2f}, area={section.area:.3f}, "
                                  f"center={cross_section.center}, radius={cross_section.radius:.3f}")
                
            except Exception as e:
                print(f"  Error at z={z_pos:.2f}: {e}")
        
        print(f"✅ Created {len(self.cross_sections)} cross-sections and nodes")
    
    def _create_cross_section_and_node(self, section_2d, z_pos: float, 
                                      slice_idx: int, cs_idx: int) -> CrossSection:
        """Create a cross-section and corresponding graph node."""
        # Get boundary points in 2D
        if hasattr(section_2d, 'vertices'):
            boundary_2d = section_2d.vertices
        else:
            # Fallback: sample points from the boundary
            boundary_2d = self._sample_boundary_points(section_2d)
        
        # Fit circle to boundary points
        center_2d, fitted_radius = fit_circle_to_points(boundary_2d, self.circle_fitting_method)
        
        # Calculate radius using specified method
        if self.radius_method == 'equivalent_area':
            radius = calculate_radius_from_area(section_2d.area)
        elif self.radius_method == 'average_distance':
            radius = calculate_radius_from_boundary(boundary_2d, center_2d)
        elif self.radius_method == 'fitted_circle':
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
            boundary_points=boundary_2d
        )
        
        # Create GraphNode
        node_id = f"node_{slice_idx}_{cs_idx}"
        node = GraphNode(
            id=node_id,
            z_position=z_pos,
            center=center_3d,
            radius=radius,
            cross_section=cross_section,
            slice_index=slice_idx,
            cross_section_index=cs_idx
        )
        
        # Add node to graph
        self.graph.add_graph_node(node)
        
        return cross_section
    
    def _sample_boundary_points(self, section_2d, n_points: int = 50) -> np.ndarray:
        """Sample points from the boundary of a 2D section."""
        # This is a fallback method if vertices are not directly available
        try:
            # Try to get boundary points from the section
            if hasattr(section_2d, 'boundary'):
                boundary = section_2d.boundary
                if hasattr(boundary, 'vertices'):
                    return boundary.vertices
            
            # Fallback: create a rough circular approximation
            center = np.array([0, 0])  # Will be refined by circle fitting
            radius_est = np.sqrt(section_2d.area / np.pi)
            angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
            points = center + radius_est * np.column_stack([np.cos(angles), np.sin(angles)])
            return points
            
        except:
            # Ultimate fallback
            center = np.array([0, 0])
            radius_est = np.sqrt(section_2d.area / np.pi)
            angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
            points = center + radius_est * np.column_stack([np.cos(angles), np.sin(angles)])
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
                f"Found {len(overlaps)} overlapping cross-sections:\n" + 
                "\n".join(overlap_info)
            )
        
        print("✅ No overlapping cross-sections detected")
    
    def _build_node_connectivity(self, mesh: trimesh.Trimesh):
        """Build connectivity between nodes in adjacent slices.
        
        This is the trickiest part - determining which nodes connect through 
        contiguous volume paths.
        """
        print("Building node connectivity through volume analysis...")
        
        # Group nodes by slice
        nodes_by_slice = {}
        for node in self.graph.nodes_list:
            slice_idx = node.slice_index
            if slice_idx not in nodes_by_slice:
                nodes_by_slice[slice_idx] = []
            nodes_by_slice[slice_idx].append(node)
        
        connections_found = 0
        
        # Check connectivity between adjacent slices
        for slice_idx in sorted(nodes_by_slice.keys())[:-1]:
            next_slice_idx = slice_idx + 1
            
            if next_slice_idx in nodes_by_slice:
                current_nodes = nodes_by_slice[slice_idx]
                next_nodes = nodes_by_slice[next_slice_idx]
                
                # Check each pair of nodes from adjacent slices
                for node1 in current_nodes:
                    for node2 in next_nodes:
                        if self._nodes_connected_through_volume(node1, node2, mesh):
                            # We'll create the actual edge later in _create_cylinder_edges
                            # For now, just mark them as connected in NetworkX graph
                            if not self.graph.has_edge(node1.id, node2.id):
                                self.graph.add_edge(node1.id, node2.id, temp_connection=True)
                                connections_found += 1
                                print(f"  Connected {node1.id} ↔ {node2.id}")
        
        print(f"✅ Found {connections_found} node connections")
    
    def _nodes_connected_through_volume(self, node1: GraphNode, node2: GraphNode, 
                                       mesh: trimesh.Trimesh) -> bool:
        """Determine if two nodes are connected through contiguous volume.
        
        This uses a combination of geometric and volumetric analysis.
        """
        # Basic geometric check: are the cross-sections close enough?
        center1_2d = node1.center[:2]
        center2_2d = node2.center[:2]
        distance_2d = np.linalg.norm(center1_2d - center2_2d)
        
        # If centers are too far apart, they're probably not connected
        max_distance = (node1.radius + node2.radius) * 2.0  # Generous threshold
        if distance_2d > max_distance:
            return False
        
        # Volume-based connectivity check
        # Sample points between the two cross-sections and check if they're inside the mesh
        z1, z2 = node1.z_position, node2.z_position
        
        # Create interpolated points along the potential connection
        n_samples = 10
        z_samples = np.linspace(z1, z2, n_samples)
        
        # Interpolate centers and radii
        center_samples = np.array([
            np.array([center1_2d[0], center1_2d[1], z]) + 
            (z - z1) / (z2 - z1) * np.array([center2_2d[0] - center1_2d[0], 
                                           center2_2d[1] - center1_2d[1], 0])
            for z in z_samples
        ])
        
        # Check if sample points are inside the mesh
        inside_count = 0
        for center_sample in center_samples:
            if mesh.contains([center_sample])[0]:
                inside_count += 1
        
        # Require most sample points to be inside the mesh
        connectivity_threshold = 0.7  # 70% of samples must be inside
        return (inside_count / n_samples) >= connectivity_threshold
    
    def _create_cylinder_edges(self):
        """Create GraphEdge objects for all connected node pairs."""
        print("Creating cylinder edges with metadata...")
        
        edge_count = 0
        for node1_id, node2_id in self.graph.edges():
            node1 = self.graph.get_node_by_id(node1_id)
            node2 = self.graph.get_node_by_id(node2_id)
            
            if node1 and node2:
                # Calculate edge properties
                center_line = np.array([node1.center, node2.center])
                length = np.linalg.norm(node2.center - node1.center)
                
                # Approximate cylinder volume (truncated cone)
                r1, r2 = node1.radius, node2.radius
                volume = (np.pi * length / 3) * (r1**2 + r1*r2 + r2**2)
                
                # Create GraphEdge
                edge_id = f"edge_{node1_id}_{node2_id}"
                edge = GraphEdge(
                    id=edge_id,
                    node1_id=node1_id,
                    node2_id=node2_id,
                    length=length,
                    radius1=r1,
                    radius2=r2,
                    center_line=center_line,
                    volume=volume
                )
                
                # Add to graph
                self.graph.add_graph_edge(edge)
                edge_count += 1
        
        print(f"✅ Created {edge_count} cylinder edges")


class MeshSegmenter:
    """Legacy systematic mesh segmentation using cross-sectional cuts."""

    def __init__(self):
        self.original_mesh = None
        self.slice_height = None
        self.cross_sections: List[CrossSection] = []
        self.slices: List[Dict] = []  # List of slice boundary info
        self.segments: List[Segment] = []
        self.graph = SegmentGraph()

    def segment_mesh(self, mesh: trimesh.Trimesh, slice_height: float, min_volume: float = 1e-6, return_segment_graph: bool = False):
        """
        Segment mesh into volumetric segments.

        Args:
            mesh: Input mesh (must be single closed volume)
            slice_height: Height of each slice
            min_volume: Minimum segment volume threshold
            return_segment_graph: If True, returns a SegmentGraph instance instead of segments list

        Returns:
            List[Segment] or SegmentGraph: List of segments or a SegmentGraph instance if return_segment_graph=True
        """
        # Validate slice_height
        if slice_height <= 0:
            raise ValueError(f"slice_height must be positive, got {slice_height}")
        
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
        self._build_graph()

        # Step 4: Validate conservation
        self._validate_conservation()

        # Return SegmentGraph if requested, otherwise return segments list
        if return_segment_graph:
            return self.get_segment_graph()
        else:
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
                    # Check if this is a single segment representing the entire mesh
                    # (no actual slicing occurred, so zero internal surface area is expected)
                    is_whole_mesh = (
                        len(self.cross_sections) == 0 and  # No cross-sections were created (no cuts made)
                        abs(component.volume - self.original_mesh.volume) / self.original_mesh.volume < 0.05  # Volume is ~same as original (5% tolerance)
                    )
                    if not is_whole_mesh:
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

    def _build_graph(self):
        """Step 3: Build graph based on shared cross-section boundaries."""
        # Clear any existing graph data but keep the SegmentGraph instance
        self.graph.clear()
        
        # Add all segments as nodes with their properties
        for segment in self.segments:
            self.graph.add_node(
                segment.id,
                segment=segment,
                volume=segment.volume,
                external_surface_area=segment.external_surface_area,
                internal_surface_area=segment.internal_surface_area,
                centroid=segment.centroid,
                z_min=segment.z_min,
                z_max=segment.z_max,
                slice_index=segment.slice_index,
                segment_index=segment.segment_index
            )

        connections_found = 0

        # For each cross-section, connect segments that share boundary faces
        for cross_section_idx in range(len(self.cross_sections)):
            z_position = self.cross_sections[cross_section_idx].z_position

            # Find all segments that have faces at this cross-section
            segments_at_cut = []
            for segment in self.segments:
                if self._segment_has_faces_at_z(segment, z_position):
                    segments_at_cut.append(segment)

            # Connect segments that share the same cross-sectional boundary
            if len(segments_at_cut) > 1:
                connections = self._find_shared_boundary_connections(segments_at_cut, z_position)
                for seg1_id, seg2_id in connections:
                    if not self.graph.has_edge(seg1_id, seg2_id):
                        self.graph.add_edge(seg1_id, seg2_id)
                        connections_found += 1
                        print(f"  Connected {seg1_id} ↔ {seg2_id}")

        print(f"✅ Built connectivity graph: {len(self.graph.nodes)} nodes, " f"{connections_found} edges")
        
        # Update the segments list in the SegmentGraph instance
        self.graph.segments = self.segments
        self.graph.segment_dict = {segment.id: segment for segment in self.segments}

    def _segment_has_faces_at_z(self, segment: Segment, z_position: float) -> bool:
        """Check if a segment has faces at the given z-position."""
        z_tolerance = self.slice_height * 0.01 if self.slice_height else 0.01

        # Check if any face centers are at the z-position
        face_centers = segment.mesh.triangles_center
        for center in face_centers:
            if abs(center[2] - z_position) < z_tolerance:
                return True
        return False

    def _find_shared_boundary_connections(self, segments: List[Segment], z_position: float) -> List[Tuple[str, str]]:
        """Find connections between segments that share boundary faces at z_position."""
        connections = []
        z_tolerance = self.slice_height * 0.01 if self.slice_height else 0.01

        # Get boundary faces for each segment at this z-position
        segment_boundary_faces = {}
        for segment in segments:
            boundary_faces = []
            face_centers = segment.mesh.triangles_center
            triangles = segment.mesh.triangles

            for i, center in enumerate(face_centers):
                if abs(center[2] - z_position) < z_tolerance:
                    boundary_faces.append(triangles[i])

            if boundary_faces:
                segment_boundary_faces[segment.id] = boundary_faces

        # Check for overlapping boundary faces between segment pairs
        segment_ids = list(segment_boundary_faces.keys())
        for i in range(len(segment_ids)):
            for j in range(i + 1, len(segment_ids)):
                seg1_id = segment_ids[i]
                seg2_id = segment_ids[j]

                if self._boundary_faces_overlap(segment_boundary_faces[seg1_id], segment_boundary_faces[seg2_id]):
                    connections.append((seg1_id, seg2_id))

        return connections

    def _boundary_faces_overlap(self, faces1: List[np.ndarray], faces2: List[np.ndarray]) -> bool:
        """Check if boundary faces from two segments overlap significantly."""
        if not faces1 or not faces2:
            return False

        # Use triangle centroids and check for proximity in XY plane
        tolerance = 0.5  # Spatial tolerance for overlap detection

        for face1 in faces1:
            centroid1 = np.mean(face1[:, :2], axis=0)  # XY only

            for face2 in faces2:
                centroid2 = np.mean(face2[:, :2], axis=0)  # XY only

                distance = np.linalg.norm(centroid1 - centroid2)
                if distance < tolerance:
                    return True

        return False

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
        if self.graph is None:
            return []
        return list(self.graph.get_connected_segments(segment_id))

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
                "num_components": len(self.get_connected_components()) if self.graph else 0,
                "num_edges": len(self.graph.edges) if self.graph else 0,
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

        source_face_types = source_mesh.face_attributes["face_type"]

        if len(components) == 1:
            # Single component case - direct transfer
            component = components[0]
            if not hasattr(component, "face_attributes"):
                component.face_attributes = {}

            if len(component.faces) == len(source_mesh.faces):
                component.face_attributes["face_type"] = source_face_types.copy()
                print(f"    🔄 Transferred face attributes: {len(component.faces)} faces")
            else:
                print(f"    ⚠️ Face count mismatch: source={len(source_mesh.faces)}, component={len(component.faces)}")
        else:
            # Multiple components case - map faces based on geometric matching
            print(f"    🔄 Transferring face attributes to {len(components)} components...")

            for comp_idx, component in enumerate(components):
                if not hasattr(component, "face_attributes"):
                    component.face_attributes = {}

                # Map component faces to source faces by geometric similarity
                comp_face_types = self._map_faces_to_source(component, source_mesh, source_face_types)
                component.face_attributes["face_type"] = comp_face_types

                external_count = np.sum(comp_face_types == "external")
                internal_count = np.sum(comp_face_types == "internal")
                print(f"      Component {comp_idx}: {external_count} external + {internal_count} internal faces")

    def _map_faces_to_source(
        self, component: trimesh.Trimesh, source_mesh: trimesh.Trimesh, source_face_types: np.ndarray
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
                if abs(z_coord - source_mesh.bounds[0, 2]) < 0.01 or abs(z_coord - source_mesh.bounds[1, 2]) < 0.01:
                    face_types[i] = "internal"
                else:
                    face_types[i] = "external"

        return face_types

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
        if self.graph is None:
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
                # Smaller, more reasonable node sizes based on volume
                base_size = 50  # Smaller base size
                volume_factor = min(3.0, max(0.5, segment.volume / 100))  # Scale factor
                node_sizes.append(int(base_size * volume_factor))

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

            # Draw the graph with improved styling
            # Draw edges first so they appear behind nodes
            nx.draw_networkx_edges(
                self.graph, pos, ax=ax1, edge_color="darkblue", width=2.5, alpha=0.7, style="-"
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
                self.graph, pos, ax=ax1, font_size=7, font_weight="bold", font_color="white"
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
            self.graph, pos_network, ax=ax_network, node_color=colors, node_size=node_sizes, alpha=0.8
        )

        nx.draw_networkx_edges(
            self.graph, pos_network, ax=ax_network, edge_color="gray", width=2, alpha=0.6
        )

        nx.draw_networkx_labels(self.graph, pos_network, ax=ax_network, font_size=8, font_weight="bold")

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

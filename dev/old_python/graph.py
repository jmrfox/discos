"""
Graph construction for compartment connectivity.

Builds the connectivity graph between regions across adjacent z-levels
to create the compartmental model structure.
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Optional, Set
from scipy.spatial.distance import cdist
from .core import Compartment, CompartmentGraph
from .regions import Region
import warnings


class GraphBuilder:
    """
    Constructs connectivity graphs from detected regions.
    """

    def __init__(self):
        self.compartments = {}
        self.connectivity_graph = CompartmentGraph()
        self.region_to_compartment = {}  # Maps region ID to compartment ID

    def build_compartment_graph(
        self,
        regions: List[Region],
        connection_method: str = "overlap",
        min_overlap_ratio: float = 0.1,
        max_connection_distance: float = None,
    ) -> CompartmentGraph:
        """
        Build compartment graph from regions.

        Args:
            regions: List of detected regions
            connection_method: Method for determining connections ('overlap', 'distance', 'hybrid')
            min_overlap_ratio: Minimum overlap ratio for connections
            max_connection_distance: Maximum distance for connections

        Returns:
            Constructed compartment graph
        """
        # Create compartments from regions
        self._create_compartments_from_regions(regions)

        # Build connections between adjacent z-levels
        self._build_connections(connection_method, min_overlap_ratio, max_connection_distance)

        print(f"Built graph with {len(self.compartments)} compartments")
        print(f"Total connections: {self.connectivity_graph.graph.number_of_edges()}")

        return self.connectivity_graph

    def _create_compartments_from_regions(self, regions: List[Region]):
        """Create compartments from regions."""
        self.compartments = {}
        self.region_to_compartment = {}

        for region in regions:
            # Only create compartments from outer regions (not holes)
            if not region.is_outer:
                continue

            compartment_id = f"comp_{region.slice_index}_{region.id}"

            # Estimate volume (area * slice thickness)
            # Note: This is a rough approximation - could be improved with
            # interpolation between adjacent slices
            volume = region.area * 1.0  # Assuming 1 µm slice thickness for now

            # Create centroid in 3D
            centroid_3d = np.array([region.centroid[0], region.centroid[1], region.z_level])

            compartment = Compartment(
                id=compartment_id,
                z_level=region.slice_index,
                area=region.area,
                volume=volume,
                centroid=centroid_3d,
                boundary_points=region.boundary,
            )

            self.compartments[compartment_id] = compartment
            self.connectivity_graph.add_compartment(compartment)
            self.region_to_compartment[region.id] = compartment_id

    def _build_connections(self, method: str, min_overlap_ratio: float, max_distance: float):
        """Build connections between compartments."""
        # Group compartments by z-level
        z_levels = {}
        for comp in self.compartments.values():
            z_level = comp.z_level
            if z_level not in z_levels:
                z_levels[z_level] = []
            z_levels[z_level].append(comp)

        # Sort z-levels
        sorted_z_levels = sorted(z_levels.keys())

        # Connect adjacent z-levels
        for i in range(len(sorted_z_levels) - 1):
            z1, z2 = sorted_z_levels[i], sorted_z_levels[i + 1]
            comps1 = z_levels[z1]
            comps2 = z_levels[z2]

            connections = self._find_connections_between_levels(comps1, comps2, method, min_overlap_ratio, max_distance)

            for comp1_id, comp2_id, conductance, area in connections:
                self.connectivity_graph.add_connection(comp1_id, comp2_id, conductance, area)

    def _find_connections_between_levels(
        self,
        comps1: List[Compartment],
        comps2: List[Compartment],
        method: str,
        min_overlap_ratio: float,
        max_distance: float,
    ) -> List[Tuple[str, str, float, float]]:
        """Find connections between compartments at adjacent z-levels."""
        connections = []

        if method == "overlap":
            connections = self._find_overlap_connections(comps1, comps2, min_overlap_ratio)
        elif method == "distance":
            connections = self._find_distance_connections(comps1, comps2, max_distance)
        elif method == "hybrid":
            # Try overlap first, then distance for unconnected compartments
            overlap_connections = self._find_overlap_connections(comps1, comps2, min_overlap_ratio)
            connections.extend(overlap_connections)

            # Find compartments without connections
            connected_1 = set(conn[0] for conn in overlap_connections)
            connected_2 = set(conn[1] for conn in overlap_connections)

            unconnected_1 = [c for c in comps1 if c.id not in connected_1]
            unconnected_2 = [c for c in comps2 if c.id not in connected_2]

            if unconnected_1 and unconnected_2:
                distance_connections = self._find_distance_connections(unconnected_1, unconnected_2, max_distance)
                connections.extend(distance_connections)
        else:
            raise ValueError(f"Unknown connection method: {method}")

        return connections

    def _find_overlap_connections(
        self, comps1: List[Compartment], comps2: List[Compartment], min_overlap_ratio: float
    ) -> List[Tuple[str, str, float, float]]:
        """Find connections based on boundary overlap."""
        connections = []

        for comp1 in comps1:
            for comp2 in comps2:
                overlap_area = self._compute_overlap_area(comp1.boundary_points, comp2.boundary_points)

                if overlap_area > 0:
                    # Compute overlap ratio relative to smaller compartment
                    min_area = min(comp1.area, comp2.area)
                    overlap_ratio = overlap_area / min_area

                    if overlap_ratio >= min_overlap_ratio:
                        # Compute connection properties
                        conductance = self._compute_axial_conductance(overlap_area)
                        connections.append((comp1.id, comp2.id, conductance, overlap_area))

        return connections

    def _find_distance_connections(
        self, comps1: List[Compartment], comps2: List[Compartment], max_distance: float
    ) -> List[Tuple[str, str, float, float]]:
        """Find connections based on centroid distance."""
        connections = []

        if max_distance is None:
            return connections

        # Compute distance matrix between centroids
        centroids1 = np.array([comp.centroid[:2] for comp in comps1])  # 2D projection
        centroids2 = np.array([comp.centroid[:2] for comp in comps2])

        if len(centroids1) == 0 or len(centroids2) == 0:
            return connections

        distances = cdist(centroids1, centroids2)

        # Find connections within distance threshold
        for i, comp1 in enumerate(comps1):
            for j, comp2 in enumerate(comps2):
                if distances[i, j] <= max_distance:
                    # Use distance to compute connection strength
                    strength = 1.0 - (distances[i, j] / max_distance)

                    # Estimate connection area based on compartment sizes and distance
                    estimated_area = min(comp1.area, comp2.area) * strength * 0.1
                    conductance = self._compute_axial_conductance(estimated_area)

                    connections.append((comp1.id, comp2.id, conductance, estimated_area))

        return connections

    def _compute_overlap_area(self, boundary1: np.ndarray, boundary2: np.ndarray) -> float:
        """
        Compute overlap area between two 2D boundaries.

        This is a simplified implementation - could be improved with
        proper polygon clipping algorithms like Sutherland-Hodgman.
        """
        try:
            # Simple approach: sample points from one boundary and check if they're
            # inside the other boundary

            # Sample points from boundary1
            sample_indices = np.linspace(0, len(boundary1) - 1, min(20, len(boundary1)), dtype=int)
            sample_points = boundary1[sample_indices]

            # Count points inside boundary2
            inside_count = 0
            for point in sample_points:
                if self._point_in_polygon(point, boundary2):
                    inside_count += 1

            # Estimate overlap as fraction of area
            overlap_fraction = inside_count / len(sample_points) if len(sample_points) > 0 else 0

            # Compute area of smaller boundary as reference
            area1 = self._compute_polygon_area(boundary1)
            area2 = self._compute_polygon_area(boundary2)
            reference_area = min(abs(area1), abs(area2))

            return overlap_fraction * reference_area

        except Exception as e:
            warnings.warn(f"Failed to compute overlap area: {str(e)}")
            return 0.0

    def _point_in_polygon(self, point: np.ndarray, polygon: np.ndarray) -> bool:
        """Check if point is inside polygon using ray casting."""
        x, y = point[0], point[1]
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def _compute_polygon_area(self, polygon: np.ndarray) -> float:
        """Compute polygon area using shoelace formula."""
        if len(polygon) < 3:
            return 0
        x, y = polygon[:, 0], polygon[:, 1]
        return 0.5 * (np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def _compute_axial_conductance(self, cross_sectional_area: float) -> float:
        """
        Compute axial conductance between compartments.

        Args:
            cross_sectional_area: Cross-sectional area of connection (µm²)

        Returns:
            Axial conductance (mS)
        """
        # Typical values for neuronal cytoplasm
        resistivity = 100.0  # Ω⋅cm (cytoplasmic resistivity)
        length = 1.0  # µm (assumed connection length)

        # Convert units: area in µm², length in µm, resistivity in Ω⋅cm
        # Result should be in mS
        area_cm2 = cross_sectional_area * 1e-8  # µm² to cm²
        length_cm = length * 1e-4  # µm to cm

        resistance = resistivity * length_cm / area_cm2  # Ω
        conductance = 1000.0 / resistance  # mS (1/Ω converted to mS)

        return max(conductance, 1e-6)  # Minimum conductance to avoid numerical issues

    def get_connectivity_statistics(self) -> Dict:
        """Get statistics about the connectivity graph."""
        graph = self.connectivity_graph.graph

        if graph.number_of_nodes() == 0:
            return {"error": "No nodes in graph"}

        # Basic graph statistics
        stats = {
            "num_compartments": graph.number_of_nodes(),
            "num_connections": graph.number_of_edges(),
            "avg_degree": 2 * graph.number_of_edges() / graph.number_of_nodes() if graph.number_of_nodes() > 0 else 0,
            "is_connected": nx.is_connected(graph.to_undirected()),
            "num_components": nx.number_connected_components(graph.to_undirected()),
        }

        # Degree distribution
        degrees = [graph.degree(n) for n in graph.nodes()]
        if degrees:
            stats["degree_stats"] = {
                "mean": np.mean(degrees),
                "std": np.std(degrees),
                "min": np.min(degrees),
                "max": np.max(degrees),
            }

        # Connection statistics
        if graph.number_of_edges() > 0:
            conductances = [graph[u][v]["conductance"] for u, v in graph.edges()]
            areas = [graph[u][v]["area"] for u, v in graph.edges()]

            stats["conductance_stats"] = {
                "mean": np.mean(conductances),
                "std": np.std(conductances),
                "min": np.min(conductances),
                "max": np.max(conductances),
            }

            stats["connection_area_stats"] = {
                "mean": np.mean(areas),
                "std": np.std(areas),
                "min": np.min(areas),
                "max": np.max(areas),
            }

        return stats

    def validate_graph(self) -> Dict[str, List[str]]:
        """Validate the constructed graph and return issues."""
        issues = {"errors": [], "warnings": []}

        graph = self.connectivity_graph.graph

        # Check for isolated nodes
        isolated = list(nx.isolates(graph))
        if isolated:
            issues["warnings"].append(f"Found {len(isolated)} isolated compartments: {isolated[:5]}...")

        # Check for very small conductances
        small_conductances = []
        for u, v, data in graph.edges(data=True):
            if data["conductance"] < 1e-9:
                small_conductances.append((u, v))

        if small_conductances:
            issues["warnings"].append(f"Found {len(small_conductances)} very small conductances")

        # Check connectivity
        if not nx.is_connected(graph.to_undirected()):
            issues["warnings"].append("Graph is not fully connected")

        # Check for self-loops
        self_loops = list(nx.selfloop_edges(graph))
        if self_loops:
            issues["errors"].append(f"Found {len(self_loops)} self-loops")

        return issues

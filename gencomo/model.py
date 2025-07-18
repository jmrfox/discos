"""
Multi-compartmental neurite model for GenCoMo.

This module implements a multi-compartmental neurite model that:
1. Uses the mesh submodule to process and segment a mesh
2. Creates a SegmentGraph to represent the connectivity of segments
3. Provides methods for configuring and simulating compartmental models

The workflow is:
1. Process a mesh using MeshProcessor
2. Segment the mesh using MeshSegmenter, which produces a SegmentGraph
3. Configure the model with biophysical parameters
4. Simulate the model using the Simulator
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
import warnings

from .mesh import MeshProcessor, MeshSegmenter
from .mesh.segmentation import Segment
from .ode import ODESystem
from .simulation import Simulator


class SegmentGraph:
    """
    Graph representation of segmented neurite structure.
    
    This class wraps a NetworkX graph to represent the connectivity
    between segments in a compartmental model. It provides methods
    for analyzing the graph structure and accessing properties of
    segments based on their position and connectivity.
    """
    
    def __init__(self, segments: List[Segment] = None, connectivity_graph: nx.Graph = None):
        """
        Initialize a SegmentGraph from segments and/or a connectivity graph.
        
        Args:
            segments: List of Segment objects from MeshSegmenter
            connectivity_graph: NetworkX graph representing segment connectivity
        """
        self.segments = segments or []
        self.segment_dict = {segment.id: segment for segment in self.segments}
        
        # Initialize graph
        if connectivity_graph is not None:
            self.graph = connectivity_graph.copy()
        else:
            self.graph = nx.Graph()
            
        # Add nodes and attributes if segments are provided
        if segments:
            for segment in segments:
                if segment.id not in self.graph:
                    self.graph.add_node(
                        segment.id,
                        volume=segment.volume,
                        external_surface_area=segment.external_surface_area,
                        internal_surface_area=segment.internal_surface_area,
                        centroid=segment.centroid,
                        z_min=segment.z_min,
                        z_max=segment.z_max,
                        slice_index=segment.slice_index,
                        segment_index=segment.segment_index
                    )
    
    @classmethod
    def from_mesh_segmenter(cls, segmenter: MeshSegmenter) -> 'SegmentGraph':
        """
        Create a SegmentGraph from a MeshSegmenter instance.
        
        Args:
            segmenter: MeshSegmenter instance with completed segmentation
            
        Returns:
            SegmentGraph instance
        """
        if not segmenter.segments or not segmenter.connectivity_graph:
            raise ValueError("MeshSegmenter must have completed segmentation before creating SegmentGraph")
        
        return cls(segments=segmenter.segments, connectivity_graph=segmenter.connectivity_graph)
    
    def get_segment(self, segment_id: str) -> Optional[Segment]:
        """Get a segment by its ID."""
        return self.segment_dict.get(segment_id)
    
    def get_connected_segments(self, segment_id: str) -> List[str]:
        """Get IDs of segments connected to the given segment."""
        if segment_id not in self.graph:
            return []
        return list(self.graph.neighbors(segment_id))
    
    def get_path_between_segments(self, source_id: str, target_id: str) -> List[str]:
        """Find shortest path between two segments."""
        try:
            return nx.shortest_path(self.graph, source=source_id, target=target_id)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
    
    def get_connected_components(self) -> List[List[str]]:
        """Get connected components in the graph."""
        return [list(component) for component in nx.connected_components(self.graph)]
    
    def set_segment_properties(self, segment_id: str, properties: Dict[str, Any]) -> None:
        """
        Set properties for a specific segment.
        
        Args:
            segment_id: ID of the segment to update
            properties: Dictionary of properties to set
        """
        if segment_id not in self.graph:
            raise ValueError(f"Segment {segment_id} not found in graph")
            
        for key, value in properties.items():
            self.graph.nodes[segment_id][key] = value
            
    def get_segment_properties(self, segment_id: str) -> Dict[str, Any]:
        """
        Get all properties of a specific segment.
        
        Args:
            segment_id: ID of the segment
            
        Returns:
            Dictionary of segment properties
        """
        if segment_id not in self.graph:
            raise ValueError(f"Segment {segment_id} not found in graph")
            
        return dict(self.graph.nodes[segment_id])
    
    def visualize(self, 
                 color_by: str = 'slice_index', 
                 show_plot: bool = True,
                 save_path: str = None,
                 figsize: tuple = (12, 10),
                 node_scale: float = 1000.0,
                 repulsion_strength: float = 0.1,
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
            for node, data in self.graph.nodes(data=True):
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
                        for node1 in self.graph.nodes():
                            force = np.zeros(2)
                            for node2 in self.graph.nodes():
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
            if color_by in nx.get_node_attributes(self.graph, color_by):
                property_values = list(nx.get_node_attributes(self.graph, color_by).values())
                node_colors = property_values
                cmap = cm.viridis
            else:
                # Default: color by slice_index
                slice_indices = [data.get('slice_index', 0) for _, data in self.graph.nodes(data=True)]
                node_colors = slice_indices
                cmap = cm.viridis
            
            # Node sizes based on volume with better scaling
            volumes = [data.get('volume', 1.0) for _, data in self.graph.nodes(data=True)]
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
                self.graph, 
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


class NeuriteModel:
    """
    Multi-compartmental neurite model.
    
    This class represents a multi-compartmental model of a neurite structure,
    built from a segmented mesh. It provides methods for configuring the model
    with biophysical parameters and simulating its behavior.
    """
    
    def __init__(self, name: str = "NeuriteModel"):
        """
        Initialize a new NeuriteModel.
        
        Args:
            name: Name of the model
        """
        self.name = name
        self.segment_graph = None
        self.ode_system = None
        self.simulator = None
        self.biophysical_parameters = {}
        
    def build_from_mesh(self, 
                       mesh_path: str, 
                       slice_height: float = 1.0,
                       min_volume: float = 1e-6) -> 'NeuriteModel':
        """
        Build model from a mesh file.
        
        Args:
            mesh_path: Path to mesh file
            slice_height: Height of each slice for segmentation
            min_volume: Minimum segment volume threshold
            
        Returns:
            Self for method chaining
        """
        # Process mesh
        processor = MeshProcessor()
        mesh = processor.load_mesh(mesh_path)
        mesh = processor.preprocess_mesh(mesh)
        
        # Segment mesh and get segment graph
        segmenter = MeshSegmenter()
        segments = segmenter.segment_mesh(mesh, slice_height, min_volume)
        self.segment_graph = segmenter.get_segment_graph()
        
        return self
    
    def build_from_segmenter(self, segmenter: MeshSegmenter) -> 'NeuriteModel':
        """
        Build model from a MeshSegmenter instance.
        
        Args:
            segmenter: MeshSegmenter instance with completed segmentation
            
        Returns:
            Self for method chaining
        """
        self.segment_graph = segmenter.get_segment_graph()
        return self
    
    def set_biophysical_parameters(self, parameters: Dict[str, Any]) -> 'NeuriteModel':
        """
        Set biophysical parameters for the model.
        
        Args:
            parameters: Dictionary of parameter names and values
            
        Returns:
            Self for method chaining
        """
        self.biophysical_parameters.update(parameters)
        return self
    
    def create_ode_system(self) -> ODESystem:
        """
        Create an ODESystem for the model.
        
        Returns:
            ODESystem instance
        """
        if not self.segment_graph:
            raise ValueError("Model must be built before creating ODE system")
        
        # Create ODE system based on segment graph
        # This is a placeholder implementation
        self.ode_system = ODESystem()
        
        # Configure ODE system based on segment graph and biophysical parameters
        # (Implementation details would depend on the ODESystem interface)
        
        return self.ode_system
    
    def create_simulator(self) -> Simulator:
        """
        Create a Simulator for the model.
        
        Returns:
            Simulator instance
        """
        if not self.ode_system:
            self.create_ode_system()
            
        self.simulator = Simulator(self.ode_system)
        return self.simulator
    
    def run_simulation(self, 
                      t_start: float = 0.0,
                      t_end: float = 100.0,
                      dt: float = 0.1,
                      **kwargs) -> Dict[str, np.ndarray]:
        """
        Run a simulation of the model.
        
        Args:
            t_start: Start time
            t_end: End time
            dt: Time step
            **kwargs: Additional arguments for the simulator
            
        Returns:
            Dictionary of simulation results
        """
        if not self.simulator:
            self.create_simulator()
            
        results = self.simulator.run(t_start=t_start, t_end=t_end, dt=dt, **kwargs)
        return results
    
    def visualize_model(self, 
                       color_by: str = 'slice_index',
                       show_plot: bool = True,
                       save_path: str = None) -> Any:
        """
        Visualize the model.
        
        Args:
            color_by: Property to color nodes by ('slice_index', 'volume', or any other node attribute)
            show_plot: Whether to display the plot
            save_path: Path to save the plot
            
        Returns:
            Figure object
        """
        if not self.segment_graph:
            raise ValueError("Model must be built before visualization")
            
        return self.segment_graph.visualize(
            color_by=color_by,
            show_plot=show_plot,
            save_path=save_path
        )

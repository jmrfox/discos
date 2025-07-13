"""
Core data structures for GenCoMo compartmental modeling.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import networkx as nx


@dataclass
class Compartment:
    """
    Represents a single compartment in the neuronal model.

    Attributes:
        id: Unique identifier for the compartment
        z_level: Z-axis level this compartment belongs to
        area: Surface area of the compartment membrane (µm²)
        volume: Volume of the compartment (µm³)
        centroid: 3D coordinates of the compartment centroid (µm)
        boundary_points: Points defining the compartment boundary
        membrane_potential: Current membrane potential (mV)
        capacitance: Membrane capacitance (µF)
        conductances: Dictionary of ionic conductances (mS)
        currents: Dictionary of ionic currents (nA)
    """

    id: str
    z_level: int
    area: float
    volume: float
    centroid: np.ndarray
    boundary_points: np.ndarray
    membrane_potential: float = -70.0  # mV
    capacitance: float = 1.0  # µF/cm²
    conductances: Dict[str, float] = None
    currents: Dict[str, float] = None

    def __post_init__(self):
        if self.conductances is None:
            self.conductances = {}
        if self.currents is None:
            self.currents = {}


class CompartmentGraph:
    """
    Graph representation of compartment connectivity.
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.compartments: Dict[str, Compartment] = {}

    def add_compartment(self, compartment: Compartment):
        """Add a compartment to the graph."""
        self.compartments[compartment.id] = compartment
        self.graph.add_node(compartment.id, compartment=compartment)

    def add_connection(self, comp1_id: str, comp2_id: str, conductance: float, area: float):
        """
        Add a bidirectional connection between two compartments.

        Args:
            comp1_id: ID of first compartment
            comp2_id: ID of second compartment
            conductance: Axial conductance between compartments (mS)
            area: Cross-sectional area of connection (µm²)
        """
        # Add bidirectional edges for electrical connectivity
        self.graph.add_edge(comp1_id, comp2_id, conductance=conductance, area=area)
        self.graph.add_edge(comp2_id, comp1_id, conductance=conductance, area=area)

    def get_neighbors(self, comp_id: str) -> List[str]:
        """Get neighboring compartment IDs."""
        return list(self.graph.neighbors(comp_id))

    def get_connection_conductance(self, comp1_id: str, comp2_id: str) -> float:
        """Get axial conductance between two compartments."""
        if self.graph.has_edge(comp1_id, comp2_id):
            return self.graph[comp1_id][comp2_id]["conductance"]
        return 0.0

    def get_compartments_by_z_level(self, z_level: int) -> List[Compartment]:
        """Get all compartments at a specific z-level."""
        return [comp for comp in self.compartments.values() if comp.z_level == z_level]


class Neuron:
    """
    Complete neuron model with mesh-based compartments.
    """

    def __init__(self, name: str = "neuron"):
        self.name = name
        self.compartment_graph = CompartmentGraph()
        self.mesh = None
        self.z_slices = None
        self.simulation_params = {
            "dt": 0.025,  # ms
            "duration": 100.0,  # ms
            "temperature": 6.3,  # °C (for rate constants)
        }

    def set_mesh(self, mesh):
        """Set the neuronal mesh."""
        self.mesh = mesh

    def get_compartment(self, comp_id: str) -> Optional[Compartment]:
        """Get compartment by ID."""
        return self.compartment_graph.compartments.get(comp_id)

    def get_all_compartments(self) -> List[Compartment]:
        """Get all compartments."""
        return list(self.compartment_graph.compartments.values())

    def get_z_levels(self) -> List[int]:
        """Get all z-levels with compartments."""
        return sorted(list(set(comp.z_level for comp in self.get_all_compartments())))

    def set_simulation_params(self, **params):
        """Set simulation parameters."""
        self.simulation_params.update(params)

    def get_membrane_potentials(self) -> Dict[str, float]:
        """Get current membrane potentials for all compartments."""
        return {comp_id: comp.membrane_potential for comp_id, comp in self.compartment_graph.compartments.items()}

    def set_membrane_potential(self, comp_id: str, voltage: float):
        """Set membrane potential for a specific compartment."""
        if comp_id in self.compartment_graph.compartments:
            self.compartment_graph.compartments[comp_id].membrane_potential = voltage

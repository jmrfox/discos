"""
Tests for core GenCoMo functionality.
"""

import pytest
import numpy as np
from gencomo.core import Compartment, CompartmentGraph, Neuron


class TestCompartment:
    """Test Compartment class."""

    def test_compartment_creation(self):
        """Test basic compartment creation."""
        centroid = np.array([1.0, 2.0, 3.0])
        boundary = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

        comp = Compartment(
            id="test_comp", z_level=1, area=10.0, volume=5.0, centroid=centroid, boundary_points=boundary
        )

        assert comp.id == "test_comp"
        assert comp.z_level == 1
        assert comp.area == 10.0
        assert comp.volume == 5.0
        assert np.array_equal(comp.centroid, centroid)
        assert np.array_equal(comp.boundary_points, boundary)
        assert comp.membrane_potential == -70.0  # Default value

    def test_compartment_defaults(self):
        """Test compartment default values."""
        comp = Compartment(
            id="test", z_level=0, area=1.0, volume=1.0, centroid=np.array([0, 0, 0]), boundary_points=np.array([[0, 0]])
        )

        assert comp.conductances == {}
        assert comp.currents == {}
        assert comp.capacitance == 1.0


class TestCompartmentGraph:
    """Test CompartmentGraph class."""

    def test_graph_creation(self):
        """Test basic graph creation."""
        graph = CompartmentGraph()
        assert len(graph.compartments) == 0
        assert graph.graph.number_of_nodes() == 0

    def test_add_compartment(self):
        """Test adding compartments to graph."""
        graph = CompartmentGraph()

        comp = Compartment(
            id="comp1",
            z_level=0,
            area=5.0,
            volume=2.0,
            centroid=np.array([0, 0, 0]),
            boundary_points=np.array([[0, 0]]),
        )

        graph.add_compartment(comp)

        assert len(graph.compartments) == 1
        assert "comp1" in graph.compartments
        assert graph.graph.number_of_nodes() == 1

    def test_add_connection(self):
        """Test adding connections between compartments."""
        graph = CompartmentGraph()

        # Add two compartments
        comp1 = Compartment("comp1", 0, 5.0, 2.0, np.array([0, 0, 0]), np.array([[0, 0]]))
        comp2 = Compartment("comp2", 1, 5.0, 2.0, np.array([0, 0, 1]), np.array([[0, 0]]))

        graph.add_compartment(comp1)
        graph.add_compartment(comp2)

        # Add connection
        graph.add_connection("comp1", "comp2", conductance=0.1, area=1.0)

        assert graph.graph.number_of_edges() == 1
        assert graph.get_connection_conductance("comp1", "comp2") == 0.1

    def test_get_neighbors(self):
        """Test getting neighboring compartments."""
        graph = CompartmentGraph()

        # Create a small network
        for i in range(3):
            comp = Compartment(f"comp{i}", i, 5.0, 2.0, np.array([0, 0, i]), np.array([[0, 0]]))
            graph.add_compartment(comp)

        # Connect comp0 -> comp1 -> comp2
        graph.add_connection("comp0", "comp1", 0.1, 1.0)
        graph.add_connection("comp1", "comp2", 0.1, 1.0)

        neighbors_0 = graph.get_neighbors("comp0")
        neighbors_1 = graph.get_neighbors("comp1")
        neighbors_2 = graph.get_neighbors("comp2")

        assert neighbors_0 == ["comp1"]
        assert neighbors_1 == ["comp2"]
        assert neighbors_2 == []

    def test_get_compartments_by_z_level(self):
        """Test getting compartments by z-level."""
        graph = CompartmentGraph()

        # Add compartments at different z-levels
        comp1 = Compartment("comp1", 0, 5.0, 2.0, np.array([0, 0, 0]), np.array([[0, 0]]))
        comp2 = Compartment("comp2", 0, 5.0, 2.0, np.array([1, 0, 0]), np.array([[0, 0]]))
        comp3 = Compartment("comp3", 1, 5.0, 2.0, np.array([0, 0, 1]), np.array([[0, 0]]))

        graph.add_compartment(comp1)
        graph.add_compartment(comp2)
        graph.add_compartment(comp3)

        z0_comps = graph.get_compartments_by_z_level(0)
        z1_comps = graph.get_compartments_by_z_level(1)

        assert len(z0_comps) == 2
        assert len(z1_comps) == 1
        assert all(comp.z_level == 0 for comp in z0_comps)
        assert all(comp.z_level == 1 for comp in z1_comps)


class TestNeuron:
    """Test Neuron class."""

    def test_neuron_creation(self):
        """Test basic neuron creation."""
        neuron = Neuron("test_neuron")

        assert neuron.name == "test_neuron"
        assert isinstance(neuron.compartment_graph, CompartmentGraph)
        assert neuron.mesh is None
        assert neuron.z_slices is None

    def test_neuron_default_params(self):
        """Test default simulation parameters."""
        neuron = Neuron()

        assert neuron.simulation_params["dt"] == 0.025
        assert neuron.simulation_params["duration"] == 100.0
        assert neuron.simulation_params["temperature"] == 6.3

    def test_set_simulation_params(self):
        """Test setting simulation parameters."""
        neuron = Neuron()

        neuron.set_simulation_params(dt=0.01, duration=200.0, new_param=42)

        assert neuron.simulation_params["dt"] == 0.01
        assert neuron.simulation_params["duration"] == 200.0
        assert neuron.simulation_params["temperature"] == 6.3  # Unchanged
        assert neuron.simulation_params["new_param"] == 42

    def test_compartment_operations(self):
        """Test compartment-related operations."""
        neuron = Neuron()

        # Add a compartment
        comp = Compartment("test_comp", 0, 5.0, 2.0, np.array([0, 0, 0]), np.array([[0, 0]]))
        neuron.compartment_graph.add_compartment(comp)

        # Test getting compartment
        retrieved = neuron.get_compartment("test_comp")
        assert retrieved is not None
        assert retrieved.id == "test_comp"

        # Test getting all compartments
        all_comps = neuron.get_all_compartments()
        assert len(all_comps) == 1
        assert all_comps[0].id == "test_comp"

        # Test getting z-levels
        z_levels = neuron.get_z_levels()
        assert z_levels == [0]

    def test_membrane_potential_operations(self):
        """Test membrane potential get/set operations."""
        neuron = Neuron()

        comp = Compartment("test_comp", 0, 5.0, 2.0, np.array([0, 0, 0]), np.array([[0, 0]]))
        neuron.compartment_graph.add_compartment(comp)

        # Test getting membrane potentials
        potentials = neuron.get_membrane_potentials()
        assert potentials["test_comp"] == -70.0  # Default value

        # Test setting membrane potential
        neuron.set_membrane_potential("test_comp", -55.0)
        potentials = neuron.get_membrane_potentials()
        assert potentials["test_comp"] == -55.0

        # Test setting for non-existent compartment
        neuron.set_membrane_potential("nonexistent", -60.0)
        # Should not raise error, just do nothing
        assert len(potentials) == 1


if __name__ == "__main__":
    pytest.main([__file__])

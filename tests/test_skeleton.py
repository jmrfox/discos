"""
Unit tests for skeletonization utilities in `discos/skeleton.py`.

Covers segmentation of:
- Cylinder
- Torus
- Idealized neuron
"""

import os
import sys

# Ensure local package import (when running tests from repo)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from discos import create_cylinder_mesh, create_demo_neuron_mesh, create_torus_mesh
from discos.skeleton import SkeletonGraph, skeletonize

# ---- Fixtures ---------------------------------------------------------------


@pytest.fixture
def cylinder_mesh():
    # Simple z-aligned cylinder centered at origin
    return create_cylinder_mesh(length=40.0, radius=3.0, resolution=24, axis="z")


@pytest.fixture
def torus_mesh():
    # Default torus has axis of symmetry along x after rotation in demo
    # Keep moderate segment counts for speed
    return create_torus_mesh(
        major_radius=20.0,
        minor_radius=5.0,
        major_segments=20,
        minor_segments=12,
        axis="x",
    )


@pytest.fixture
def neuron_mesh():
    # Idealized neuron with soma + dendrites + axon
    return create_demo_neuron_mesh(
        soma_radius=8.0,
        dendrite_length=40.0,
        dendrite_radius=2.0,
        axon_length=60.0,
        axon_radius=1.5,
        num_dendrites=4,
        dendrite_angle=30.0,
    )


# ---- Helpers ----------------------------------------------------------------


def _sum_edge_volumes(G: SkeletonGraph) -> float:
    return float(sum(data.get("volume", 0.0) for _, _, data in G.edges(data=True)))


def _assert_edges_are_between_adjacent_slices(G: SkeletonGraph):
    for u, v, data in G.edges(data=True):
        si_u = G.nodes[u]["slice_index"]
        si_v = G.nodes[v]["slice_index"]
        assert (
            abs(si_u - si_v) == 1
        ), f"Edge {u}-{v} connects non-adjacent slices {si_u} and {si_v}"


# ---- Tests: Cylinder --------------------------------------------------------


def test_cylinder_skeleton_basic(cylinder_mesh):
    n_slices = 4
    G = skeletonize(cylinder_mesh, n_slices=n_slices, validate=True)

    # Graph shape expectations
    assert isinstance(G, SkeletonGraph)
    assert G.number_of_nodes() == n_slices + 1
    assert G.number_of_edges() == n_slices

    # Each node should be near x=y=0 and radius near true radius
    true_r = cylinder_mesh.metadata["radius"]
    for node, attrs in G.nodes(data=True):
        cx, cy, _ = attrs["center"]
        r = attrs["radius"]
        assert np.allclose([cx, cy], [0.0, 0.0], atol=1e-2)
        assert abs(r - true_r) / true_r < 0.1  # within 10%

    # z should be monotonic across slice indices
    z_by_slice = [
        np.mean(
            [
                attrs["z"]
                for n, attrs in G.nodes(data=True)
                if attrs["slice_index"] == si
            ]
        )
        for si in range(n_slices + 1)
    ]
    assert all(z_by_slice[i] < z_by_slice[i + 1] for i in range(len(z_by_slice) - 1))

    # Edge volumes positive and adjacency rule respected
    assert _sum_edge_volumes(G) > 0
    _assert_edges_are_between_adjacent_slices(G)


# ---- Tests: Torus -----------------------------------------------------------


def test_torus_skeleton_topology(torus_mesh):
    n_slices = 12
    G = skeletonize(torus_mesh, n_slices=n_slices, validate=True)

    assert isinstance(G, SkeletonGraph)
    assert G.number_of_nodes() > n_slices + 1  # expect multiple nodes per slice overall
    assert G.number_of_edges() > n_slices

    # Single connected component expected for a single torus body
    import networkx as nx

    assert nx.is_connected(G)

    # Expect at least one cycle for toroidal topology
    cycles = nx.cycle_basis(G)
    assert len(cycles) >= 1

    # Volumes positive and adjacency respected
    assert _sum_edge_volumes(G) > 0
    _assert_edges_are_between_adjacent_slices(G)


# ---- Tests: Idealized Neuron ------------------------------------------------


def test_neuron_skeleton_branching(neuron_mesh):
    n_slices = 16
    # Skip if mesh isn't watertight (boolean union backends may be unavailable)
    if not getattr(neuron_mesh, "is_watertight", False):
        pytest.skip("Neuron mesh not watertight; skipping skeletonization test")

    G = skeletonize(neuron_mesh, n_slices=n_slices, validate=True)

    assert isinstance(G, SkeletonGraph)
    assert G.number_of_nodes() > 0
    assert G.number_of_edges() > 0

    # Expect branching: some node with degree >= 3
    degrees = dict(G.degree())
    assert max(degrees.values()) >= 3

    # Expect that some slice has multiple cross-sections (nodes)
    from collections import Counter

    slices = [attrs["slice_index"] for _, attrs in G.nodes(data=True)]
    counts = Counter(slices)
    assert any(c > 1 for c in counts.values())

    # Edge rule: adjacent slices only
    _assert_edges_are_between_adjacent_slices(G)

    # Volumes positive
    assert _sum_edge_volumes(G) > 0

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
from discos.skeleton import (
    CrossSection,
    Segment,
    SkeletonGraph,
    _check_overlap_in_slice,
    _fit_circle_algebraic,
    _line_of_sight_inside,
    _polygon_area_and_centroid,
    skeletonize,
)

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


# ---- Tests: Helper functions and dataclasses ---------------------------------


def test_polygon_area_and_centroid_square():
    # Unit square with centroid at (0.5, 0.5)
    pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    area, ctr = _polygon_area_and_centroid(pts)
    assert np.isclose(area, 1.0)
    assert np.allclose(ctr, [0.5, 0.5])


def test_fit_circle_algebraic_perfect_circle():
    # Points on a circle centered at (2, -1) with radius 3
    rng = np.random.default_rng(0)
    angles = np.linspace(0, 2 * np.pi, 50, endpoint=False)
    pts = np.stack([2 + 3 * np.cos(angles), -1 + 3 * np.sin(angles)], axis=1)
    # small noise
    pts += 1e-6 * rng.standard_normal(size=pts.shape)
    center, r = _fit_circle_algebraic(pts)
    assert np.allclose(center, [2, -1], atol=1e-3)
    assert np.isclose(r, 3.0, atol=1e-3)


def test_check_overlap_in_slice_raises_on_overlap():
    # Two cross-sections at same z with overlapping radii
    z = 0.0
    cs1 = CrossSection(
        z=z,
        area=np.pi * 1.0**2,
        center=np.array([0.0, 0.0, z], dtype=float),
        radius=1.0,
        boundary_2d=None,
        slice_index=0,
        index_within_slice=0,
    )
    cs2 = CrossSection(
        z=z,
        area=np.pi * 1.0**2,
        center=np.array([1.2, 0.0, z], dtype=float),  # centers 1.2 apart
        radius=1.0,
        boundary_2d=None,
        slice_index=0,
        index_within_slice=1,
    )
    with pytest.raises(ValueError):
        _check_overlap_in_slice([cs1, cs2], tolerance=1e-6)


def test_check_overlap_in_slice_allows_separated():
    z = 0.0
    cs1 = CrossSection(
        z=z,
        area=np.pi * 1.0**2,
        center=np.array([0.0, 0.0, z], dtype=float),
        radius=1.0,
        boundary_2d=None,
        slice_index=0,
        index_within_slice=0,
    )
    cs2 = CrossSection(
        z=z,
        area=np.pi * 1.0**2,
        center=np.array([2.5, 0.0, z], dtype=float),  # far enough apart
        radius=1.0,
        boundary_2d=None,
        slice_index=0,
        index_within_slice=1,
    )
    # Should not raise
    _check_overlap_in_slice([cs1, cs2], tolerance=1e-6)


def test_line_of_sight_inside_true_for_cylinder_axis(cylinder_mesh):
    # For a z-aligned cylinder: pick two points along the axis, strictly inside
    zmin, zmax = float(cylinder_mesh.bounds[0, 2]), float(cylinder_mesh.bounds[1, 2])
    p0 = np.array([0.0, 0.0, zmin + 0.25 * (zmax - zmin)], dtype=float)
    p1 = np.array([0.0, 0.0, zmin + 0.75 * (zmax - zmin)], dtype=float)
    assert _line_of_sight_inside(cylinder_mesh, p0, p1, n_samples=10) is True


def test_line_of_sight_inside_false_outside(cylinder_mesh):
    # One point inside near center, another far outside radially
    zc = float(np.mean([cylinder_mesh.bounds[0, 2], cylinder_mesh.bounds[1, 2]]))
    p0 = np.array([0.0, 0.0, zc], dtype=float)
    p1 = np.array([1e3, 0.0, zc], dtype=float)
    assert _line_of_sight_inside(cylinder_mesh, p0, p1, n_samples=10) is False


def test_dataclasses_construction_and_fields():
    ctr = np.array([1.0, 2.0, 3.0], dtype=float)
    cs = CrossSection(
        z=3.0,
        area=12.34,
        center=ctr,
        radius=5.6,
        boundary_2d=None,
        slice_index=7,
        index_within_slice=2,
    )
    assert cs.z == 3.0
    assert np.allclose(cs.center, ctr)
    assert cs.radius == 5.6
    assert cs.slice_index == 7 and cs.index_within_slice == 2

    seg = Segment(
        u_id="u",
        v_id="v",
        length=10.0,
        r1=1.0,
        r2=2.0,
        volume=20.0,
        center_line=np.vstack([np.zeros(3), np.ones(3)]),
    )
    assert seg.u_id == "u" and seg.v_id == "v"
    assert np.isclose(seg.length, 10.0) and np.isclose(seg.r1, 1.0) and np.isclose(seg.r2, 2.0)
    assert np.isclose(seg.volume, 20.0)
    assert seg.center_line.shape == (2, 3)


def test_skeletongraph_from_mesh_node_and_edge_attrs(cylinder_mesh):
    G = skeletonize(cylinder_mesh, n_slices=5, validate=False)
    # Node attributes present and typed
    for _, attrs in G.nodes(data=True):
        assert "center" in attrs and isinstance(attrs["center"], np.ndarray)
        assert "radius" in attrs and isinstance(attrs["radius"], float)
        assert "z" in attrs and isinstance(attrs["z"], float)
        assert "slice_index" in attrs and isinstance(attrs["slice_index"], int)
        assert "index_within_slice" in attrs and isinstance(attrs["index_within_slice"], int)
        assert "area" in attrs and isinstance(attrs["area"], float)
    # Edge attributes present and typed
    for _, _, data in G.edges(data=True):
        assert "length" in data and isinstance(data["length"], float)
        assert "r1" in data and isinstance(data["r1"], float)
        assert "r2" in data and isinstance(data["r2"], float)
        assert "volume" in data and isinstance(data["volume"], float)
        assert "center_line" in data and isinstance(data["center_line"], np.ndarray)
        assert data["center_line"].shape == (2, 3)

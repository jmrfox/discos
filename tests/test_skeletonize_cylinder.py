import numpy as np
import pytest
import trimesh

from discos.skeleton import skeletonize


def test_cylinder_connectivity_and_boundaries():
    # Cylinder: radius 1, height 4 centered at z=0 => z in [-2, 2]
    mesh = trimesh.creation.cylinder(radius=1.0, height=4.0)

    n_slices = 4
    skel = skeletonize(mesh, n_slices=n_slices, validate_volume=True, verbose=False)

    G = skel.to_networkx()

    # Expected nodes: n_cuts + 2 terminals = (n_slices-1) + 2
    expected_nodes = (n_slices - 1) + 2
    assert G.number_of_nodes() == expected_nodes

    # Expected edges: one per band for a cylinder = n_slices
    assert G.number_of_edges() == n_slices

    # Check there is an edge whose upper z matches the top bounding plane
    zs = [attrs.get("z") for _, attrs in G.nodes(data=True) if "z" in attrs]
    assert zs, "Nodes should carry 'z' attribute"
    zmax = float(max(zs))

    has_top_edge = any(np.isclose(eattrs.get("z_upper", -np.inf), zmax) for _, _, eattrs in G.edges(data=True))
    assert has_top_edge, "No edge found that reaches the top bounding plane (z_upper == zmax)"

    # Each node should have boundary_2d attached for plotting
    missing_boundary = [n for n, a in G.nodes(data=True) if a.get("boundary_2d") is None]
    assert not missing_boundary, f"Nodes missing boundary_2d: {missing_boundary}"

    # Basic sanity on boundary array shapes
    for n, a in G.nodes(data=True):
        b = a.get("boundary_2d")
        assert isinstance(b, np.ndarray) and b.ndim == 2 and b.shape[1] == 2 and b.shape[0] >= 2

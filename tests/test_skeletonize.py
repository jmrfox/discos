import numpy as np
import pytest
import networkx as nx
from discos.skeleton import skeletonize
from discos.demo import create_cylinder_mesh, create_torus_mesh


SLICE_COUNTS = [3, 4, 5, 6, 7, 8, 9, 10, 16, 32, 64]


def _cycle_count(G: nx.Graph) -> int:
    # Cyclomatic number equals size of a cycle basis for undirected graphs
    return len(nx.cycle_basis(G))


@pytest.mark.parametrize("n_slices", SLICE_COUNTS)
def test_cylinder_single_component_no_cycles(n_slices: int):
    # Simple solid cylinder centered at origin with height spanning z in [-2, 2]
    mesh = create_cylinder_mesh(length=4.0, radius=1.0, resolution=32, axis="z")

    skel = skeletonize(mesh, n_slices=n_slices, validate_volume=True, verbose=False)
    G = skel.to_networkx()

    # Single connected component for all slice counts
    assert nx.number_connected_components(G) == 1

    # Cylinder should be a tree (no cycles)
    assert _cycle_count(G) == 0


@pytest.mark.parametrize("n_slices", SLICE_COUNTS)
def test_torus_single_component_one_cycle(n_slices: int):
    # Donut (torus) around Z; expect exactly one cycle in the skeleton graph
    # Choose radii to be well-resolved
    torus = create_torus_mesh(
        major_radius=2.0,
        minor_radius=0.6,
        major_segments=128,
        minor_segments=64,
        axis="z",
    )

    skel = skeletonize(torus, n_slices=n_slices, validate_volume=True, verbose=False)
    G = skel.to_networkx()

    # Single connected component across all slice counts
    assert nx.number_connected_components(G) == 1

    # A torus should yield a single topological cycle
    assert _cycle_count(G) == 1

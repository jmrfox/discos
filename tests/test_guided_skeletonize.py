import networkx as nx
import numpy as np
import pytest

from discos import create_cylinder_mesh, create_torus_mesh, PolylinesSkeleton
from discos.skeleton import skeletonize, PolylineGuidanceConfig


def _cycle_count(G: nx.Graph) -> int:
    return len(nx.cycle_basis(G))


def _vertical_polyline_for_mesh(mesh, n: int = 41, x: float = 0.0, y: float = 0.0) -> np.ndarray:
    V = np.asarray(mesh.vertices, dtype=float)
    zmin = float(V[:, 2].min())
    zmax = float(V[:, 2].max())
    dz = max(zmax - zmin, 1e-9)
    # Extend slightly beyond mesh bounds to ensure crossings at bounding planes
    z0 = zmin - 0.1 * dz
    z1 = zmax + 0.1 * dz
    zline = np.linspace(z0, z1, int(n))
    xline = np.full_like(zline, float(x))
    yline = np.full_like(zline, float(y))
    return np.column_stack([xline, yline, zline]).astype(float)


@pytest.mark.parametrize("n_slices", [5, 9])
def test_cylinder_guided_edges_marked_guided(n_slices: int):
    # Simple cylinder centered at origin
    mesh = create_cylinder_mesh(length=4.0, radius=1.0, resolution=24)

    # Polyline along the cylinder axis (x=y=0) crossing all slice planes
    pl = _vertical_polyline_for_mesh(mesh, n=61, x=0.0, y=0.0)
    pls = PolylinesSkeleton([pl])

    # Enable guidance with default tolerances
    gcfg = PolylineGuidanceConfig(use_guidance=True)

    skel = skeletonize(
        mesh,
        n_slices=n_slices,
        polylines=pls,
        guidance=gcfg,
        validate_volume=True,
        verbose=False,
    )
    G = skel.to_networkx()

    # One connected component, cylinder should be acyclic
    assert nx.number_connected_components(G) == 1
    assert _cycle_count(G) == 0

    # All edges should be guided and have supporting polyline ids
    for u, v, data in G.edges(data=True):
        assert data.get("chosen_by") == "guided"
        supp = data.get("polyline_support")
        assert isinstance(supp, list) and len(supp) > 0


@pytest.mark.parametrize("n_slices", [5, 9])
def test_torus_guidance_preserves_topology_and_support(n_slices: int):
    # Torus; choose radii to be well-resolved. Use default axis from demo to
    # match prior topology tests and avoid fragmentation under guidance.
    torus = create_torus_mesh(
        major_radius=np.pi,
        minor_radius=np.pi / 3,
        major_segments=24,
        minor_segments=12,
    )

    # Vertical polyline through origin (torus cross-section centroids lie at origin)
    pl = _vertical_polyline_for_mesh(torus, n=81, x=0.0, y=0.0)
    pls = PolylinesSkeleton([pl])

    gcfg = PolylineGuidanceConfig(use_guidance=True)

    skel = skeletonize(
        torus,
        n_slices=n_slices,
        polylines=pls,
        guidance=gcfg,
        validate_volume=True,
        verbose=False,
    )
    G = skel.to_networkx()

    # One connected component; torus yields exactly one cycle
    assert nx.number_connected_components(G) == 1
    assert _cycle_count(G) == 1

    # If any edges were selected via guidance, they must carry non-empty support
    guided_edges = [(u, v, d) for u, v, d in G.edges(data=True) if d.get("chosen_by") == "guided"]
    for _, _, data in guided_edges:
        supp = data.get("polyline_support")
        assert isinstance(supp, list) and len(supp) > 0

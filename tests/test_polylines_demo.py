import os
from pathlib import Path

import numpy as np
import pytest
import trimesh

from discos import PolylinesSkeleton, MeshManager, data_path


def _load_mesh_if_exists(rel_path: str) -> trimesh.Trimesh:
    """Load a mesh via MeshManager if the file exists; otherwise skip the test."""
    p = data_path(rel_path)
    if not Path(p).exists():
        pytest.skip(f"Demo mesh not found: {p}")
    mm = MeshManager(verbose=False)
    return mm.load_mesh(str(p))


def test_cylinder_polylines_snap_reduces_distance():
    # Files
    mesh_rel = os.path.join("mesh", "demo", "cylinder.obj")
    poly_rel = os.path.join("polylines", "cylinder.polylines.txt")

    # Load mesh (skip if missing) and polylines (must be in repo)
    mesh = _load_mesh_if_exists(mesh_rel)
    poly_path = data_path(poly_rel)
    pls = PolylinesSkeleton.from_txt(str(poly_path))

    # Offset polylines to ensure a measurable distance to surface
    offset = np.array([0.0, 0.0, 2.0], dtype=float)
    pls.translate(offset)

    # Measure mean distance before
    from trimesh.proximity import closest_point  # type: ignore

    P_before = np.vstack(pls.polylines)
    _, dist_before, _ = closest_point(mesh, P_before)
    mean_before = float(np.mean(dist_before))

    # Snap to surface (project all points)
    moved, mean_move = pls.snap_to_mesh_surface(mesh, project_outside_only=False)
    assert moved > 0
    assert mean_move >= 0.0

    # Measure mean distance after; should be near zero and <= before
    P_after = np.vstack(pls.polylines)
    _, dist_after, _ = closest_point(mesh, P_after)
    mean_after = float(np.mean(dist_after))

    assert mean_after <= mean_before + 1e-6
    assert mean_after < 1e-3


def test_torus_polylines_copy_transforms_from_mesh_composite():
    # Files
    mesh_rel = os.path.join("mesh", "demo", "torus.obj")
    poly_rel = os.path.join("polylines", "torus.polylines.txt")

    # Load mesh and polylines
    mesh = _load_mesh_if_exists(mesh_rel)
    poly_path = data_path(poly_rel)
    pls = PolylinesSkeleton.from_txt(str(poly_path))

    # Prepare a MeshManager and apply transforms
    mm = MeshManager(mesh=mesh, verbose=False)
    mm.center_mesh("centroid")
    mm.scale_mesh(0.75)

    # Record original polylines for manual transform
    original = [pl.copy() for pl in pls.polylines]

    # Copy composite transform and compare to manual application
    pls.copy_transforms_from_mesh(mm, mode="composite")
    M = mm.get_composite_matrix()

    def apply(M: np.ndarray, points: np.ndarray) -> np.ndarray:
        ones = np.ones((points.shape[0], 1), dtype=float)
        vh = np.hstack([points, ones])
        v2 = (M @ vh.T).T[:, :3]
        return v2

    manual = [apply(M, pl) for pl in original]
    for a, b in zip(pls.polylines, manual):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)

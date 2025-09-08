import os
import numpy as np
import trimesh

from discos import PolylinesSkeleton, MeshManager, data_path


def load_ts2_polylines() -> PolylinesSkeleton:
    poly_path = data_path(os.path.join("polylines", "TS2.polylines.txt"))
    return PolylinesSkeleton.from_txt(str(poly_path))


def load_ts2_mesh() -> trimesh.Trimesh:
    mesh_path = data_path(os.path.join("mesh", "processed", "TS2_repaired.obj"))
    mm = MeshManager(verbose=False)
    mesh = mm.load_mesh(str(mesh_path))
    return mesh


def test_load_polylines_from_txt():
    pls = load_ts2_polylines()
    assert isinstance(pls, PolylinesSkeleton)
    assert len(pls.polylines) > 0
    assert pls.total_points() > 0
    # Ensure all polylines are (N,3)
    for pl in pls.polylines:
        assert pl.ndim == 2 and pl.shape[1] == 3


def test_transform_and_undo():
    pls = load_ts2_polylines()
    c0 = pls.centroid()
    assert c0 is not None
    t = np.array([1.5, -2.0, 3.25], dtype=float)
    pls.translate(t)
    c1 = pls.centroid()
    assert c1 is not None
    np.testing.assert_allclose(c1, c0 + t, rtol=1e-6, atol=1e-6)
    pls.undo_last_transform()
    c2 = pls.centroid()
    np.testing.assert_allclose(c2, c0, rtol=1e-6, atol=1e-6)


def test_copy_transforms_from_mesh_composite():
    # Prepare a mesh manager and apply some transforms
    mesh_path = data_path(os.path.join("mesh", "processed", "TS2_repaired.obj"))
    mm = MeshManager(verbose=False)
    mm.load_mesh(str(mesh_path))
    mm.center_mesh("centroid")
    mm.scale_mesh(0.5)
    # Prepare polylines
    pls = load_ts2_polylines()
    original = [pl.copy() for pl in pls.polylines]
    # Copy composite transform from mesh
    pls.copy_transforms_from_mesh(mm, mode="composite")
    M = mm.get_composite_matrix()
    # Manually transform originals and compare
    def apply(M, points):
        ones = np.ones((points.shape[0], 1), dtype=float)
        vh = np.hstack([points, ones])
        v2 = (M @ vh.T).T[:, :3]
        return v2
    manual = [apply(M, pl) for pl in original]
    for a, b in zip(pls.polylines, manual):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)


def test_snap_to_mesh_surface_reduces_distance():
    mesh = load_ts2_mesh()
    pls = load_ts2_polylines()
    # Offset polylines slightly to ensure some distance from the surface
    pls.translate([0.0, 0.0, 10.0])
    # Measure distances before
    from trimesh.proximity import closest_point

    P_before = np.vstack(pls.polylines)
    CP_before, dist_before, _ = closest_point(mesh, P_before)
    mean_before = float(np.mean(dist_before))
    # Snap (project all points regardless of sign)
    moved, mean_move = pls.snap_to_mesh_surface(mesh, project_outside_only=False)
    assert moved > 0
    assert mean_move >= 0.0
    # Measure distances after
    P_after = np.vstack(pls.polylines)
    CP_after, dist_after, _ = closest_point(mesh, P_after)
    mean_after = float(np.mean(dist_after))
    # Expect reduction in mean distance to surface
    assert mean_after <= mean_before + 1e-6
    # After snapping, points coincide with their closest points (near-zero dist)
    assert mean_after < 1e-3

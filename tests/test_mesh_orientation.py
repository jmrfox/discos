import numpy as np
import trimesh

from discos.mesh import MeshManager


def _furthest_pair_vector(points: np.ndarray) -> np.ndarray:
    # Simple O(n^2) search for furthest pair; sufficient for small tests
    n = points.shape[0]
    best_i, best_j = 0, 1
    best_d2 = -1.0
    for i in range(n - 1):
        diffs = points[i + 1 :] - points[i]
        if diffs.size == 0:
            continue
        d2 = np.einsum("ij,ij->i", diffs, diffs)
        j_rel = int(np.argmax(d2))
        if d2[j_rel] > best_d2:
            best_d2 = float(d2[j_rel])
            best_i = i
            best_j = i + 1 + j_rel
    return points[best_j] - points[best_i]


def test_align_furthest_points_with_z_on_ellipsoid():
    # Create a prolate ellipsoid by scaling an icosphere
    sphere = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    T_scale = np.eye(4)
    T_scale[0, 0] = 1.0
    T_scale[1, 1] = 1.2
    T_scale[2, 2] = 4.0  # major axis along Z before rotation
    ellipsoid = sphere.copy()
    ellipsoid.apply_transform(T_scale)

    # Apply a random rotation so the major axis is not aligned with Z
    R = trimesh.transformations.random_rotation_matrix()
    ellipsoid.apply_transform(R)

    mgr = MeshManager(mesh=ellipsoid)
    mgr.align_furthest_points_with_z()

    # After alignment, the furthest-pair vector should be parallel to +Z
    v = _furthest_pair_vector(mgr.mesh.vertices)
    v = v / (np.linalg.norm(v) + 1e-12)
    # dot product with +Z should be close to 1.0 (allow small numerical tolerance)
    assert v[2] > 0.99, f"Furthest pair not aligned with +Z: v={v}"

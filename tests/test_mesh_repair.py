"""
Tests for robust mesh repair pipeline.
"""

import os
import sys

# Add the parent directory to path for importing discos
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import trimesh

from discos.demo import create_cylinder_mesh
from discos.mesh import MeshManager


def make_holed_cylinder(length=20.0, radius=2.0, resolution=24) -> trimesh.Trimesh:
    """Create a cylinder and remove a cap region to introduce a hole."""
    m = create_cylinder_mesh(length=length, radius=radius, resolution=resolution)
    m = m.copy()
    # Remove faces near the max-z to create a hole
    zmax = m.vertices[:, 2].max()
    face_z = m.vertices[m.faces].mean(axis=1)[:, 2]
    to_remove = np.where(face_z > zmax - 1e-6)[0]
    if len(to_remove) == 0:
        # Fallback: remove top 2% faces by z
        thresh = np.quantile(face_z, 0.98)
        to_remove = np.where(face_z >= thresh)[0]
    mask = np.ones(len(m.faces), dtype=bool)
    mask[to_remove] = False
    m.update_faces(mask)
    m.remove_unreferenced_vertices()
    return m


class TestMeshRepair:
    def test_repair_holed_cylinder_watertight(self):
        m = make_holed_cylinder()
        assert not m.is_watertight

        mgr = MeshManager(m, verbose=False)
        repaired = mgr.repair_mesh_pymeshfix(
            join_components=True,
            remove_small_components=False,
            keep_largest_component=True,
            min_component_faces=30,
            verbose=False,
        )
        assert isinstance(repaired, trimesh.Trimesh)
        assert repaired.is_watertight
        assert len(repaired.faces) > 0 and len(repaired.vertices) > 0
        vol = float(repaired.volume)
        assert vol > 0

    def test_keep_largest_component(self):
        # Create a large and small cylinder far apart, then concatenate
        big = create_cylinder_mesh(length=20.0, radius=2.0, resolution=24)
        small = create_cylinder_mesh(length=5.0, radius=0.5, resolution=16)
        small = small.copy()
        small.apply_translation([100.0, 0.0, 0.0])
        combo = trimesh.util.concatenate([big, small])
        assert len(combo.split(only_watertight=False)) >= 2

        mgr = MeshManager(combo, verbose=False)
        repaired = mgr.repair_mesh_pymeshfix(
            keep_largest_component=True, min_component_faces=30, verbose=False
        )
        comps = repaired.split(only_watertight=False)
        assert len(comps) == 1

    def test_manager_integration(self):
        m = make_holed_cylinder()
        mgr = MeshManager(m, verbose=False)
        repaired = mgr.repair_mesh_pymeshfix(verbose=False)
        assert isinstance(repaired, trimesh.Trimesh)
        assert repaired.is_watertight


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

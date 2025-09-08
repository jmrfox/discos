"""
Polylines-based skeleton handler.

Provides a `PolylinesSkeleton` class to:
- Load/save skeleton polylines from/to text format lines: `N x1 y1 z1 x2 y2 z2 ...`
- Manage a transform stack similar to `MeshManager` (apply, track, undo, reset).
- Copy transforms from a `MeshManager` (either composite matrix or full stack).
- Snap/project polyline points to the nearest mesh surface to handle small
  outside deviations.

Note: This class is designed to guide the skeletonization process optionally.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import trimesh

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class Transform:
    name: str
    M: np.ndarray  # 4x4 homogeneous
    params: Optional[Dict[str, Any]]
    timestamp: datetime
    is_uniform_scale: bool = False
    uniform_scale: Optional[float] = None


class PolylinesSkeleton:
    """
    Container for a set of 3D polylines with transform management.

    Attributes:
        polylines: List of (N_i, 3) arrays of float64
        original_polylines: Deep copy of original coordinates for reset/undo
        transform_stack: list of `Transform` records in applied order
        M_world_from_local: Composite transform mapping local->world
        M_local_from_world: Inverse composite mapping world->local (if invertible)
    """

    def __init__(self, polylines: Optional[Sequence[np.ndarray]] = None):
        self.polylines: List[np.ndarray] = []
        if polylines is not None:
            for pl in polylines:
                arr = np.asarray(pl, dtype=float)
                if arr.ndim != 2 or arr.shape[1] != 3:
                    raise ValueError("Each polyline must be an (N,3) array")
                self.polylines.append(arr.copy())
        self.original_polylines: List[np.ndarray] = [pl.copy() for pl in self.polylines]

        self.transform_stack: List[Transform] = []
        self.M_world_from_local: np.ndarray = np.eye(4, dtype=float)
        self.M_local_from_world: np.ndarray = np.eye(4, dtype=float)

    # ---------------------------------------------------------------------
    # IO
    # ---------------------------------------------------------------------
    @staticmethod
    def from_txt(path: str) -> "PolylinesSkeleton":
        """
        Load polylines from a `.polylines.txt` file where each line encodes one
        polyline as: `N x1 y1 z1 x2 y2 z2 ... xN yN zN`.
        """
        polylines: List[np.ndarray] = []
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                s = line.strip()
                if not s:
                    continue
                parts = s.split()
                try:
                    n = int(float(parts[0]))
                except Exception as e:
                    raise ValueError(f"Invalid header count on line {line_no}") from e
                coords = parts[1:]
                if len(coords) != 3 * n:
                    raise ValueError(
                        f"Line {line_no}: expected {3*n} coordinate values, got {len(coords)}"
                    )
                vals = np.array([float(c) for c in coords], dtype=float)
                pl = vals.reshape(n, 3)
                polylines.append(pl)
        return PolylinesSkeleton(polylines)

    def to_txt(self, path: str) -> None:
        """Save polylines to the `.polylines.txt` format described above."""
        with open(path, "w", encoding="utf-8") as f:
            for pl in self.polylines:
                n = int(pl.shape[0])
                flat = " ".join(f"{v:.17g}" for v in pl.reshape(-1))
                f.write(f"{n} {flat}\n")

    # ---------------------------------------------------------------------
    # Basic properties
    # ---------------------------------------------------------------------
    def copy(self) -> "PolylinesSkeleton":
        cps = PolylinesSkeleton([pl.copy() for pl in self.polylines])
        cps.original_polylines = [pl.copy() for pl in self.original_polylines]
        cps.transform_stack = list(self.transform_stack)
        cps.M_world_from_local = self.M_world_from_local.copy()
        cps.M_local_from_world = self.M_local_from_world.copy()
        return cps

    def as_arrays(self) -> List[np.ndarray]:
        return [pl.copy() for pl in self.polylines]

    def total_points(self) -> int:
        return int(sum(pl.shape[0] for pl in self.polylines))

    def bounds(self) -> Optional[Dict[str, Tuple[float, float]]]:
        if not self.polylines:
            return None
        all_pts = np.vstack(self.polylines)
        lo = all_pts.min(axis=0)
        hi = all_pts.max(axis=0)
        return {"x": (float(lo[0]), float(hi[0])), "y": (float(lo[1]), float(hi[1])), "z": (float(lo[2]), float(hi[2]))}

    def centroid(self) -> Optional[np.ndarray]:
        if not self.polylines:
            return None
        all_pts = np.vstack(self.polylines)
        return all_pts.mean(axis=0)

    # ---------------------------------------------------------------------
    # Transform management (modeled after MeshManager)
    # ---------------------------------------------------------------------
    def _apply_transform_inplace(
        self,
        M: np.ndarray,
    ) -> None:
        M = np.asarray(M, dtype=float)
        if M.shape != (4, 4):
            raise ValueError("Transform matrix must be 4x4")
        for i, pl in enumerate(self.polylines):
            if pl.size == 0:
                continue
            ones = np.ones((pl.shape[0], 1), dtype=float)
            vh = np.hstack([pl, ones])
            v2 = (M @ vh.T).T[:, :3]
            self.polylines[i] = v2

    def _apply_and_record_transform(
        self,
        name: str,
        M: np.ndarray,
        *,
        params: Optional[Dict[str, Any]] = None,
        is_uniform_scale: bool = False,
        uniform_scale: Optional[float] = None,
    ) -> None:
        self._apply_transform_inplace(M)
        # Update composite matrices
        self.M_world_from_local = M @ self.M_world_from_local
        try:
            self.M_local_from_world = np.linalg.inv(self.M_world_from_local)
        except Exception:
            pass
        # Record
        self.transform_stack.append(
            Transform(
                name=name,
                M=M.copy(),
                params=None if params is None else dict(params),
                timestamp=datetime.now(),
                is_uniform_scale=bool(is_uniform_scale),
                uniform_scale=float(uniform_scale) if uniform_scale is not None else None,
            )
        )

    def apply_transform(
        self, M: np.ndarray, *, name: str = "custom", params: Optional[Dict[str, Any]] = None
    ) -> None:
        self._apply_and_record_transform(name, M, params=params)

    def get_composite_matrix(self) -> np.ndarray:
        return self.M_world_from_local.copy()

    def get_inverse_matrix(self) -> np.ndarray:
        return self.M_local_from_world.copy()

    # Convenience transforms
    def translate(self, t: Iterable[float]) -> None:
        tx, ty, tz = [float(c) for c in t]
        T = np.eye(4, dtype=float)
        T[:3, 3] = [tx, ty, tz]
        self._apply_and_record_transform("translate", T, params={"t": [tx, ty, tz]})

    def scale(self, s: float) -> None:
        sf = float(s)
        S = np.eye(4, dtype=float)
        S[0, 0] = sf
        S[1, 1] = sf
        S[2, 2] = sf
        self._apply_and_record_transform(
            "scale", S, params={"scale": sf}, is_uniform_scale=True, uniform_scale=sf
        )

    def center_on_centroid(self) -> None:
        c = self.centroid()
        if c is None:
            return
        T = np.eye(4, dtype=float)
        T[:3, 3] = -np.asarray(c, dtype=float)
        self._apply_and_record_transform("center", T, params={"center": c.tolist()})

    def align_principal_axis_with_z(self, target_axis: Optional[np.ndarray] = None) -> None:
        if not self.polylines:
            return
        if target_axis is None:
            target_axis = np.array([0.0, 0.0, 1.0], dtype=float)
        pts = np.vstack(self.polylines).astype(float)
        ctr = pts.mean(axis=0)
        V = pts - ctr
        cov = np.cov(V.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        principal = eigvecs[:, int(np.argmax(eigvals))]
        # Build rotation from principal to target
        v1 = principal / (np.linalg.norm(principal) + 1e-12)
        v2 = target_axis / (np.linalg.norm(target_axis) + 1e-12)
        if np.allclose(v1, v2):
            return
        if np.allclose(v1, -v2):
            # 180-deg around a perpendicular axis
            axis = np.array([1.0, 0.0, 0.0], dtype=float)
            if abs(v1[0]) > 0.9:
                axis = np.array([0.0, 1.0, 0.0], dtype=float)
            angle = np.pi
        else:
            axis = np.cross(v1, v2)
            axis = axis / (np.linalg.norm(axis) + 1e-12)
            dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
            angle = float(np.arccos(dot))
        K = np.array([[0, -axis[2], axis[1]],[axis[2], 0, -axis[0]],[-axis[1], axis[0], 0]], dtype=float)
        R3 = np.eye(3, dtype=float) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        # rotate about centroid
        T1 = np.eye(4, dtype=float)
        T2 = np.eye(4, dtype=float)
        T1[:3, 3] = -ctr
        T2[:3, 3] = ctr
        R4 = np.eye(4, dtype=float)
        R4[:3, :3] = R3
        M = T2 @ R4 @ T1
        self._apply_and_record_transform("align_principal_axis_with_z", M, params={"centroid": ctr.tolist()})

    # Undo / reset
    def reset_transforms(self) -> None:
        self.polylines = [pl.copy() for pl in self.original_polylines]
        self.transform_stack.clear()
        self.M_world_from_local = np.eye(4, dtype=float)
        self.M_local_from_world = np.eye(4, dtype=float)

    def undo_last_transform(self) -> None:
        if not self.transform_stack:
            return
        # Recompute from original for numerical stability
        last = self.transform_stack.pop()
        _ = last  # unused except pop
        self.polylines = [pl.copy() for pl in self.original_polylines]
        self.M_world_from_local = np.eye(4, dtype=float)
        self.M_local_from_world = np.eye(4, dtype=float)
        for t in self.transform_stack:
            self._apply_transform_inplace(t.M)
            self.M_world_from_local = t.M @ self.M_world_from_local
        try:
            self.M_local_from_world = np.linalg.inv(self.M_world_from_local)
        except Exception:
            pass

    # ---------------------------------------------------------------------
    # Interop with MeshManager
    # ---------------------------------------------------------------------
    def copy_transforms_from_mesh(self, mesh_manager: Any, *, mode: str = "stack") -> None:
        """
        Copy transforms from a `MeshManager`.

        Args:
            mesh_manager: instance of `discos.mesh.MeshManager`
            mode: "stack" (default) to apply each recorded transform in order,
                  or "composite" to apply the final composite matrix once.
        """
        # Lazy import type to avoid circular import at runtime
        try:
            from .mesh import MeshManager  # type: ignore
        except Exception:
            MeshManager = None  # type: ignore
        if MeshManager is not None and not isinstance(mesh_manager, MeshManager):
            logger.warning("copy_transforms_from_mesh: object is not a MeshManager instance")
        if mode == "composite":
            M = np.asarray(getattr(mesh_manager, "M_world_from_local", np.eye(4)), dtype=float)
            self._apply_and_record_transform("mesh_composite_copy", M, params={"source": "MeshManager"})
            return
        # Default: apply each transform in order
        stack = getattr(mesh_manager, "transform_stack", [])
        for t in stack:
            M = np.asarray(getattr(t, "M", None), dtype=float)
            if M.shape == (4, 4):
                self._apply_and_record_transform(
                    getattr(t, "name", "mesh_transform_copy"),
                    M,
                    params=getattr(t, "params", None),
                    is_uniform_scale=bool(getattr(t, "is_uniform_scale", False)),
                    uniform_scale=(getattr(t, "uniform_scale", None)),
                )

    # ---------------------------------------------------------------------
    # Projection / snapping to mesh surface
    # ---------------------------------------------------------------------
    def snap_to_mesh_surface(
        self,
        mesh: trimesh.Trimesh,
        *,
        project_outside_only: bool = True,
        max_distance: Optional[float] = None,
    ) -> Tuple[int, float]:
        """
        Project points to the nearest surface point on `mesh`.

        Args:
            project_outside_only: If True, use signed distance to project only
                points outside (positive sign). If signed distance is unavailable,
                falls back to projecting all points.
            max_distance: If provided, only move points whose distance to the
                surface exceeds this threshold. Units are mesh units.

        Returns:
            (n_moved, mean_move_distance)
        """
        if mesh is None or len(getattr(mesh, "vertices", [])) == 0:
            return 0, 0.0

        all_pts = [pl for pl in self.polylines if pl.size > 0]
        if not all_pts:
            return 0, 0.0

        # Flatten to a single array and track mapping back
        counts = [pl.shape[0] for pl in self.polylines]
        P = np.vstack(self.polylines) if self.polylines else np.zeros((0, 3), dtype=float)

        # Try signed distance to detect outside points
        use_mask = None
        if project_outside_only:
            try:
                from trimesh.proximity import signed_distance  # type: ignore

                d = signed_distance(mesh, P)
                use_mask = d > 0  # outside
            except Exception:
                use_mask = None

        try:
            from trimesh.proximity import closest_point  # type: ignore

            CP, dist, tri = closest_point(mesh, P)
            _ = tri  # unused
        except Exception:
            # Fallback: KDTree over vertices only
            V = np.asarray(mesh.vertices, dtype=float)
            if V.size == 0:
                return 0, 0.0
            from scipy.spatial import cKDTree  # type: ignore

            tree = cKDTree(V)
            dd, idx = tree.query(P, k=1)
            CP = V[idx]
            dist = dd

        if use_mask is None:
            mask = np.ones(P.shape[0], dtype=bool)
        else:
            mask = use_mask

        if max_distance is not None:
            mask = mask & (dist >= float(max_distance))

        moved = 0
        total_move = 0.0
        # Re-split CP back to polylines and update selectively
        start = 0
        new_polylines: List[np.ndarray] = []
        for cnt in counts:
            segP = P[start : start + cnt]
            segCP = CP[start : start + cnt]
            segDist = dist[start : start + cnt]
            segMask = mask[start : start + cnt]
            start += cnt
            out = segP.copy()
            sel = np.nonzero(segMask)[0]
            if len(sel) > 0:
                out[sel] = segCP[sel]
                moved += int(len(sel))
                total_move += float(np.sum(segDist[sel]))
            new_polylines.append(out)
        self.polylines = new_polylines
        mean_move = (total_move / moved) if moved > 0 else 0.0
        return moved, mean_move

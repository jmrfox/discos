"""
Robust mesh repair utilities for biological meshes.

This module provides a staged repair pipeline that attempts lightweight
repairs first and falls back to a voxel-based watertightization when needed.

Key goals:
- Preserve geometry when possible.
- Produce watertight output suitable for downstream analysis/skeletonization.
"""
from typing import Dict, Optional, Tuple

import numpy as np
import trimesh


class MeshRepairer:
    """
    Staged mesh repair pipeline.

    Parameters
    ----------
    keep_largest_component : bool
        If True, keep only the largest connected component at the end of light repair.
    min_component_faces : int
        Components with fewer faces than this will be discarded during cleanup when
        keep_largest_component is True.
    grid_size : int
        Target voxel grid resolution along the largest bbox dimension for fallback.
        Effective voxel pitch is computed from bbox extent / grid_size.
    taubin_smooth_iters : int
        Number of Taubin smoothing iterations to apply after voxel fallback (0 to skip).
    verbose : bool
        Print a short repair summary.
    """

    def __init__(
        self,
        keep_largest_component: bool = True,
        min_component_faces: int = 50,
        grid_size: int = 256,
        taubin_smooth_iters: int = 10,
        verbose: bool = False,
    ) -> None:
        self.keep_largest_component = keep_largest_component
        self.min_component_faces = int(min_component_faces)
        self.grid_size = int(grid_size)
        self.taubin_smooth_iters = int(taubin_smooth_iters)
        self.verbose = verbose

    # ------------- Public API -------------
    def repair(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Run the robust repair pipeline and return a new (repaired) mesh.
        The input mesh is not modified.
        """
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError("mesh must be a trimesh.Trimesh")

        m = mesh.copy()
        log: Dict[str, str] = {}

        # Stage 0: basic sanitation
        self._basic_cleanup(m, log)

        # Stage 1: orientation and normals
        self._fix_orientation_and_normals(m, log)

        # Stage 2: duplicates/degenerates and hole filling
        self._remove_bad_faces_and_fill_holes(m, log)

        # Stage 3: connected components policy
        self._prune_components(m, log)

        # Final validate
        self._final_process(m, log)

        # Optional PyMeshFix stage (if available) before voxel fallback
        if not m.is_watertight:
            m2 = self._pymeshfix_stage(m, log)
            if isinstance(m2, trimesh.Trimesh):
                m = m2

        # If still not watertight, voxel fallback
        if not m.is_watertight:
            # Use the original mesh for voxel fallback to avoid artifacts
            m = self._voxel_fallback(mesh=mesh, log=log)
            # Ensure watertight if possible by repairing any residual openings
            self._ensure_watertight(m, log)

        # Attach log metadata
        if not hasattr(m, "metadata") or m.metadata is None:
            m.metadata = {}
        m.metadata.setdefault("repair", {})
        m.metadata["repair"].update(log)

        if self.verbose:
            self._print_summary(m, log)

        return m

    # ------------- Internal steps -------------
    def _basic_cleanup(self, m: trimesh.Trimesh, log: Dict[str, str]) -> None:
        try:
            # Remove NaN/Inf
            m.remove_infinite_values()
            # Remove exact duplicate faces, unreferenced vertices (modern API)
            try:
                m.update_faces(m.unique_faces())
            except Exception:
                # Back-compat
                m.remove_duplicate_faces()
            m.remove_unreferenced_vertices()
            log["basic_cleanup"] = "removed duplicates and unreferenced vertices"
        except Exception as e:
            log["basic_cleanup_error"] = str(e)

    def _fix_orientation_and_normals(self, m: trimesh.Trimesh, log: Dict[str, str]) -> None:
        try:
            # Negative volume -> invert
            if hasattr(m, "volume") and m.volume < 0:
                v0 = float(m.volume)
                m.invert()
                log["invert"] = f"inverted faces (volume {v0:.3g} -> {float(m.volume):.3g})"
        except Exception as e:
            log["invert_error"] = str(e)

        try:
            if not m.is_winding_consistent:
                m.fix_normals()
                if m.is_winding_consistent:
                    log["normals"] = "fixed winding"
                else:
                    log["normals"] = "attempted to fix winding; still inconsistent"
        except Exception as e:
            log["normals_error"] = str(e)

    def _remove_bad_faces_and_fill_holes(self, m: trimesh.Trimesh, log: Dict[str, str]) -> None:
        try:
            before = len(m.faces)
            try:
                m.update_faces(m.nondegenerate_faces())
            except Exception:
                # Back-compat
                m.remove_degenerate_faces()
            after = len(m.faces)
            if after != before:
                log["degenerate"] = f"removed {before - after} degenerate faces"
        except Exception as e:
            log["degenerate_error"] = str(e)

        # Fill holes if not watertight
        try:
            if not m.is_watertight:
                m.fill_holes()
                log["fill_holes"] = "called fill_holes"
                # Try trimesh.repair.fill_holes as well for stubborn cases
                try:
                    import trimesh.repair as _tr

                    _tr.fill_holes(m)
                    log["fill_holes_repair"] = "trimesh.repair.fill_holes"
                except Exception:
                    pass
        except Exception as e:
            log["fill_holes_error"] = str(e)

    def _prune_components(self, m: trimesh.Trimesh, log: Dict[str, str]) -> None:
        if not self.keep_largest_component:
            return
        try:
            comps = m.split(only_watertight=False)
            if len(comps) > 1:
                # Keep the largest by (abs volume if available) else by face count
                def comp_key(c: trimesh.Trimesh) -> Tuple[float, int]:
                    vol = abs(float(c.volume)) if hasattr(c, "volume") else 0.0
                    return (vol, len(c.faces))

                comps_filtered = [c for c in comps if len(c.faces) >= self.min_component_faces]
                comps_use = comps_filtered if len(comps_filtered) > 0 else comps
                largest = max(comps_use, key=comp_key)
                m.vertices = largest.vertices.copy()
                m.faces = largest.faces.copy()
                # Update cached properties
                m.process(validate=False)
                log["components"] = f"kept largest component of {len(comps)}"
        except Exception as e:
            log["components_error"] = str(e)

    def _final_process(self, m: trimesh.Trimesh, log: Dict[str, str]) -> None:
        try:
            m.process(validate=True)
            log["final_process"] = "processed and validated"
        except Exception as e:
            log["final_process_error"] = str(e)

    def _pymeshfix_stage(self, m: trimesh.Trimesh, log: Dict[str, str]) -> trimesh.Trimesh:
        """Attempt to repair with PyMeshFix if installed.

        This often produces a watertight manifold surface without heavy voxelization.
        """
        try:
            from pymeshfix import MeshFix  # type: ignore
        except Exception:
            log["pymeshfix"] = "not available"
            return m

        try:
            v = np.asarray(m.vertices, dtype=np.float64)
            f = np.asarray(m.faces, dtype=np.int64)
            mf = MeshFix(v, f)
            # Try modern signature; fallback to default if not supported
            try:
                mf.repair(verbose=False, joincomp=True, remove_small_components=False)
            except TypeError:
                mf.repair()

            v2 = np.asarray(getattr(mf, "v", None), dtype=np.float64)
            f2 = np.asarray(getattr(mf, "f", None), dtype=np.int64)
            if v2.size == 0 or f2.size == 0:
                log["pymeshfix_error"] = "empty output"
                return m

            m2 = trimesh.Trimesh(vertices=v2, faces=f2, process=True)
            # Post-fix normals and validate
            try:
                if not m2.is_winding_consistent:
                    m2.fix_normals()
                m2.process(validate=True)
            except Exception:
                pass

            log["pymeshfix"] = f"applied (faces {len(m.faces)} -> {len(m2.faces)})"
            return m2
        except Exception as e:
            log["pymeshfix_error"] = str(e)
            return m

    def _voxel_fallback(self, mesh: trimesh.Trimesh, log: Dict[str, str]) -> trimesh.Trimesh:
        # Compute pitch from bbox and grid_size
        bounds = mesh.bounds if hasattr(mesh, "bounds") else None
        if bounds is None:
            base_pitch = 1.0
        else:
            extents = np.asarray(bounds[1] - bounds[0], dtype=float)
            max_extent = float(np.max(extents)) if np.all(np.isfinite(extents)) else 1.0
            base_pitch = max_extent / max(self.grid_size, 16)
            if not np.isfinite(base_pitch) or base_pitch <= 0:
                base_pitch = 1.0

        # Prefer 'ray' which is robust; try 'subdivide' only if 'ray' fails
        methods = ["ray", "subdivide"]
        factors = [1.0, 0.5, 0.25, 0.125]
        last_mesh: Optional[trimesh.Trimesh] = None
        last_err: Optional[str] = None
        for method in methods:
            for f in factors:
                pitch = base_pitch * f
                # Create voxel grid; catch any errors (e.g., remesh max_iter)
                try:
                    try:
                        vg = mesh.voxelized(pitch, method=method)  # type: ignore[arg-type]
                    except TypeError:
                        # Older trimesh doesn't accept method kwarg
                        vg = mesh.voxelized(pitch)
                except Exception as e:
                    last_err = f"voxelized failed (method={method}, pitch={pitch:.4g}): {e}"
                    continue
                try:
                    vg_filled = vg.fill()
                except Exception:
                    vg_filled = vg
                try:
                    mc = vg_filled.marching_cubes
                    mc.process(validate=True)
                    # Cleanup faces/verts to ensure manifold closure
                    try:
                        mc.update_faces(mc.unique_faces())
                        mc.update_faces(mc.nondegenerate_faces())
                        mc.remove_unreferenced_vertices()
                        mc.process(validate=True)
                    except Exception:
                        pass
                    # Keep largest component
                    try:
                        parts = mc.split(only_watertight=False)
                        if len(parts) > 1:
                            def comp_key(c: trimesh.Trimesh):
                                vol = abs(float(getattr(c, 'volume', 0.0)))
                                return (vol, len(c.faces))
                            mc = max(parts, key=comp_key)
                            mc.process(validate=True)
                    except Exception:
                        pass
                    # Optional smoothing
                    if self.taubin_smooth_iters > 0:
                        try:
                            from trimesh.smoothing import filter_taubin
                            filter_taubin(mc, lamb=0.5, nu=-0.53, iterations=self.taubin_smooth_iters)
                            mc.process(validate=True)
                        except Exception:
                            pass
                    # Try to enforce watertightness on the result
                    try:
                        import trimesh.repair as _tr
                        try:
                            _tr.stitch(mc)
                        except Exception:
                            pass
                        _tr.fill_holes(mc)
                        mc.fix_normals()
                        mc.process(validate=True)
                    except Exception:
                        pass
                    last_mesh = mc
                    log["fallback"] = f"voxelized(method={method}, pitch={pitch:.4g}) -> marching_cubes"
                    if mc.is_watertight:
                        return mc
                except Exception as e:
                    last_err = str(e)

        if last_mesh is not None:
            return last_mesh

        # If everything failed, return a processed copy
        log["fallback_error"] = f"voxel fallback failed: {last_err or 'unknown'}"
        m2 = mesh.copy()
        try:
            m2.process(validate=True)
        except Exception:
            pass
        return m2

    def _ensure_watertight(self, m: trimesh.Trimesh, log: Dict[str, str]) -> None:
        """Try to enforce watertightness with additional repairs."""
        try:
            if not m.is_watertight:
                try:
                    import trimesh.repair as _tr

                    _tr.fill_holes(m)
                except Exception:
                    pass
                if not m.is_watertight:
                    try:
                        m.fix_normals()
                        m.process(validate=True)
                    except Exception:
                        pass
                if not m.is_watertight:
                    # As a last resort, cap open boundaries by triangulating their outlines
                    try:
                        boundaries = m.boundary_edges
                        if boundaries is not None and len(boundaries) > 0:
                            # Let trimesh attempt automatic hole filling again
                            m.fill_holes()
                    except Exception:
                        pass
        except Exception as e:
            log["ensure_watertight_error"] = str(e)
        # Final conservative fallback: convex hull is always watertight
        if not m.is_watertight:
            try:
                hull = m.convex_hull
                hull.process(validate=True)
                log["ensure_watertight_convex_hull"] = "used convex hull fallback"
                m.vertices = hull.vertices.copy()
                m.faces = hull.faces.copy()
                m.process(validate=True)
            except Exception as e:
                log["ensure_watertight_convex_hull_error"] = str(e)

    # ------------- Reporting -------------
    def _print_summary(self, m: trimesh.Trimesh, log: Dict[str, str]) -> None:
        try:
            print("🔧 Robust Mesh Repair Summary:")
            for k, v in log.items():
                print(f"  • {k}: {v}")
            print("📊 Final status:")
            vol = getattr(m, "volume", None)
            if isinstance(vol, (int, float)):
                print(f"  • volume: {vol:.3g}")
            print(f"  • watertight: {m.is_watertight}")
            print(f"  • faces: {len(m.faces)}  vertices: {len(m.vertices)}")
        except Exception:
            pass

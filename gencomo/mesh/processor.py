"""
Main mesh processor class combining all mesh processing functionality.
"""

import numpy as np
import trimesh
import open3d as o3d
from typing import Optional, Tuple, Dict, Any, Union
import warnings
import os
from pathlib import Path

from .utils import analyze_mesh, repair_mesh


class MeshProcessor:
    """
    Unified mesh processor handling loading, processing, analysis, and preprocessing.

    This class combines the functionality of the original MeshProcessor and MeshPreprocessor
    classes to provide a comprehensive mesh processing interface.
    """

    def __init__(self, verbose: bool = True):
        # Core mesh attributes
        self.mesh = None
        self.original_mesh = None
        self.bounds = None

        # Preprocessing attributes
        self.verbose = verbose
        self.stats = {
            "processed": 0,
            "successful": 0,
            "failed": 0,
            "volume_fixed": 0,
            "watertight_fixed": 0,
            "degenerate_removed": 0,
        }

    def log(self, message: str, level: str = "INFO"):
        """Log messages if verbose mode is enabled."""
        if self.verbose:
            prefix = {"INFO": "ℹ️", "SUCCESS": "✅", "WARNING": "⚠️", "ERROR": "❌", "PROCESSING": "🔧"}.get(level, "📝")
            print(f"{prefix} {message}")

    # =================================================================
    # MESH LOADING AND BASIC OPERATIONS
    # =================================================================

    def load_mesh(self, filepath: str, file_format: Optional[str] = None) -> trimesh.Trimesh:
        """
        Load a mesh from file.

        Args:
            filepath: Path to mesh file
            file_format: Optional format specification (auto-detected if None)

        Returns:
            Loaded trimesh object
        """
        try:
            if file_format:
                mesh = trimesh.load(filepath, file_type=file_format)
            else:
                mesh = trimesh.load(filepath)

            # Ensure we have a single mesh
            if isinstance(mesh, trimesh.Scene):
                # If it's a scene, try to get the first geometry
                geometries = list(mesh.geometry.values())
                if geometries:
                    mesh = geometries[0]
                else:
                    raise ValueError("No geometry found in mesh scene")

            if not isinstance(mesh, trimesh.Trimesh):
                raise ValueError(f"Loaded object is not a mesh: {type(mesh)}")

            self.mesh = mesh
            self.original_mesh = mesh.copy()
            self.bounds = self._compute_bounds()

            if self.verbose:
                print(f"Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
                print(f"Bounds: {self.bounds}")

            return mesh

        except Exception as e:
            raise ValueError(f"Failed to load mesh from {filepath}: {str(e)}")

    def load_from_arrays(self, vertices: np.ndarray, faces: np.ndarray) -> trimesh.Trimesh:
        """
        Create mesh from vertex and face arrays.

        Args:
            vertices: Nx3 array of vertex coordinates
            faces: Mx3 array of face vertex indices

        Returns:
            Created trimesh object
        """
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        self.mesh = mesh
        self.original_mesh = mesh.copy()
        self.bounds = self._compute_bounds()
        return mesh

    def _compute_bounds(self) -> Optional[Dict[str, Tuple[float, float]]]:
        """Compute mesh bounding box."""
        if self.mesh is None:
            return None

        min_coords = self.mesh.vertices.min(axis=0)
        max_coords = self.mesh.vertices.max(axis=0)

        return {
            "x": (min_coords[0], max_coords[0]),
            "y": (min_coords[1], max_coords[1]),
            "z": (min_coords[2], max_coords[2]),
        }

    def get_z_range(self) -> Tuple[float, float]:
        """Get the z-axis range of the mesh."""
        if self.bounds is None:
            raise ValueError("No mesh loaded")
        return self.bounds["z"]

    def center_mesh(self, center_on: str = "centroid") -> trimesh.Trimesh:
        """
        Center the mesh.

        Args:
            center_on: 'centroid', 'bounds_center', or 'origin'

        Returns:
            Centered mesh
        """
        if self.mesh is None:
            raise ValueError("No mesh loaded")

        if center_on == "centroid":
            center = self.mesh.centroid
        elif center_on == "bounds_center":
            center = self.mesh.bounds.mean(axis=0)
        elif center_on == "origin":
            center = np.array([0, 0, 0])
        else:
            raise ValueError(f"Unknown center_on option: {center_on}")

        self.mesh.vertices -= center
        self.bounds = self._compute_bounds()

        return self.mesh

    def scale_mesh(self, scale_factor: float) -> trimesh.Trimesh:
        """
        Scale the mesh uniformly.

        Args:
            scale_factor: Scaling factor

        Returns:
            Scaled mesh
        """
        if self.mesh is None:
            raise ValueError("No mesh loaded")

        self.mesh.vertices *= scale_factor
        self.bounds = self._compute_bounds()

        return self.mesh

    def align_with_z_axis(self, target_axis: Optional[np.ndarray] = None) -> trimesh.Trimesh:
        """
        Align the mesh's principal axis with the z-axis.

        Args:
            target_axis: Target direction (default: [0, 0, 1])

        Returns:
            Aligned mesh
        """
        if self.mesh is None:
            raise ValueError("No mesh loaded")

        if target_axis is None:
            target_axis = np.array([0, 0, 1])

        # Compute principal axis using PCA
        vertices_centered = self.mesh.vertices - self.mesh.centroid
        covariance = np.cov(vertices_centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)

        # Principal axis is the eigenvector with largest eigenvalue
        principal_axis = eigenvectors[:, np.argmax(eigenvalues)]

        # Compute rotation to align principal axis with target
        rotation_matrix = self._rotation_matrix_between_vectors(principal_axis, target_axis)

        # Apply rotation
        self.mesh.vertices = (rotation_matrix @ vertices_centered.T).T + self.mesh.centroid
        self.bounds = self._compute_bounds()

        return self.mesh

    def _rotation_matrix_between_vectors(self, vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
        """Compute rotation matrix to rotate vec1 to vec2."""
        # Normalize vectors
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)

        # Check if vectors are already aligned
        if np.allclose(vec1, vec2):
            return np.eye(3)
        if np.allclose(vec1, -vec2):
            # 180-degree rotation - find perpendicular axis
            perp = np.array([1, 0, 0]) if abs(vec1[0]) < 0.9 else np.array([0, 1, 0])
            axis = np.cross(vec1, perp)
            axis = axis / np.linalg.norm(axis)
            return self._rodrigues_rotation(axis, np.pi)

        # General case using Rodrigues' formula
        axis = np.cross(vec1, vec2)
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(np.clip(np.dot(vec1, vec2), -1, 1))

        return self._rodrigues_rotation(axis, angle)

    def _rodrigues_rotation(self, axis: np.ndarray, angle: float) -> np.ndarray:
        """Rodrigues' rotation formula."""
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K

    def smooth_mesh(self, iterations: int = 1) -> trimesh.Trimesh:
        """
        Apply Laplacian smoothing to the mesh.

        Args:
            iterations: Number of smoothing iterations

        Returns:
            Smoothed mesh
        """
        if self.mesh is None:
            raise ValueError("No mesh loaded")

        self.mesh = self.mesh.smoothed(iterations=iterations)
        return self.mesh

    def compute_mesh_properties(self) -> Dict[str, Any]:
        """Compute basic mesh properties."""
        if self.mesh is None:
            raise ValueError("No mesh loaded")

        return {
            "volume": self.mesh.volume,
            "surface_area": self.mesh.area,
            "is_watertight": self.mesh.is_watertight,
            "is_winding_consistent": self.mesh.is_winding_consistent,
            "num_vertices": len(self.mesh.vertices),
            "num_faces": len(self.faces),
            "bounds": self.bounds,
            "centroid": self.mesh.centroid.tolist(),
        }

    def repair_mesh_basic(self) -> trimesh.Trimesh:
        """Attempt to repair common mesh issues (basic version)."""
        if self.mesh is None:
            raise ValueError("No mesh loaded")

        # Remove duplicate vertices
        self.mesh.remove_duplicate_faces()
        self.mesh.remove_degenerate_faces()

        # Try to make watertight if not already
        if not self.mesh.is_watertight:
            try:
                self.mesh.fill_holes()
            except:
                warnings.warn("Could not automatically repair mesh holes")

        return self.mesh

    # =================================================================
    # MESH ANALYSIS AND PREPROCESSING
    # =================================================================

    def analyze_mesh_issues(self, mesh: Optional[trimesh.Trimesh] = None, filename: str = "") -> Dict:
        """Analyze mesh for common issues."""
        if mesh is None:
            mesh = self.mesh
        if mesh is None:
            raise ValueError("No mesh provided or loaded")

        analysis = analyze_mesh(mesh)
        issues = []

        # Check for negative volume
        volume = analysis.get("volume", 0)
        if volume is not None and volume < 0:
            issues.append("negative_volume")

        # Check watertightness
        if not analysis.get("is_watertight", False):
            issues.append("not_watertight")

        # Check winding consistency
        if not analysis.get("is_winding_consistent", False):
            issues.append("inconsistent_winding")

        # Check for degenerate faces
        if len(mesh.faces) > 0:
            degenerate_count = 0
            for face in mesh.faces:
                if len(np.unique(face)) < 3:
                    degenerate_count += 1
            if degenerate_count > 0:
                issues.append(f"{degenerate_count}_degenerate_faces")

        # Check for very small faces
        if hasattr(mesh, "area_faces"):
            very_small = np.sum(mesh.area_faces < 1e-10)
            if very_small > 0:
                issues.append(f"{very_small}_tiny_faces")

        return {
            "analysis": analysis,
            "issues": issues,
            "volume": volume,
            "surface_area": analysis.get("surface_area", 0),
            "num_vertices": analysis.get("num_vertices", 0),
            "num_faces": analysis.get("num_faces", 0),
        }

    def fix_negative_volume(self, mesh: Optional[trimesh.Trimesh] = None) -> Tuple[trimesh.Trimesh, bool]:
        """Fix negative volume by correcting face winding order."""
        if mesh is None:
            mesh = self.mesh
        if mesh is None:
            raise ValueError("No mesh provided or loaded")

        if mesh.volume >= 0:
            return mesh, False

        self.log("Fixing negative volume by correcting face winding...", "PROCESSING")

        # Create a copy to work on
        fixed_mesh = mesh.copy()

        # Method 1: Flip all face normals
        try:
            fixed_mesh.faces = np.fliplr(fixed_mesh.faces)

            # Check if this fixed the volume
            if fixed_mesh.volume > 0:
                self.log(f"Volume fixed: {mesh.volume:.3f} → {fixed_mesh.volume:.3f}", "SUCCESS")
                return fixed_mesh, True
            else:
                # If still negative, revert
                fixed_mesh.faces = np.fliplr(fixed_mesh.faces)
        except Exception as e:
            self.log(f"Face flipping failed: {e}", "WARNING")

        # Method 2: Use trimesh's fix_normals
        try:
            fixed_mesh.fix_normals()
            if fixed_mesh.volume > 0:
                self.log(f"Volume fixed with fix_normals: {mesh.volume:.3f} → {fixed_mesh.volume:.3f}", "SUCCESS")
                return fixed_mesh, True
        except Exception as e:
            self.log(f"fix_normals failed: {e}", "WARNING")

        # Method 3: Try manual mesh repair
        try:
            fixed_mesh = repair_mesh(fixed_mesh, fix_normals=True)
            if fixed_mesh.volume > 0:
                self.log(f"Volume fixed with repair_mesh: {mesh.volume:.3f} → {fixed_mesh.volume:.3f}", "SUCCESS")
                return fixed_mesh, True
        except Exception as e:
            self.log(f"repair_mesh failed: {e}", "WARNING")

        self.log("Could not fix negative volume", "WARNING")
        return mesh, False

    def preprocess_mesh(
        self,
        mesh: Optional[trimesh.Trimesh] = None,
        fix_volume: bool = True,
        fix_holes: bool = True,
        remove_duplicates: bool = True,
        fix_normals: bool = True,
        remove_degenerate: bool = True,
    ) -> trimesh.Trimesh:
        """
        Preprocess a mesh to fix common issues.

        Args:
            mesh: Input mesh to preprocess (uses loaded mesh if None)
            fix_volume: Whether to fix negative volume issues
            fix_holes: Whether to attempt hole filling
            remove_duplicates: Whether to remove duplicate faces/vertices
            fix_normals: Whether to fix normal orientation
            remove_degenerate: Whether to remove degenerate faces

        Returns:
            Preprocessed mesh
        """
        if mesh is None:
            mesh = self.mesh
        if mesh is None:
            raise ValueError("No mesh provided or loaded")

        # Analyze initial state
        initial_analysis = self.analyze_mesh_issues(mesh)
        if self.verbose:
            self.log(f"Initial state: {len(initial_analysis['issues'])} issues found")
            for issue in initial_analysis["issues"]:
                self.log(f"  • {issue}", "WARNING")

        # Start preprocessing
        processed_mesh = mesh.copy()
        volume_fixed = False
        watertight_fixed = False

        # Step 1: Fix negative volume
        if fix_volume and "negative_volume" in initial_analysis["issues"]:
            processed_mesh, volume_fixed = self.fix_negative_volume(processed_mesh)
            if volume_fixed:
                self.stats["volume_fixed"] += 1

        # Step 2: General mesh repair
        try:
            repaired_mesh = repair_mesh(
                processed_mesh,
                fix_holes=fix_holes,
                remove_duplicates=remove_duplicates,
                fix_normals=fix_normals,
                remove_degenerate=remove_degenerate,
            )

            # Check if watertightness was improved
            if not processed_mesh.is_watertight and repaired_mesh.is_watertight:
                watertight_fixed = True
                self.stats["watertight_fixed"] += 1

            processed_mesh = repaired_mesh

        except Exception as e:
            self.log(f"Mesh repair failed: {e}", "WARNING")

        # Step 3: Final validation
        if self.verbose:
            final_analysis = self.analyze_mesh_issues(processed_mesh)
            self.log(f"Final state: {len(final_analysis['issues'])} issues remaining")
            if final_analysis["issues"]:
                for issue in final_analysis["issues"]:
                    self.log(f"  • {issue}", "WARNING")

        # Update internal mesh if preprocessing the loaded mesh
        if mesh is self.mesh:
            self.mesh = processed_mesh
            self.bounds = self._compute_bounds()

        return processed_mesh

    # =================================================================
    # FILE PROCESSING OPERATIONS
    # =================================================================

    def preprocess_single_mesh(self, input_path: str, output_path: str) -> bool:
        """Preprocess a single mesh file."""
        try:
            self.log(f"Processing: {input_path}")
            self.stats["processed"] += 1

            # Load mesh
            mesh = trimesh.load_mesh(input_path)

            # Handle scene objects
            if isinstance(mesh, trimesh.Scene):
                geometries = list(mesh.geometry.values())
                if geometries:
                    mesh = geometries[0]
                else:
                    raise ValueError("No geometry found in mesh scene")

            if not isinstance(mesh, trimesh.Trimesh):
                raise ValueError(f"Loaded object is not a mesh: {type(mesh)}")

            # Preprocess the mesh
            processed_mesh = self.preprocess_mesh(mesh)

            # Save processed mesh
            output_dir = os.path.dirname(output_path)
            if output_dir:  # Only create directory if output_path includes a directory
                os.makedirs(output_dir, exist_ok=True)
            processed_mesh.export(output_path)

            self.log(f"Saved to: {output_path}")
            self.stats["successful"] += 1
            return True

        except Exception as e:
            self.log(f"Failed to process {input_path}: {e}", "ERROR")
            self.stats["failed"] += 1
            return False

    def preprocess_directory(self, input_dir: str, output_dir: str, pattern: str = "*.obj") -> None:
        """Preprocess all mesh files in a directory."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)

        if not input_path.exists():
            self.log(f"Input directory does not exist: {input_dir}", "ERROR")
            return

        # Find all mesh files
        mesh_files = list(input_path.glob(pattern))
        if not mesh_files:
            self.log(f"No mesh files found matching pattern: {pattern}", "WARNING")
            return

        self.log(f"Found {len(mesh_files)} mesh files to process")

        # Process each file
        for mesh_file in mesh_files:
            output_file = output_path / f"{mesh_file.stem}_processed{mesh_file.suffix}"
            self.preprocess_single_mesh(str(mesh_file), str(output_file))

    def analyze_single_mesh(self, input_path: str) -> None:
        """Analyze a single mesh file without preprocessing."""
        try:
            self.log(f"Analyzing mesh: {input_path}")

            # Load mesh
            mesh = trimesh.load_mesh(input_path)
            if isinstance(mesh, trimesh.Scene):
                mesh = list(mesh.geometry.values())[0]

            # Analyze issues
            analysis = self.analyze_mesh_issues(mesh, Path(input_path).name)

            self.log(f"Analysis complete for {input_path}")

        except Exception as e:
            self.log(f"Failed to analyze {input_path}: {e}", "ERROR")

    def analyze_directory(self, input_dir: str, pattern: str = "*.obj") -> None:
        """Analyze all mesh files in a directory without preprocessing."""
        input_path = Path(input_dir)

        if not input_path.exists():
            self.log(f"Input directory does not exist: {input_dir}", "ERROR")
            return

        # Find all mesh files
        mesh_files = list(input_path.glob(pattern))
        if not mesh_files:
            self.log(f"No mesh files found matching pattern: {pattern}", "WARNING")
            return

        self.log(f"Found {len(mesh_files)} mesh files to analyze")

        # Analyze each file
        for mesh_file in mesh_files:
            self.analyze_single_mesh(str(mesh_file))

    def print_statistics(self):
        """Print processing statistics."""
        print("\n" + "=" * 60)
        print("📊 PREPROCESSING STATISTICS")
        print("=" * 60)
        print(f"Total files processed: {self.stats['processed']}")
        print(f"Successful: {self.stats['successful']}")
        print(f"Failed: {self.stats['failed']}")
        print(f"Volume issues fixed: {self.stats['volume_fixed']}")
        print(f"Watertightness fixed: {self.stats['watertight_fixed']}")

        if self.stats["processed"] > 0:
            success_rate = (self.stats["successful"] / self.stats["processed"]) * 100
            print(f"Success rate: {success_rate:.1f}%")

"""
Z-axis slicing utilities for mesh compartmentalization.

Provides functionality to slice neuronal meshes along the z-axis and extract
cross-sectional contours for compartment generation.
"""

import numpy as np
import trimesh
from typing import List, Tuple, Dict, Optional
from scipy.spatial.distance import cdist
import cv2
from skimage import measure
import warnings


class ZAxisSlicer:
    """
    Handles slicing of neuronal meshes along the z-axis.
    """

    def __init__(self, mesh: trimesh.Trimesh):
        self.mesh = mesh
        self.slices = []
        self.z_levels = []
        self.slice_spacing = None

    def create_slices(
        self, num_slices: int = None, slice_spacing: float = None, z_min: float = None, z_max: float = None
    ) -> List[Dict]:
        """
        Create z-axis slices of the mesh.

        Args:
            num_slices: Number of slices (alternative to slice_spacing)
            slice_spacing: Distance between slices in µm
            z_min: Minimum z-coordinate (default: mesh minimum)
            z_max: Maximum z-coordinate (default: mesh maximum)

        Returns:
            List of slice dictionaries with contour information
        """
        # Get mesh z-bounds
        mesh_z_min, mesh_z_max = self.mesh.vertices[:, 2].min(), self.mesh.vertices[:, 2].max()

        if z_min is None:
            z_min = mesh_z_min
        if z_max is None:
            z_max = mesh_z_max

        # Determine z-levels
        if slice_spacing is not None:
            self.slice_spacing = slice_spacing
            self.z_levels = np.arange(z_min, z_max + slice_spacing, slice_spacing)
        elif num_slices is not None:
            self.z_levels = np.linspace(z_min, z_max, num_slices)
            self.slice_spacing = (z_max - z_min) / (num_slices - 1) if num_slices > 1 else 0
        else:
            raise ValueError("Must specify either num_slices or slice_spacing")

        self.slices = []

        for i, z in enumerate(self.z_levels):
            slice_data = self._create_single_slice(z, i)
            if slice_data is not None:
                self.slices.append(slice_data)

        print(f"Created {len(self.slices)} slices from z={z_min:.2f} to z={z_max:.2f}")
        return self.slices

    def _create_single_slice(self, z_level: float, slice_index: int) -> Optional[Dict]:
        """
        Create a single slice at the specified z-level.

        Args:
            z_level: Z-coordinate for the slice
            slice_index: Index of this slice

        Returns:
            Dictionary containing slice information
        """
        try:
            # Create intersection with plane at z_level
            slice_mesh = self.mesh.section(plane_origin=[0, 0, z_level], plane_normal=[0, 0, 1])

            if slice_mesh is None:
                return None

            # Extract 2D paths (contours)
            if hasattr(slice_mesh, "entities"):
                paths_2d = []
                for entity in slice_mesh.entities:
                    if hasattr(entity, "points"):
                        # Get the points and project to 2D (remove z-coordinate)
                        points = slice_mesh.vertices[entity.points][:, :2]
                        paths_2d.append(points)
            else:
                # Fallback: if no entities, try to get vertices directly
                if hasattr(slice_mesh, "vertices") and len(slice_mesh.vertices) > 0:
                    paths_2d = [slice_mesh.vertices[:, :2]]
                else:
                    return None

            if not paths_2d:
                return None

            slice_data = {
                "z_level": z_level,
                "slice_index": slice_index,
                "contours": paths_2d,
                "raw_slice": slice_mesh,
                "num_contours": len(paths_2d),
            }

            return slice_data

        except Exception as e:
            warnings.warn(f"Failed to create slice at z={z_level}: {str(e)}")
            return None

    def get_slice_contours(self, slice_index: int) -> List[np.ndarray]:
        """Get contours for a specific slice."""
        if slice_index >= len(self.slices):
            raise IndexError(f"Slice index {slice_index} out of range")
        return self.slices[slice_index]["contours"]

    def get_slice_at_z(self, z_level: float, tolerance: float = None) -> Optional[Dict]:
        """
        Get slice closest to specified z-level.

        Args:
            z_level: Target z-coordinate
            tolerance: Maximum distance tolerance

        Returns:
            Slice dictionary or None if not found
        """
        if not self.slices:
            return None

        if tolerance is None:
            tolerance = self.slice_spacing / 2 if self.slice_spacing else float("inf")

        distances = [abs(slice_data["z_level"] - z_level) for slice_data in self.slices]
        min_dist_idx = np.argmin(distances)

        if distances[min_dist_idx] <= tolerance:
            return self.slices[min_dist_idx]
        return None

    def visualize_slice(self, slice_index: int, show_plot: bool = True) -> Optional[np.ndarray]:
        """
        Create a visualization of a specific slice.

        Args:
            slice_index: Index of slice to visualize
            show_plot: Whether to display the plot

        Returns:
            Image array if matplotlib is available
        """
        try:
            import matplotlib.pyplot as plt

            if slice_index >= len(self.slices):
                raise IndexError(f"Slice index {slice_index} out of range")

            slice_data = self.slices[slice_index]
            contours = slice_data["contours"]

            fig, ax = plt.subplots(figsize=(8, 8))

            for i, contour in enumerate(contours):
                if len(contour) > 2:
                    # Close the contour
                    closed_contour = np.vstack([contour, contour[0]])
                    ax.plot(closed_contour[:, 0], closed_contour[:, 1], label=f"Contour {i}", linewidth=2)

            ax.set_aspect("equal")
            ax.set_title(f'Slice {slice_index} at z={slice_data["z_level"]:.2f}')
            ax.set_xlabel("X (µm)")
            ax.set_ylabel("Y (µm)")
            ax.grid(True, alpha=0.3)
            ax.legend()

            if show_plot:
                plt.show()

            return fig

        except ImportError:
            print("Matplotlib not available for visualization")
            return None

    def compute_slice_properties(self, slice_index: int) -> Dict:
        """
        Compute geometric properties of a slice.

        Args:
            slice_index: Index of slice to analyze

        Returns:
            Dictionary of slice properties
        """
        if slice_index >= len(self.slices):
            raise IndexError(f"Slice index {slice_index} out of range")

        slice_data = self.slices[slice_index]
        contours = slice_data["contours"]

        properties = {
            "z_level": slice_data["z_level"],
            "num_contours": len(contours),
            "total_perimeter": 0,
            "total_area": 0,
            "contour_areas": [],
            "contour_perimeters": [],
            "centroids": [],
        }

        for contour in contours:
            if len(contour) < 3:
                continue

            # Compute area using shoelace formula
            area = self._compute_polygon_area(contour)
            properties["contour_areas"].append(abs(area))
            properties["total_area"] += abs(area)

            # Compute perimeter
            perimeter = self._compute_polygon_perimeter(contour)
            properties["contour_perimeters"].append(perimeter)
            properties["total_perimeter"] += perimeter

            # Compute centroid
            centroid = self._compute_polygon_centroid(contour)
            properties["centroids"].append(centroid)

        return properties

    def _compute_polygon_area(self, points: np.ndarray) -> float:
        """Compute area of polygon using shoelace formula."""
        if len(points) < 3:
            return 0
        x, y = points[:, 0], points[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def _compute_polygon_perimeter(self, points: np.ndarray) -> float:
        """Compute perimeter of polygon."""
        if len(points) < 2:
            return 0
        diff = np.diff(points, axis=0, append=points[0:1])
        return np.sum(np.sqrt(np.sum(diff**2, axis=1)))

    def _compute_polygon_centroid(self, points: np.ndarray) -> np.ndarray:
        """Compute centroid of polygon."""
        if len(points) < 3:
            return points.mean(axis=0) if len(points) > 0 else np.array([0, 0])
        return points.mean(axis=0)

    def get_slice_summary(self) -> Dict:
        """Get summary statistics for all slices."""
        if not self.slices:
            return {}

        z_levels = [s["z_level"] for s in self.slices]
        num_contours = [s["num_contours"] for s in self.slices]

        return {
            "num_slices": len(self.slices),
            "z_range": (min(z_levels), max(z_levels)),
            "slice_spacing": self.slice_spacing,
            "total_contours": sum(num_contours),
            "avg_contours_per_slice": np.mean(num_contours),
            "max_contours_per_slice": max(num_contours),
            "min_contours_per_slice": min(num_contours),
        }

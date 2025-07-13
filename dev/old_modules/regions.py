"""
Region detection and analysis for slice contours.

Identifies closed regions within each z-slice and computes their properties
for compartment generation.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.spatial.distance import cdist
import warnings
from dataclasses import dataclass


@dataclass
class Region:
    """Represents a closed region in a slice."""

    id: str
    slice_index: int
    z_level: float
    boundary: np.ndarray
    area: float
    perimeter: float
    centroid: np.ndarray
    is_outer: bool = False  # True for outer boundary, False for inner (holes)


class RegionDetector:
    """
    Detects and analyzes closed regions in mesh slices.
    """

    def __init__(self):
        self.regions = []
        self.slice_regions = {}  # slice_index -> list of regions

    def detect_regions(self, slices: List[Dict], min_area: float = 0.1, hole_detection: bool = True) -> List[Region]:
        """
        Detect closed regions in all slices.

        Args:
            slices: List of slice dictionaries from ZAxisSlicer
            min_area: Minimum area threshold for valid regions
            hole_detection: Whether to detect holes within regions

        Returns:
            List of detected regions
        """
        self.regions = []
        self.slice_regions = {}

        for slice_data in slices:
            slice_regions = self._detect_regions_in_slice(slice_data, min_area, hole_detection)

            slice_index = slice_data["slice_index"]
            self.slice_regions[slice_index] = slice_regions
            self.regions.extend(slice_regions)

        print(f"Detected {len(self.regions)} regions across {len(slices)} slices")
        return self.regions

    def _detect_regions_in_slice(self, slice_data: Dict, min_area: float, hole_detection: bool) -> List[Region]:
        """Detect regions in a single slice."""
        contours = slice_data["contours"]
        slice_index = slice_data["slice_index"]
        z_level = slice_data["z_level"]

        regions = []

        if not contours:
            return regions

        # Analyze each contour
        contour_data = []
        for i, contour in enumerate(contours):
            if len(contour) < 3:
                continue

            area = self._compute_area(contour)
            if abs(area) < min_area:
                continue

            perimeter = self._compute_perimeter(contour)
            centroid = self._compute_centroid(contour)

            contour_data.append(
                {
                    "index": i,
                    "contour": contour,
                    "area": area,
                    "abs_area": abs(area),
                    "perimeter": perimeter,
                    "centroid": centroid,
                    "is_clockwise": area < 0,  # Negative area indicates clockwise
                }
            )

        if not contour_data:
            return regions

        # Sort by area (largest first)
        contour_data.sort(key=lambda x: x["abs_area"], reverse=True)

        # Detect containment relationships if hole detection is enabled
        if hole_detection and len(contour_data) > 1:
            containment = self._compute_containment_matrix(contour_data)
        else:
            containment = np.zeros((len(contour_data), len(contour_data)))

        # Create regions
        for i, data in enumerate(contour_data):
            # Determine if this is an outer boundary or a hole
            is_outer = True
            if hole_detection:
                # Count how many other contours contain this one
                num_containers = np.sum(containment[:, i])
                is_outer = (num_containers % 2) == 0  # Even number means outer

            region_id = f"slice_{slice_index}_region_{len(regions)}"

            region = Region(
                id=region_id,
                slice_index=slice_index,
                z_level=z_level,
                boundary=data["contour"],
                area=data["abs_area"],
                perimeter=data["perimeter"],
                centroid=data["centroid"],
                is_outer=is_outer,
            )

            regions.append(region)

        return regions

    def _compute_containment_matrix(self, contour_data: List[Dict]) -> np.ndarray:
        """
        Compute which contours contain which other contours.

        Returns:
            Binary matrix where [i,j] = 1 if contour i contains contour j
        """
        n = len(contour_data)
        containment = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    if self._contour_contains_contour(contour_data[i]["contour"], contour_data[j]["contour"]):
                        containment[i, j] = 1

        return containment

    def _contour_contains_contour(self, outer: np.ndarray, inner: np.ndarray) -> bool:
        """Check if outer contour contains inner contour."""
        # Sample a few points from inner contour
        sample_indices = np.linspace(0, len(inner) - 1, min(5, len(inner)), dtype=int)
        sample_points = inner[sample_indices]

        # Check if all sample points are inside outer contour
        for point in sample_points:
            if not self._point_in_polygon(point, outer):
                return False
        return True

    def _point_in_polygon(self, point: np.ndarray, polygon: np.ndarray) -> bool:
        """Ray casting algorithm to check if point is inside polygon."""
        x, y = point[0], point[1]
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def _compute_area(self, contour: np.ndarray) -> float:
        """Compute signed area using shoelace formula."""
        if len(contour) < 3:
            return 0
        x, y = contour[:, 0], contour[:, 1]
        return 0.5 * (np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def _compute_perimeter(self, contour: np.ndarray) -> float:
        """Compute perimeter length."""
        if len(contour) < 2:
            return 0
        diff = np.diff(contour, axis=0, append=contour[0:1])
        return np.sum(np.sqrt(np.sum(diff**2, axis=1)))

    def _compute_centroid(self, contour: np.ndarray) -> np.ndarray:
        """Compute centroid coordinates."""
        return contour.mean(axis=0)

    def get_regions_at_slice(self, slice_index: int) -> List[Region]:
        """Get all regions for a specific slice."""
        return self.slice_regions.get(slice_index, [])

    def get_outer_regions_at_slice(self, slice_index: int) -> List[Region]:
        """Get only outer (non-hole) regions for a specific slice."""
        return [r for r in self.get_regions_at_slice(slice_index) if r.is_outer]

    def get_region_by_id(self, region_id: str) -> Optional[Region]:
        """Get region by ID."""
        for region in self.regions:
            if region.id == region_id:
                return region
        return None

    def compute_region_statistics(self) -> Dict:
        """Compute statistics across all detected regions."""
        if not self.regions:
            return {}

        areas = [r.area for r in self.regions]
        perimeters = [r.perimeter for r in self.regions]

        outer_regions = [r for r in self.regions if r.is_outer]
        hole_regions = [r for r in self.regions if not r.is_outer]

        stats = {
            "total_regions": len(self.regions),
            "outer_regions": len(outer_regions),
            "hole_regions": len(hole_regions),
            "area_stats": {
                "mean": np.mean(areas),
                "std": np.std(areas),
                "min": np.min(areas),
                "max": np.max(areas),
                "total": np.sum(areas),
            },
            "perimeter_stats": {
                "mean": np.mean(perimeters),
                "std": np.std(perimeters),
                "min": np.min(perimeters),
                "max": np.max(perimeters),
                "total": np.sum(perimeters),
            },
        }

        # Per-slice statistics
        slice_counts = {}
        for region in self.regions:
            slice_idx = region.slice_index
            if slice_idx not in slice_counts:
                slice_counts[slice_idx] = {"total": 0, "outer": 0, "holes": 0}
            slice_counts[slice_idx]["total"] += 1
            if region.is_outer:
                slice_counts[slice_idx]["outer"] += 1
            else:
                slice_counts[slice_idx]["holes"] += 1

        if slice_counts:
            total_counts = [counts["total"] for counts in slice_counts.values()]
            outer_counts = [counts["outer"] for counts in slice_counts.values()]

            stats["regions_per_slice"] = {
                "mean": np.mean(total_counts),
                "std": np.std(total_counts),
                "min": np.min(total_counts),
                "max": np.max(total_counts),
            }

            stats["outer_regions_per_slice"] = {
                "mean": np.mean(outer_counts),
                "std": np.std(outer_counts),
                "min": np.min(outer_counts),
                "max": np.max(outer_counts),
            }

        return stats

    def filter_regions(
        self,
        min_area: float = None,
        max_area: float = None,
        min_perimeter: float = None,
        max_perimeter: float = None,
        outer_only: bool = False,
    ) -> List[Region]:
        """
        Filter regions based on geometric criteria.

        Args:
            min_area: Minimum area threshold
            max_area: Maximum area threshold
            min_perimeter: Minimum perimeter threshold
            max_perimeter: Maximum perimeter threshold
            outer_only: Only return outer regions (no holes)

        Returns:
            Filtered list of regions
        """
        filtered = self.regions

        if outer_only:
            filtered = [r for r in filtered if r.is_outer]

        if min_area is not None:
            filtered = [r for r in filtered if r.area >= min_area]

        if max_area is not None:
            filtered = [r for r in filtered if r.area <= max_area]

        if min_perimeter is not None:
            filtered = [r for r in filtered if r.perimeter >= min_perimeter]

        if max_perimeter is not None:
            filtered = [r for r in filtered if r.perimeter <= max_perimeter]

        return filtered

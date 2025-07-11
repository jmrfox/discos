"""
Voxel-based 3D modeling for GenCoMo using voxelmap.

This module provides a VoxelModel class that wraps voxelmap functionality
for working with 3D voxel data (equivalent to z-stack format) in GenCoMo.
"""

import numpy as np
from typing import Optional, Tuple, List, Union, Dict, Any, Sequence
import warnings


class VoxelModel:
    """
    A 3D voxel model for neuronal morphologies using voxelmap.

    This class provides a high-level interface for working with 3D voxel data,
    including creation, manipulation, analysis, and conversion operations.
    """

    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        resolution: Union[float, Tuple[float, float, float]] = 1.0,
        origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a VoxelModel.

        Args:
            data: 3D numpy array (z, y, x) with voxel data
            resolution: Voxel resolution in μm (single value or (z, y, x) tuple)
            origin: Origin coordinates in μm (x, y, z)
            metadata: Additional metadata dictionary
        """
        try:
            import voxelmap

            self._voxelmap = voxelmap
        except ImportError:
            raise ImportError("voxelmap is required for VoxelModel. Install with: pip install voxelmap")

        # Handle resolution
        if isinstance(resolution, (int, float)):
            self.resolution = (float(resolution), float(resolution), float(resolution))
        else:
            self.resolution = tuple(float(r) for r in resolution)

        self.origin = tuple(float(o) for o in origin)
        self.metadata = metadata or {}

        # Initialize voxel data
        if data is not None:
            self.data = np.asarray(data, dtype=np.uint8)
            self._validate_data()
        else:
            self.data = None

        # Initialize voxelmap object
        self._voxelmap_obj = None
        self._update_voxelmap()

    def _validate_data(self):
        """Validate the voxel data."""
        if self.data is None:
            return

        if self.data.ndim != 3:
            raise ValueError(f"Voxel data must be 3D, got {self.data.ndim}D")

        if self.data.dtype != np.uint8:
            warnings.warn("Converting voxel data to uint8")
            self.data = self.data.astype(np.uint8)

    def _update_voxelmap(self):
        """Update the internal voxelmap object."""
        if self.data is not None:
            # Create voxelmap object with proper scaling
            self._voxelmap_obj = self._voxelmap.VoxelMap(data=self.data, voxel_size=self.resolution, origin=self.origin)

    @classmethod
    def from_zstack(cls, z_stack: np.ndarray, metadata: Dict[str, Any]) -> "VoxelModel":
        """
        Create a VoxelModel from GenCoMo z-stack format.

        Args:
            z_stack: 3D numpy array (z, y, x) from GenCoMo
            metadata: Metadata dictionary from GenCoMo

        Returns:
            VoxelModel instance
        """
        # Extract resolution information
        z_res = metadata.get("z_resolution", 1.0)
        xy_res = metadata.get("xy_resolution", 1.0)
        resolution = (z_res, xy_res, xy_res)

        # Extract origin information
        x_coords = metadata.get("x_coords", None)
        y_coords = metadata.get("y_coords", None)
        z_coords = metadata.get("z_coords", None)

        if x_coords is not None and y_coords is not None and z_coords is not None:
            origin = (x_coords[0], y_coords[0], z_coords[0])
        else:
            origin = (0.0, 0.0, 0.0)

        return cls(data=z_stack, resolution=resolution, origin=origin, metadata=metadata)

    def to_zstack(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Convert to GenCoMo z-stack format.

        Returns:
            Tuple of (z_stack, metadata) in GenCoMo format
        """
        if self.data is None:
            raise ValueError("No voxel data to convert")

        # Create coordinate arrays
        nz, ny, nx = self.data.shape
        z_res, y_res, x_res = self.resolution

        x_coords = np.linspace(self.origin[0], self.origin[0] + nx * x_res, nx)
        y_coords = np.linspace(self.origin[1], self.origin[1] + ny * y_res, ny)
        z_coords = np.linspace(self.origin[2], self.origin[2] + nz * z_res, nz)

        # Create metadata
        metadata = {
            "z_resolution": z_res,
            "xy_resolution": x_res,  # Assume x and y have same resolution
            "x_coords": x_coords,
            "y_coords": y_coords,
            "z_coords": z_coords,
            "bounds": {
                "x_range": (x_coords[0], x_coords[-1]),
                "y_range": (y_coords[0], y_coords[-1]),
                "z_range": (z_coords[0], z_coords[-1]),
            },
            "shape": self.data.shape,
            "total_voxels": self.data.size,
            "neuron_voxels": np.sum(self.data),
            "volume_um3": np.sum(self.data) * z_res * y_res * x_res,
            **self.metadata,
        }

        return self.data.copy(), metadata

    @property
    def shape(self) -> Optional[Tuple[int, int, int]]:
        """Get the shape of the voxel data."""
        return self.data.shape if self.data is not None else None

    @property
    def volume(self) -> float:
        """Get the volume of filled voxels in μm³."""
        if self.data is None:
            return 0.0
        return np.sum(self.data) * np.prod(self.resolution)

    @property
    def bounding_box(self) -> Optional[Tuple[float, float, float, float, float, float]]:
        """Get bounding box as (x_min, x_max, y_min, y_max, z_min, z_max)."""
        if self.data is None:
            return None

        nz, ny, nx = self.data.shape
        z_res, y_res, x_res = self.resolution

        return (
            self.origin[0],
            self.origin[0] + nx * x_res,
            self.origin[1],
            self.origin[1] + ny * y_res,
            self.origin[2],
            self.origin[2] + nz * z_res,
        )

    def rotate(self, angle_x: float = 0.0, angle_y: float = 0.0, angle_z: float = 0.0, order: int = 1) -> "VoxelModel":
        """
        Rotate the voxel model about coordinate axes.

        Args:
            angle_x: Rotation angle about x-axis in degrees
            angle_y: Rotation angle about y-axis in degrees
            angle_z: Rotation angle about z-axis in degrees
            order: Interpolation order (0=nearest, 1=linear, etc.)

        Returns:
            New VoxelModel with rotated data
        """
        if self.data is None:
            raise ValueError("No voxel data to rotate")

        # Use GenCoMo's rotation function
        from .z_stack import rotate_zstack

        # Convert to z-stack format
        z_stack, metadata = self.to_zstack()

        # Apply rotation
        rotated_data, rotated_metadata = rotate_zstack(
            z_stack, metadata, angle_x=angle_x, angle_y=angle_y, angle_z=angle_z, order=order
        )

        # Create new VoxelModel
        return VoxelModel.from_zstack(rotated_data, rotated_metadata)

    def rotate_arbitrary_axis(self, axis: Sequence[float], angle: float, order: int = 1) -> "VoxelModel":
        """
        Rotate about an arbitrary axis.

        Args:
            axis: 3D vector defining rotation axis [x, y, z]
            angle: Rotation angle in degrees
            order: Interpolation order

        Returns:
            New VoxelModel with rotated data
        """
        if self.data is None:
            raise ValueError("No voxel data to rotate")

        # Use GenCoMo's rotation function
        from .z_stack import rotate_zstack_arbitrary_axis

        # Convert to z-stack format
        z_stack, metadata = self.to_zstack()

        # Apply rotation
        rotated_data, rotated_metadata = rotate_zstack_arbitrary_axis(
            z_stack, metadata, axis=np.array(axis), angle=angle, order=order
        )

        # Create new VoxelModel
        return VoxelModel.from_zstack(rotated_data, rotated_metadata)

    def translate(self, offset: Sequence[float]) -> "VoxelModel":
        """
        Translate the voxel model.

        Args:
            offset: Translation offset as (x, y, z) in μm

        Returns:
            New VoxelModel with translated origin
        """
        offset = tuple(float(o) for o in offset)
        new_origin = tuple(self.origin[i] + offset[i] for i in range(3))

        return VoxelModel(
            data=self.data.copy() if self.data is not None else None,
            resolution=self.resolution,
            origin=new_origin,
            metadata=self.metadata.copy(),
        )

    def scale(self, factor: Union[float, Sequence[float]]) -> "VoxelModel":
        """
        Scale the voxel model by changing resolution.

        Args:
            factor: Scaling factor (single value or (x, y, z) tuple)

        Returns:
            New VoxelModel with scaled resolution
        """
        if isinstance(factor, (int, float)):
            factor = (factor, factor, factor)
        else:
            factor = tuple(float(f) for f in factor)

        new_resolution = tuple(self.resolution[i] * factor[i] for i in range(3))

        return VoxelModel(
            data=self.data.copy() if self.data is not None else None,
            resolution=new_resolution,
            origin=self.origin,
            metadata=self.metadata.copy(),
        )

    def crop(self, bounds: Tuple[int, int, int, int, int, int]) -> "VoxelModel":
        """
        Crop the voxel model to specified bounds.

        Args:
            bounds: Crop bounds as (z_min, z_max, y_min, y_max, x_min, x_max)

        Returns:
            New VoxelModel with cropped data
        """
        if self.data is None:
            raise ValueError("No voxel data to crop")

        z_min, z_max, y_min, y_max, x_min, x_max = bounds

        # Validate bounds
        nz, ny, nx = self.data.shape
        z_min = max(0, min(z_min, nz))
        z_max = max(z_min, min(z_max, nz))
        y_min = max(0, min(y_min, ny))
        y_max = max(y_min, min(y_max, ny))
        x_min = max(0, min(x_min, nx))
        x_max = max(x_min, min(x_max, nx))

        # Crop data
        cropped_data = self.data[z_min:z_max, y_min:y_max, x_min:x_max]

        # Update origin
        z_res, y_res, x_res = self.resolution
        new_origin = (self.origin[0] + x_min * x_res, self.origin[1] + y_min * y_res, self.origin[2] + z_min * z_res)

        return VoxelModel(
            data=cropped_data, resolution=self.resolution, origin=new_origin, metadata=self.metadata.copy()
        )

    def dilate(self, radius: int = 1) -> "VoxelModel":
        """
        Dilate (expand) the voxel structure.

        Args:
            radius: Dilation radius in voxels

        Returns:
            New VoxelModel with dilated data
        """
        if self.data is None:
            raise ValueError("No voxel data to dilate")

        try:
            from scipy.ndimage import binary_dilation
        except ImportError:
            raise ImportError("scipy is required for morphological operations")

        # Create spherical structuring element
        from scipy.ndimage import generate_binary_structure

        struct = generate_binary_structure(3, 1)

        # Apply dilation
        dilated_data = binary_dilation(self.data.astype(bool), structure=struct, iterations=radius).astype(np.uint8)

        return VoxelModel(
            data=dilated_data, resolution=self.resolution, origin=self.origin, metadata=self.metadata.copy()
        )

    def erode(self, radius: int = 1) -> "VoxelModel":
        """
        Erode (shrink) the voxel structure.

        Args:
            radius: Erosion radius in voxels

        Returns:
            New VoxelModel with eroded data
        """
        if self.data is None:
            raise ValueError("No voxel data to erode")

        try:
            from scipy.ndimage import binary_erosion
        except ImportError:
            raise ImportError("scipy is required for morphological operations")

        # Create spherical structuring element
        from scipy.ndimage import generate_binary_structure

        struct = generate_binary_structure(3, 1)

        # Apply erosion
        eroded_data = binary_erosion(self.data.astype(bool), structure=struct, iterations=radius).astype(np.uint8)

        return VoxelModel(
            data=eroded_data, resolution=self.resolution, origin=self.origin, metadata=self.metadata.copy()
        )

    def union(self, other: "VoxelModel") -> "VoxelModel":
        """
        Compute union with another VoxelModel.

        Args:
            other: Another VoxelModel

        Returns:
            New VoxelModel with union of both models
        """
        if self.data is None or other.data is None:
            raise ValueError("Both models must have voxel data")

        if self.data.shape != other.data.shape:
            raise ValueError("Models must have same shape for union")

        union_data = np.logical_or(self.data, other.data).astype(np.uint8)

        return VoxelModel(
            data=union_data, resolution=self.resolution, origin=self.origin, metadata=self.metadata.copy()
        )

    def intersection(self, other: "VoxelModel") -> "VoxelModel":
        """
        Compute intersection with another VoxelModel.

        Args:
            other: Another VoxelModel

        Returns:
            New VoxelModel with intersection of both models
        """
        if self.data is None or other.data is None:
            raise ValueError("Both models must have voxel data")

        if self.data.shape != other.data.shape:
            raise ValueError("Models must have same shape for intersection")

        intersection_data = np.logical_and(self.data, other.data).astype(np.uint8)

        return VoxelModel(
            data=intersection_data, resolution=self.resolution, origin=self.origin, metadata=self.metadata.copy()
        )

    def difference(self, other: "VoxelModel") -> "VoxelModel":
        """
        Compute difference with another VoxelModel.

        Args:
            other: Another VoxelModel to subtract

        Returns:
            New VoxelModel with difference (self - other)
        """
        if self.data is None or other.data is None:
            raise ValueError("Both models must have voxel data")

        if self.data.shape != other.data.shape:
            raise ValueError("Models must have same shape for difference")

        difference_data = np.logical_and(self.data, np.logical_not(other.data)).astype(np.uint8)

        return VoxelModel(
            data=difference_data, resolution=self.resolution, origin=self.origin, metadata=self.metadata.copy()
        )

    def analyze_properties(self) -> Dict[str, Any]:
        """
        Analyze properties of the voxel model.

        Returns:
            Dictionary of analysis results
        """
        if self.data is None:
            return {}

        # Use GenCoMo's analysis function
        from .z_stack import analyze_zstack_properties

        z_stack, metadata = self.to_zstack()
        return analyze_zstack_properties(z_stack, metadata)

    def visualize_3d(self, **kwargs) -> Optional[object]:
        """
        Create 3D visualization of the voxel model.

        Args:
            **kwargs: Arguments passed to visualization function

        Returns:
            Figure object
        """
        if self.data is None:
            raise ValueError("No voxel data to visualize")

        # Use GenCoMo's visualization function
        from .z_stack import visualize_zstack_3d

        z_stack, metadata = self.to_zstack()
        return visualize_zstack_3d(z_stack, metadata, **kwargs)

    def visualize_slices(self, **kwargs) -> Optional[object]:
        """
        Create interactive slice viewer for the voxel model.

        Args:
            **kwargs: Arguments passed to visualization function

        Returns:
            Interactive figure object
        """
        if self.data is None:
            raise ValueError("No voxel data to visualize")

        # Use GenCoMo's visualization function
        from .z_stack import visualize_zstack_slices

        z_stack, metadata = self.to_zstack()
        return visualize_zstack_slices(z_stack, metadata, **kwargs)

    def save(self, filepath: str, format: str = "npz") -> None:
        """
        Save the voxel model to file.

        Args:
            filepath: Output file path
            format: File format ('npz', 'npy', 'h5')
        """
        if self.data is None:
            raise ValueError("No voxel data to save")

        # Use GenCoMo's save function
        from .z_stack import save_zstack_data

        z_stack, metadata = self.to_zstack()
        save_zstack_data(z_stack, metadata, filepath, format)

    @classmethod
    def load(cls, filepath: str, format: str = "npz") -> "VoxelModel":
        """
        Load a voxel model from file.

        Args:
            filepath: Input file path
            format: File format ('npz', 'npy', 'h5')

        Returns:
            VoxelModel instance
        """
        # Use GenCoMo's load function
        from .z_stack import load_zstack_data

        z_stack, metadata = load_zstack_data(filepath, format)
        return cls.from_zstack(z_stack, metadata)

    def __repr__(self) -> str:
        """String representation."""
        if self.data is None:
            return "VoxelModel(empty)"

        return f"VoxelModel(shape={self.shape}, " f"resolution={self.resolution}, " f"volume={self.volume:.1f}μm³)"

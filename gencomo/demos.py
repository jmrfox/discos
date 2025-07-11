"""
Demo mesh generation functions for GenCoMo.

Provides specific example mesh generators for testing and demonstration purposes.
These functions create standard neuronal morphologies for tutorials and examples.
"""

import numpy as np
from typing import Tuple, Union, Dict, Any
import warnings

# Z-stack demo functions


def create_cylinder_zstack(
    length: float = 100.0,
    radius: float = 5.0,
    z_resolution: float = 1.0,
    xy_resolution: float = 0.5,
    padding: float = 2.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Create a cylindrical neuron as a z-stack of binary arrays.

    Args:
        length: Cylinder length along z-axis (µm)
        radius: Cylinder radius (µm)
        z_resolution: Resolution along z-axis (µm per slice)
        xy_resolution: Resolution in x-y plane (µm per pixel)
        padding: Padding around cylinder (µm)

    Returns:
        Tuple of (z_stack, metadata)
    """
    # Calculate grid dimensions
    x_min, x_max = -radius - padding, radius + padding
    y_min, y_max = -radius - padding, radius + padding
    z_min, z_max = -padding, length + padding

    nx = int(np.ceil((x_max - x_min) / xy_resolution))
    ny = int(np.ceil((y_max - y_min) / xy_resolution))
    nz = int(np.ceil((z_max - z_min) / z_resolution))

    # Create coordinate grids
    x_coords = np.linspace(x_min, x_max, nx)
    y_coords = np.linspace(y_min, y_max, ny)
    z_coords = np.linspace(z_min, z_max, nz)

    # Initialize z-stack
    z_stack = np.zeros((nz, ny, nx), dtype=np.uint8)

    # Fill cylinder analytically
    for k, z in enumerate(z_coords):
        if 0 <= z <= length:  # Within cylinder z-range
            for j, y in enumerate(y_coords):
                for i, x in enumerate(x_coords):
                    # Check if point is inside cylinder
                    distance_from_axis = np.sqrt(x**2 + y**2)
                    z_stack[k, j, i] = 1 if distance_from_axis <= radius else 0

    # Create metadata
    metadata = {
        "morphology_type": "cylinder",
        "z_resolution": z_resolution,
        "xy_resolution": xy_resolution,
        "x_coords": x_coords,
        "y_coords": y_coords,
        "z_coords": z_coords,
        "bounds": {"x_range": (x_min, x_max), "y_range": (y_min, y_max), "z_range": (z_min, z_max)},
        "shape": z_stack.shape,
        "cylinder_params": {"length": length, "radius": radius},
        "total_voxels": z_stack.size,
        "neuron_voxels": np.sum(z_stack),
        "volume_um3": np.sum(z_stack) * xy_resolution * xy_resolution * z_resolution,
    }

    return z_stack, metadata


def create_branching_zstack(
    trunk_length: float = 60.0,
    trunk_radius: float = 5.0,
    branch_length: float = 40.0,
    branch_radius: float = 3.0,
    branch_angle: float = 45.0,
    z_resolution: float = 1.0,
    xy_resolution: float = 0.5,
    padding: float = 3.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Create a Y-shaped neuron as a z-stack of binary arrays.

    Args:
        trunk_length: Length of main trunk (µm)
        trunk_radius: Radius of trunk (µm)
        branch_length: Length of each branch (µm)
        branch_radius: Radius of branches (µm)
        branch_angle: Angle of branches from vertical (degrees)
        z_resolution: Resolution along z-axis (µm per slice)
        xy_resolution: Resolution in x-y plane (µm per pixel)
        padding: Padding around structure (µm)

    Returns:
        Tuple of (z_stack, metadata)
    """
    # Convert angle to radians
    angle_rad = np.radians(branch_angle)

    # Calculate bounds
    max_branch_x = branch_length * np.sin(angle_rad)
    max_radius = max(trunk_radius, branch_radius)

    x_min = -max_branch_x - max_radius - padding
    x_max = max_branch_x + max_radius + padding
    y_min = -max_radius - padding
    y_max = max_radius + padding
    z_min = -padding
    z_max = trunk_length + branch_length * np.cos(angle_rad) + padding

    nx = int(np.ceil((x_max - x_min) / xy_resolution))
    ny = int(np.ceil((y_max - y_min) / xy_resolution))
    nz = int(np.ceil((z_max - z_min) / z_resolution))

    # Create coordinate grids
    x_coords = np.linspace(x_min, x_max, nx)
    y_coords = np.linspace(y_min, y_max, ny)
    z_coords = np.linspace(z_min, z_max, nz)

    # Initialize z-stack
    z_stack = np.zeros((nz, ny, nx), dtype=np.uint8)

    # Fill Y-shape analytically
    for k, z in enumerate(z_coords):
        for j, y in enumerate(y_coords):
            for i, x in enumerate(x_coords):
                inside = False

                # Check trunk
                if 0 <= z <= trunk_length:
                    dist_from_trunk = np.sqrt(x**2 + y**2)
                    if dist_from_trunk <= trunk_radius:
                        inside = True

                # Check branches
                if z > trunk_length:
                    branch_z = z - trunk_length
                    max_branch_z = branch_length * np.cos(angle_rad)

                    if branch_z <= max_branch_z:
                        # Branch 1 (positive x direction)
                        branch1_center_x = branch_z * np.tan(angle_rad)
                        dist_from_branch1 = np.sqrt((x - branch1_center_x) ** 2 + y**2)

                        # Branch 2 (negative x direction)
                        branch2_center_x = -branch_z * np.tan(angle_rad)
                        dist_from_branch2 = np.sqrt((x - branch2_center_x) ** 2 + y**2)

                        if dist_from_branch1 <= branch_radius or dist_from_branch2 <= branch_radius:
                            inside = True

                z_stack[k, j, i] = 1 if inside else 0

    # Create metadata
    metadata = {
        "morphology_type": "y_shaped",
        "z_resolution": z_resolution,
        "xy_resolution": xy_resolution,
        "x_coords": x_coords,
        "y_coords": y_coords,
        "z_coords": z_coords,
        "bounds": {"x_range": (x_min, x_max), "y_range": (y_min, y_max), "z_range": (z_min, z_max)},
        "shape": z_stack.shape,
        "y_params": {
            "trunk_length": trunk_length,
            "trunk_radius": trunk_radius,
            "branch_length": branch_length,
            "branch_radius": branch_radius,
            "branch_angle": branch_angle,
        },
        "total_voxels": z_stack.size,
        "neuron_voxels": np.sum(z_stack),
        "volume_um3": np.sum(z_stack) * xy_resolution * xy_resolution * z_resolution,
    }

    return z_stack, metadata


def create_torus_zstack(
    major_radius: float = 8.0,
    minor_radius: float = 3.0,
    center: Tuple[float, float, float] = (0, 0, 0),
    z_resolution: float = 0.5,
    xy_resolution: float = 0.5,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Create a torus (donut) z-stack morphology.

    Args:
        major_radius: Distance from center to tube center (μm)
        minor_radius: Radius of the tube (μm)
        center: Center position (x, y, z) in μm
        z_resolution: Resolution along z-axis (μm per slice)
        xy_resolution: Resolution in x-y plane (μm per pixel)

    Returns:
        Tuple of (z_stack, metadata)
    """
    # Calculate grid bounds
    max_radius = major_radius + minor_radius
    padding = 2.0

    x_min, y_min, z_min = -max_radius - padding, -max_radius - padding, -minor_radius - padding
    x_max, y_max, z_max = max_radius + padding, max_radius + padding, minor_radius + padding

    # Calculate grid dimensions
    nx = int(np.ceil((x_max - x_min) / xy_resolution))
    ny = int(np.ceil((y_max - y_min) / xy_resolution))
    nz = int(np.ceil((z_max - z_min) / z_resolution))

    # Create coordinate grids
    x_coords = np.linspace(x_min, x_max, nx)
    y_coords = np.linspace(y_min, y_max, ny)
    z_coords = np.linspace(z_min, z_max, nz)

    # Initialize z-stack
    z_stack = np.zeros((nz, ny, nx), dtype=np.uint8)

    # Fill torus
    for k, z in enumerate(z_coords):
        for j, y in enumerate(y_coords):
            for i, x in enumerate(x_coords):
                # Distance from z-axis
                rho = np.sqrt(x**2 + y**2)

                # Distance from torus center circle
                distance_to_tube = np.sqrt((rho - major_radius) ** 2 + z**2)

                # Inside torus if distance to tube < minor_radius
                if distance_to_tube <= minor_radius:
                    z_stack[k, j, i] = 1

    # Create metadata
    metadata = {
        "major_radius": major_radius,
        "minor_radius": minor_radius,
        "center": center,
        "z_resolution": z_resolution,
        "xy_resolution": xy_resolution,
        "x_coords": x_coords,
        "y_coords": y_coords,
        "z_coords": z_coords,
        "bounds": (x_min, x_max, y_min, y_max, z_min, z_max),
        "shape": z_stack.shape,
        "morphology_type": "torus",
        "total_voxels": z_stack.size,
        "neuron_voxels": np.sum(z_stack),
        "volume_um3": np.sum(z_stack) * xy_resolution * xy_resolution * z_resolution,
    }

    return z_stack, metadata

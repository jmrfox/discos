"""
Demo mesh generation functions for GenCoMo.

Provides specific example mesh generators for testing and demonstration purposes.
These functions create standard neuronal morphologies for tutorials and examples.
"""

import numpy as np
import trimesh
from typing import Tuple, Union, Dict, Any
import warnings


def create_cylinder_mesh(
    length: float = 100.0,
    radius: float = 5.0,
    num_segments: int = 16,
    num_z_divisions: int = 50,
    return_mesh: bool = True,
) -> Union[trimesh.Trimesh, Tuple[np.ndarray, np.ndarray]]:
    """
    Create a simple cylindrical mesh.

    Args:
        length: Cylinder length along z-axis (µm)
        radius: Cylinder radius (µm)
        num_segments: Number of circumferential segments
        num_z_divisions: Number of divisions along z-axis
        return_mesh: If True, return trimesh object; if False, return (vertices, faces) tuple

    Returns:
        Cylindrical trimesh object or (vertices, faces) tuple
    """
    # Create vertices
    theta = np.linspace(0, 2 * np.pi, num_segments, endpoint=False)
    z_levels = np.linspace(0, length, num_z_divisions)

    vertices = []
    faces = []

    # Generate vertices
    for z in z_levels:
        for angle in theta:
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            vertices.append([x, y, z])

    vertices = np.array(vertices)

    # Generate faces (connect adjacent rings)
    for i in range(num_z_divisions - 1):
        for j in range(num_segments):
            # Current and next ring base indices
            curr_base = i * num_segments
            next_base = (i + 1) * num_segments

            # Four vertices of current quad
            v1 = curr_base + j
            v2 = curr_base + (j + 1) % num_segments
            v3 = next_base + j
            v4 = next_base + (j + 1) % num_segments

            # Two triangular faces per quad
            faces.append([v1, v2, v3])
            faces.append([v2, v4, v3])

    # Add end caps
    center_bottom = len(vertices)
    center_top = center_bottom + 1
    vertices = np.vstack([vertices, [0, 0, 0], [0, 0, length]])

    # Bottom cap
    for j in range(num_segments):
        v1 = j
        v2 = (j + 1) % num_segments
        faces.append([center_bottom, v2, v1])

    # Top cap
    top_ring_base = (num_z_divisions - 1) * num_segments
    for j in range(num_segments):
        v1 = top_ring_base + j
        v2 = top_ring_base + (j + 1) % num_segments
        faces.append([center_top, v1, v2])

    mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces))

    if return_mesh:
        return mesh
    else:
        return vertices, np.array(faces)


def create_y_shaped_mesh(
    trunk_length: float = 60.0,
    trunk_radius: float = 5.0,
    branch_length: float = 40.0,
    branch_radius: float = 3.0,
    branch_angle: float = 45.0,
    num_segments: int = 12,
    return_mesh: bool = True,
) -> Union[trimesh.Trimesh, Tuple[np.ndarray, np.ndarray]]:
    """
    Create a Y-shaped neuronal mesh with a main trunk and two branches.

    Args:
        trunk_length: Length of main trunk (µm)
        trunk_radius: Radius of main trunk (µm)
        branch_length: Length of each branch (µm)
        branch_radius: Radius of each branch (µm)
        branch_angle: Angle of branches from vertical (degrees)
        num_segments: Number of circumferential segments

    Returns:
        Y-shaped trimesh object
    """
    # Create main trunk
    trunk = create_cylinder_mesh(
        length=trunk_length, radius=trunk_radius, num_segments=num_segments, num_z_divisions=30
    )

    # Create first branch
    branch1 = create_cylinder_mesh(
        length=branch_length, radius=branch_radius, num_segments=num_segments, num_z_divisions=20
    )

    # Rotate and translate first branch
    angle_rad = np.radians(branch_angle)
    rotation_matrix = trimesh.transformations.rotation_matrix(angle_rad, [0, 1, 0])
    branch1.apply_transform(rotation_matrix)

    # Position branch at top of trunk
    branch1_offset = np.array([0, 0, trunk_length])
    branch1.vertices += branch1_offset

    # Create second branch (mirror of first)
    branch2 = branch1.copy()
    mirror_matrix = trimesh.transformations.rotation_matrix(np.pi, [0, 0, 1])
    # Apply mirror around trunk center
    branch2.vertices -= branch1_offset
    branch2.apply_transform(mirror_matrix)
    branch2.vertices += branch1_offset

    # Combine all parts
    combined = trimesh.util.concatenate([trunk, branch1, branch2])

    # Smooth the mesh to reduce artifacts at junctions
    combined = combined.smoothed()

    if return_mesh:
        return combined
    else:
        return combined.vertices, combined.faces


def create_mesh_with_hole(
    length: float = 80.0,
    outer_radius: float = 6.0,
    hole_radius: float = 2.0,
    hole_position: float = 40.0,
    hole_direction: str = "x",
    num_segments: int = 16,
    return_mesh: bool = True,
) -> Union[trimesh.Trimesh, Tuple[np.ndarray, np.ndarray]]:
    """
    Create a cylindrical mesh with a hole drilled perpendicular to its main axis.
    This creates a torus-like topology embedded in a cylinder.

    Args:
        length: Total length of cylinder along z-axis (µm)
        outer_radius: Outer radius of cylinder (µm)
        hole_radius: Radius of the perpendicular hole (µm)
        hole_position: Z-position where hole center is located (µm)
        hole_direction: Direction of hole ('x' or 'y' perpendicular to z-axis)
        num_segments: Number of circumferential segments

    Returns:
        Mesh with perpendicular hole (torus-like topology)
    """
    # Create outer cylinder along z-axis
    outer_mesh = create_cylinder_mesh(
        length=length, radius=outer_radius, num_segments=num_segments, num_z_divisions=50, return_mesh=True
    )

    # Create hole cylinder that goes perpendicular to the main cylinder
    # The hole needs to be long enough to completely penetrate the outer cylinder
    hole_length = outer_radius * 2.5  # Make sure it goes all the way through

    hole_mesh = create_cylinder_mesh(
        length=hole_length,
        radius=hole_radius,
        num_segments=max(8, num_segments // 2),
        num_z_divisions=20,
        return_mesh=True,
    )

    # Rotate the hole cylinder to be perpendicular to the main cylinder
    if hole_direction == "x":
        # Rotate 90 degrees around y-axis to align hole with x-axis
        rotation_matrix = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
    elif hole_direction == "y":
        # Rotate 90 degrees around x-axis to align hole with y-axis
        rotation_matrix = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
    else:
        raise ValueError(f"hole_direction must be 'x' or 'y', got {hole_direction}")

    hole_mesh.apply_transform(rotation_matrix)

    # Position the hole at the specified z-position
    hole_mesh.vertices[:, 2] += hole_position

    try:
        # Use boolean difference to create the hole
        result = outer_mesh.difference(hole_mesh)

        # If boolean operation fails, return the outer mesh with warning
        if result is None or len(result.vertices) == 0:
            warnings.warn("Boolean operation failed, returning mesh without hole")
            if return_mesh:
                return outer_mesh
            else:
                return outer_mesh.vertices, outer_mesh.faces

        if return_mesh:
            return result
        else:
            return result.vertices, result.faces

    except Exception as e:
        warnings.warn(f"Could not create hole: {e}. Returning mesh without hole.")
        if return_mesh:
            return outer_mesh
        else:
            return outer_mesh.vertices, outer_mesh.faces


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


def create_y_shaped_zstack(
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


def create_hole_zstack(
    length: float = 80.0,
    outer_radius: float = 6.0,
    hole_radius: float = 3.0,
    hole_position: float = 40.0,
    hole_direction: str = "x",
    z_resolution: float = 1.0,
    xy_resolution: float = 0.5,
    padding: float = 2.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Create a cylinder with perpendicular hole as a z-stack of binary arrays.

    Args:
        length: Cylinder length along z-axis (µm)
        outer_radius: Outer radius of cylinder (µm)
        hole_radius: Radius of perpendicular hole (µm)
        hole_position: Z-position where hole is centered (µm)
        hole_direction: Direction of hole ('x' or 'y')
        z_resolution: Resolution along z-axis (µm per slice)
        xy_resolution: Resolution in x-y plane (µm per pixel)
        padding: Padding around structure (µm)

    Returns:
        Tuple of (z_stack, metadata)
    """
    # Calculate bounds
    x_min = y_min = -outer_radius - padding
    x_max = y_max = outer_radius + padding
    z_min = -padding
    z_max = length + padding

    nx = int(np.ceil((x_max - x_min) / xy_resolution))
    ny = int(np.ceil((y_max - y_min) / xy_resolution))
    nz = int(np.ceil((z_max - z_min) / z_resolution))

    # Create coordinate grids
    x_coords = np.linspace(x_min, x_max, nx)
    y_coords = np.linspace(y_min, y_max, ny)
    z_coords = np.linspace(z_min, z_max, nz)

    # Initialize z-stack
    z_stack = np.zeros((nz, ny, nx), dtype=np.uint8)

    # Fill cylinder with hole analytically
    for k, z in enumerate(z_coords):
        for j, y in enumerate(y_coords):
            for i, x in enumerate(x_coords):
                inside = False

                # Check if inside outer cylinder
                if 0 <= z <= length:
                    dist_from_axis = np.sqrt(x**2 + y**2)
                    if dist_from_axis <= outer_radius:
                        inside = True

                        # Check if inside hole
                        if hole_direction == "x":
                            # Hole goes along x-axis
                            hole_dist = np.sqrt((z - hole_position) ** 2 + y**2)
                        elif hole_direction == "y":
                            # Hole goes along y-axis
                            hole_dist = np.sqrt((z - hole_position) ** 2 + x**2)
                        else:
                            raise ValueError(f"Invalid hole_direction: {hole_direction}")

                        if hole_dist <= hole_radius:
                            inside = False  # Inside the hole, so outside the neuron

                z_stack[k, j, i] = 1 if inside else 0

    # Create metadata
    metadata = {
        "morphology_type": "cylinder_with_hole",
        "z_resolution": z_resolution,
        "xy_resolution": xy_resolution,
        "x_coords": x_coords,
        "y_coords": y_coords,
        "z_coords": z_coords,
        "bounds": {"x_range": (x_min, x_max), "y_range": (y_min, y_max), "z_range": (z_min, z_max)},
        "shape": z_stack.shape,
        "hole_params": {
            "length": length,
            "outer_radius": outer_radius,
            "hole_radius": hole_radius,
            "hole_position": hole_position,
            "hole_direction": hole_direction,
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


def save_test_meshes(output_dir: str = "test_meshes"):
    """
    Generate and save test meshes to files.

    Args:
        output_dir: Directory to save mesh files
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    print("Generating test meshes...")

    # Create cylinder mesh
    print("  Creating cylinder mesh...")
    cylinder = create_cylinder_mesh(length=100.0, radius=5.0)
    cylinder.export(os.path.join(output_dir, "cylinder.stl"))

    # Create Y-shaped mesh
    print("  Creating Y-shaped mesh...")
    y_shape = create_y_shaped_mesh(trunk_length=60.0, branch_length=40.0)
    y_shape.export(os.path.join(output_dir, "y_neuron.stl"))

    # Create mesh with hole
    print("  Creating mesh with hole...")
    with_hole = create_mesh_with_hole(length=80.0, hole_radius=2.0)
    with_hole.export(os.path.join(output_dir, "cylinder_with_hole.stl"))

    print(f"Test meshes saved to {output_dir}/")

    return {
        "cylinder": os.path.join(output_dir, "cylinder.stl"),
        "y_neuron": os.path.join(output_dir, "y_neuron.stl"),
        "with_hole": os.path.join(output_dir, "cylinder_with_hole.stl"),
    }


def save_test_zstacks(output_dir: str = "test_zstacks"):
    """
    Generate and save test z-stacks to files.

    Args:
        output_dir: Directory to save z-stack files
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    print("Generating test z-stacks...")

    # Create cylinder z-stack
    print("  Creating cylinder z-stack...")
    cylinder_zstack, cylinder_metadata = create_cylinder_zstack(length=100.0, radius=5.0)
    np.savez_compressed(
        os.path.join(output_dir, "cylinder_zstack.npz"), z_stack=cylinder_zstack, metadata=cylinder_metadata
    )

    # Create Y-shaped z-stack
    print("  Creating Y-shaped z-stack...")
    y_zstack, y_metadata = create_y_shaped_zstack(trunk_length=60.0, branch_length=40.0)
    np.savez_compressed(os.path.join(output_dir, "y_zstack.npz"), z_stack=y_zstack, metadata=y_metadata)

    # Create z-stack with hole
    print("  Creating z-stack with hole...")
    hole_zstack, hole_metadata = create_hole_zstack(length=80.0, hole_radius=3.0)
    np.savez_compressed(os.path.join(output_dir, "hole_zstack.npz"), z_stack=hole_zstack, metadata=hole_metadata)

    # Create torus z-stack
    print("  Creating torus z-stack...")
    torus_zstack, torus_metadata = create_torus_zstack(major_radius=8.0, minor_radius=3.0)
    np.savez_compressed(os.path.join(output_dir, "torus_zstack.npz"), z_stack=torus_zstack, metadata=torus_metadata)

    print(f"Test z-stacks saved to {output_dir}/")

    return {
        "cylinder": os.path.join(output_dir, "cylinder_zstack.npz"),
        "y_shaped": os.path.join(output_dir, "y_zstack.npz"),
        "with_hole": os.path.join(output_dir, "hole_zstack.npz"),
        "torus": os.path.join(output_dir, "torus_zstack.npz"),
    }

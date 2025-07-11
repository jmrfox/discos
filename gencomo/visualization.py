"""
3D mesh generation and visualization utilities for GenCoMo.

Provides functions to create test meshes and visualize neuronal morphologies.
"""

import numpy as np
import trimesh
from typing import Optional, Tuple, List, Union, Dict, Any
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


def visualize_mesh_3d(
    mesh_data: Union[trimesh.Trimesh, Tuple[np.ndarray, np.ndarray]] = None,
    vertices: Optional[np.ndarray] = None,
    faces: Optional[np.ndarray] = None,
    title: str = "Neuronal Mesh",
    show_wireframe: bool = False,
    color: str = "lightblue",
    backend: str = "plotly",
) -> Optional[object]:
    """
    Visualize a 3D mesh using various backends.

    Args:
        mesh_data: Either a Trimesh object or (vertices, faces) tuple
        vertices: Vertex array (alternative to mesh_data)
        faces: Face array (alternative to mesh_data)
        title: Plot title
        show_wireframe: Whether to show wireframe
        color: Mesh color
        backend: Visualization backend ('matplotlib', 'plotly', 'trimesh')

    Returns:
        Figure object (depends on backend)
    """
    # Handle different input formats
    if mesh_data is not None:
        if isinstance(mesh_data, tuple):
            vertices, faces = mesh_data
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        else:
            mesh = mesh_data
    elif vertices is not None and faces is not None:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    else:
        raise ValueError("Must provide either mesh_data or both vertices and faces")

    if backend == "matplotlib":
        return _visualize_matplotlib(mesh, title, show_wireframe, color)
    elif backend == "plotly":
        return _visualize_plotly(mesh, title, color)
    elif backend == "trimesh":
        return _visualize_trimesh(mesh, title)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def _visualize_matplotlib(mesh: trimesh.Trimesh, title: str, show_wireframe: bool, color: str):
    """Visualize using matplotlib 3D."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Create 3D collection from mesh faces
        vertices = mesh.vertices
        faces = mesh.faces

        # Create face collection
        face_collection = []
        for face in faces:
            face_vertices = vertices[face]
            face_collection.append(face_vertices)

        poly3d = Poly3DCollection(
            face_collection, alpha=0.7, facecolor=color, edgecolor="black" if show_wireframe else None
        )
        ax.add_collection3d(poly3d)

        # Set axis limits
        ax.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
        ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
        ax.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())

        ax.set_xlabel("X (µm)")
        ax.set_ylabel("Y (µm)")
        ax.set_zlabel("Z (µm)")
        ax.set_title(title)

        plt.tight_layout()
        return fig

    except ImportError:
        print("Matplotlib not available for 3D visualization")
        return None


def _visualize_plotly(mesh: trimesh.Trimesh, title: str, color: str):
    """Visualize using plotly."""
    try:
        import plotly.graph_objects as go

        vertices = mesh.vertices
        faces = mesh.faces

        fig = go.Figure(
            data=[
                go.Mesh3d(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    z=vertices[:, 2],
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    color=color,
                    opacity=0.8,
                    name="Mesh",
                )
            ]
        )

        fig.update_layout(
            title=title, scene=dict(xaxis_title="X (µm)", yaxis_title="Y (µm)", zaxis_title="Z (µm)", aspectmode="data")
        )

        return fig

    except ImportError:
        print("Plotly not available for 3D visualization")
        return None


def _visualize_trimesh(mesh: trimesh.Trimesh, title: str):
    """Visualize using trimesh's built-in viewer."""
    try:
        scene = trimesh.Scene([mesh])
        return scene.show(caption=title)
    except Exception as e:
        print(f"Trimesh visualization failed: {e}")
        return None


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
    with_hole = create_mesh_with_hole(length=80.0, hole_length=30.0)
    with_hole.export(os.path.join(output_dir, "cylinder_with_hole.stl"))

    print(f"Test meshes saved to {output_dir}/")

    return {
        "cylinder": os.path.join(output_dir, "cylinder.stl"),
        "y_neuron": os.path.join(output_dir, "y_neuron.stl"),
        "with_hole": os.path.join(output_dir, "cylinder_with_hole.stl"),
    }


def analyze_mesh_properties(mesh_data: Union[trimesh.Trimesh, Tuple[np.ndarray, np.ndarray]]) -> dict:
    """
    Analyze and return mesh properties for diagnostic purposes.

    Args:
        mesh_data: Either a Trimesh object or (vertices, faces) tuple

    Returns:
        Dictionary of mesh properties
    """
    # Convert to mesh object if needed
    if isinstance(mesh_data, tuple):
        vertices, faces = mesh_data
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    else:
        mesh = mesh_data

    properties = {
        "num_vertices": len(mesh.vertices),
        "num_faces": len(mesh.faces),
        "volume": mesh.volume if mesh.is_volume else None,
        "surface_area": mesh.area,
        "is_watertight": mesh.is_watertight,
        "is_winding_consistent": mesh.is_winding_consistent,
        "bounds": {
            "x_range": (mesh.vertices[:, 0].min(), mesh.vertices[:, 0].max()),
            "y_range": (mesh.vertices[:, 1].min(), mesh.vertices[:, 1].max()),
            "z_range": (mesh.vertices[:, 2].min(), mesh.vertices[:, 2].max()),
        },
        "centroid": mesh.centroid.tolist() if hasattr(mesh, "centroid") else None,
        "bounding_box_volume": mesh.bounding_box.volume,
        "convex_hull_volume": mesh.convex_hull.volume if hasattr(mesh, "convex_hull") else None,
    }

    return properties


def mesh_to_zstack(
    mesh: trimesh.Trimesh, z_resolution: float = 1.0, xy_resolution: float = 0.5, padding: float = 5.0
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Convert a mesh to a z-stack of binary arrays.

    Args:
        mesh: Input trimesh object
        z_resolution: Resolution along z-axis (µm per slice)
        xy_resolution: Resolution in x-y plane (µm per pixel)
        padding: Padding around mesh bounds (µm)

    Returns:
        Tuple of (z_stack, metadata) where:
        - z_stack: 3D numpy array (z, y, x) with 1 inside neuron, 0 outside
        - metadata: Dict with spatial information
    """
    # Get mesh bounds with padding
    bounds = mesh.bounds
    x_min, y_min, z_min = bounds[0] - padding
    x_max, y_max, z_max = bounds[1] + padding

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

    print(f"Creating z-stack: {nz} slices of {ny}x{nx} pixels")
    print(f"Spatial resolution: {xy_resolution:.1f} µm/pixel, {z_resolution:.1f} µm/slice")

    # Fill z-stack by checking point containment
    for k, z in enumerate(z_coords):
        if k % 10 == 0:  # Progress indicator
            print(f"  Processing slice {k+1}/{nz}")

        for j, y in enumerate(y_coords):
            for i, x in enumerate(x_coords):
                point = np.array([x, y, z])
                z_stack[k, j, i] = 1 if mesh.contains([point])[0] else 0

    # Create metadata
    metadata = {
        "z_resolution": z_resolution,
        "xy_resolution": xy_resolution,
        "x_coords": x_coords,
        "y_coords": y_coords,
        "z_coords": z_coords,
        "bounds": {"x_range": (x_min, x_max), "y_range": (y_min, y_max), "z_range": (z_min, z_max)},
        "shape": z_stack.shape,
        "total_voxels": z_stack.size,
        "neuron_voxels": np.sum(z_stack),
        "volume_um3": np.sum(z_stack) * xy_resolution * xy_resolution * z_resolution,
    }

    return z_stack, metadata


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


def visualize_zstack_3d(
    z_stack: np.ndarray,
    metadata: Dict[str, Any],
    title: str = "Z-stack Neuron Morphology",
    color: str = "lightblue",
    opacity: float = 0.7,
    backend: str = "plotly",
) -> Optional[object]:
    """
    Visualize a z-stack as a 3D volume rendering.

    Args:
        z_stack: 3D binary array (z, y, x)
        metadata: Metadata dict with coordinate information
        title: Plot title
        color: Volume color
        opacity: Volume opacity
        backend: Visualization backend

    Returns:
        Figure object
    """
    if backend == "plotly":
        try:
            import plotly.graph_objects as go

            # Get coordinates
            x_coords = metadata["x_coords"]
            y_coords = metadata["y_coords"]
            z_coords = metadata["z_coords"]

            # Create meshgrid for volume
            Z, Y, X = np.meshgrid(z_coords, y_coords, x_coords, indexing="ij")

            # Flatten arrays for scatter plot
            mask = z_stack == 1
            x_points = X[mask]
            y_points = Y[mask]
            z_points = Z[mask]

            # Create 3D scatter plot
            fig = go.Figure(
                data=go.Scatter3d(
                    x=x_points,
                    y=y_points,
                    z=z_points,
                    mode="markers",
                    marker=dict(size=2, color=color, opacity=opacity),
                    name="Neuron Volume",
                )
            )

            fig.update_layout(
                title=title,
                scene=dict(xaxis_title="X (µm)", yaxis_title="Y (µm)", zaxis_title="Z (µm)", aspectmode="data"),
            )

            return fig

        except ImportError:
            print("Plotly not available for z-stack visualization")
            return None

    else:
        raise ValueError(f"Backend {backend} not supported for z-stack visualization")


def save_zstack_data(z_stack: np.ndarray, metadata: Dict[str, Any], filepath: str, format: str = "npz") -> None:
    """
    Save z-stack data to file.

    Args:
        z_stack: 3D binary array
        metadata: Metadata dictionary
        filepath: Output file path
        format: File format ('npz', 'npy', 'h5')
    """
    if format == "npz":
        np.savez_compressed(filepath, z_stack=z_stack, metadata=metadata)
    elif format == "npy":
        np.save(filepath, {"z_stack": z_stack, "metadata": metadata})
    elif format == "h5":
        try:
            import h5py

            with h5py.File(filepath, "w") as f:
                f.create_dataset("z_stack", data=z_stack, compression="gzip")

                # Save metadata
                meta_group = f.create_group("metadata")
                for key, value in metadata.items():
                    if isinstance(value, dict):
                        sub_group = meta_group.create_group(key)
                        for sub_key, sub_value in value.items():
                            sub_group.create_dataset(sub_key, data=sub_value)
                    else:
                        meta_group.create_dataset(key, data=value)
        except ImportError:
            raise ImportError("h5py required for HDF5 format")
    else:
        raise ValueError(f"Unknown format: {format}")


def load_zstack_data(filepath: str, format: str = "npz") -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load z-stack data from file.

    Args:
        filepath: Input file path
        format: File format ('npz', 'npy', 'h5')

    Returns:
        Tuple of (z_stack, metadata)
    """
    if format == "npz":
        data = np.load(filepath, allow_pickle=True)
        return data["z_stack"], data["metadata"].item()
    elif format == "npy":
        data = np.load(filepath, allow_pickle=True).item()
        return data["z_stack"], data["metadata"]
    elif format == "h5":
        try:
            import h5py

            with h5py.File(filepath, "r") as f:
                z_stack = f["z_stack"][:]

                # Load metadata
                metadata = {}
                meta_group = f["metadata"]
                for key in meta_group.keys():
                    if isinstance(meta_group[key], h5py.Group):
                        metadata[key] = {}
                        for sub_key in meta_group[key].keys():
                            metadata[key][sub_key] = meta_group[key][sub_key][:]
                    else:
                        metadata[key] = meta_group[key][:]

                return z_stack, metadata
        except ImportError:
            raise ImportError("h5py required for HDF5 format")
    else:
        raise ValueError(f"Unknown format: {format}")


def load_mesh_file_to_zstack(
    filepath: str, z_resolution: float = 1.0, xy_resolution: float = 0.5, padding: float = 5.0
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load a mesh file and convert it directly to z-stack format.
    This is the recommended way to import mesh data into GenCoMo.

    Args:
        filepath: Path to mesh file (.stl, .obj, .ply, etc.)
        z_resolution: Resolution along z-axis (µm per slice)
        xy_resolution: Resolution in x-y plane (µm per pixel)
        padding: Padding around mesh bounds (µm)

    Returns:
        Tuple of (z_stack, metadata)
    """
    print(f"Loading mesh file: {filepath}")
    mesh = trimesh.load(filepath)

    # Handle scene objects
    if isinstance(mesh, trimesh.Scene):
        geometries = list(mesh.geometry.values())
        if geometries:
            mesh = geometries[0]
        else:
            raise ValueError("No geometry found in mesh file")

    print(f"Converting mesh to GenCoMo's native z-stack format...")

    # Convert to z-stack
    z_stack, metadata = mesh_to_zstack(mesh, z_resolution, xy_resolution, padding)

    # Add mesh file information to metadata
    metadata["source_file"] = filepath
    metadata["morphology_type"] = "imported_from_mesh"
    metadata["original_mesh_volume"] = mesh.volume if hasattr(mesh, "volume") else None
    metadata["original_mesh_vertices"] = len(mesh.vertices)
    metadata["original_mesh_faces"] = len(mesh.faces)

    return z_stack, metadata


def analyze_zstack_properties(z_stack: np.ndarray, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze properties of a z-stack morphology.

    Args:
        z_stack: 3D binary array (z, y, x)
        metadata: Metadata dictionary

    Returns:
        Dictionary of morphology properties
    """
    properties = {
        "shape": z_stack.shape,
        "total_voxels": z_stack.size,
        "neuron_voxels": np.sum(z_stack),
        "volume_um3": metadata.get(
            "volume_um3", np.sum(z_stack) * metadata.get("xy_resolution", 1.0) ** 2 * metadata.get("z_resolution", 1.0)
        ),
        "z_resolution": metadata.get("z_resolution", 1.0),
        "xy_resolution": metadata.get("xy_resolution", 1.0),
        "bounds": metadata.get("bounds", {}),
        "morphology_type": metadata.get("morphology_type", "unknown"),
        "fill_ratio": np.sum(z_stack) / z_stack.size,
        "z_extent": z_stack.shape[0] * metadata.get("z_resolution", 1.0),
        "xy_extent": {
            "x": z_stack.shape[2] * metadata.get("xy_resolution", 1.0),
            "y": z_stack.shape[1] * metadata.get("xy_resolution", 1.0),
        },
    }

    # Add slice-by-slice analysis
    slice_areas = []
    for k in range(z_stack.shape[0]):
        slice_area = np.sum(z_stack[k]) * metadata.get("xy_resolution", 1.0) ** 2
        slice_areas.append(slice_area)

    properties["slice_areas"] = slice_areas
    properties["max_cross_section"] = max(slice_areas) if slice_areas else 0
    properties["min_cross_section"] = min([a for a in slice_areas if a > 0]) if slice_areas else 0

    return properties

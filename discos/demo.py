"""
Demo mesh generation functions for DISCOS.

Provides example mesh generators for neuronal morphologies using trimesh.
These functions create standard geometries for tutorials and demonstrations.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import trimesh


def create_cylinder_mesh(
    length: float = 100.0,
    radius: float = 5.0,
    resolution: int = 16,
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    axis: str = "z",
) -> trimesh.Trimesh:
    """
    Create a cylindrical mesh representing a simple neuronal process.

    Args:
        length: Cylinder length (μm)
        radius: Cylinder radius (μm)
        resolution: Number of circumferential segments
        center: Center position (x, y, z) in μm
        axis: Primary axis ('x', 'y', or 'z')

    Returns:
        Trimesh cylinder object
    """
    # Create cylinder along z-axis first
    cylinder = trimesh.creation.cylinder(
        radius=radius, height=length, sections=resolution
    )

    # Rotate to desired axis if needed
    if axis.lower() == "x":
        # Rotate 90 degrees around y-axis to align with x
        rotation = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
        cylinder.apply_transform(rotation)
    elif axis.lower() == "y":
        # Rotate 90 degrees around x-axis to align with y
        rotation = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
        cylinder.apply_transform(rotation)
    # z-axis is default, no rotation needed

    # Translate to center position
    if center != (0.0, 0.0, 0.0):
        translation = trimesh.transformations.translation_matrix(center)
        cylinder.apply_transform(translation)

    # Add metadata
    cylinder.metadata["morphology_type"] = "cylinder"
    cylinder.metadata["length"] = length
    cylinder.metadata["radius"] = radius
    cylinder.metadata["volume_theoretical"] = np.pi * radius**2 * length
    cylinder.metadata["surface_area_theoretical"] = (
        2 * np.pi * radius * (radius + length)
    )
    cylinder.metadata["center"] = center
    cylinder.metadata["axis"] = axis

    return cylinder


def create_torus_mesh(
    major_radius: float = 20.0,
    minor_radius: float = 5.0,
    major_segments: int = 20,
    minor_segments: int = 12,
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    axis: str = "x",
) -> trimesh.Trimesh:
    """
    Create a torus mesh representing a ring-shaped neuronal structure.

    Args:
        major_radius: Radius from center of torus to center of tube (μm)
        minor_radius: Radius of the tube itself (μm)
        major_segments: Number of segments around the major radius
        minor_segments: Number of segments around the minor radius (tube)
        center: Center position (x, y, z) in μm
        axis: Axis of symmetry ('x', 'y', or 'z') - torus rotates around this axis

    Returns:
        Trimesh torus object
    """
    # Create torus using trimesh (default: axis of symmetry along z)
    # Handle API differences across trimesh versions by trying multiple signatures
    torus = None
    candidate_kwargs = [
        # Newer style keyword names
        {
            "major_radius": major_radius,
            "minor_radius": minor_radius,
            "major_sections": major_segments,
            "minor_sections": minor_segments,
        },
        {
            "major_radius": major_radius,
            "minor_radius": minor_radius,
        },
        # Older style keyword names
        {
            "radius": major_radius,
            "tube_radius": minor_radius,
            "major_sections": major_segments,
            "minor_sections": minor_segments,
        },
        {
            "radius": major_radius,
            "tube_radius": minor_radius,
        },
    ]

    last_err: Optional[BaseException] = None
    for kwargs in candidate_kwargs:
        try:
            torus = trimesh.creation.torus(**kwargs)
            break
        except TypeError as e:
            last_err = e
            continue
        except Exception as e:
            last_err = e
            continue

    if torus is None:
        raise TypeError(
            f"Failed to construct torus with trimesh.creation.torus using multiple signatures: {last_err}"
        )

    # Rotate to desired axis of symmetry if needed
    if axis.lower() == "x":
        # Rotate 90 degrees around y-axis to make x the axis of symmetry
        rotation = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
        torus.apply_transform(rotation)
    elif axis.lower() == "y":
        # Rotate 90 degrees around x-axis to make y the axis of symmetry
        rotation = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
        torus.apply_transform(rotation)
    # z-axis is default, no rotation needed

    # Translate to center position if needed
    if center != (0.0, 0.0, 0.0):
        translation = trimesh.transformations.translation_matrix(center)
        torus.apply_transform(translation)

    # Add metadata
    torus.metadata["morphology_type"] = "torus"
    torus.metadata["major_radius"] = major_radius
    torus.metadata["minor_radius"] = minor_radius
    torus.metadata["center"] = center
    torus.metadata["axis"] = axis

    # Calculate theoretical properties
    volume_theoretical = 2 * np.pi**2 * major_radius * minor_radius**2
    surface_area_theoretical = 4 * np.pi**2 * major_radius * minor_radius

    torus.metadata["volume_theoretical"] = volume_theoretical
    torus.metadata["surface_area_theoretical"] = surface_area_theoretical

    return torus


def _clean_exterior_mesh(combined_mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Clean up a mesh to remove interior artifacts while preserving geometry.
    Only removes duplicate faces and merges very close vertices.
    """
    try:
        # Remove duplicate faces (this helps with overlapping geometry)
        combined_mesh.update_faces(combined_mesh.unique_faces())

        # Remove unreferenced vertices
        combined_mesh.remove_unreferenced_vertices()

        # Merge vertices that are very close together (helps at junctions)
        # Use a small tolerance to avoid changing the geometry significantly
        combined_mesh.merge_vertices(merge_tex=True, merge_norm=True)

        return combined_mesh

    except Exception as e:
        warnings.warn(f"Mesh cleanup failed: {e}")
        return combined_mesh


def create_branching_mesh(
    trunk_length: float = 60.0,
    trunk_radius: float = 5.0,
    branch_length: float = 40.0,
    branch_radius: float = 3.0,
    branch_angle: float = 45.0,
    num_branches: int = 2,
    resolution: int = 12,
    smooth_junctions: bool = True,
) -> trimesh.Trimesh:
    """
    Create a branching mesh representing a dendritic or axonal tree.

    Args:
        trunk_length: Length of main trunk (μm)
        trunk_radius: Radius of main trunk (μm)
        branch_length: Length of each branch (μm)
        branch_radius: Radius of each branch (μm)
        branch_angle: Angle of branches from trunk axis (degrees)
        num_branches: Number of branches (2 for Y-shape, more for multi-branch)
        resolution: Number of circumferential segments
        smooth_junctions: Whether to smooth branch junctions

    Returns:
        Trimesh object with branching structure
    """
    # Create main trunk along z-axis
    trunk = trimesh.creation.cylinder(
        radius=trunk_radius, height=trunk_length, sections=resolution
    )

    # Position trunk so top is at z=trunk_length
    trunk_translation = trimesh.transformations.translation_matrix(
        [0, 0, trunk_length / 2]
    )
    trunk.apply_transform(trunk_translation)

    # Create branches
    branches = []
    for i in range(num_branches):
        # Create branch cylinder (initially centered at origin along z-axis)
        branch = trimesh.creation.cylinder(
            radius=branch_radius, height=branch_length, sections=max(8, resolution // 2)
        )

        # Calculate branch angle in radians
        angle_rad = np.radians(branch_angle)

        # Calculate rotation angle around z-axis for this branch
        azimuth_angle = (2 * np.pi * i) / num_branches

        # Step 1: Rotate the branch around its center to desired angle
        # Rotate around axis perpendicular to both z-axis and the desired branch direction
        rotation_axis = [np.cos(azimuth_angle), np.sin(azimuth_angle), 0]
        rotation = trimesh.transformations.rotation_matrix(angle_rad, rotation_axis)
        branch.apply_transform(rotation)

        # Step 2: Calculate where the bottom of the rotated branch is now
        # The original bottom was at [0, 0, -branch_length/2]
        original_bottom = np.array(
            [0, 0, -branch_length / 2, 1]
        )  # homogeneous coordinates
        rotated_bottom = rotation @ original_bottom

        # Step 3: Translate so the bottom of the branch is at trunk top
        # We want rotated_bottom to be at [0, 0, trunk_length]
        target_bottom = np.array([0, 0, trunk_length])
        translation_vector = target_bottom - rotated_bottom[:3]
        translation = trimesh.transformations.translation_matrix(translation_vector)
        branch.apply_transform(translation)

        branches.append(
            branch
        )  # Combine using boolean union operations for clean exterior surface
    all_meshes = [trunk] + branches

    try:
        # Use boolean union to combine all overlapping shapes cleanly
        combined = trimesh.boolean.union(all_meshes)
    except Exception as e:
        # If union fails, fall back to concatenation
        warnings.warn(f"Boolean union failed: {e}, using concatenation")
        combined = trimesh.util.concatenate(all_meshes)

    # Smooth junctions if requested
    if smooth_junctions:
        try:
            # Apply light smoothing to reduce sharp edges at junctions
            combined = combined.smoothed()
        except:
            # If smoothing fails, continue with unsmoothed mesh
            warnings.warn("Could not smooth branch junctions, using unsmoothed mesh")

    # Add metadata
    combined.metadata["morphology_type"] = "branching"
    combined.metadata["trunk_length"] = trunk_length
    combined.metadata["trunk_radius"] = trunk_radius
    combined.metadata["branch_length"] = branch_length
    combined.metadata["branch_radius"] = branch_radius
    combined.metadata["branch_angle"] = branch_angle
    combined.metadata["num_branches"] = num_branches
    combined.metadata["total_branches"] = num_branches
    combined.metadata["smooth_junctions"] = smooth_junctions

    # Calculate theoretical properties
    trunk_volume = np.pi * trunk_radius**2 * trunk_length
    branch_volume = np.pi * branch_radius**2 * branch_length
    total_volume = trunk_volume + num_branches * branch_volume

    combined.metadata["volume_theoretical"] = total_volume
    combined.metadata["trunk_volume"] = trunk_volume
    combined.metadata["branch_volume"] = branch_volume

    return combined


def save_demo_meshes(output_dir: str = "data/mesh") -> Dict[str, str]:
    """
    Generate and save example meshes to files.

    Args:
        output_dir: Directory to save mesh files

    Returns:
        Dictionary mapping mesh names to file paths
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    print("🔧 Generating demo meshes...")

    # Create cylinder mesh
    print("  Creating cylinder mesh...")
    cylinder = create_cylinder_mesh(length=100.0, radius=5.0, resolution=16)
    cylinder_path = os.path.join(output_dir, "cylinder.stl")
    cylinder.export(cylinder_path)

    # Create torus mesh
    print("  Creating torus mesh...")
    torus = create_torus_mesh(major_radius=20.0, minor_radius=5.0)
    torus_path = os.path.join(output_dir, "torus.stl")
    torus.export(torus_path)

    # Create Y-branching mesh
    print("  Creating Y-branching mesh...")
    y_branch = create_branching_mesh(
        trunk_length=60.0,
        trunk_radius=5.0,
        branch_length=40.0,
        branch_radius=3.0,
        branch_angle=45.0,
        num_branches=2,
    )
    y_branch_path = os.path.join(output_dir, "y_branch.stl")
    y_branch.export(y_branch_path)

    # Create multi-branching mesh
    print("  Creating multi-branching mesh...")
    multi_branch = create_branching_mesh(
        trunk_length=80.0,
        trunk_radius=6.0,
        branch_length=35.0,
        branch_radius=3.5,
        branch_angle=60.0,
        num_branches=4,
    )
    multi_branch_path = os.path.join(output_dir, "multi_branch.stl")
    multi_branch.export(multi_branch_path)

    print(f"✅ Demo meshes saved to {output_dir}/")

    return {
        "cylinder": cylinder_path,
        "torus": torus_path,
        "y_branch": y_branch_path,
        "multi_branch": multi_branch_path,
    }


def create_demo_neuron_mesh(
    soma_radius: float = 10.0,
    dendrite_length: float = 50.0,
    dendrite_radius: float = 2.0,
    axon_length: float = 100.0,
    axon_radius: float = 1.5,
    num_dendrites: int = 4,
    dendrite_angle: float = 30.0,
) -> trimesh.Trimesh:
    """
    Create a simplified neuron mesh with soma, dendrites, and axon.

    Args:
        soma_radius: Radius of soma (cell body) in μm
        dendrite_length: Length of each dendrite in μm
        dendrite_radius: Radius of dendrites in μm
        axon_length: Length of axon in μm
        axon_radius: Radius of axon in μm
        num_dendrites: Number of dendrites
        dendrite_angle: Angle of dendrites from vertical (degrees)

    Returns:
        Trimesh object representing a simple neuron
    """
    # Create soma (cell body) as sphere
    soma = trimesh.creation.icosphere(subdivisions=2, radius=soma_radius)

    # Create axon extending downward
    axon = trimesh.creation.cylinder(
        radius=axon_radius, height=axon_length, sections=12
    )

    # Position axon extending down from soma
    axon_translation = trimesh.transformations.translation_matrix(
        [0, 0, -axon_length / 2]
    )
    axon.apply_transform(axon_translation)

    # Create dendrites extending upward
    dendrites = []
    for i in range(num_dendrites):
        dendrite = trimesh.creation.cylinder(
            radius=dendrite_radius, height=dendrite_length, sections=8
        )

        # Calculate rotation for this dendrite
        azimuth_angle = (2 * np.pi * i) / num_dendrites
        tilt_angle = np.radians(dendrite_angle)

        # Step 1: Rotate dendrite around its center to desired angle
        # Rotate around axis perpendicular to both z-axis and the desired dendrite direction
        rotation_axis = [np.cos(azimuth_angle), np.sin(azimuth_angle), 0]
        rotation = trimesh.transformations.rotation_matrix(tilt_angle, rotation_axis)
        dendrite.apply_transform(rotation)

        # Step 2: Calculate where the bottom of the rotated dendrite is now
        # The original bottom was at [0, 0, -dendrite_length/2]
        original_bottom = np.array(
            [0, 0, -dendrite_length / 2, 1]
        )  # homogeneous coordinates
        rotated_bottom = rotation @ original_bottom

        # Step 3: Translate so the bottom of the dendrite is at soma surface
        # We want rotated_bottom to be at soma surface in the direction of the dendrite
        dendrite_direction = np.array(
            [
                np.sin(tilt_angle) * np.cos(azimuth_angle),
                np.sin(tilt_angle) * np.sin(azimuth_angle),
                np.cos(tilt_angle),
            ]
        )
        target_bottom = (
            dendrite_direction * soma_radius * 0.8
        )  # Point on soma surface. included a factor to ensure proper overlap
        translation_vector = target_bottom - rotated_bottom[:3]
        translation = trimesh.transformations.translation_matrix(translation_vector)
        dendrite.apply_transform(translation)

        dendrites.append(dendrite)

    # Ensure all components are watertight before union
    for i, component in enumerate([soma, axon] + dendrites):
        if not component.is_watertight:
            component.fill_holes()

    # Combine using boolean union operations for clean exterior surface
    try:
        # Start with soma and progressively add components
        neuron = soma
        for component in [axon] + dendrites:
            neuron = trimesh.boolean.union([neuron, component])
            if neuron is None or len(neuron.vertices) == 0:
                raise ValueError("Boolean union resulted in empty mesh")
    except Exception as e:
        # If union fails, fall back to concatenation
        warnings.warn(f"Boolean union failed: {e}, using concatenation")
        all_components = [soma, axon] + dendrites
        neuron = trimesh.util.concatenate(all_components)

    # Light cleanup to remove any duplicate faces from concatenation fallback
    neuron = _clean_exterior_mesh(neuron)

    # Add metadata
    neuron.metadata["morphology_type"] = "neuron"
    neuron.metadata["soma_radius"] = soma_radius
    neuron.metadata["dendrite_length"] = dendrite_length
    neuron.metadata["dendrite_radius"] = dendrite_radius
    neuron.metadata["axon_length"] = axon_length
    neuron.metadata["axon_radius"] = axon_radius
    neuron.metadata["num_dendrites"] = num_dendrites
    neuron.metadata["dendrite_angle"] = dendrite_angle

    return neuron


if __name__ == "__main__":
    # Generate and save demo meshes when run as script
    paths = save_demo_meshes()
    print(f"Demo meshes created:")
    for name, path in paths.items():
        print(f"  {name}: {path}")

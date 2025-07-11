"""
Script to generate test meshes for GenCoMo tutorials.
"""

import numpy as np
import os


def create_simple_cylinder_mesh(length=100.0, radius=5.0, num_segments=16, num_z_divisions=50):
    """Create a simple cylindrical mesh and save as STL."""

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

    return vertices, np.array(faces)


def save_stl(vertices, faces, filename):
    """Save mesh as STL file."""
    with open(filename, "w") as f:
        f.write("solid mesh\n")

        for face in faces:
            # Calculate normal (simple approach)
            v1, v2, v3 = vertices[face]
            normal = np.cross(v2 - v1, v3 - v1)
            normal = normal / np.linalg.norm(normal)

            f.write("  facet normal {:.6f} {:.6f} {:.6f}\n".format(normal[0], normal[1], normal[2]))
            f.write("    outer loop\n")
            f.write("      vertex {:.6f} {:.6f} {:.6f}\n".format(v1[0], v1[1], v1[2]))
            f.write("      vertex {:.6f} {:.6f} {:.6f}\n".format(v2[0], v2[1], v2[2]))
            f.write("      vertex {:.6f} {:.6f} {:.6f}\n".format(v3[0], v3[1], v3[2]))
            f.write("    endloop\n")
            f.write("  endfacet\n")

        f.write("endsolid mesh\n")


def create_y_shaped_mesh():
    """Create a Y-shaped mesh."""
    # Create main trunk
    trunk_vertices, trunk_faces = create_simple_cylinder_mesh(
        length=60.0, radius=5.0, num_segments=12, num_z_divisions=30
    )

    # Create branches (simplified - just add two cylinders at angles)
    branch_length = 40.0
    branch_radius = 3.0

    # Branch 1 - angled to the right
    branch1_vertices, branch1_faces = create_simple_cylinder_mesh(
        length=branch_length, radius=branch_radius, num_segments=12, num_z_divisions=20
    )

    # Rotate and translate branch1
    angle = np.radians(45)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]])

    branch1_vertices = np.dot(branch1_vertices, rotation_matrix.T)
    branch1_vertices[:, 2] += 60.0  # Move to top of trunk

    # Branch 2 - angled to the left
    branch2_vertices = branch1_vertices.copy()
    branch2_vertices[:, 0] *= -1  # Mirror in x-direction

    # Combine all vertices and faces
    offset1 = len(trunk_vertices)
    offset2 = offset1 + len(branch1_vertices)

    all_vertices = np.vstack([trunk_vertices, branch1_vertices, branch2_vertices])
    all_faces = np.vstack(
        [trunk_faces, branch1_faces + offset1, branch1_faces + offset2]  # Reuse branch1 topology for branch2
    )

    return all_vertices, all_faces


def create_cylinder_with_hole():
    """Create a cylinder with a hole through it (simplified)."""
    # Create outer cylinder
    outer_vertices, outer_faces = create_simple_cylinder_mesh(
        length=80.0, radius=6.0, num_segments=16, num_z_divisions=40
    )

    # For simplicity, we'll just remove some faces to create a "hole" effect
    # In a real implementation, you'd use proper boolean operations

    # Remove some faces in the middle section to simulate a hole
    hole_start_z = 20.0
    hole_end_z = 50.0

    # Filter faces that don't intersect with hole region
    filtered_faces = []
    for face in outer_faces:
        face_z_coords = outer_vertices[face][:, 2]
        face_center_z = np.mean(face_z_coords)

        # Keep faces outside hole region or end caps
        if (
            face_center_z < hole_start_z + 5
            or face_center_z > hole_end_z - 5
            or face_center_z < 5
            or face_center_z > 75
        ):
            filtered_faces.append(face)

    return outer_vertices, np.array(filtered_faces)


def main():
    """Generate all test meshes."""
    os.makedirs("test_meshes", exist_ok=True)

    print("Generating test meshes...")

    # Generate cylinder
    print("  Creating cylinder...")
    vertices, faces = create_simple_cylinder_mesh()
    save_stl(vertices, faces, "test_meshes/cylinder.stl")

    # Generate Y-shaped neuron
    print("  Creating Y-shaped neuron...")
    vertices, faces = create_y_shaped_mesh()
    save_stl(vertices, faces, "test_meshes/y_neuron.stl")

    # Generate cylinder with hole
    print("  Creating cylinder with hole...")
    vertices, faces = create_cylinder_with_hole()
    save_stl(vertices, faces, "test_meshes/cylinder_with_hole.stl")

    print("Test meshes generated successfully!")
    print("Files created:")
    print("  - test_meshes/cylinder.stl")
    print("  - test_meshes/y_neuron.stl")
    print("  - test_meshes/cylinder_with_hole.stl")


if __name__ == "__main__":
    main()

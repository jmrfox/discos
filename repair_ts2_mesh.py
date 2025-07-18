"""
Repair the TS2_alone.obj mesh to fix negative volume and make it watertight.
"""

import trimesh
import numpy as np
from pathlib import Path

# Import the project utilities
from gencomo.utils import project_root, data_path

def repair_mesh(input_path, output_path=None):
    """
    Repair a mesh with negative volume and non-watertight issues.
    
    Args:
        input_path: Path to the input mesh file
        output_path: Path to save the repaired mesh (default: input_path with _repaired suffix)
    
    Returns:
        The repaired mesh
    """
    print(f"Loading mesh from {input_path}")
    mesh = trimesh.load(input_path)
    
    # Print original mesh properties
    print(f"\nOriginal mesh:")
    print(f"  Volume: {mesh.volume}")
    print(f"  Is watertight: {mesh.is_watertight}")
    print(f"  Is winding consistent: {mesh.is_winding_consistent}")
    print(f"  Face normals sum: {mesh.face_normals.sum(axis=0)}")
    print(f"  Bounds: {mesh.bounds}")
    print(f"  Number of faces: {len(mesh.faces)}")
    print(f"  Number of vertices: {len(mesh.vertices)}")
    
    # Step 1: Fix face winding to ensure normals point outward
    print("\nFixing face winding...")
    if mesh.volume < 0:
        mesh.invert()
        print("  Inverted faces to fix negative volume")
    
    # Step 2: Make the mesh watertight
    print("\nMaking mesh watertight...")
    try:
        # Try to fix holes
        mesh.fill_holes()
        print("  Filled holes in the mesh")
    except Exception as e:
        print(f"  Error filling holes: {e}")
    
    # Step 3: Try to ensure the mesh is a single component
    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        print(f"\nMesh has {len(components)} components, keeping the largest...")
        # Keep the largest component by volume
        volumes = [abs(c.volume) for c in components]
        largest_idx = np.argmax(volumes)
        mesh = components[largest_idx]
        print(f"  Kept component with volume {mesh.volume}")
    
    # Step 4: Final repair using trimesh's built-in repair function
    print("\nPerforming final repairs...")
    mesh.process()
    
    # Print repaired mesh properties
    print(f"\nRepaired mesh:")
    print(f"  Volume: {mesh.volume}")
    print(f"  Is watertight: {mesh.is_watertight}")
    print(f"  Is winding consistent: {mesh.is_winding_consistent}")
    print(f"  Face normals sum: {mesh.face_normals.sum(axis=0)}")
    print(f"  Number of faces: {len(mesh.faces)}")
    print(f"  Number of vertices: {len(mesh.vertices)}")
    
    # Save the repaired mesh
    if output_path is None:
        # Create output path with _repaired suffix
        input_path = Path(input_path)
        output_path = input_path.parent / f"{input_path.stem}_repaired{input_path.suffix}"
    
    mesh.export(output_path)
    print(f"\nSaved repaired mesh to {output_path}")
    
    return mesh

if __name__ == "__main__":
    # Get the paths
    input_path = project_root() / "data" / "TS2_alone.obj"
    output_path = project_root() / "data" / "TS2_alone_repaired.obj"
    
    # Repair the mesh
    repaired_mesh = repair_mesh(input_path, output_path)

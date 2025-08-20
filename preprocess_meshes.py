#!/usr/bin/env python3
"""
Mesh preprocessing script for DISCOS project.

This script loads OBJ mesh files from data/raw, processes them using MeshManager
to center, normalize, and repair the meshes, then saves the processed meshes
to data/processed.

Usage:
    python preprocess_meshes.py
"""

import os
import sys
from pathlib import Path
import numpy as np
from typing import List, Tuple

# Add the discos package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'discos'))

from discos.mesh import MeshManager


def get_obj_files(directory: str) -> List[Path]:
    """
    Get all OBJ files from the specified directory.
    
    Args:
        directory: Path to directory containing OBJ files
        
    Returns:
        List of Path objects for OBJ files
    """
    directory_path = Path(directory)
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory {directory} does not exist")
    
    obj_files = list(directory_path.glob("*.obj"))
    return sorted(obj_files)


def normalize_mesh_size(mesh_manager: MeshManager, target_size: float = 1.0) -> None:
    """
    Normalize mesh to fit within a unit cube or sphere.
    
    Args:
        mesh_manager: MeshManager instance with loaded mesh
        target_size: Target maximum dimension (default: 1.0 for unit size)
    """
    if mesh_manager.mesh is None:
        raise ValueError("No mesh loaded in MeshManager")
    
    # Get the current bounding box dimensions
    bounds = mesh_manager.mesh.bounds
    dimensions = bounds[1] - bounds[0]  # max - min for each axis
    max_dimension = np.max(dimensions)
    
    if max_dimension > 0:
        # Calculate scale factor to fit within target size
        scale_factor = target_size / max_dimension
        mesh_manager.scale_mesh(scale_factor)
        mesh_manager.log(f"Normalized mesh: max dimension {max_dimension:.3f} → {target_size:.3f} (scale: {scale_factor:.3f})")
    else:
        mesh_manager.log("Warning: Mesh has zero dimensions, skipping normalization", "WARNING")


def process_single_mesh(input_path: Path, output_path: Path, verbose: bool = True) -> Tuple[bool, str]:
    """
    Process a single mesh file: load, center, normalize, repair, and save.
    
    Args:
        input_path: Path to input OBJ file
        output_path: Path to output OBJ file
        verbose: Whether to print processing details
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        # Initialize MeshManager
        mesh_manager = MeshManager(verbose=verbose)
        
        # Load the mesh
        mesh_manager.log(f"Loading mesh from {input_path.name}", "PROCESSING")
        mesh_manager.load_mesh(str(input_path))
        
        # Analyze mesh before processing
        if verbose:
            mesh_manager.log("Initial mesh analysis:")
            analysis = mesh_manager.analyze_mesh()
            mesh_manager.log(f"  Vertices: {analysis['vertex_count']}, Faces: {analysis['face_count']}")
            mesh_manager.log(f"  Volume: {analysis['volume']:.6f}")
            mesh_manager.log(f"  Watertight: {analysis['is_watertight']}")
            mesh_manager.log(f"  Bounds: {analysis['bounds']}")
        
        # Step 1: Center the mesh
        mesh_manager.log("Centering mesh on centroid", "PROCESSING")
        mesh_manager.center_mesh(center_on="centroid")
        
        # Step 2: Normalize the mesh size
        mesh_manager.log("Normalizing mesh size", "PROCESSING")
        normalize_mesh_size(mesh_manager, target_size=1.0)
        
        # Step 3: Repair the mesh if necessary
        mesh_manager.log("Repairing mesh", "PROCESSING")
        mesh_manager.repair_mesh(
            fix_holes=True,
            remove_duplicates=True,
            fix_normals=True,
            remove_degenerate=True,
            fix_negative_volume=True,
            keep_largest_component=False,
            verbose=verbose
        )
        
        # Final analysis
        if verbose:
            mesh_manager.log("Final mesh analysis:")
            final_analysis = mesh_manager.analyze_mesh()
            mesh_manager.log(f"  Vertices: {final_analysis['vertex_count']}, Faces: {final_analysis['face_count']}")
            mesh_manager.log(f"  Volume: {final_analysis['volume']:.6f}")
            mesh_manager.log(f"  Watertight: {final_analysis['is_watertight']}")
            mesh_manager.log(f"  Bounds: {final_analysis['bounds']}")
        
        # Save the processed mesh
        mesh_manager.log(f"Saving processed mesh to {output_path.name}", "PROCESSING")
        mesh_manager.mesh.export(str(output_path))
        
        return True, f"Successfully processed {input_path.name}"
        
    except Exception as e:
        error_msg = f"Failed to process {input_path.name}: {str(e)}"
        if verbose:
            print(f"❌ {error_msg}")
        return False, error_msg


def main():
    """Main preprocessing function."""
    # Define directories
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    
    # Create output directory if it doesn't exist
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all OBJ files from raw directory
    try:
        obj_files = get_obj_files(raw_dir)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return
    
    if not obj_files:
        print(f"❌ No OBJ files found in {raw_dir}")
        return
    
    print(f"🔍 Found {len(obj_files)} OBJ files to process")
    print(f"📁 Input directory: {raw_dir.absolute()}")
    print(f"📁 Output directory: {processed_dir.absolute()}")
    print("=" * 60)
    
    # Process each mesh file
    successful = 0
    failed = 0
    
    for obj_file in obj_files:
        print(f"\n🔧 Processing: {obj_file.name}")
        print("-" * 40)
        
        # Define output path
        output_file = processed_dir / obj_file.name
        
        # Process the mesh
        success, message = process_single_mesh(obj_file, output_file, verbose=True)
        
        if success:
            successful += 1
            print(f"✅ {message}")
        else:
            failed += 1
            print(f"❌ {message}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 PROCESSING SUMMARY")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Total files: {len(obj_files)}")
    
    if successful > 0:
        print(f"\n🎉 Processed meshes saved to: {processed_dir.absolute()}")


if __name__ == "__main__":
    main()

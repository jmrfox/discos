"""
Test script for the fixed visualize_mesh_slice_interactive function.
This script creates a cylinder mesh and visualizes it with the interactive slice visualization.
"""

import sys
import os
import numpy as np

# Add the parent directory to the path so we can import gencomo
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import gencomo modules
import gencomo
import trimesh

def create_cylinder_mesh(radius=1.0, height=2.0, sections=32):
    """Create a simple cylinder mesh for testing."""
    # Create a cylinder using trimesh
    cylinder = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    
    # Center the cylinder at the origin
    cylinder.vertices -= cylinder.center_mass
    
    return cylinder

def main():
    """Main function to test the visualization."""
    print("Creating cylinder mesh...")
    cylinder = create_cylinder_mesh(radius=1.0, height=2.0)
    
    print("Cylinder mesh created with:")
    print(f"  - {len(cylinder.vertices)} vertices")
    print(f"  - {len(cylinder.faces)} faces")
    print(f"  - Z range: {cylinder.vertices[:, 2].min():.2f} to {cylinder.vertices[:, 2].max():.2f}")
    
    print("\nVisualizing cylinder with interactive slice...")
    fig = gencomo.mesh.utils.visualize_mesh_slice_interactive(
        cylinder,
        title="Interactive Cylinder Cross-Sections",
        slice_color="darkblue",
        mesh_color="lightblue",
        mesh_opacity=0.3,
        debug=True  # Enable debug output
    )
    
    # Show the figure
    fig.show()
    
    print("\nDone! The figure should be displayed in your browser.")
    print("Use the slider to change the Z-level and see the cross-section change.")

if __name__ == "__main__":
    main()

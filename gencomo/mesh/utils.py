"""
Mesh utility functions for GenCoMo.

This module contains standalone utility functions for mesh analysis, repair,
visualization, and preprocessing.
"""

import numpy as np
import trimesh
from typing import Optional, Tuple, Dict, Any, Union


def analyze_mesh(mesh_data: Union[trimesh.Trimesh, Tuple[np.ndarray, np.ndarray]]) -> dict:
    """
    Analyze and return mesh properties for diagnostic purposes.
    This function performs pure analysis without modifying the input mesh.

    Args:
        mesh_data: Either a Trimesh object or (vertices, faces) tuple

    Returns:
        Dictionary of mesh properties including volume, watertightness, winding consistency,
        face count, vertex count, bounds, and potential issues.
    """
    # Convert to mesh object if needed
    if isinstance(mesh_data, tuple):
        vertices, faces = mesh_data
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    else:
        # Work with a copy to avoid side effects
        mesh = mesh_data.copy()
    
    # Initialize results dictionary
    results = {
        "face_count": len(mesh.faces),
        "vertex_count": len(mesh.vertices),
        "bounds": mesh.bounds.tolist() if hasattr(mesh, "bounds") else None,
        "is_watertight": mesh.is_watertight,
        "is_winding_consistent": mesh.is_winding_consistent,
        "issues": [],
    }
    
    # Calculate volume (report actual value, even if negative)
    try:
        results["volume"] = mesh.volume
        if mesh.volume < 0:
            results["issues"].append("Negative volume detected - face normals may be inverted")
    except Exception as e:
        results["volume"] = None
        results["issues"].append(f"Volume calculation failed: {str(e)}")
    
    # Check for non-manifold edges
    try:
        if hasattr(mesh, "is_manifold"):
            results["is_manifold"] = mesh.is_manifold
            if not mesh.is_manifold:
                results["issues"].append("Non-manifold edges detected")
    except Exception:
        results["is_manifold"] = None

    # Calculate topological properties using trimesh's built-in methods
    try:
        # Use trimesh's built-in euler_number property for correct topology calculation
        # For a sphere: euler_number = 2
        # For a torus: euler_number = 0
        # For a double torus: euler_number = -2
        # Genus = (2 - euler_number) / 2
        
        results["euler_characteristic"] = mesh.euler_number
        
        # Only calculate genus for closed (watertight) meshes
        if mesh.is_watertight:
            # For a closed orientable surface: genus = (2 - euler_number) / 2
            results["genus"] = int((2 - mesh.euler_number) / 2)
            
            # Sanity check - genus should be non-negative for simple shapes
            if results["genus"] < 0:
                results["genus"] = 0  # Default to 0 for simple shapes like spheres, cylinders
                results["issues"].append("Calculated negative genus, defaulting to 0")
        else:
            # For non-watertight meshes, genus is not well-defined
            results["genus"] = None
            results["issues"].append("Genus undefined for non-watertight mesh")
    except Exception as e:
        results["genus"] = None
        results["euler_characteristic"] = None
        results["issues"].append(f"Topology calculation failed: {str(e)}")


    # Analyze face normals
    try:
        if hasattr(mesh, "face_normals") and mesh.face_normals is not None:
            # Get statistics on face normal directions
            results["normal_stats"] = {
                "mean": mesh.face_normals.mean(axis=0).tolist(),
                "std": mesh.face_normals.std(axis=0).tolist(),
                "sum": mesh.face_normals.sum(axis=0).tolist(),
            }
            
            # Check if normals are predominantly pointing inward (negative volume)
            if results.get("volume", 0) < 0:
                results["normal_direction"] = "inward"
            else:
                results["normal_direction"] = "outward"
    except Exception as e:
        results["normal_stats"] = None
        results["issues"].append(f"Normal analysis failed: {str(e)}")
    
    # Check for duplicate vertices and faces
    try:
        unique_verts = np.unique(mesh.vertices, axis=0)
        results["duplicate_vertices"] = len(mesh.vertices) - len(unique_verts)
        if results["duplicate_vertices"] > 0:
            results["issues"].append(f"Found {results['duplicate_vertices']} duplicate vertices")
    except Exception:
        results["duplicate_vertices"] = None
    
    # Check for degenerate faces (zero area)
    try:
        if hasattr(mesh, "area_faces"):
            degenerate_count = np.sum(mesh.area_faces < 1e-8)
            results["degenerate_faces"] = int(degenerate_count)
            if degenerate_count > 0:
                results["issues"].append(f"Found {degenerate_count} degenerate faces")
    except Exception:
        results["degenerate_faces"] = None
    
    # Check for connected components
    try:
        components = mesh.split(only_watertight=False)
        results["component_count"] = len(components)
        if len(components) > 1:
            results["issues"].append(f"Mesh has {len(components)} disconnected components")
    except Exception:
        results["component_count"] = None
    
    return results


def print_mesh_analysis(mesh_data: Union[trimesh.Trimesh, Tuple[np.ndarray, np.ndarray]], verbose: bool = False) -> None:
    """
    Analyze a mesh and print a formatted report of its properties.
    
    Args:
        mesh_data: Either a Trimesh object or (vertices, faces) tuple
        verbose: Whether to print detailed information
    """
    analysis = analyze_mesh(mesh_data)
    
    print("Mesh Analysis Report")
    print("====================")
    
    # Basic properties
    print(f"\nGeometry:")
    print(f"  * Vertices: {analysis['vertex_count']}")
    print(f"  * Faces: {analysis['face_count']}")
    if analysis.get('component_count') is not None:
        print(f"  * Components: {analysis['component_count']}")
    if analysis.get('volume') is not None:
        print(f"  * Volume: {analysis['volume']:.2f}")
    if analysis.get('bounds') is not None:
        min_bound, max_bound = analysis['bounds']
        print(f"  * Bounds: [{min_bound[0]:.1f}, {min_bound[1]:.1f}, {min_bound[2]:.1f}] to [{max_bound[0]:.1f}, {max_bound[1]:.1f}, {max_bound[2]:.1f}]")
    
    # Mesh quality
    print(f"\nMesh Quality:")
    print(f"  * Watertight: {analysis['is_watertight']}")
    print(f"  * Winding Consistent: {analysis['is_winding_consistent']}")
    if analysis.get('is_manifold') is not None:
        print(f"  * Manifold: {analysis['is_manifold']}")
    if analysis.get('normal_direction') is not None:
        print(f"  * Normal Direction: {analysis['normal_direction']}")
    if analysis.get('duplicate_vertices') is not None:
        print(f"  * Duplicate Vertices: {analysis['duplicate_vertices']}")
    if analysis.get('degenerate_faces') is not None:
        print(f"  * Degenerate Faces: {analysis['degenerate_faces']}")
    
    # Topology
    if analysis.get('genus') is not None or analysis.get('euler_characteristic') is not None:
        print(f"\nTopology:")
        if analysis.get('genus') is not None:
            print(f"  * Genus: {analysis['genus']}")
        if analysis.get('euler_characteristic') is not None:
            print(f"  * Euler Characteristic: {analysis['euler_characteristic']}")
    
    # Issues
    if analysis['issues']:
        print(f"\nIssues Detected ({len(analysis['issues'])}):")
        for i, issue in enumerate(analysis['issues']):
            print(f"  {i+1}. {issue}")
    else:
        print(f"\nNo issues detected")
    
    # Detailed stats
    if verbose and analysis.get('normal_stats') is not None:
        print(f"\nNormal Statistics:")
        mean = analysis['normal_stats']['mean']
        sum_val = analysis['normal_stats']['sum']
        print(f"  * Mean: [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
        print(f"  * Sum: [{sum_val[0]:.4f}, {sum_val[1]:.4f}, {sum_val[2]:.4f}]")
    
    print("\nRecommendation:")
    if analysis['issues']:
        print("  Consider using repair_mesh() to fix the detected issues.")
    else:
        print("  Mesh appears to be in good condition.")
    print("====================")


def repair_mesh(
    mesh_data: Union[trimesh.Trimesh, Tuple[np.ndarray, np.ndarray]],
    fix_holes: bool = True,
    remove_duplicates: bool = True,
    fix_normals: bool = True,
    remove_degenerate: bool = True,
    fix_negative_volume: bool = True,
    keep_largest_component: bool = False,
    verbose: bool = True,
) -> trimesh.Trimesh:
    """
    Attempt to repair common mesh issues to improve watertightness and quality.

    Args:
        mesh_data: Either a Trimesh object or (vertices, faces) tuple
        fix_holes: Whether to attempt filling holes
        remove_duplicates: Whether to remove duplicate faces and vertices
        fix_normals: Whether to fix face normal consistency
        remove_degenerate: Whether to remove degenerate faces
        fix_negative_volume: Whether to invert faces if mesh has negative volume
        keep_largest_component: Whether to keep only the largest connected component
        verbose: Whether to print repair summary

    Returns:
        Repaired mesh (new copy, original is not modified)
    """
    # Convert to mesh object if needed and create a copy
    if isinstance(mesh_data, tuple):
        vertices, faces = mesh_data
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    else:
        mesh = mesh_data.copy()  # Work on a copy

    repair_log = []
    
    # Fix negative volume by inverting faces if needed
    if fix_negative_volume:
        try:
            # Check if the mesh has a negative volume
            if hasattr(mesh, "volume") and mesh.volume < 0:
                initial_volume = mesh.volume
                mesh.invert()
                repair_log.append(f"Inverted faces to fix negative volume: {initial_volume:.2f} → {mesh.volume:.2f}")
        except Exception as e:
            repair_log.append(f"Failed to fix negative volume: {e}")

    # Remove duplicate and degenerate faces
    if remove_duplicates:
        try:
            initial_faces = len(mesh.faces)
            mesh.remove_duplicate_faces()
            removed_faces = initial_faces - len(mesh.faces)
            if removed_faces > 0:
                repair_log.append(f"Removed {removed_faces} duplicate faces")
        except Exception as e:
            repair_log.append(f"Failed to remove duplicate faces: {e}")

    if remove_degenerate:
        try:
            initial_faces = len(mesh.faces)
            mesh.remove_degenerate_faces()
            removed_faces = initial_faces - len(mesh.faces)
            if removed_faces > 0:
                repair_log.append(f"Removed {removed_faces} degenerate faces")
        except Exception as e:
            repair_log.append(f"Failed to remove degenerate faces: {e}")

    # Fix winding consistency
    if fix_normals:
        try:
            if not mesh.is_winding_consistent:
                mesh.fix_normals()
                if mesh.is_winding_consistent:
                    repair_log.append("Fixed face normal winding consistency")
                else:
                    repair_log.append("Attempted to fix normals but still inconsistent")
        except Exception as e:
            repair_log.append(f"Failed to fix normals: {e}")

    # Attempt to fill holes
    if fix_holes:
        try:
            if not mesh.is_watertight:
                initial_watertight = mesh.is_watertight
                mesh.fill_holes()
                if mesh.is_watertight and not initial_watertight:
                    repair_log.append("Successfully filled holes - mesh is now watertight")
                elif mesh.is_watertight:
                    repair_log.append("Mesh was already watertight")
                else:
                    repair_log.append("Attempted to fill holes but mesh still not watertight")
        except Exception as e:
            repair_log.append(f"Failed to fill holes: {e}")
    
    # Keep only the largest component if requested
    if keep_largest_component:
        try:
            components = mesh.split(only_watertight=False)
            if len(components) > 1:
                # Keep the largest component by volume or face count
                volumes = [abs(c.volume) if hasattr(c, "volume") else len(c.faces) for c in components]
                largest_idx = np.argmax(volumes)
                mesh = components[largest_idx]
                repair_log.append(f"Kept largest of {len(components)} components (volume: {volumes[largest_idx]:.2f})")
        except Exception as e:
            repair_log.append(f"Failed to isolate largest component: {e}")
    
    # Final processing to ensure consistency
    try:
        mesh.process(validate=True)
        repair_log.append("Applied final mesh processing and validation")
    except Exception as e:
        repair_log.append(f"Final processing failed: {e}")

    # Store repair log as mesh metadata
    if not hasattr(mesh, "metadata"):
        mesh.metadata = {}
    mesh.metadata["repair_log"] = repair_log

    # Print repair summary
    if verbose:
        if repair_log:
            print("🔧 Mesh Repair Summary:")
            for log_entry in repair_log:
                print(f"  • {log_entry}")
            
            # Print final mesh status
            print("\n📊 Final Mesh Status:")
            print(f"  • Volume: {mesh.volume if hasattr(mesh, 'volume') else 'N/A'}")
            print(f"  • Watertight: {mesh.is_watertight}")
            print(f"  • Winding consistent: {mesh.is_winding_consistent}")
            print(f"  • Faces: {len(mesh.faces)}")
            print(f"  • Vertices: {len(mesh.vertices)}")
        else:
            print("🔧 No repairs needed - mesh is in good condition")

    return mesh


def preprocess_mesh(mesh: trimesh.Trimesh, verbose: bool = True, **kwargs) -> trimesh.Trimesh:
    """
    Convenience function to preprocess a mesh with default settings.

    Args:
        mesh: Input mesh to preprocess
        verbose: Whether to print processing information
        **kwargs: Additional arguments passed to MeshProcessor.preprocess_mesh()

    Returns:
        Preprocessed mesh
    """
    from .processor import MeshProcessor

    processor = MeshProcessor(verbose=verbose)
    return processor.preprocess_mesh(mesh, **kwargs)


def visualize_mesh_3d(
    mesh_data: Union[trimesh.Trimesh, Tuple[np.ndarray, np.ndarray]],
    title: str = "3D Mesh Visualization",
    color: str = "lightblue",
    backend: str = "auto",
    show_axes: bool = True,
    show_wireframe: bool = False,
) -> Optional[object]:
    """
    Create a 3D visualization of a mesh.

    Args:
        mesh_data: Either a Trimesh object or (vertices, faces) tuple
        title: Plot title
        color: Mesh color (named color or RGB tuple)
        backend: Visualization backend ('plotly', 'matplotlib', 'trimesh', or 'auto')
        show_axes: Whether to show coordinate axes
        show_wireframe: Whether to show wireframe overlay

    Returns:
        Figure object (backend-dependent) or None if visualization fails
    """
    # Convert to mesh object if needed
    if isinstance(mesh_data, tuple):
        vertices, faces = mesh_data
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    else:
        mesh = mesh_data

    if backend == "auto":
        # Try plotly first, then fallback to others
        try:
            import plotly.graph_objects as go

            backend = "plotly"
        except ImportError:
            try:
                import matplotlib.pyplot as plt

                backend = "matplotlib"
            except ImportError:
                backend = "trimesh"

    if backend == "plotly":
        return _visualize_mesh_plotly(mesh, title, color, show_axes, show_wireframe)
    elif backend == "matplotlib":
        return _visualize_mesh_matplotlib(mesh, title, color, show_axes, show_wireframe)
    elif backend == "trimesh":
        return _visualize_mesh_trimesh(mesh, title, color)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def _visualize_mesh_plotly(mesh, title, color, show_axes, show_wireframe):
    """Plotly-based mesh visualization."""
    try:
        import plotly.graph_objects as go

        vertices = mesh.vertices
        faces = mesh.faces

        # Create mesh trace
        mesh_trace = go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            opacity=0.8,
            color=color,
            name="Mesh",
        )

        fig = go.Figure(data=[mesh_trace])

        # Add wireframe if requested
        if show_wireframe:
            # Create wireframe edges
            edge_trace = []
            for face in faces:
                for i in range(3):
                    v1, v2 = face[i], face[(i + 1) % 3]
                    edge_trace.extend(
                        [
                            vertices[v1][0],
                            vertices[v2][0],
                            None,
                            vertices[v1][1],
                            vertices[v2][1],
                            None,
                            vertices[v1][2],
                            vertices[v2][2],
                            None,
                        ]
                    )

            if edge_trace:
                fig.add_trace(
                    go.Scatter3d(
                        x=edge_trace[::3],
                        y=edge_trace[1::3],
                        z=edge_trace[2::3],
                        mode="lines",
                        line=dict(color="black", width=1),
                        name="Wireframe",
                    )
                )

        # Configure layout
        fig.update_layout(
            title=title,
            scene=dict(
                aspectmode="data",
                xaxis=dict(visible=show_axes),
                yaxis=dict(visible=show_axes),
                zaxis=dict(visible=show_axes),
            ),
        )

        return fig

    except ImportError:
        print("Plotly not available")
        return None
    except Exception as e:
        print(f"Plotly visualization failed: {e}")
        return None


def _visualize_mesh_matplotlib(mesh, title, color, show_axes, show_wireframe):
    """Matplotlib-based mesh visualization."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        vertices = mesh.vertices
        faces = mesh.faces

        # Create mesh surface
        poly3d = Poly3DCollection(
            vertices[faces], alpha=0.7, facecolor=color, edgecolor="black" if show_wireframe else None
        )
        ax.add_collection3d(poly3d)

        # Set equal aspect ratio
        ax.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
        ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
        ax.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())

        ax.set_xlabel("X (µm)")
        ax.set_ylabel("Y (µm)")
        ax.set_zlabel("Z (µm)")
        ax.set_title(title)

        if not show_axes:
            ax.set_axis_off()

        plt.tight_layout()
        return fig

    except ImportError:
        print("Matplotlib not available")
        return None
    except Exception as e:
        print(f"Matplotlib visualization failed: {e}")
        return None


def _visualize_mesh_trimesh(mesh, title, color):
    """Trimesh-based mesh visualization."""
    try:
        # Create a copy for visualization
        viz_mesh = mesh.copy()

        # Set mesh color
        if color:
            # Convert color name to RGB if needed
            color_map = {
                "lightblue": [173, 216, 230],
                "orange": [255, 165, 0],
                "lightgreen": [144, 238, 144],
                "lightcoral": [240, 128, 128],
                "purple": [128, 0, 128],
                "red": [255, 0, 0],
                "blue": [0, 0, 255],
                "green": [0, 255, 0],
                "yellow": [255, 255, 0],
                "pink": [255, 192, 203],
                "cyan": [0, 255, 255],
            }

            if color in color_map:
                rgb_color = color_map[color]
            else:
                # Default to light blue if color not found
                rgb_color = [173, 216, 230]

            # Set visual properties
            if hasattr(viz_mesh, "visual") and hasattr(viz_mesh.visual, "face_colors"):
                viz_mesh.visual.face_colors = rgb_color + [255]  # RGBA

        # Create scene with the mesh
        scene = trimesh.Scene([viz_mesh])

        # Set scene metadata
        scene.metadata["title"] = title

        return scene.show()

    except Exception as e:
        print(f"Trimesh visualization failed: {e}")
        return None



def visualize_mesh_slice_interactive(
    mesh_data: Union[trimesh.Trimesh, Tuple[np.ndarray, np.ndarray]],
    title: str = "Interactive Mesh Slice",
    z_range: Optional[Tuple[float, float]] = None,
    num_slices: int = 50,
    slice_color: str = "red",
    mesh_color: str = "lightblue",
    mesh_opacity: float = 0.3,
) -> Optional[object]:
    """
    Create an interactive 3D visualization of a mesh with a controllable slice plane.
    
    This function displays a 3D mesh and calculates the intersection of the mesh
    with an xy-plane at a user-controlled z-value. The intersection is shown as a
    colored line on the mesh. A slider allows the user to interactively change the
    z-value of the intersection plane.
    
    Args:
        mesh_data: Either a Trimesh object or (vertices, faces) tuple
        title: Plot title
        z_range: Tuple of (min_z, max_z) for slice range. Auto-detected if None.
        num_slices: Number of positions for the slider
        slice_color: Color for the intersection line
        mesh_color: Color for the 3D mesh
        mesh_opacity: Opacity of the 3D mesh (0-1)
    
    Returns:
        Plotly figure with interactive slider for controlling the z-value
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Plotly is required for interactive visualization")
        return None
        
    # Convert input to trimesh object if needed
    if isinstance(mesh_data, tuple):
        vertices, faces = mesh_data
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    else:
        mesh = mesh_data.copy()
    
    # Determine z-range if not provided
    if z_range is None:
        z_min, z_max = mesh.vertices[:, 2].min(), mesh.vertices[:, 2].max()
        # Add small padding
        padding = (z_max - z_min) * 0.05
        z_min -= padding
        z_max += padding
    else:
        z_min, z_max = z_range
    
    # Create the base figure with the mesh
    fig = go.Figure()
    
    # Add the mesh to the figure
    fig.add_trace(go.Mesh3d(
        x=mesh.vertices[:, 0],
        y=mesh.vertices[:, 1],
        z=mesh.vertices[:, 2],
        i=mesh.faces[:, 0],
        j=mesh.faces[:, 1],
        k=mesh.faces[:, 2],
        opacity=mesh_opacity,
        color=mesh_color,
        name="Mesh"
    ))
    
    # Function to create a slice at a given z-value
    def create_slice_trace(z_value):
        # Calculate intersection with plane at z_value
        section = mesh.section(plane_origin=[0, 0, z_value], plane_normal=[0, 0, 1])
        
        # If no intersection, return None
        if section is None or not hasattr(section, 'entities') or len(section.entities) == 0:
            return None
            
        # Process all entities in the section to get 3D coordinates
        all_points = []
        
        for entity in section.entities:
            if hasattr(entity, 'points') and len(entity.points) > 0:
                # Get the actual 2D coordinates
                points_2d = section.vertices[entity.points]
                
                # Convert to 3D by adding z_value
                points_3d = np.column_stack([points_2d, np.full(len(points_2d), z_value)])
                
                # Add closing point if needed (to complete the loop)
                if len(points_2d) > 2 and not np.array_equal(points_2d[0], points_2d[-1]):
                    closing_point = np.array([points_2d[0][0], points_2d[0][1], z_value])
                    points_3d = np.vstack([points_3d, closing_point])
                
                # Add to all points list
                all_points.extend(points_3d.tolist())
                
                # Add None to create a break between separate entities
                all_points.append([None, None, None])
        
        # If we have points, create a scatter trace
        if all_points:
            x_coords = [p[0] if p is not None else None for p in all_points]
            y_coords = [p[1] if p is not None else None for p in all_points]
            z_coords = [p[2] if p is not None else None for p in all_points]
            
            return go.Scatter3d(
                x=x_coords,
                y=y_coords,
                z=z_coords,
                mode='lines',
                line=dict(color=slice_color, width=5),
                name=f'Slice at z={z_value:.2f}'
            )
        
        return None
    
    # Create initial slice
    initial_z = (z_min + z_max) / 2
    initial_slice = create_slice_trace(initial_z)
    
    # Add initial slice to figure if it exists
    if initial_slice:
        fig.add_trace(initial_slice)
    
    # Create frames for animation
    frames = []
    for i, z_val in enumerate(np.linspace(z_min, z_max, num_slices)):
        # Create a slice at this z-value
        slice_trace = create_slice_trace(z_val)
        
        # If we have a valid slice, add it to frames
        if slice_trace:
            frame_data = [fig.data[0], slice_trace]  # Mesh and slice
        else:
            frame_data = [fig.data[0]]  # Just the mesh
            
        frames.append(go.Frame(
            data=frame_data,
            name=f"frame_{i}",
            traces=[0, 1]  # Update both traces
        ))
    
    # Create slider steps
    steps = []
    for i, z_val in enumerate(np.linspace(z_min, z_max, num_slices)):
        step = dict(
            args=[
                [f"frame_{i}"],
                {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}
            ],
            label=f"{z_val:.2f}",
            method="animate"
        )
        steps.append(step)
    
    # Configure the slider
    sliders = [dict(
        active=num_slices // 2,  # Start in the middle
        currentvalue={"prefix": "Z-value: ", "visible": True, "xanchor": "right"},
        pad={"t": 50, "b": 10},
        len=0.9,
        x=0.1,
        y=0,
        steps=steps
    )]
    
    # Configure the figure layout
    fig.update_layout(
        title=title,
        scene=dict(
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        height=800,  # Taller to make room for slider
        margin=dict(l=50, r=50, b=100, t=100),  # Add margin at bottom for slider
        sliders=sliders,
        # Add animation controls
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=0,
            x=0,
            xanchor="left",
            yanchor="top",
            pad=dict(t=60, r=10),
            buttons=[dict(
                label="Play",
                method="animate",
                args=[None, {"frame": {"duration": 200, "redraw": True}, "fromcurrent": True}]
            ), dict(
                label="Pause",
                method="animate",
                args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]
            ), dict(
                label="Reset View",
                method="relayout",
                args=[{"scene.camera.eye": dict(x=1.5, y=1.5, z=1.5)}]
            )]
        )]
    )
    
    # Set frames
    fig.frames = frames
    
    return fig


def visualize_mesh_slice_grid(
    mesh_data: Union[trimesh.Trimesh, Tuple[np.ndarray, np.ndarray]] = None,
    vertices: Optional[np.ndarray] = None,
    faces: Optional[np.ndarray] = None,
    title: str = "Mesh Slice Grid",
    num_slices: int = 9,
    z_range: Optional[Tuple[float, float]] = None,
) -> Optional[object]:
    """
    Create a grid visualization showing multiple cross-sections of a 3D mesh.

    Args:
        mesh_data: Either a Trimesh object or (vertices, faces) tuple
        vertices: Vertex array (alternative to mesh_data)
        faces: Face array (alternative to mesh_data)
        title: Plot title
        num_slices: Number of slices to show (should be perfect square for grid)
        z_range: Tuple of (min_z, max_z) for slice range. Auto-detected if None.

    Returns:
        Plotly figure with subplot grid
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import math
    except ImportError:
        print("Plotly not available for slice grid visualization")
        return None

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

    # Determine grid size
    grid_size = int(math.sqrt(num_slices))
    if grid_size * grid_size != num_slices:
        grid_size = int(math.ceil(math.sqrt(num_slices)))
        num_slices = grid_size * grid_size

    # Determine Z range
    if z_range is None:
        z_min, z_max = mesh.vertices[:, 2].min(), mesh.vertices[:, 2].max()
    else:
        z_min, z_max = z_range

    # Create Z levels
    z_levels = np.linspace(z_min, z_max, num_slices)

    # Create subplots
    fig = make_subplots(
        rows=grid_size,
        cols=grid_size,
        subplot_titles=[f"Z = {z:.2f}" for z in z_levels],
        specs=[[{"type": "xy"}] * grid_size for _ in range(grid_size)],
    )

    # Generate slices and add to subplots
    for i, z_level in enumerate(z_levels):
        row = i // grid_size + 1
        col = i % grid_size + 1

        try:
            # Get 2D cross-section
            slice_2d = mesh.section(plane_origin=[0, 0, z_level], plane_normal=[0, 0, 1])

            if slice_2d is not None and hasattr(slice_2d, "entities") and len(slice_2d.entities) > 0:
                # Plot each entity in the slice
                for entity in slice_2d.entities:
                    if hasattr(entity, "points"):
                        points = slice_2d.vertices[entity.points]
                        # Close the loop
                        points_closed = np.vstack([points, points[0]])

                        fig.add_trace(
                            go.Scatter(
                                x=points_closed[:, 0],
                                y=points_closed[:, 1],
                                mode="lines",
                                line=dict(color="red", width=2),
                                showlegend=False,
                            ),
                            row=row,
                            col=col,
                        )

            # Set equal aspect ratio for each subplot
            fig.update_xaxes(scaleanchor="y", scaleratio=1, row=row, col=col)
            fig.update_xaxes(title_text="X (µm)", row=row, col=col)
            fig.update_yaxes(title_text="Y (µm)", row=row, col=col)

        except Exception as e:
            # If slicing fails, just leave subplot empty
            pass

    fig.update_layout(
        title=title,
        height=150 * grid_size,
        showlegend=False,
    )

    return fig

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
        Dictionary of mesh properties including topological genus
    """
    # Convert to mesh object if needed
    if isinstance(mesh_data, tuple):
        vertices, faces = mesh_data
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    else:
        # Work with a copy to avoid side effects
        mesh = mesh_data.copy()

    # Calculate volume robustly without modifying the mesh
    volume = None
    try:
        if mesh.is_volume and mesh.is_watertight:
            volume = mesh.volume
        elif hasattr(mesh, "volume") and mesh.volume > 0:
            volume = abs(mesh.volume)  # Take absolute value in case of negative volume
        else:
            # Fallback: try to calculate volume even if not marked as watertight
            try:
                vol = mesh.volume
                if vol is not None and not np.isnan(vol):
                    volume = abs(vol)
            except:
                volume = None
    except:
        volume = None

    # Calculate topological genus using Euler characteristic
    # Genus = (2 - V + E - F) / 2, where V = vertices, E = edges, F = faces
    genus = None
    edges = None
    euler_char = None
    try:
        V = len(mesh.vertices)
        F = len(mesh.faces)
        # Calculate number of edges
        edges = set()
        for face in mesh.faces:
            for i in range(3):
                edge = tuple(sorted([face[i], face[(i + 1) % 3]]))
                edges.add(edge)
        E = len(edges)

        # Euler characteristic for a surface: χ = V - E + F
        euler_char = V - E + F

        # For a closed surface: genus = (2 - χ) / 2
        # For multiple connected components: genus = (2 - χ + 2*(components - 1)) / 2
        # Simplifying for single component: genus = (2 - χ) / 2
        genus = (2 - euler_char) // 2

        # Ensure genus is non-negative
        if genus < 0:
            genus = 0

    except Exception as e:
        genus = None

    properties = {
        "num_vertices": len(mesh.vertices),
        "num_faces": len(mesh.faces),
        "num_edges": len(edges) if edges is not None else None,
        "volume": volume,
        "surface_area": mesh.area,
        "is_watertight": mesh.is_watertight,
        "is_winding_consistent": mesh.is_winding_consistent,
        "euler_characteristic": euler_char,
        "genus": genus,
        "bounds": {
            "x_range": (mesh.vertices[:, 0].min(), mesh.vertices[:, 0].max()),
            "y_range": (mesh.vertices[:, 1].min(), mesh.vertices[:, 1].max()),
            "z_range": (mesh.vertices[:, 2].min(), mesh.vertices[:, 2].max()),
        },
        "centroid": mesh.centroid.tolist() if hasattr(mesh, "centroid") else None,
        "bounding_box_volume": mesh.bounding_box.volume,
    }

    # Add convex hull volume safely
    try:
        if hasattr(mesh, "convex_hull") and mesh.convex_hull is not None:
            properties["convex_hull_volume"] = mesh.convex_hull.volume
        else:
            properties["convex_hull_volume"] = None
    except:
        properties["convex_hull_volume"] = None

    return properties


def repair_mesh(
    mesh_data: Union[trimesh.Trimesh, Tuple[np.ndarray, np.ndarray]],
    fix_holes: bool = True,
    remove_duplicates: bool = True,
    fix_normals: bool = True,
    remove_degenerate: bool = True,
) -> trimesh.Trimesh:
    """
    Attempt to repair common mesh issues to improve watertightness and quality.

    Args:
        mesh_data: Either a Trimesh object or (vertices, faces) tuple
        fix_holes: Whether to attempt filling holes
        remove_duplicates: Whether to remove duplicate faces and vertices
        fix_normals: Whether to fix face normal consistency
        remove_degenerate: Whether to remove degenerate faces

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

    # Store repair log as mesh metadata
    if not hasattr(mesh, "metadata"):
        mesh.metadata = {}
    mesh.metadata["repair_log"] = repair_log

    # Print repair summary
    if repair_log:
        print("🔧 Mesh Repair Summary:")
        for log_entry in repair_log:
            print(f"  • {log_entry}")
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
    Create an interactive visualization showing cross-sections of a 3D mesh.

    Users can use a slider to explore different Z-slice levels and see how
    the geometry changes along the Z-axis.

    Args:
        mesh_data: Either a Trimesh object or (vertices, faces) tuple
        title: Plot title
        z_range: Tuple of (min_z, max_z) for slice range. Auto-detected if None.
        num_slices: Number of slice levels to create
        slice_color: Color for the slice lines
        mesh_color: Color for the background mesh
        mesh_opacity: Opacity of the background mesh (0-1)

    Returns:
        Interactive Plotly figure with slider
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("Plotly not available for interactive slice visualization")
        return None

    # Handle different input formats
    if isinstance(mesh_data, tuple):
        vertices, faces = mesh_data
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    else:
        mesh = mesh_data

    # Determine Z range
    if z_range is None:
        z_min, z_max = mesh.vertices[:, 2].min(), mesh.vertices[:, 2].max()
        # Add small padding
        z_padding = (z_max - z_min) * 0.05
        z_min -= z_padding
        z_max += z_padding
    else:
        z_min, z_max = z_range

    # Create slice levels
    slice_levels = np.linspace(z_min, z_max, num_slices)

    try:
        # Create the base mesh visualization
        fig = go.Figure()

        # Add the mesh with reduced opacity
        fig.add_trace(
            go.Mesh3d(
                x=mesh.vertices[:, 0],
                y=mesh.vertices[:, 1],
                z=mesh.vertices[:, 2],
                i=mesh.faces[:, 0],
                j=mesh.faces[:, 1],
                k=mesh.faces[:, 2],
                opacity=mesh_opacity,
                color=mesh_color,
                name="Mesh",
                showlegend=True,
            )
        )

        # Create frames for animation/slider
        frames = []
        steps = []

        for i, z_level in enumerate(slice_levels):
            # Calculate intersection with plane at z_level
            try:
                slice_2d = mesh.section(plane_origin=[0, 0, z_level], plane_normal=[0, 0, 1])

                if slice_2d is not None and hasattr(slice_2d, "entities") and len(slice_2d.entities) > 0:
                    # Convert 2D slice to 3D coordinates
                    slice_coords = []
                    for entity in slice_2d.entities:
                        if hasattr(entity, "points"):
                            points_3d = np.column_stack([entity.points, np.full(len(entity.points), z_level)])
                            slice_coords.extend(points_3d.tolist())
                            slice_coords.append([None, None, None])  # Separator for discontinuous lines

                    if slice_coords:
                        x_coords = [p[0] if p[0] is not None else None for p in slice_coords]
                        y_coords = [p[1] if p[1] is not None else None for p in slice_coords]
                        z_coords = [p[2] if p[2] is not None else None for p in slice_coords]

                        frame_data = [
                            go.Mesh3d(
                                x=mesh.vertices[:, 0],
                                y=mesh.vertices[:, 1],
                                z=mesh.vertices[:, 2],
                                i=mesh.faces[:, 0],
                                j=mesh.faces[:, 1],
                                k=mesh.faces[:, 2],
                                opacity=mesh_opacity,
                                color=mesh_color,
                                name="Mesh",
                            ),
                            go.Scatter3d(
                                x=x_coords,
                                y=y_coords,
                                z=z_coords,
                                mode="lines",
                                line=dict(color=slice_color, width=4),
                                name=f"Slice at Z={z_level:.2f}",
                            ),
                        ]
                    else:
                        # No intersection at this level
                        frame_data = [
                            go.Mesh3d(
                                x=mesh.vertices[:, 0],
                                y=mesh.vertices[:, 1],
                                z=mesh.vertices[:, 2],
                                i=mesh.faces[:, 0],
                                j=mesh.faces[:, 1],
                                k=mesh.faces[:, 2],
                                opacity=mesh_opacity,
                                color=mesh_color,
                                name="Mesh",
                            )
                        ]
                else:
                    # No slice available
                    frame_data = [
                        go.Mesh3d(
                            x=mesh.vertices[:, 0],
                            y=mesh.vertices[:, 1],
                            z=mesh.vertices[:, 2],
                            i=mesh.faces[:, 0],
                            j=mesh.faces[:, 1],
                            k=mesh.faces[:, 2],
                            opacity=mesh_opacity,
                            color=mesh_color,
                            name="Mesh",
                        )
                    ]

            except Exception as e:
                # Fallback for slicing errors
                frame_data = [
                    go.Mesh3d(
                        x=mesh.vertices[:, 0],
                        y=mesh.vertices[:, 1],
                        z=mesh.vertices[:, 2],
                        i=mesh.faces[:, 0],
                        j=mesh.faces[:, 1],
                        k=mesh.faces[:, 2],
                        opacity=mesh_opacity,
                        color=mesh_color,
                        name="Mesh",
                    )
                ]

            frames.append(go.Frame(data=frame_data, name=str(i)))

            steps.append(
                {
                    "args": [[str(i)], {"frame": {"duration": 100, "redraw": True}, "mode": "immediate"}],
                    "label": f"{z_level:.1f}",
                    "method": "animate",
                }
            )

        # Add slider
        sliders = [{"active": 0, "currentvalue": {"prefix": "Z-level: "}, "pad": {"t": 50}, "steps": steps}]

        fig.frames = frames
        fig.update_layout(
            title=title,
            sliders=sliders,
            scene=dict(aspectmode="data", camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))),
            updatemenus=[
                {
                    "type": "buttons",
                    "showactive": False,
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [None, {"frame": {"duration": 200}, "fromcurrent": True}],
                        },
                        {
                            "label": "Pause",
                            "method": "animate",
                            "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}],
                        },
                    ],
                }
            ],
        )

        return fig

    except Exception as e:
        print(f"Failed to create interactive slice visualization: {e}")
        # Fallback to simple mesh visualization
        return visualize_mesh_3d(mesh_data, title=title, color=mesh_color)

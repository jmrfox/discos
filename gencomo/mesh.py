"""
Mesh processing utilities for GenCoMo.

Handles loading, processing, and analyzing neuronal meshes from various formats.
"""

import numpy as np
import trimesh
import open3d as o3d
from typing import Optional, Tuple, Dict, Any, Union
import warnings


class MeshProcessor:
    """
    Handles mesh loading, processing, and basic geometric operations.
    """

    def __init__(self):
        self.mesh = None
        self.original_mesh = None
        self.bounds = None

    def load_mesh(self, filepath: str, file_format: Optional[str] = None) -> trimesh.Trimesh:
        """
        Load a mesh from file.

        Args:
            filepath: Path to mesh file
            file_format: Optional format specification (auto-detected if None)

        Returns:
            Loaded trimesh object
        """
        try:
            if file_format:
                mesh = trimesh.load(filepath, file_type=file_format)
            else:
                mesh = trimesh.load(filepath)

            # Ensure we have a single mesh
            if isinstance(mesh, trimesh.Scene):
                # If it's a scene, try to get the first geometry
                geometries = list(mesh.geometry.values())
                if geometries:
                    mesh = geometries[0]
                else:
                    raise ValueError("No geometry found in mesh scene")

            if not isinstance(mesh, trimesh.Trimesh):
                raise ValueError(f"Loaded object is not a mesh: {type(mesh)}")

            self.mesh = mesh
            self.original_mesh = mesh.copy()
            self.bounds = self._compute_bounds()

            print(f"Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
            print(f"Bounds: {self.bounds}")

            return mesh

        except Exception as e:
            raise ValueError(f"Failed to load mesh from {filepath}: {str(e)}")

    def load_from_arrays(self, vertices: np.ndarray, faces: np.ndarray) -> trimesh.Trimesh:
        """
        Create mesh from vertex and face arrays.

        Args:
            vertices: Nx3 array of vertex coordinates
            faces: Mx3 array of face vertex indices

        Returns:
            Created trimesh object
        """
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        self.mesh = mesh
        self.original_mesh = mesh.copy()
        self.bounds = self._compute_bounds()
        return mesh

    def _compute_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Compute mesh bounding box."""
        if self.mesh is None:
            return None

        min_coords = self.mesh.vertices.min(axis=0)
        max_coords = self.mesh.vertices.max(axis=0)

        return {
            "x": (min_coords[0], max_coords[0]),
            "y": (min_coords[1], max_coords[1]),
            "z": (min_coords[2], max_coords[2]),
        }

    def get_z_range(self) -> Tuple[float, float]:
        """Get the z-axis range of the mesh."""
        if self.bounds is None:
            raise ValueError("No mesh loaded")
        return self.bounds["z"]

    def center_mesh(self, center_on: str = "centroid") -> trimesh.Trimesh:
        """
        Center the mesh.

        Args:
            center_on: 'centroid', 'bounds_center', or 'origin'

        Returns:
            Centered mesh
        """
        if self.mesh is None:
            raise ValueError("No mesh loaded")

        if center_on == "centroid":
            center = self.mesh.centroid
        elif center_on == "bounds_center":
            center = self.mesh.bounds.mean(axis=0)
        elif center_on == "origin":
            center = np.array([0, 0, 0])
        else:
            raise ValueError(f"Unknown center_on option: {center_on}")

        self.mesh.vertices -= center
        self.bounds = self._compute_bounds()

        return self.mesh

    def scale_mesh(self, scale_factor: float) -> trimesh.Trimesh:
        """
        Scale the mesh uniformly.

        Args:
            scale_factor: Scaling factor

        Returns:
            Scaled mesh
        """
        if self.mesh is None:
            raise ValueError("No mesh loaded")

        self.mesh.vertices *= scale_factor
        self.bounds = self._compute_bounds()

        return self.mesh

    def align_with_z_axis(self, target_axis: np.ndarray = None) -> trimesh.Trimesh:
        """
        Align the mesh's principal axis with the z-axis.

        Args:
            target_axis: Target direction (default: [0, 0, 1])

        Returns:
            Aligned mesh
        """
        if self.mesh is None:
            raise ValueError("No mesh loaded")

        if target_axis is None:
            target_axis = np.array([0, 0, 1])

        # Compute principal axis using PCA
        vertices_centered = self.mesh.vertices - self.mesh.centroid
        covariance = np.cov(vertices_centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)

        # Principal axis is the eigenvector with largest eigenvalue
        principal_axis = eigenvectors[:, np.argmax(eigenvalues)]

        # Compute rotation to align principal axis with target
        rotation_matrix = self._rotation_matrix_between_vectors(principal_axis, target_axis)

        # Apply rotation
        self.mesh.vertices = (rotation_matrix @ vertices_centered.T).T + self.mesh.centroid
        self.bounds = self._compute_bounds()

        return self.mesh

    def _rotation_matrix_between_vectors(self, vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
        """Compute rotation matrix to rotate vec1 to vec2."""
        # Normalize vectors
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)

        # Check if vectors are already aligned
        if np.allclose(vec1, vec2):
            return np.eye(3)
        if np.allclose(vec1, -vec2):
            # 180-degree rotation - find perpendicular axis
            perp = np.array([1, 0, 0]) if abs(vec1[0]) < 0.9 else np.array([0, 1, 0])
            axis = np.cross(vec1, perp)
            axis = axis / np.linalg.norm(axis)
            return self._rodrigues_rotation(axis, np.pi)

        # General case using Rodrigues' formula
        axis = np.cross(vec1, vec2)
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(np.clip(np.dot(vec1, vec2), -1, 1))

        return self._rodrigues_rotation(axis, angle)

    def _rodrigues_rotation(self, axis: np.ndarray, angle: float) -> np.ndarray:
        """Rodrigues' rotation formula."""
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K

    def smooth_mesh(self, iterations: int = 1) -> trimesh.Trimesh:
        """
        Apply Laplacian smoothing to the mesh.

        Args:
            iterations: Number of smoothing iterations

        Returns:
            Smoothed mesh
        """
        if self.mesh is None:
            raise ValueError("No mesh loaded")

        self.mesh = self.mesh.smoothed(iterations=iterations)
        return self.mesh

    def compute_mesh_properties(self) -> Dict[str, Any]:
        """Compute basic mesh properties."""
        if self.mesh is None:
            raise ValueError("No mesh loaded")

        return {
            "volume": self.mesh.volume,
            "surface_area": self.mesh.area,
            "is_watertight": self.mesh.is_watertight,
            "is_winding_consistent": self.mesh.is_winding_consistent,
            "num_vertices": len(self.mesh.vertices),
            "num_faces": len(self.mesh.faces),
            "bounds": self.bounds,
            "centroid": self.mesh.centroid.tolist(),
        }

    def repair_mesh(self) -> trimesh.Trimesh:
        """Attempt to repair common mesh issues."""
        if self.mesh is None:
            raise ValueError("No mesh loaded")

        # Remove duplicate vertices
        self.mesh.remove_duplicate_faces()
        self.mesh.remove_degenerate_faces()

        # Try to make watertight if not already
        if not self.mesh.is_watertight:
            try:
                self.mesh.fill_holes()
            except:
                warnings.warn("Could not automatically repair mesh holes")

        return self.mesh


# Mesh visualization functions


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
            title=title,
            scene=dict(
                xaxis_title="X (µm)",
                yaxis_title="Y (µm)",
                zaxis_title="Z (µm)",
                aspectmode="data",  # Use data aspect ratio to preserve real geometry
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),  # Better default viewing angle
            ),
            width=800,
            height=600,
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


# Mesh utility functions


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


def visualize_mesh_slice_interactive(
    mesh_data: Union[trimesh.Trimesh, Tuple[np.ndarray, np.ndarray]] = None,
    vertices: Optional[np.ndarray] = None,
    faces: Optional[np.ndarray] = None,
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
        vertices: Vertex array (alternative to mesh_data)
        faces: Face array (alternative to mesh_data)
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

    # Determine Z range
    if z_range is None:
        z_min, z_max = mesh.vertices[:, 2].min(), mesh.vertices[:, 2].max()
        # Add small padding
        z_padding = (z_max - z_min) * 0.05
        z_range = (z_min - z_padding, z_max + z_padding)
    else:
        z_min, z_max = z_range

    # Create Z levels for slicing
    z_levels = np.linspace(z_min, z_max, num_slices)

    # Create base mesh trace
    mesh_vertices = mesh.vertices
    mesh_faces = mesh.faces

    base_mesh = go.Mesh3d(
        x=mesh_vertices[:, 0],
        y=mesh_vertices[:, 1],
        z=mesh_vertices[:, 2],
        i=mesh_faces[:, 0],
        j=mesh_faces[:, 1],
        k=mesh_faces[:, 2],
        color=mesh_color,
        opacity=mesh_opacity,
        name="Mesh",
    )

    # Create slice traces for each Z level
    slice_traces = []
    for i, z_level in enumerate(z_levels):
        try:
            # Get 2D cross-section at this Z level
            slice_2d = mesh.section(plane_origin=[0, 0, z_level], plane_normal=[0, 0, 1])

            if slice_2d is not None and hasattr(slice_2d, "entities") and len(slice_2d.entities) > 0:
                # Convert slice to 3D coordinates - handle multiple disconnected regions
                all_x_coords = []
                all_y_coords = []
                all_z_coords = []

                for entity in slice_2d.entities:
                    if hasattr(entity, "points"):
                        points = slice_2d.vertices[entity.points]
                        # Add Z coordinate
                        points_3d = np.column_stack([points, np.full(len(points), z_level)])

                        # Add points for this entity
                        all_x_coords.extend(points_3d[:, 0])
                        all_y_coords.extend(points_3d[:, 1])
                        all_z_coords.extend(points_3d[:, 2])

                        # Close the loop for this entity
                        if len(points_3d) > 0:
                            all_x_coords.append(points_3d[0, 0])
                            all_y_coords.append(points_3d[0, 1])
                            all_z_coords.append(points_3d[0, 2])

                        # Add line break between entities (None values)
                        all_x_coords.append(None)
                        all_y_coords.append(None)
                        all_z_coords.append(None)

                if all_x_coords:
                    slice_trace = go.Scatter3d(
                        x=all_x_coords,
                        y=all_y_coords,
                        z=all_z_coords,
                        mode="lines",
                        line=dict(color=slice_color, width=4),
                        name=f"Slice Z={z_level:.2f}",
                        visible=(i == num_slices // 2),  # Start with middle slice visible
                    )
                    slice_traces.append(slice_trace)
                else:
                    # Empty slice
                    slice_trace = go.Scatter3d(
                        x=[],
                        y=[],
                        z=[],
                        mode="lines",
                        name=f"Slice Z={z_level:.2f}",
                        visible=(i == num_slices // 2),
                    )
                    slice_traces.append(slice_trace)
            else:
                # No intersection at this level
                slice_trace = go.Scatter3d(
                    x=[],
                    y=[],
                    z=[],
                    mode="lines",
                    name=f"Slice Z={z_level:.2f}",
                    visible=(i == num_slices // 2),
                )
                slice_traces.append(slice_trace)

        except Exception as e:
            # If slicing fails, create empty trace
            slice_trace = go.Scatter3d(
                x=[],
                y=[],
                z=[],
                mode="lines",
                name=f"Slice Z={z_level:.2f}",
                visible=(i == num_slices // 2),
            )
            slice_traces.append(slice_trace)

    # Create figure with all traces
    fig = go.Figure(data=[base_mesh] + slice_traces)

    # Create slider steps
    steps = []
    for i, z_level in enumerate(z_levels):
        step = dict(
            method="update",
            args=[{"visible": [True] + [j == i for j in range(len(slice_traces))]}],
            label=f"{z_level:.2f}",
        )
        steps.append(step)

    # Add slider
    sliders = [dict(active=num_slices // 2, currentvalue={"prefix": "Z-slice: "}, pad={"t": 50}, steps=steps)]

    fig.update_layout(
        title=f"{title}<br><sub>Use slider to explore different Z-levels</sub>",
        scene=dict(
            xaxis_title="X (µm)",
            yaxis_title="Y (µm)",
            zaxis_title="Z (µm)",
            aspectmode="data",  # Use data aspect ratio to preserve real geometry
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),  # Better default viewing angle
        ),
        sliders=sliders,
        width=800,
        height=700,
    )

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

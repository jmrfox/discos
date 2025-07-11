"""
Z-stack operations for GenCoMo.

This module provides functions for creating, analyzing, and manipulating
z-stack morphologies - GenCoMo's primary format for neuronal geometries.
"""

import numpy as np
import trimesh
from typing import Optional, Tuple, List, Union, Dict, Any
import warnings


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
    # Calculate basic properties
    voxel_size = metadata.get("xy_resolution", 1.0) ** 2 * metadata.get("z_resolution", 1.0)
    volume = np.sum(z_stack) * voxel_size

    # Estimate surface area using basic edge detection
    surface_voxels = 0
    for z in range(z_stack.shape[0]):
        for y in range(z_stack.shape[1]):
            for x in range(z_stack.shape[2]):
                if z_stack[z, y, x] == 1:
                    # Check if this voxel is on the surface (has at least one neighbor that's 0)
                    is_surface = False
                    for dz, dy, dx in [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]:
                        nz, ny, nx = z + dz, y + dy, x + dx
                        if (
                            nz < 0
                            or nz >= z_stack.shape[0]
                            or ny < 0
                            or ny >= z_stack.shape[1]
                            or nx < 0
                            or nx >= z_stack.shape[2]
                            or z_stack[nz, ny, nx] == 0
                        ):
                            is_surface = True
                            break
                    if is_surface:
                        surface_voxels += 1

    surface_area = surface_voxels * metadata.get("xy_resolution", 1.0) ** 2

    properties = {
        "shape": z_stack.shape,
        "total_voxels": z_stack.size,
        "neuron_voxels": np.sum(z_stack),
        "volume": volume,
        "surface_area": surface_area,
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
        "connected_components": 1,  # For now, assume single component
        "euler_number": 2,  # For simple topologies
        "bounding_box": [
            0,
            z_stack.shape[2] * metadata.get("xy_resolution", 1.0),
            0,
            z_stack.shape[1] * metadata.get("xy_resolution", 1.0),
            0,
            z_stack.shape[0] * metadata.get("z_resolution", 1.0),
        ],
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


def visualize_zstack_slices(
    z_stack: np.ndarray,
    metadata: Dict[str, Any],
    title: str = "Z-stack Slice Viewer",
    colormap: str = "viridis",
    backend: str = "plotly",
) -> Optional[object]:
    """
    Create an interactive slice viewer for z-stack data.

    Allows users to flip through z-slices interactively to explore
    the internal structure of neuronal morphologies.

    Args:
        z_stack: 3D binary array (z, y, x)
        metadata: Metadata dict with coordinate information
        title: Plot title
        colormap: Colormap for visualization ('viridis', 'gray', 'hot', etc.)
        backend: Visualization backend ('plotly', 'matplotlib')

    Returns:
        Interactive figure object
    """
    if backend == "plotly":
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            # Get coordinates
            x_coords = metadata.get("x_coords", np.arange(z_stack.shape[2]))
            y_coords = metadata.get("y_coords", np.arange(z_stack.shape[1]))
            z_coords = metadata.get("z_coords", np.arange(z_stack.shape[0]))

            xy_resolution = metadata.get("xy_resolution", 1.0)
            z_resolution = metadata.get("z_resolution", 1.0)

            # Create figure with slider
            fig = go.Figure()

            # Add initial slice (middle slice)
            initial_slice = z_stack.shape[0] // 2

            # Add all slices as traces (initially hidden except the first one)
            for k in range(z_stack.shape[0]):
                slice_data = z_stack[k, :, :]

                fig.add_trace(
                    go.Heatmap(
                        z=slice_data,
                        x=x_coords,
                        y=y_coords,
                        colorscale=colormap,
                        name=f"Z-slice {k}",
                        visible=(k == initial_slice),
                        hovertemplate="<b>X:</b> %{x:.1f} µm<br>"
                        + "<b>Y:</b> %{y:.1f} µm<br>"
                        + "<b>Value:</b> %{z}<br>"
                        + "<extra></extra>",
                        showscale=True,  # Always show colorbar
                        colorbar=dict(title="Neuron<br>Presence", tickvals=[0, 1], ticktext=["Outside", "Inside"]),
                    )
                )

            # Create slider steps
            steps = []
            for k in range(z_stack.shape[0]):
                z_pos = z_coords[k] if len(z_coords) > k else k * z_resolution

                # Create visibility array
                visibility = [False] * z_stack.shape[0]
                visibility[k] = True

                step = dict(
                    method="update",
                    args=[
                        {"visible": visibility},
                        {"title": f"{title}<br>Z = {z_pos:.1f} µm (slice {k+1}/{z_stack.shape[0]})"},
                    ],
                    label=f"{z_pos:.1f}",
                )
                steps.append(step)

            # Create slider
            sliders = [
                dict(
                    active=initial_slice,
                    currentvalue={"prefix": "Z-position (µm): "},
                    pad={"t": 50},
                    steps=steps,
                    len=0.9,
                    x=0.05,
                )
            ]

            # Update layout
            fig.update_layout(
                sliders=sliders,
                title=f"{title}<br>Z = {z_coords[initial_slice]:.1f} µm (slice {initial_slice+1}/{z_stack.shape[0]})",
                xaxis=dict(title="X (µm)", scaleanchor="y", scaleratio=1),
                yaxis=dict(title="Y (µm)", autorange="reversed"),  # Match typical image orientation
                width=700,
                height=600,
            )

            return fig

        except ImportError:
            print("Plotly not available for interactive z-stack visualization")
            return None

    elif backend == "matplotlib":
        try:
            import matplotlib.pyplot as plt
            from matplotlib.widgets import Slider
            import matplotlib.patches as patches

            # Get coordinates and resolution info
            x_coords = metadata.get("x_coords", np.arange(z_stack.shape[2]))
            y_coords = metadata.get("y_coords", np.arange(z_stack.shape[1]))
            z_coords = metadata.get("z_coords", np.arange(z_stack.shape[0]))

            # Create figure and axis
            fig, ax = plt.subplots(figsize=(10, 8))
            plt.subplots_adjust(bottom=0.2)

            # Initial slice
            initial_slice = z_stack.shape[0] // 2

            # Create extent for proper coordinate mapping
            extent = [x_coords[0], x_coords[-1], y_coords[-1], y_coords[0]]

            # Display initial slice
            im = ax.imshow(z_stack[initial_slice, :, :], extent=extent, cmap=colormap, vmin=0, vmax=1, aspect="equal")

            ax.set_xlabel("X (µm)")
            ax.set_ylabel("Y (µm)")
            ax.set_title(f"{title}\nZ = {z_coords[initial_slice]:.1f} µm (slice {initial_slice+1}/{z_stack.shape[0]})")

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Neuron Presence")
            cbar.set_ticks([0, 1])
            cbar.set_ticklabels(["Outside", "Inside"])

            # Add slider
            ax_slider = plt.axes([0.1, 0.05, 0.8, 0.03])
            slider = Slider(ax_slider, "Z-slice", 0, z_stack.shape[0] - 1, valinit=initial_slice, valfmt="%d")

            def update_slice(val):
                slice_idx = int(slider.val)
                im.set_array(z_stack[slice_idx, :, :])
                ax.set_title(f"{title}\nZ = {z_coords[slice_idx]:.1f} µm (slice {slice_idx+1}/{z_stack.shape[0]})")
                fig.canvas.draw()

            slider.on_changed(update_slice)

            return fig

        except ImportError:
            print("Matplotlib not available for z-stack slice visualization")
            return None

    else:
        raise ValueError(f"Backend {backend} not supported for z-stack slice visualization")


def compare_zstack_slices(
    z_stack: np.ndarray,
    metadata: Dict[str, Any],
    slice_indices: List[int],
    titles: Optional[List[str]] = None,
    colormap: str = "viridis",
    backend: str = "plotly",
) -> Optional[object]:
    """
    Compare multiple z-stack slices side by side.

    Args:
        z_stack: 3D binary array (z, y, x)
        metadata: Metadata dict with coordinate information
        slice_indices: List of z-slice indices to compare
        titles: Optional list of titles for each slice
        colormap: Colormap for visualization
        backend: Visualization backend ('plotly', 'matplotlib')

    Returns:
        Figure object with side-by-side slice comparison
    """
    if backend == "plotly":
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            # Get coordinates
            x_coords = metadata.get("x_coords", np.arange(z_stack.shape[2]))
            y_coords = metadata.get("y_coords", np.arange(z_stack.shape[1]))
            z_coords = metadata.get("z_coords", np.arange(z_stack.shape[0]))

            n_slices = len(slice_indices)
            cols = min(3, n_slices)  # Max 3 columns
            rows = (n_slices + cols - 1) // cols  # Calculate needed rows

            # Create subplot titles
            if titles is None:
                subplot_titles = [f"Z = {z_coords[i]:.1f} µm" for i in slice_indices]
            else:
                subplot_titles = titles

            fig = make_subplots(
                rows=rows, cols=cols, subplot_titles=subplot_titles, vertical_spacing=0.1, horizontal_spacing=0.05
            )

            # Add each slice as a heatmap
            for idx, slice_idx in enumerate(slice_indices):
                row = idx // cols + 1
                col = idx % cols + 1

                slice_data = z_stack[slice_idx, :, :]

                fig.add_trace(
                    go.Heatmap(
                        z=slice_data,
                        x=x_coords,
                        y=y_coords,
                        colorscale=colormap,
                        showscale=(idx == 0),  # Only show colorbar for first subplot
                        hovertemplate="<b>X:</b> %{x:.1f} µm<br>"
                        + "<b>Y:</b> %{y:.1f} µm<br>"
                        + "<b>Value:</b> %{z}<br>"
                        + "<extra></extra>",
                        colorbar=(
                            dict(title="Neuron<br>Presence", tickvals=[0, 1], ticktext=["Outside", "Inside"], len=0.8)
                            if idx == 0
                            else None
                        ),
                    ),
                    row=row,
                    col=col,
                )

            # Update layout
            fig.update_layout(
                title="Z-stack Slice Comparison",
                height=300 * rows,
                width=250 * cols + 100,
            )

            # Update axes to maintain aspect ratio
            for idx in range(len(slice_indices)):
                row = idx // cols + 1
                col = idx % cols + 1
                fig.update_xaxes(title_text="X (µm)", row=row, col=col, scaleanchor=f"y{row}{col}", scaleratio=1)
                fig.update_yaxes(title_text="Y (µm)", row=row, col=col, autorange="reversed")

            return fig

        except ImportError:
            print("Plotly not available for z-stack slice comparison")
            return None

    elif backend == "matplotlib":
        try:
            import matplotlib.pyplot as plt

            # Get coordinates
            x_coords = metadata.get("x_coords", np.arange(z_stack.shape[2]))
            y_coords = metadata.get("y_coords", np.arange(z_stack.shape[1]))
            z_coords = metadata.get("z_coords", np.arange(z_stack.shape[0]))

            n_slices = len(slice_indices)
            cols = min(3, n_slices)  # Max 3 columns
            rows = (n_slices + cols - 1) // cols

            fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
            if n_slices == 1:
                axes = [axes]
            elif rows == 1:
                axes = [axes] if cols == 1 else axes
            else:
                axes = axes.flatten()

            # Create extent for proper coordinate mapping
            extent = [x_coords[0], x_coords[-1], y_coords[-1], y_coords[0]]

            for idx, slice_idx in enumerate(slice_indices):
                if idx < len(axes):
                    ax = axes[idx]
                    slice_data = z_stack[slice_idx, :, :]

                    im = ax.imshow(slice_data, extent=extent, cmap=colormap, vmin=0, vmax=1, aspect="equal")

                    title = titles[idx] if titles else f"Z = {z_coords[slice_idx]:.1f} µm"
                    ax.set_title(title)
                    ax.set_xlabel("X (µm)")
                    ax.set_ylabel("Y (µm)")

                    # Add colorbar to first subplot
                    if idx == 0:
                        cbar = plt.colorbar(im, ax=ax)
                        cbar.set_label("Neuron Presence")
                        cbar.set_ticks([0, 1])
                        cbar.set_ticklabels(["Outside", "Inside"])

            # Hide unused subplots
            for idx in range(len(slice_indices), len(axes)):
                axes[idx].set_visible(False)

            plt.tight_layout()
            return fig

        except ImportError:
            print("Matplotlib not available for z-stack slice comparison")
            return None

    else:
        raise ValueError(f"Backend {backend} not supported for z-stack slice comparison")


def rotate_zstack(
    z_stack: np.ndarray,
    metadata: Dict[str, Any],
    angle_x: float = 0.0,
    angle_y: float = 0.0,
    angle_z: float = 0.0,
    order: int = 1,
    mode: str = "constant",
    cval: float = 0.0,
    prefilter: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Rotate a z-stack by given angles about the x, y, and/or z axes.

    Args:
        z_stack: 3D numpy array (z, y, x) with 1 inside neuron, 0 outside
        metadata: Dictionary with spatial information including resolutions
        angle_x: Rotation angle about x-axis in degrees
        angle_y: Rotation angle about y-axis in degrees
        angle_z: Rotation angle about z-axis in degrees
        order: Interpolation order (0=nearest, 1=linear, 2=cubic, etc.)
        mode: How to handle points outside boundary ('constant', 'nearest', 'reflect', 'wrap')
        cval: Constant value used for points outside boundary when mode='constant'
        prefilter: Whether to prefilter the input array

    Returns:
        Tuple of (rotated_z_stack, updated_metadata)
    """
    try:
        from scipy.spatial.transform import Rotation
        from scipy.ndimage import affine_transform
    except ImportError:
        raise ImportError("scipy is required for z-stack rotation. Install with: pip install scipy")

    # Convert angles to radians
    angles_rad = np.deg2rad([angle_x, angle_y, angle_z])

    # Create rotation object using intrinsic rotations (xyz order)
    rotation = Rotation.from_euler("xyz", angles_rad, degrees=False)
    rotation_matrix = rotation.as_matrix()

    # Get current resolutions
    z_res = metadata.get("z_resolution", 1.0)
    xy_res = metadata.get("xy_resolution", 1.0)

    # Create scaling matrix to handle anisotropic voxels
    # Convert to isotropic coordinates for rotation, then back
    min_res = min(z_res, xy_res)
    scale_z = z_res / min_res
    scale_xy = xy_res / min_res

    # Scaling matrices
    scale_to_iso = np.diag([1 / scale_z, 1 / scale_xy, 1 / scale_xy])
    scale_from_iso = np.diag([scale_z, scale_xy, scale_xy])

    # Combined transformation matrix
    transform_matrix = scale_from_iso @ rotation_matrix @ scale_to_iso

    # Get center of z-stack for rotation
    center = np.array(z_stack.shape) / 2.0

    # Calculate offset to rotate around center
    offset = center - transform_matrix @ center

    # Apply rotation using affine transformation
    print(f"🔄 Rotating z-stack by ({angle_x:.1f}°, {angle_y:.1f}°, {angle_z:.1f}°) about (x,y,z) axes...")

    rotated_z_stack = affine_transform(
        z_stack.astype(np.float32),
        matrix=transform_matrix,
        offset=offset,
        output_shape=z_stack.shape,
        order=order,
        mode=mode,
        cval=cval,
        prefilter=prefilter,
    )

    # Convert back to binary and uint8
    if z_stack.dtype == np.uint8:
        # Threshold to maintain binary nature
        rotated_z_stack = (rotated_z_stack > 0.5).astype(np.uint8)

    # Update metadata
    updated_metadata = metadata.copy()

    # Update coordinate arrays if they exist
    if all(key in metadata for key in ["x_coords", "y_coords", "z_coords"]):
        # For now, keep the same coordinate arrays since we're rotating in place
        # A more sophisticated approach would recalculate the bounds
        updated_metadata["rotation_applied"] = {
            "angle_x_degrees": angle_x,
            "angle_y_degrees": angle_y,
            "angle_z_degrees": angle_z,
            "rotation_matrix": rotation_matrix.tolist(),
            "rotation_order": "xyz",
        }

    # Update bounds if they exist
    if "bounds" in metadata:
        # Keep original bounds for now - in practice you might want to recalculate
        # based on the rotated geometry
        updated_metadata["original_bounds"] = metadata["bounds"]

    print(f"✓ Z-stack rotation complete!")
    print(f"✓ Output shape: {rotated_z_stack.shape}")
    print(f"✓ Voxel count: {np.sum(rotated_z_stack):,} inside neuron")

    return rotated_z_stack, updated_metadata


def rotate_zstack_arbitrary_axis(
    z_stack: np.ndarray,
    metadata: Dict[str, Any],
    axis: np.ndarray,
    angle: float,
    order: int = 1,
    mode: str = "constant",
    cval: float = 0.0,
    prefilter: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Rotate a z-stack by a given angle about an arbitrary axis.

    Args:
        z_stack: 3D numpy array (z, y, x) with 1 inside neuron, 0 outside
        metadata: Dictionary with spatial information
        axis: 3D vector defining the rotation axis [x, y, z]
        angle: Rotation angle in degrees
        order: Interpolation order (0=nearest, 1=linear, 2=cubic, etc.)
        mode: How to handle points outside boundary
        cval: Constant value for points outside boundary when mode='constant'
        prefilter: Whether to prefilter the input array

    Returns:
        Tuple of (rotated_z_stack, updated_metadata)
    """
    try:
        from scipy.spatial.transform import Rotation
        from scipy.ndimage import affine_transform
    except ImportError:
        raise ImportError("scipy is required for z-stack rotation. Install with: pip install scipy")

    # Normalize the axis vector
    axis = np.array(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)

    # Create rotation object using axis-angle representation
    rotation = Rotation.from_rotvec(np.deg2rad(angle) * axis)
    rotation_matrix = rotation.as_matrix()

    # Get current resolutions
    z_res = metadata.get("z_resolution", 1.0)
    xy_res = metadata.get("xy_resolution", 1.0)

    # Create scaling matrix to handle anisotropic voxels
    min_res = min(z_res, xy_res)
    scale_z = z_res / min_res
    scale_xy = xy_res / min_res

    # Scaling matrices
    scale_to_iso = np.diag([1 / scale_z, 1 / scale_xy, 1 / scale_xy])
    scale_from_iso = np.diag([scale_z, scale_xy, scale_xy])

    # Combined transformation matrix
    transform_matrix = scale_from_iso @ rotation_matrix @ scale_to_iso

    # Get center of z-stack for rotation
    center = np.array(z_stack.shape) / 2.0

    # Calculate offset to rotate around center
    offset = center - transform_matrix @ center

    # Apply rotation
    print(f"🔄 Rotating z-stack by {angle:.1f}° about axis [{axis[0]:.2f}, {axis[1]:.2f}, {axis[2]:.2f}]...")

    rotated_z_stack = affine_transform(
        z_stack.astype(np.float32),
        matrix=transform_matrix,
        offset=offset,
        output_shape=z_stack.shape,
        order=order,
        mode=mode,
        cval=cval,
        prefilter=prefilter,
    )

    # Convert back to binary and uint8 if original was uint8
    if z_stack.dtype == np.uint8:
        rotated_z_stack = (rotated_z_stack > 0.5).astype(np.uint8)

    # Update metadata
    updated_metadata = metadata.copy()
    updated_metadata["rotation_applied"] = {
        "axis": axis.tolist(),
        "angle_degrees": angle,
        "rotation_matrix": rotation_matrix.tolist(),
        "rotation_type": "arbitrary_axis",
    }

    print(f"✓ Z-stack rotation complete!")
    print(f"✓ Output shape: {rotated_z_stack.shape}")
    print(f"✓ Voxel count: {np.sum(rotated_z_stack):,} inside neuron")

    return rotated_z_stack, updated_metadata

import numpy as np
import pytest

from discos import (
    MeshManager,
    PolylinesSkeleton,
    TraceOptions,
    build_traced_skeleton_graph,
    data_path,
)


TORUS_MINOR_RADIUS = 5.0  # as defined in scripts/make_demo_meshes.py


def _estimate_torus_axis_and_center(mesh) -> tuple[int, np.ndarray]:
    V = np.asarray(mesh.vertices, dtype=float)
    mins = V.min(axis=0)
    maxs = V.max(axis=0)
    ranges = maxs - mins
    axis_idx = int(np.argmin(ranges))  # torus axis is the smallest extent
    center = (mins + maxs) * 0.5
    return axis_idx, center


def _build_torus_centerline_polyline(mesh) -> np.ndarray:
    axis_idx, center = _estimate_torus_axis_and_center(mesh)
    # Project vertices to plane orthogonal to torus axis to estimate major radius
    V = np.asarray(mesh.vertices, dtype=float)
    if axis_idx == 0:  # x-axis
        R_samples = np.sqrt((V[:, 1] - center[1]) ** 2 + (V[:, 2] - center[2]) ** 2)
        major_R = float(np.median(R_samples))
        angles = np.linspace(0.0, 2.0 * np.pi, 361)
        poly = np.column_stack(
            [
                np.full_like(angles, center[0]),
                center[1] + major_R * np.cos(angles),
                center[2] + major_R * np.sin(angles),
            ]
        )
    elif axis_idx == 1:  # y-axis
        R_samples = np.sqrt((V[:, 0] - center[0]) ** 2 + (V[:, 2] - center[2]) ** 2)
        major_R = float(np.median(R_samples))
        angles = np.linspace(0.0, 2.0 * np.pi, 361)
        poly = np.column_stack(
            [
                center[0] + major_R * np.cos(angles),
                np.full_like(angles, center[1]),
                center[2] + major_R * np.sin(angles),
            ]
        )
    else:  # z-axis
        R_samples = np.sqrt((V[:, 0] - center[0]) ** 2 + (V[:, 1] - center[1]) ** 2)
        major_R = float(np.median(R_samples))
        angles = np.linspace(0.0, 2.0 * np.pi, 361)
        poly = np.column_stack(
            [
                center[0] + major_R * np.cos(angles),
                center[1] + major_R * np.sin(angles),
                np.full_like(angles, center[2]),
            ]
        )
    return poly.astype(float)


def _load_torus_and_polylines():
    mm = MeshManager()
    mesh_path = str(data_path("mesh/demo/torus.obj"))
    mesh = mm.load_mesh(mesh_path)
    # Build a synthetic centerline polyline around the torus' major circle
    centerline = _build_torus_centerline_polyline(mesh)
    pls = PolylinesSkeleton([centerline])
    return mm, pls


@pytest.mark.parametrize(
    "mode",
    [
        "equivalent_area",
        "equivalent_perimeter",
        "section_median",
        "section_circle_fit",
        "nearest_surface",
    ],
)
def test_trace_radius_modes_on_torus(mode):
    mm, pls = _load_torus_and_polylines()
    opts = TraceOptions(
        spacing=1.0,
        radius_mode=mode,
        snap_polylines_to_mesh=False,
        section_probe_eps=1e-2,  # more robust probing window
        section_probe_tries=10,
    )

    skel = build_traced_skeleton_graph(mm, pls, options=opts)

    # Basic sanity
    assert len(skel.junctions) > 0, "No junctions created by tracing"

    radii = np.array([j.radius for j in skel.junctions.values()], dtype=float)
    radii = radii[np.isfinite(radii) & (radii > 0)]
    assert radii.size > 0, "No positive radii produced"

    r_med = float(np.median(radii))

    # For a torus with minor radius 5.0, the local tube radius should be ~5
    # Use a tolerant bound to account for discretization and slice approximations
    assert abs(r_med - TORUS_MINOR_RADIUS) <= 1.0, (
        f"radius_mode={mode}: median radius {r_med:.3f} deviates too much from expected {TORUS_MINOR_RADIUS}"
    )

    # Additionally, require the bulk of radii lie near expected (robustness check)
    frac_within = float(np.mean(np.abs(radii - TORUS_MINOR_RADIUS) <= 1.5))
    assert frac_within >= 0.7, (
        f"radius_mode={mode}: only {frac_within*100:.1f}% radii within ±1.5 of {TORUS_MINOR_RADIUS}"
    )

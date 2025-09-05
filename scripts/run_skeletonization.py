from pathlib import Path

import networkx as nx

from discos.mesh import MeshManager
from discos.skeleton import skeletonize


def main():
    """
    Skeletonize all OBJ meshes under data/mesh/ recursively.

    For each mesh, run skeletonization with n_slices in [8, 16], print basic
    graph metrics, and export SWC files to data/swc/.

    Usage (no CLI):
        uv run scripts/run_skeletonization.py
    """
    mesh_root = Path("data/mesh")
    obj_paths = sorted(mesh_root.rglob("*.obj")) if mesh_root.exists() else []
    if not obj_paths:
        print(f"No OBJ meshes found under {mesh_root.resolve()}")
        return

    n_slices_list = [8, 16]
    out_dir = Path("data/swc")
    out_dir.mkdir(parents=True, exist_ok=True)

    for mesh_path in obj_paths:
        print(f"\n=== Mesh: {mesh_path} ===")

        # Load mesh via MeshManager
        mm = MeshManager(verbose=False)
        mm.load_mesh(str(mesh_path))
        mesh = mm.to_trimesh()

        for n in n_slices_list:
            print("\n--- Skeletonization: n_slices =", n, "---")
            skel = skeletonize(mesh, n_slices=n, validate_volume=False, verbose=False)
            G = skel.to_networkx()

            num_nodes = G.number_of_nodes()
            num_edges = G.number_of_edges()
            num_components = nx.number_connected_components(G)

            zero_conn_segments = [
                s
                for s in skel.segments
                if len(s.lower_junction_ids) == 0 or len(s.upper_junction_ids) == 0
            ]

            print(f"Nodes: {num_nodes}")
            print(f"Edges: {num_edges}")
            print(f"Connected components: {num_components}")
            print(
                f"Segments: {len(skel.segments)} (segments with zero connectivity: {len(zero_conn_segments)})"
            )

            if zero_conn_segments:
                # Summarize by slice index
                by_slice = {}
                for s in zero_conn_segments:
                    by_slice.setdefault(s.slice_index, 0)
                    by_slice[s.slice_index] += 1
                summary = ", ".join(
                    f"slice {k}: {v}" for k, v in sorted(by_slice.items())
                )
                print(f"Zero-connectivity segments by slice: {summary}")

            # Always export SWC outputs
            stem = mesh_path.stem
            out_path = out_dir / f"{stem}_n{n}.swc"
            skel.to_swc(str(out_path))
            print(f"SWC written: {out_path}")


if __name__ == "__main__":
    main()

from discos.demo import create_torus_mesh
from discos.skeleton import skeletonize
import networkx as nx
import numpy as np
import os
import json

if __name__ == "__main__":
    torus = create_torus_mesh(
        major_radius=2.0,
        minor_radius=0.6,
        major_segments=128,
        minor_segments=64,
        axis="z",
    )
    skel = skeletonize(
        torus,
        n_slices=64,
        validate_volume=True,
        verbose=True,
        enforce_connected=False,
    )
    G = skel.to_networkx()
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    comps = nx.number_connected_components(G)
    cycles = len(nx.cycle_basis(G))
    degs = [d for _, d in G.degree()]
    avg_deg = float(np.mean(degs)) if degs else 0.0
    print(f"nodes {n_nodes} edges {n_edges} components {comps} cycles {cycles} avg_deg {avg_deg:.3f}")

    # Nodes per slice
    per_slice = {}
    for n, a in G.nodes(data=True):
        si = a.get("slice_index")
        per_slice[si] = per_slice.get(si, 0) + 1
    print("nodes_per_slice:", sorted(per_slice.items()))

    # Edges per band
    per_band = {}
    for u, v, e in G.edges(data=True):
        bi = e.get("slice_index")
        per_band[bi] = per_band.get(bi, 0) + 1
    print("edges_per_band:", sorted(per_band.items()) )

    # Write a JSON report to data/diagnostics for reliable inspection
    out_dir = os.path.join("data", "diagnostics")
    os.makedirs(out_dir, exist_ok=True)
    report = {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "components": int(comps),
        "cycles": int(cycles),
        "avg_degree": avg_deg,
        "nodes_per_slice": {int(k) if k is not None else -1: int(v) for k, v in per_slice.items()},
        "edges_per_band": {int(k) if k is not None else -1: int(v) for k, v in per_band.items()},
    }
    out_path = os.path.join(out_dir, "debug_torus_64.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {out_path}")

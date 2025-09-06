"""
Batch diagnostics for DISCOS skeletonization across varying n_slices on two meshes.

- Inputs: data/mesh/processed/TS1_wrapped.obj, TS2_repaired.obj
- For each mesh, test a range of n_slices values and record:
  * success/failure
  * error message (if any)
  * number of nodes, edges
  * number of connected components
  * notes (e.g., power-of-two warning)
- Outputs: JSON report(s) under data/diagnostics/

Run with:
    uv run scripts/diagnose_skeletonization.py

This script is intentionally non-interactive and has no CLI per project rules.
"""
from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import trimesh

# Import discos after adjusting path if needed
try:
    from discos.skeleton import skeletonize
except Exception:
    # If running as a plain script without package install
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from discos.skeleton import skeletonize  # type: ignore

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MESH_DIR = os.path.join(DATA_DIR, "mesh", "processed")
DIAG_DIR = os.path.join(DATA_DIR, "diagnostics")
os.makedirs(DIAG_DIR, exist_ok=True)

MESHES = [
    ("TS1_wrapped", os.path.join(MESH_DIR, "TS1_wrapped.obj")),
    ("TS2_repaired", os.path.join(MESH_DIR, "TS2_repaired.obj")),
]

# Choose a spread of slice counts to probe behavior, emphasizing small counts and powers of two
N_SLICES_LIST = [
    3, 4, 5, 6, 7, 8,
    9, 10, 12, 14, 16,
    18, 20, 24, 28, 32,
]


@dataclass
class Result:
    n_slices: int
    ok: bool
    msg: str
    nodes: int
    edges: int
    components: int
    has_power2: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def analyze_mesh(mesh_name: str, mesh_path: str) -> Dict[str, Any]:
    if not os.path.exists(mesh_path):
        return {
            "mesh": mesh_name,
            "mesh_path": mesh_path,
            "error": f"mesh not found: {mesh_path}",
        }

    try:
        mesh = trimesh.load(mesh_path, force='mesh')
        if mesh is None or not hasattr(mesh, 'vertices'):
            raise ValueError("failed to load trimesh")
    except Exception as e:
        return {
            "mesh": mesh_name,
            "mesh_path": mesh_path,
            "error": f"load error: {e}",
        }

    results: List[Result] = []

    for n_slices in N_SLICES_LIST:
        has_power2 = (n_slices & (n_slices - 1) == 0)
        try:
            skel = skeletonize(
                mesh,
                n_slices,
                validate_volume=True,
                volume_tol=0.08,  # slightly looser for diagnostics
                verbosity=1,
                enforce_connected=False,  # we want to observe components without failing
                connect_isolated_terminals=True,
            )
            G = skel.to_networkx()
            nodes = int(G.number_of_nodes())
            edges = int(G.number_of_edges())
            try:
                components = int(len(list(__import__('networkx').connected_components(G))))
            except Exception:
                components = 1 if nodes <= 1 else 2
            results.append(
                Result(
                    n_slices=n_slices,
                    ok=True,
                    msg="",
                    nodes=nodes,
                    edges=edges,
                    components=components,
                    has_power2=bool(has_power2),
                )
            )
        except Exception as e:
            results.append(
                Result(
                    n_slices=n_slices,
                    ok=False,
                    msg=str(e),
                    nodes=0,
                    edges=0,
                    components=0,
                    has_power2=bool(has_power2),
                )
            )

    # Persist JSON report
    out_path = os.path.join(DIAG_DIR, f"diagnostics_{mesh_name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mesh": mesh_name,
                "mesh_path": mesh_path,
                "n": int(len(results)),
                "results": [r.to_dict() for r in results],
            },
            f,
            indent=2,
        )

    # Print a concise summary to stdout
    print(f"\n=== {mesh_name} ===")
    ok_counts = sum(1 for r in results if r.ok)
    print(f"tested n_slices values: {len(results)}; successes: {ok_counts}; failures: {len(results)-ok_counts}")
    # Show a compact line for each
    for r in results:
        status = "OK " if r.ok else "ERR"
        p2 = "*" if r.has_power2 else " "
        tail = f"nodes={r.nodes} edges={r.edges} comps={r.components}" if r.ok else r.msg
        print(f"n={r.n_slices:>2}{p2}  {status}  {tail}")

    return {
        "mesh": mesh_name,
        "path": mesh_path,
        "json": out_path,
        "ok": ok_counts,
        "total": len(results),
    }


if __name__ == "__main__":
    summaries: List[Dict[str, Any]] = []
    for name, path in MESHES:
        summaries.append(analyze_mesh(name, path))

    print("\n=== Summary ===")
    for s in summaries:
        print(f"{s['mesh']}: {s['ok']}/{s['total']} success -> {s['json']}")

import numpy as np
from pathlib import Path

from discos.skeleton import SkeletonGraph, Junction, Segment


def _build_diamond_cycle_graph() -> SkeletonGraph:
    """Construct a small graph with a diamond-shaped cycle:
    j0 (z=0) connects to j1, j2 (z=1), which both connect to j3 (z=2).
    Edges: (j0-j1), (j0-j2), (j1-j3), (j2-j3) -> one cycle.
    """
    skel = SkeletonGraph()

    # Create four junctions at z=0,1,1,2
    j0 = Junction(id=0, z=0.0, center=np.array([0.0, 0.0, 0.0]), radius=1.0,
                  area=float(np.pi * 1.0**2), slice_index=0, cross_section_index=0)
    j1 = Junction(id=1, z=1.0, center=np.array([ -1.0, 0.0, 1.0]), radius=1.0,
                  area=float(np.pi * 1.0**2), slice_index=1, cross_section_index=0)
    j2 = Junction(id=2, z=1.0, center=np.array([ +1.0, 0.0, 1.0]), radius=1.0,
                  area=float(np.pi * 1.0**2), slice_index=1, cross_section_index=1)
    j3 = Junction(id=3, z=2.0, center=np.array([0.0, 0.0, 2.0]), radius=1.0,
                  area=float(np.pi * 1.0**2), slice_index=2, cross_section_index=0)

    for j in (j0, j1, j2, j3):
        skel.add_junction(j)

    # Two segments: slice 0->1 connects j0 to both j1 and j2; slice 1->2 connects j1 and j2 to j3
    seg0 = Segment(
        id=0,
        slice_index=0,
        z_lower=0.0,
        z_upper=1.0,
        volume=1.0,
        surface_area=1.0,
        lower_junction_ids=[j0.id],
        upper_junction_ids=[j1.id, j2.id],
    )
    seg1 = Segment(
        id=1,
        slice_index=1,
        z_lower=1.0,
        z_upper=2.0,
        volume=1.0,
        surface_area=1.0,
        lower_junction_ids=[j1.id, j2.id],
        upper_junction_ids=[j3.id],
    )
    skel.add_segment_edges(seg0)
    skel.add_segment_edges(seg1)

    return skel


def test_to_swc_breaks_cycles_and_annotates(tmp_path: Path = None):
    skel = _build_diamond_cycle_graph()

    # Choose output directory
    out_dir = Path("tests") / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    swc_path = out_dir / "diamond_cycle_default.swc"

    # Export with defaults (type_index=5)
    skel.to_swc(str(swc_path))

    # Read back
    text = swc_path.read_text(encoding="utf-8")

    # Header annotations for cycle removal should be present
    assert "Removed" in text
    assert "# cycle_edge_removed" in text

    # Parse data rows
    rows = [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
    assert len(rows) == 4  # 4 nodes expected

    # Each row: n T x y z R p
    parsed = []
    for line in rows:
        parts = line.split()
        assert len(parts) == 7
        n, T, x, y, z, R, p = parts
        n = int(n)
        T = int(T)
        p = int(p)
        parsed.append((n, T, p))

    # Default type_index should be 5 for all
    assert all(T == 5 for _, T, _ in parsed)

    # Parents must be -1 (root) or index of a previously written node
    seen = set()
    parent_count = 0
    for n, T, p in parsed:
        if p == -1:
            pass
        else:
            assert p in seen, "Parent must appear before child in SWC file"
            parent_count += 1
        seen.add(n)

    # A tree with 4 nodes must have exactly 3 parent links
    assert parent_count == 3


def test_to_swc_type_index_override(tmp_path: Path = None):
    skel = _build_diamond_cycle_graph()

    out_dir = Path("tests") / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    swc_path = out_dir / "diamond_cycle_type3.swc"

    # Export with a custom type index
    skel.to_swc(str(swc_path), type_index=3)

    text = swc_path.read_text(encoding="utf-8")
    rows = [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
    assert len(rows) == 4

    for line in rows:
        parts = line.split()
        assert len(parts) == 7
        T = int(parts[1])
        assert T == 3

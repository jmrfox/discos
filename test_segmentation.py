"""
Test script for mesh segmentation functionality.
"""

import sys
import os

sys.path.append("/home/jordan/github/gencomo")

import numpy as np
from gencomo import create_torus_mesh, MeshSegmenter


def test_segmentation():
    """Test the segmentation on a torus mesh."""
    print("Testing mesh segmentation...")

    # Create a test torus
    torus = create_torus_mesh()
    print(f"Created torus with {len(torus.vertices)} vertices, {len(torus.faces)} faces")
    print(f"Torus bounds: {torus.bounds}")
    print(f"Torus volume: {torus.volume:.4f}")

    # Create segmenter
    segmenter = MeshSegmenter()

    # Segment the mesh
    slice_width = 1.0  # 1 unit slice width (10% of height)
    segments = segmenter.segment_mesh(torus, slice_width=slice_width)

    print(f"\nSegmentation results:")
    print(f"Number of segments: {len(segments)}")

    # Print segment details
    for seg in segments[:5]:  # Show first 5 segments
        print(
            f"  {seg.id}: vol={seg.volume:.4f}, ext_area={seg.exterior_surface_area:.4f}, int_area={seg.interior_surface_area:.4f}"
        )

    # Show statistics
    stats = segmenter.compute_segmentation_statistics()
    print(f"\nStatistics:")
    print(f"  Total volume: {stats['volume_stats']['total']:.4f}")
    print(f"  Connected components: {stats['connected_components']}")
    print(f"  Segments per slice: {stats['segments_per_slice']['mean']:.2f} ± {stats['segments_per_slice']['std']:.2f}")

    # Test connectivity
    if segments:
        test_seg = segments[0]
        connected = segmenter.get_connected_segments(test_seg.id)
        print(f"\nSegment {test_seg.id} connected to: {connected}")

    print("\nSegmentation test completed!")
    return segmenter


if __name__ == "__main__":
    segmenter = test_segmentation()

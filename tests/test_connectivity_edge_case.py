#!/usr/bin/env python3
"""
Test script to verify connectivity validation catches disconnected graphs.
This simulates what would happen if there was a bug in the segmentation algorithm.
"""

import networkx as nx
from gencomo.mesh import MeshSegmenter


def test_disconnected_graph_detection():
    """Test that our validation function properly detects disconnected graphs."""
    print("🧪 Testing Disconnected Graph Detection")
    print("=" * 50)

    # Create a dummy segmenter with a disconnected graph
    segmenter = MeshSegmenter()

    # Create a disconnected graph: two separate components
    segmenter.connectivity_graph = nx.Graph()

    # Add first component (nodes 0, 1, 2 connected)
    segmenter.connectivity_graph.add_edges_from([(0, 1), (1, 2)])

    # Add second component (nodes 3, 4 connected, separate from first)
    segmenter.connectivity_graph.add_edges_from([(3, 4)])

    print(f"Created test graph:")
    print(f"  Nodes: {list(segmenter.connectivity_graph.nodes)}")
    print(f"  Edges: {list(segmenter.connectivity_graph.edges)}")

    # Import validation function
    from tests.test_segmentation import validate_connectivity_graph

    # Test our validation - this should detect the disconnection
    is_valid = validate_connectivity_graph(segmenter, "Disconnected Test")

    print(f"\n🎯 Test Result:")
    if not is_valid:
        print(f"  ✅ SUCCESS: Validation correctly detected disconnected graph")
        return True
    else:
        print(f"  ❌ FAILURE: Validation failed to detect disconnected graph")
        return False


if __name__ == "__main__":
    test_disconnected_graph_detection()

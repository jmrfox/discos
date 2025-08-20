"""
Test script for SWC export functionality.

This test script validates the SWC export functionality by:
1. Testing SWCData class functionality
2. Testing SegmentGraph.export_to_swc() method
3. Validating Arbor compatibility requirements
4. Testing cycle-breaking functionality
5. Testing soma identification and classification
6. Validating file output format and annotations
"""

import sys
import os
import tempfile
import json
from pathlib import Path

# Add the parent directory to path for importing discos
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import trimesh
import networkx as nx

from discos.demos import create_cylinder_mesh, create_torus_mesh
from discos.segmentation import MeshSegmenter, SegmentGraph, SWCData


class TestSWCData:
    """Test SWCData class functionality."""
    
    @pytest.fixture
    def sample_swc_data(self):
        """Create sample SWC data for testing."""
        entries = [
            "1 5 0.0 0.0 0.0 2.0 -1",
            "2 5 1.0 0.0 0.5 1.8 1",
            "3 5 2.0 0.0 1.0 1.5 2",
            "4 5 3.0 0.0 1.5 1.2 3"
        ]
        
        metadata = {
            'total_segments': 4,
            'original_graph_nodes': 4,
            'original_graph_edges': 4,
            'tree_edges': 3,
            'non_tree_edges': 1,
            'cycle_breaking_strategy': 'minimum_spanning_tree'
        }
        
        non_tree_edges = [
            {
                'sample_id_1': 2,
                'sample_id_2': 4,
                'original_node_1': 'seg_0_1',
                'original_node_2': 'seg_1_1',
                'centroid_1': [1.0, 0.0, 0.5],
                'centroid_2': [3.0, 0.0, 1.5]
            }
        ]
        
        return SWCData(
            entries=entries,
            metadata=metadata,
            non_tree_edges=non_tree_edges,
            root_segment='seg_0_0',
            scale_factor=1.0
        )
    
    def test_swc_data_initialization(self, sample_swc_data):
        """Test SWCData initialization."""
        assert len(sample_swc_data.entries) == 4
        assert sample_swc_data.metadata['total_segments'] == 4
        assert len(sample_swc_data.non_tree_edges) == 1
        # No soma_segments field in refactored SWCData
        assert sample_swc_data.root_segment == 'seg_0_0'
        assert sample_swc_data.scale_factor == 1.0
    
    def test_swc_data_summary(self, sample_swc_data):
        """Test SWC data summary generation."""
        summary = sample_swc_data.get_summary()
        
        assert "Total segments: 4" in summary
        # No soma segments line in refactored summary
        assert "Root segment: seg_0_0" in summary
        assert "Tree edges: 3" in summary
        assert "Non-tree edges: 1" in summary
        assert "cycle-breaking connections need post-processing" in summary
    
    def test_swc_data_file_writing(self, sample_swc_data):
        """Test SWC data file writing functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            swc_file = os.path.join(temp_dir, "test.swc")
            json_file = os.path.join(temp_dir, "test_connections.json")
            
            # Write to file
            sample_swc_data.write_to_file(swc_file)
            
            # Check SWC file was created
            assert os.path.exists(swc_file)
            
            # Check JSON file was created (since we have non-tree edges)
            assert os.path.exists(json_file)
            
            # Validate SWC file content
            with open(swc_file, 'r') as f:
                content = f.read()
                
            # Check header information
            assert "# SWC file generated from SegmentGraph for Arbor simulator" in content
            assert "# TypeID: 5=segment (all segments use type 5)" in content
            assert "# Total segments: 4" in content
            # No soma segments line in refactored header
            
            # Check cycle-breaking annotations
            assert "# CYCLE-BREAKING ANNOTATIONS:" in content
            assert "# CONNECT 2 <-> 4 (seg_0_1 <-> seg_1_1)" in content
            
            # Check data entries
            assert "1 5 0.0 0.0 0.0 2.0 -1" in content
            assert "2 5 1.0 0.0 0.5 1.8 1" in content
            
            # Validate JSON file content
            with open(json_file, 'r') as f:
                json_data = json.load(f)
                
            assert json_data['metadata']['total_segments'] == 4
            assert len(json_data['non_tree_connections']) == 1
            assert json_data['non_tree_connections'][0]['sample_id_1'] == 2
            assert json_data['non_tree_connections'][0]['sample_id_2'] == 4


class TestSegmentGraphSWCExport:
    """Test SegmentGraph SWC export functionality."""
    
    @pytest.fixture
    def simple_segment_graph(self):
        """Create a simple segment graph for testing."""
        graph = SegmentGraph()
        
        # Add nodes with properties
        graph.add_node('seg_0_0', 
                      centroid=np.array([0.0, 0.0, 0.0]),
                      volume=10.0,
                      slice_index=0)
        graph.add_node('seg_0_1', 
                      centroid=np.array([1.0, 0.0, 0.5]),
                      volume=8.0,
                      slice_index=0)
        graph.add_node('seg_1_0', 
                      centroid=np.array([2.0, 0.0, 1.0]),
                      volume=6.0,
                      slice_index=1)
        graph.add_node('seg_1_1', 
                      centroid=np.array([3.0, 0.0, 1.5]),
                      volume=4.0,
                      slice_index=1)
        
        # Add edges (tree structure)
        graph.add_edge('seg_0_0', 'seg_0_1')
        graph.add_edge('seg_0_1', 'seg_1_0')
        graph.add_edge('seg_1_0', 'seg_1_1')
        
        return graph
    
    @pytest.fixture
    def cyclic_segment_graph(self):
        """Create a segment graph with cycles for testing cycle-breaking."""
        graph = SegmentGraph()
        
        # Add nodes
        for i in range(4):
            graph.add_node(f'seg_{i}', 
                          centroid=np.array([i, 0.0, i * 0.5]),
                          volume=5.0 + i,
                          slice_index=i // 2)
         
        # Add edges to create a cycle
        graph.add_edge('seg_0', 'seg_1')
        graph.add_edge('seg_1', 'seg_2')
        graph.add_edge('seg_2', 'seg_3')
        graph.add_edge('seg_3', 'seg_0')  # Creates cycle
        graph.add_edge('seg_1', 'seg_3')  # Additional connection
        
        return graph
    
    def test_simple_swc_export(self, simple_segment_graph):
        """Test SWC export with simple tree structure."""
        swc_data = simple_segment_graph.export_to_swc(scale_factor=1.0)
        
        # Check basic properties
        assert len(swc_data.entries) == 4
        assert swc_data.scale_factor == 1.0
        assert len(swc_data.non_tree_edges) == 0  # No cycles in simple graph
        assert swc_data.metadata['total_segments'] == 4
        assert swc_data.metadata['tree_edges'] == 3
        
        # Check entries are valid SWC format
        for entry in swc_data.entries:
            parts = entry.split()
            assert len(parts) == 7  # SampleID TypeID x y z radius ParentID
            
            sample_id = int(parts[0])
            type_id = int(parts[1])
            x, y, z, radius = map(float, parts[2:6])
            parent_id = int(parts[6])
            
            # Validate ranges
            assert sample_id >= 1
            assert type_id == 5  # All segments use type 5
            assert radius > 0
            assert parent_id >= -1
    
    def test_cyclic_swc_export(self, cyclic_segment_graph):
        """Test SWC export with cycle-breaking."""
        swc_data = cyclic_segment_graph.export_to_swc(
            cycle_breaking_strategy='minimum_spanning_tree'
        )
        
        # Check that cycles were broken
        assert len(swc_data.entries) == 4
        assert len(swc_data.non_tree_edges) > 0  # Should have removed edges
        assert swc_data.metadata['tree_edges'] == 3  # Tree has n-1 edges
        assert swc_data.metadata['non_tree_edges'] > 0
        
        # Check that removed edges are properly annotated
        for edge in swc_data.non_tree_edges:
            assert 'sample_id_1' in edge
            assert 'sample_id_2' in edge
            assert 'original_node_1' in edge
            assert 'original_node_2' in edge
            assert 'centroid_1' in edge
            assert 'centroid_2' in edge
    
    def test_type_id_consistency(self, simple_segment_graph):
        """Test that all segments use type ID 5."""
        swc_data = simple_segment_graph.export_to_swc()
        
        # No soma segments in simplified approach (field removed from SWCData)
        
        # Check that all segments have type 5
        for entry in swc_data.entries:
            parts = entry.split()
            type_id = int(parts[1])
            assert type_id == 5  # All segments use type 5
    
    def test_arbor_validation(self, simple_segment_graph):
        """Test that exported SWC meets Arbor requirements."""
        swc_data = simple_segment_graph.export_to_swc()
        
        sample_ids = []
        parent_ids = []
        
        for entry in swc_data.entries:
            parts = entry.split()
            sample_id = int(parts[0])
            type_id = int(parts[1])
            parent_id = int(parts[6])
            
            sample_ids.append(sample_id)
            parent_ids.append(parent_id)
            
            # All segments should use type 5
            assert type_id == 5
        
        # Arbor requirement checks
        # 1. No duplicate sample IDs
        assert len(set(sample_ids)) == len(sample_ids)
        
        # 2. Parent IDs are less than child IDs (except root)
        for sample_id, parent_id in zip(sample_ids, parent_ids):
            if parent_id != -1:
                assert parent_id < sample_id
        
        # 3. All parent IDs refer to existing samples
        valid_sample_ids = set(sample_ids + [-1])
        for parent_id in parent_ids:
            assert parent_id in valid_sample_ids
    
    def test_scale_factor_application(self, simple_segment_graph):
        """Test that scale factor is properly applied."""
        scale_factor = 2.5
        swc_data = simple_segment_graph.export_to_swc(scale_factor=scale_factor)
        
        # Check that coordinates and radii are scaled
        for entry in swc_data.entries:
            parts = entry.split()
            x, y, z, radius = map(float, parts[2:6])
            
            # All values should be reasonable (not zero unless original was zero)
            # This is a basic sanity check - more detailed validation would require
            # comparing with unscaled version
            assert radius > 0  # Radius should always be positive after scaling
    
    def test_cycle_breaking_strategies(self, cyclic_segment_graph):
        """Test different cycle-breaking strategies."""
        strategies = ['minimum_spanning_tree', 'bfs_tree']
        
        for strategy in strategies:
            swc_data = cyclic_segment_graph.export_to_swc(
                cycle_breaking_strategy=strategy
            )
            
            # Both strategies should produce valid trees
            assert len(swc_data.entries) == 4
            assert swc_data.metadata['tree_edges'] == 3
            assert len(swc_data.non_tree_edges) > 0
            assert swc_data.metadata['cycle_breaking_strategy'] == strategy


class TestSWCIntegrationWithMeshSegmentation:
    """Test SWC export integration with mesh segmentation."""
    
    @pytest.fixture
    def segmented_cylinder(self):
        """Create a segmented cylinder for testing."""
        cylinder = create_cylinder_mesh(length=20.0, radius=3.0, resolution=16)
        segmenter = MeshSegmenter()
        segments = segmenter.segment_mesh(cylinder, slice_height=5.0)
        return segmenter.get_segment_graph()
    
    def test_cylinder_swc_export(self, segmented_cylinder):
        """Test SWC export from actual mesh segmentation."""
        swc_data = segmented_cylinder.export_to_swc()
        
        # Should have segments from cylinder segmentation
        assert len(swc_data.entries) > 0
        assert swc_data.metadata['total_segments'] > 0
        
        # All segments should have valid properties
        for entry in swc_data.entries:
            parts = entry.split()
            assert len(parts) == 7
            
            # Check that coordinates and radius are reasonable
            x, y, z, radius = map(float, parts[2:6])
            assert not np.isnan(x) and not np.isnan(y) and not np.isnan(z)
            assert not np.isinf(x) and not np.isinf(y) and not np.isinf(z)
            assert radius > 0
    
    def test_end_to_end_workflow(self, segmented_cylinder):
        """Test complete workflow from segmentation to SWC file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            swc_file = os.path.join(temp_dir, "cylinder.swc")
            
            # Export to SWC
            swc_data = segmented_cylinder.export_to_swc(scale_factor=1.0)
            
            # Write to file
            swc_data.write_to_file(swc_file)
            
            # Verify file exists and has content
            assert os.path.exists(swc_file)
            
            with open(swc_file, 'r') as f:
                content = f.read()
            
            # Check that file has proper structure
            lines = content.strip().split('\n')
            data_lines = [line for line in lines if not line.startswith('#')]
            
            assert len(data_lines) > 0
            assert len(data_lines) == len(swc_data.entries)
            
            # Check that each data line has proper format
            for line in data_lines:
                parts = line.split()
                assert len(parts) == 7
                
                # Validate that all parts can be converted to numbers
                sample_id = int(parts[0])
                type_id = int(parts[1])
                x, y, z, radius = map(float, parts[2:6])
                parent_id = int(parts[6])
                
                assert sample_id > 0
                assert type_id == 5
                assert radius > 0
                assert parent_id >= -1


class TestSWCErrorHandling:
    """Test error handling in SWC export functionality."""
    
    def test_empty_graph_export(self):
        """Test SWC export with empty graph."""
        empty_graph = SegmentGraph()
        
        with pytest.raises(ValueError, match="Cannot export empty graph"):
            empty_graph.export_to_swc()
    
    def test_invalid_cycle_breaking_strategy(self):
        """Test invalid cycle-breaking strategy."""
        graph = SegmentGraph()
        graph.add_node('seg_0', centroid=np.array([0, 0, 0]), volume=1.0)
        
        with pytest.raises(ValueError, match="Unknown cycle breaking strategy"):
            graph.export_to_swc(cycle_breaking_strategy='invalid_strategy')
    
    def test_missing_centroid_data(self):
        """Test handling of missing centroid data."""
        graph = SegmentGraph()
        graph.add_node('seg_0', volume=1.0)  # Missing centroid
        
        with pytest.raises(ValueError, match="has no centroid"):
            graph.export_to_swc()
    
    def test_swc_data_file_writing_errors(self):
        """Test SWC data file writing error handling."""
        swc_data = SWCData(
            entries=["1 5 0 0 0 1 -1"],
            metadata={'total_segments': 1},
            non_tree_edges=[],
            root_segment='seg_0',
            scale_factor=1.0
        )
        
        # Test writing to invalid directory
        invalid_path = "/invalid/path/test.swc"
        
        with pytest.raises((OSError, FileNotFoundError, PermissionError)):
            swc_data.write_to_file(invalid_path)


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])

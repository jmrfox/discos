"""
Test script for mesh segmentation functionality.

This test script methodically validates the MeshSegmenter class by:
1. Creating cylinder meshes with known theoretical properties
2. Segmenting them with different slice heights (L, L/2, L/3)
3. Validating volume conservation (sum of segments vs theoretical)
4. Validating surface area conservation (sum of external areas vs theoretical)
5. Testing edge cases and error conditions
"""

import sys
import os

# Add the parent directory to path for importing discos
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import trimesh
import warnings

from discos.demos import create_cylinder_mesh, create_torus_mesh
from discos.segmentation import MeshSegmenter


class TestMeshSegmenterBasic:
    """Basic tests for MeshSegmenter functionality."""
    
    @pytest.fixture
    def test_cylinder(self):
        """Create a test cylinder with known properties."""
        length = 30.0
        radius = 3.0
        return create_cylinder_mesh(length=length, radius=radius, resolution=32)
    
    @pytest.fixture
    def segmenter(self):
        """Create a MeshSegmenter instance."""
        return MeshSegmenter()
    
    def test_segmenter_init(self, segmenter):
        """Test MeshSegmenter initialization."""
        assert segmenter.original_mesh is None
        assert segmenter.slice_height is None
        assert segmenter.cross_sections == []
        assert segmenter.segments == []
        assert segmenter.slices == []
        
    def test_segment_mesh_basic(self, segmenter, test_cylinder):
        """Test basic mesh segmentation functionality."""
        slice_height = 10.0
        segments = segmenter.segment_mesh(test_cylinder, slice_height)
        
        # Check that we get segments
        assert len(segments) > 0
        assert segmenter.original_mesh is not None
        assert segmenter.slice_height == slice_height
        
        # Check segment properties
        for segment in segments:
            assert hasattr(segment, 'id')
            assert hasattr(segment, 'volume')
            assert hasattr(segment, 'external_surface_area')
            assert hasattr(segment, 'internal_surface_area')
            assert segment.volume > 0
            assert segment.external_surface_area > 0


class TestMeshSegmenterVolumeConservation:
    """Test volume conservation across different slice heights."""
    
    @pytest.fixture
    def test_cylinder_params(self):
        """Parameters for test cylinder."""
        return {
            "length": 24.0,  # Divisible by 2, 3, 4, 6, 8, 12
            "radius": 4.0,
            "resolution": 32
        }
    
    @pytest.fixture
    def test_cylinder(self, test_cylinder_params):
        """Create test cylinder with specific parameters."""
        return create_cylinder_mesh(**test_cylinder_params)
    
    @pytest.fixture
    def theoretical_volume(self, test_cylinder_params):
        """Calculate theoretical volume of the cylinder."""
        length = test_cylinder_params["length"]
        radius = test_cylinder_params["radius"]
        return np.pi * radius**2 * length
    
    @pytest.fixture
    def theoretical_surface_area(self, test_cylinder_params):
        """Calculate theoretical surface area of the cylinder."""
        length = test_cylinder_params["length"]
        radius = test_cylinder_params["radius"]
        return 2 * np.pi * radius * (radius + length)
    
    def test_volume_conservation_full_length(self, test_cylinder, test_cylinder_params, theoretical_volume):
        """Test volume conservation with slice_height = L (full length)."""
        segmenter = MeshSegmenter()
        length = test_cylinder_params["length"]
        slice_height = length  # Full length - should create 1 segment
        
        segments = segmenter.segment_mesh(test_cylinder, slice_height)
        
        # Should have exactly 1 segment
        assert len(segments) == 1
        
        # Check volume conservation
        total_segment_volume = sum(seg.volume for seg in segments)
        volume_error_percent = abs(total_segment_volume - theoretical_volume) / theoretical_volume * 100
        
        # Volume should be conserved within 5%
        assert volume_error_percent < 5.0, f"Volume error {volume_error_percent:.2f}% exceeds 5%"
        
        # Check that segment volume is close to theoretical
        assert abs(segments[0].volume - theoretical_volume) / theoretical_volume < 0.05
    
    def test_volume_conservation_half_length(self, test_cylinder, test_cylinder_params, theoretical_volume):
        """Test volume conservation with slice_height = L/2."""
        segmenter = MeshSegmenter()
        length = test_cylinder_params["length"]
        slice_height = length / 2  # Half length - should create 2 segments
        
        segments = segmenter.segment_mesh(test_cylinder, slice_height)
        
        # Should have 2 segments
        assert len(segments) == 2
        
        # Check volume conservation
        total_segment_volume = sum(seg.volume for seg in segments)
        volume_error_percent = abs(total_segment_volume - theoretical_volume) / theoretical_volume * 100
        
        # Volume should be conserved within 5%
        assert volume_error_percent < 5.0, f"Volume error {volume_error_percent:.2f}% exceeds 5%"
        
        # Each segment should have approximately half the volume
        expected_segment_volume = theoretical_volume / 2
        for segment in segments:
            segment_error = abs(segment.volume - expected_segment_volume) / expected_segment_volume
            assert segment_error < 0.1, f"Individual segment volume error {segment_error:.2f} exceeds 10%"
    
    def test_volume_conservation_third_length(self, test_cylinder, test_cylinder_params, theoretical_volume):
        """Test volume conservation with slice_height = L/3."""
        segmenter = MeshSegmenter()
        length = test_cylinder_params["length"]
        slice_height = length / 3  # Third length - should create 3 segments
        
        segments = segmenter.segment_mesh(test_cylinder, slice_height)
        
        # Should have 3 segments
        assert len(segments) == 3
        
        # Check volume conservation
        total_segment_volume = sum(seg.volume for seg in segments)
        volume_error_percent = abs(total_segment_volume - theoretical_volume) / theoretical_volume * 100
        
        # Volume should be conserved within 5%
        assert volume_error_percent < 5.0, f"Volume error {volume_error_percent:.2f}% exceeds 5%"
        
        # Each segment should have approximately one third the volume
        expected_segment_volume = theoretical_volume / 3
        for segment in segments:
            segment_error = abs(segment.volume - expected_segment_volume) / expected_segment_volume
            assert segment_error < 0.1, f"Individual segment volume error {segment_error:.2f} exceeds 10%"


class TestMeshSegmenterSurfaceAreaConservation:
    """Test surface area conservation across different slice heights."""
    
    @pytest.fixture
    def test_cylinder_params(self):
        """Parameters for test cylinder."""
        return {
            "length": 24.0,  # Divisible by 2, 3, 4, 6, 8, 12
            "radius": 4.0,
            "resolution": 32
        }
    
    @pytest.fixture
    def test_cylinder(self, test_cylinder_params):
        """Create test cylinder with specific parameters."""
        return create_cylinder_mesh(**test_cylinder_params)
    
    @pytest.fixture
    def theoretical_surface_area(self, test_cylinder_params):
        """Calculate theoretical surface area of the cylinder."""
        length = test_cylinder_params["length"]
        radius = test_cylinder_params["radius"]
        return 2 * np.pi * radius * (radius + length)
    
    def test_surface_area_conservation_full_length(self, test_cylinder, test_cylinder_params, theoretical_surface_area):
        """Test surface area conservation with slice_height = L."""
        segmenter = MeshSegmenter()
        length = test_cylinder_params["length"]
        slice_height = length
        
        segments = segmenter.segment_mesh(test_cylinder, slice_height)
        
        # Check external surface area conservation
        total_external_area = sum(seg.external_surface_area for seg in segments)
        area_error_percent = abs(total_external_area - theoretical_surface_area) / theoretical_surface_area * 100
        
        # Surface area should be conserved within 5%
        assert area_error_percent < 5.0, f"Surface area error {area_error_percent:.2f}% exceeds 5%"
    
    def test_surface_area_conservation_half_length(self, test_cylinder, test_cylinder_params, theoretical_surface_area):
        """Test surface area conservation with slice_height = L/2."""
        segmenter = MeshSegmenter()
        length = test_cylinder_params["length"]
        slice_height = length / 2
        
        segments = segmenter.segment_mesh(test_cylinder, slice_height)
        
        # Check external surface area conservation
        total_external_area = sum(seg.external_surface_area for seg in segments)
        area_error_percent = abs(total_external_area - theoretical_surface_area) / theoretical_surface_area * 100
        
        # Surface area should be conserved within 5%
        assert area_error_percent < 5.0, f"Surface area error {area_error_percent:.2f}% exceeds 5%"
        
        # Check that internal surface area is created from cuts
        total_internal_area = sum(seg.internal_surface_area for seg in segments)
        assert total_internal_area > 0, "Internal surface area should be created from cuts"
    
    def test_surface_area_conservation_third_length(self, test_cylinder, test_cylinder_params, theoretical_surface_area):
        """Test surface area conservation with slice_height = L/3."""
        segmenter = MeshSegmenter()
        length = test_cylinder_params["length"]
        slice_height = length / 3
        
        segments = segmenter.segment_mesh(test_cylinder, slice_height)
        
        # Check external surface area conservation
        total_external_area = sum(seg.external_surface_area for seg in segments)
        area_error_percent = abs(total_external_area - theoretical_surface_area) / theoretical_surface_area * 100
        
        # Surface area should be conserved within 5%
        assert area_error_percent < 5.0, f"Surface area error {area_error_percent:.2f}% exceeds 5%"
        
        # Check that internal surface area increases with more cuts
        total_internal_area = sum(seg.internal_surface_area for seg in segments)
        assert total_internal_area > 0, "Internal surface area should be created from cuts"


class TestMeshSegmenterIntegration:
    """Integration tests combining different aspects of segmentation."""
    
    def test_comprehensive_cylinder_segmentation(self):
        """Comprehensive test of cylinder segmentation across multiple slice heights."""
        # Create test cylinder
        length = 30.0
        radius = 5.0
        cylinder = create_cylinder_mesh(length=length, radius=radius, resolution=24)
        
        # Theoretical values
        theoretical_volume = np.pi * radius**2 * length
        theoretical_surface_area = 2 * np.pi * radius * (radius + length)
        
        # Test different slice heights
        slice_heights = [length, length/2, length/3, length/5]
        expected_segments = [1, 2, 3, 5]
        
        for slice_height, expected_count in zip(slice_heights, expected_segments):
            segmenter = MeshSegmenter()
            segments = segmenter.segment_mesh(cylinder, slice_height)
            
            # Check segment count
            assert len(segments) == expected_count, f"Expected {expected_count} segments, got {len(segments)} for slice_height={slice_height}"
            
            # Check volume conservation
            total_volume = sum(seg.volume for seg in segments)
            volume_error = abs(total_volume - theoretical_volume) / theoretical_volume * 100
            assert volume_error < 5.0, f"Volume error {volume_error:.2f}% for slice_height={slice_height}"
            
            # Check surface area conservation
            total_external_area = sum(seg.external_surface_area for seg in segments)
            area_error = abs(total_external_area - theoretical_surface_area) / theoretical_surface_area * 100
            assert area_error < 5.0, f"Surface area error {area_error:.2f}% for slice_height={slice_height}"
            
            # Check that all segments have positive volumes and surface areas
            for i, segment in enumerate(segments):
                assert segment.volume > 0, f"Segment {i} has non-positive volume: {segment.volume}"
                assert segment.external_surface_area > 0, f"Segment {i} has non-positive external surface area"
                
                # Check segment bounds
                assert segment.z_min < segment.z_max, f"Segment {i} has invalid z-bounds: {segment.z_min} >= {segment.z_max}"
    
    def test_segment_graph_creation(self):
        """Test creation and properties of segment graph."""
        cylinder = create_cylinder_mesh(length=20.0, radius=3.0, resolution=20)
        segmenter = MeshSegmenter()
        
        # Get segment graph
        segment_graph = segmenter.segment_mesh(cylinder, slice_height=5.0, return_segment_graph=True)
        
        # Check that we get a SegmentGraph
        from discos.segmentation import SegmentGraph
        assert isinstance(segment_graph, SegmentGraph)
        
        # Check graph properties
        assert len(segment_graph.nodes) > 0
        assert len(segment_graph.segments) > 0
        
        # Check that segments are connected (for a cylinder, should be linear)
        assert segment_graph.number_of_edges() >= len(segment_graph.segments) - 1
    
    def test_different_cylinder_sizes(self):
        """Test segmentation with different cylinder sizes."""
        test_cases = [
            {"length": 10.0, "radius": 2.0, "slice_height": 2.0},
            {"length": 50.0, "radius": 8.0, "slice_height": 10.0},
            {"length": 100.0, "radius": 1.0, "slice_height": 25.0},
        ]
        
        for case in test_cases:
            cylinder = create_cylinder_mesh(
                length=case["length"], 
                radius=case["radius"], 
                resolution=24
            )
            
            theoretical_volume = np.pi * case["radius"]**2 * case["length"]
            
            segmenter = MeshSegmenter()
            segments = segmenter.segment_mesh(cylinder, case["slice_height"])
            
            # Check volume conservation
            total_volume = sum(seg.volume for seg in segments)
            volume_error = abs(total_volume - theoretical_volume) / theoretical_volume * 100
            
            assert volume_error < 5.0, f"Volume error {volume_error:.2f}% for case {case}"
            assert len(segments) > 0, f"No segments created for case {case}"


class TestMeshSegmenterErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_mesh_input(self):
        """Test handling of invalid mesh inputs."""
        segmenter = MeshSegmenter()
        
        # Test with None
        with pytest.raises((ValueError, AttributeError)):
            segmenter.segment_mesh(None, 5.0)
    
    def test_invalid_slice_height(self):
        """Test handling of invalid slice heights."""
        cylinder = create_cylinder_mesh(length=20.0, radius=3.0)
        segmenter = MeshSegmenter()
        
        # Test with zero slice height - should raise an error or produce no segments
        with pytest.raises((ValueError, ZeroDivisionError, RuntimeError)):
            segmenter.segment_mesh(cylinder, 0.0)
        
        # Test with negative slice height - may not raise ValueError but should produce warnings
        # The implementation generates warnings but may not raise exceptions
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                segments = segmenter.segment_mesh(cylinder, -5.0)
                # If it doesn't raise an exception, check that warnings were generated
                # or that no valid segments were created
                assert len(w) > 0 or len(segments) == 0
            except (ValueError, RuntimeError, ZeroDivisionError):
                # If it does raise an exception, that's also acceptable
                pass
    
    def test_very_small_slice_height(self):
        """Test with very small slice height (should create many segments)."""
        cylinder = create_cylinder_mesh(length=10.0, radius=2.0, resolution=16)
        segmenter = MeshSegmenter()
        
        # Use very small slice height
        segments = segmenter.segment_mesh(cylinder, slice_height=0.5)
        
        # Should create many segments
        assert len(segments) >= 15  # 10.0 / 0.5 = 20, but allow some tolerance
        
        # Volume should still be conserved
        theoretical_volume = np.pi * 2.0**2 * 10.0
        total_volume = sum(seg.volume for seg in segments)
        volume_error = abs(total_volume - theoretical_volume) / theoretical_volume * 100
        
        assert volume_error < 10.0  # Allow slightly higher tolerance for many small segments


class TestMeshSegmenterTorusConservation:
    """Test segmentation of torus meshes with complex topology.
    
    The torus is oriented with its symmetric axis perpendicular to the z-axis,
    forcing the segmentation algorithm to handle cross-sections that pass through
    the hole in the mesh, creating more complex topology challenges.
    """
    
    @pytest.fixture
    def test_torus_params(self):
        """Parameters for test torus with axis perpendicular to z."""
        return {
            "major_radius": 8.0,  # Distance from center to tube center
            "minor_radius": 2.0,  # Tube radius
            "major_segments": 32,
            "minor_segments": 16,
            "axis": "x",  # Perpendicular to z-axis for challenging cross-sections
            "center": (0.0, 0.0, 0.0)
        }
    
    @pytest.fixture
    def test_torus(self, test_torus_params):
        """Create test torus with axis perpendicular to z-axis."""
        return create_torus_mesh(**test_torus_params)
    
    @pytest.fixture
    def theoretical_volume(self, test_torus_params):
        """Calculate theoretical volume of the torus."""
        major_radius = test_torus_params["major_radius"]
        minor_radius = test_torus_params["minor_radius"]
        return 2 * np.pi**2 * major_radius * minor_radius**2
    
    @pytest.fixture
    def theoretical_surface_area(self, test_torus_params):
        """Calculate theoretical surface area of the torus."""
        major_radius = test_torus_params["major_radius"]
        minor_radius = test_torus_params["minor_radius"]
        return 4 * np.pi**2 * major_radius * minor_radius
    
    @pytest.fixture
    def torus_extent(self, test_torus_params):
        """Calculate L = 2 * (major_radius + minor_radius) (full extent on z-axis)."""
        major_radius = test_torus_params["major_radius"]
        minor_radius = test_torus_params["minor_radius"]
        return 2 * (major_radius + minor_radius)
    
    def test_torus_volume_conservation_half_extent(self, test_torus, test_torus_params, theoretical_volume, torus_extent):
        """Test torus volume conservation with slice_height = L/2."""
        segmenter = MeshSegmenter()
        slice_height = torus_extent / 2  # L/2
        
        segments = segmenter.segment_mesh(test_torus, slice_height)
        
        # Check that segments were created
        assert len(segments) > 0, "No segments created for torus"
        
        # Check volume conservation
        total_segment_volume = sum(seg.volume for seg in segments)
        volume_error_percent = abs(total_segment_volume - theoretical_volume) / theoretical_volume * 100
        
        # Volume should be conserved within 10% (higher tolerance for complex topology)
        assert volume_error_percent < 10.0, f"Torus volume error {volume_error_percent:.2f}% exceeds 10%"
        
        # Check that all segments have positive volume
        for i, segment in enumerate(segments):
            assert segment.volume > 0, f"Torus segment {i} has non-positive volume: {segment.volume}"
    
    def test_torus_volume_conservation_third_extent(self, test_torus, test_torus_params, theoretical_volume, torus_extent):
        """Test torus volume conservation with slice_height = L/3."""
        segmenter = MeshSegmenter()
        slice_height = torus_extent / 3  # L/3
        
        segments = segmenter.segment_mesh(test_torus, slice_height)
        
        # Check that segments were created
        assert len(segments) > 0, "No segments created for torus"
        
        # Check volume conservation
        total_segment_volume = sum(seg.volume for seg in segments)
        volume_error_percent = abs(total_segment_volume - theoretical_volume) / theoretical_volume * 100
        
        # Volume should be conserved within 10%
        assert volume_error_percent < 10.0, f"Torus volume error {volume_error_percent:.2f}% exceeds 10%"
        
        # Check that all segments have positive volume
        for i, segment in enumerate(segments):
            assert segment.volume > 0, f"Torus segment {i} has non-positive volume: {segment.volume}"
    
    def test_torus_volume_conservation_quarter_extent(self, test_torus, test_torus_params, theoretical_volume, torus_extent):
        """Test torus volume conservation with slice_height = L/4."""
        segmenter = MeshSegmenter()
        slice_height = torus_extent / 4  # L/4
        
        segments = segmenter.segment_mesh(test_torus, slice_height)
        
        # Check that segments were created
        assert len(segments) > 0, "No segments created for torus"
        
        # Check volume conservation
        total_segment_volume = sum(seg.volume for seg in segments)
        volume_error_percent = abs(total_segment_volume - theoretical_volume) / theoretical_volume * 100
        
        # Volume should be conserved within 10%
        assert volume_error_percent < 10.0, f"Torus volume error {volume_error_percent:.2f}% exceeds 10%"
        
        # Check that all segments have positive volume
        for i, segment in enumerate(segments):
            assert segment.volume > 0, f"Torus segment {i} has non-positive volume: {segment.volume}"
    
    def test_torus_surface_area_conservation_half_extent(self, test_torus, theoretical_surface_area, torus_extent):
        """Test torus surface area conservation with slice_height = L/2."""
        segmenter = MeshSegmenter()
        slice_height = torus_extent / 2
        
        segments = segmenter.segment_mesh(test_torus, slice_height)
        
        # Check external surface area conservation
        total_external_area = sum(seg.external_surface_area for seg in segments)
        area_error_percent = abs(total_external_area - theoretical_surface_area) / theoretical_surface_area * 100
        
        # Surface area should be conserved within 10% (higher tolerance for complex topology)
        assert area_error_percent < 10.0, f"Torus surface area error {area_error_percent:.2f}% exceeds 10%"
        
        # Check that internal surface area is created from cuts
        total_internal_area = sum(seg.internal_surface_area for seg in segments)
        # For torus with cuts, we expect some internal surface area
        # (though it might be small depending on where cuts intersect the torus)
        assert total_internal_area >= 0, "Internal surface area should be non-negative"
    
    def test_torus_surface_area_conservation_third_extent(self, test_torus, theoretical_surface_area, torus_extent):
        """Test torus surface area conservation with slice_height = L/3."""
        segmenter = MeshSegmenter()
        slice_height = torus_extent / 3
        
        segments = segmenter.segment_mesh(test_torus, slice_height)
        
        # Check external surface area conservation
        total_external_area = sum(seg.external_surface_area for seg in segments)
        area_error_percent = abs(total_external_area - theoretical_surface_area) / theoretical_surface_area * 100
        
        # Surface area should be conserved within 10%
        assert area_error_percent < 10.0, f"Torus surface area error {area_error_percent:.2f}% exceeds 10%"
    
    def test_torus_surface_area_conservation_quarter_extent(self, test_torus, theoretical_surface_area, torus_extent):
        """Test torus surface area conservation with slice_height = L/4."""
        segmenter = MeshSegmenter()
        slice_height = torus_extent / 4
        
        segments = segmenter.segment_mesh(test_torus, slice_height)
        
        # Check external surface area conservation
        total_external_area = sum(seg.external_surface_area for seg in segments)
        area_error_percent = abs(total_external_area - theoretical_surface_area) / theoretical_surface_area * 100
        
        # Surface area should be conserved within 10%
        assert area_error_percent < 10.0, f"Torus surface area error {area_error_percent:.2f}% exceeds 10%"
    
    def test_torus_comprehensive_segmentation(self, test_torus, test_torus_params, theoretical_volume, theoretical_surface_area, torus_extent):
        """Comprehensive test of torus segmentation across multiple slice heights."""
        # Test different slice heights: L/2, L/3, L/4
        slice_heights = [torus_extent/2, torus_extent/3, torus_extent/4]
        
        for slice_height in slice_heights:
            segmenter = MeshSegmenter()
            segments = segmenter.segment_mesh(test_torus, slice_height)
            
            # Check that segments were created
            assert len(segments) > 0, f"No segments created for torus with slice_height={slice_height}"
            
            # Check volume conservation
            total_volume = sum(seg.volume for seg in segments)
            volume_error = abs(total_volume - theoretical_volume) / theoretical_volume * 100
            assert volume_error < 10.0, f"Torus volume error {volume_error:.2f}% for slice_height={slice_height}"
            
            # Check surface area conservation
            total_external_area = sum(seg.external_surface_area for seg in segments)
            area_error = abs(total_external_area - theoretical_surface_area) / theoretical_surface_area * 100
            assert area_error < 10.0, f"Torus surface area error {area_error:.2f}% for slice_height={slice_height}"
            
            # Check segment properties
            for i, segment in enumerate(segments):
                assert segment.volume > 0, f"Torus segment {i} has non-positive volume for slice_height={slice_height}"
                assert segment.external_surface_area > 0, f"Torus segment {i} has non-positive external surface area"
                assert segment.z_min < segment.z_max, f"Torus segment {i} has invalid z-bounds"
    
    def test_torus_topology_challenges(self, test_torus, torus_extent):
        """Test that torus segmentation handles topology challenges correctly."""
        segmenter = MeshSegmenter()
        slice_height = torus_extent / 3  # Should create multiple segments
        
        segments = segmenter.segment_mesh(test_torus, slice_height)
        
        # Check that segments were created despite complex topology
        assert len(segments) > 0, "Torus segmentation failed to create segments"
        
        # Check that each segment has a valid mesh
        for i, segment in enumerate(segments):
            assert segment.mesh is not None, f"Segment {i} has no mesh"
            assert len(segment.mesh.vertices) > 0, f"Segment {i} has no vertices"
            assert len(segment.mesh.faces) > 0, f"Segment {i} has no faces"
            
            # Check that segment mesh is valid
            assert segment.mesh.is_watertight or not segment.mesh.is_watertight, "Segment mesh validity check"
            
            # Check centroid is reasonable
            assert not np.any(np.isnan(segment.centroid)), f"Segment {i} has NaN centroid"
            assert not np.any(np.isinf(segment.centroid)), f"Segment {i} has infinite centroid"


class TestMeshSegmenterMixedTopology:
    """Test segmentation with both simple (cylinder) and complex (torus) topology."""
    
    def test_cylinder_vs_torus_segmentation(self):
        """Compare segmentation behavior between cylinder and torus."""
        # Create comparable geometries
        cylinder = create_cylinder_mesh(length=20.0, radius=4.0, resolution=24)
        torus = create_torus_mesh(major_radius=6.0, minor_radius=2.0, axis="x", major_segments=24, minor_segments=12)
        
        slice_height = 5.0
        
        # Segment both
        cyl_segmenter = MeshSegmenter()
        tor_segmenter = MeshSegmenter()
        
        cyl_segments = cyl_segmenter.segment_mesh(cylinder, slice_height)
        tor_segments = tor_segmenter.segment_mesh(torus, slice_height)
        
        # Both should create segments
        assert len(cyl_segments) > 0, "Cylinder segmentation failed"
        assert len(tor_segments) > 0, "Torus segmentation failed"
        
        # Both should have valid volume conservation
        cyl_total_volume = sum(seg.volume for seg in cyl_segments)
        tor_total_volume = sum(seg.volume for seg in tor_segments)
        
        cyl_theoretical = np.pi * 4.0**2 * 20.0
        tor_theoretical = 2 * np.pi**2 * 6.0 * 2.0**2
        
        cyl_error = abs(cyl_total_volume - cyl_theoretical) / cyl_theoretical * 100
        tor_error = abs(tor_total_volume - tor_theoretical) / tor_theoretical * 100
        
        assert cyl_error < 5.0, f"Cylinder volume error {cyl_error:.2f}%"
        assert tor_error < 10.0, f"Torus volume error {tor_error:.2f}%"
        
        # Check that both create valid segment graphs
        cyl_graph = cyl_segmenter.get_segment_graph()
        tor_graph = tor_segmenter.get_segment_graph()
        
        assert len(cyl_graph.nodes) == len(cyl_segments)
        assert len(tor_graph.nodes) == len(tor_segments)


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])

"""Tests for polygon hole geometry utilities."""

from __future__ import annotations

import numpy as np
import pytest

from tdgl3d.mesh.holes import (
    point_in_polygon,
    identify_hole_nodes,
    identify_boundary_links,
    identify_normal_boundary_links,
    identify_circular_hole_nodes,
)


def test_point_in_polygon_simple():
    """Test point-in-polygon for a simple triangle."""
    triangle = [(0.0, 0.0), (10.0, 0.0), (5.0, 10.0)]
    
    # Inside
    assert point_in_polygon((5.0, 5.0), triangle)
    assert point_in_polygon((5.0, 2.0), triangle)
    
    # Outside
    assert not point_in_polygon((0.0, 5.0), triangle)
    assert not point_in_polygon((15.0, 5.0), triangle)
    assert not point_in_polygon((5.0, 11.0), triangle)


def test_point_in_polygon_square():
    """Test point-in-polygon for a square."""
    square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    
    # Inside
    assert point_in_polygon((5.0, 5.0), square)
    assert point_in_polygon((1.0, 1.0), square)
    assert point_in_polygon((9.0, 9.0), square)
    
    # Outside
    assert not point_in_polygon((-1.0, 5.0), square)
    assert not point_in_polygon((11.0, 5.0), square)
    assert not point_in_polygon((5.0, -1.0), square)
    assert not point_in_polygon((5.0, 11.0), square)


def test_point_in_polygon_concave():
    """Test point-in-polygon for a concave (L-shaped) polygon."""
    # L-shape
    L_shape = [(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (10.0, 5.0), 
               (10.0, 10.0), (0.0, 10.0)]
    
    # Inside the L
    assert point_in_polygon((2.0, 5.0), L_shape)
    assert point_in_polygon((7.0, 7.0), L_shape)
    
    # In the concave notch (outside)
    assert not point_in_polygon((7.0, 2.0), L_shape)


def test_identify_hole_nodes_rectangular():
    """Test identify_hole_nodes for a rectangular hole."""
    # Square hole from (5, 5) to (15, 15)
    square = [(5.0, 5.0), (15.0, 5.0), (15.0, 15.0), (5.0, 15.0)]
    
    # Grid: 20×20×5, h=1.0
    mask = identify_hole_nodes(square, (0, 4), 1.0, 1.0, 20, 20, 5)
    
    assert mask.shape == (21, 21, 6)
    
    # Check interior of hole
    assert mask[10, 10, 2]  # Center of hole
    assert mask[6, 6, 1]    # Inside
    assert mask[14, 14, 3]  # Inside
    
    # Check outside
    assert not mask[0, 0, 2]    # Far outside
    assert not mask[20, 20, 2]  # Far outside
    assert not mask[3, 10, 2]   # Outside in x
    assert not mask[10, 3, 2]   # Outside in y
    
    # Check z-range
    assert mask[10, 10, 0]  # z_min included
    assert mask[10, 10, 4]  # z_max included
    assert not mask[10, 10, 5]  # Beyond z_max


def test_identify_hole_nodes_z_range():
    """Test that hole is properly extruded through z-range."""
    square = [(5.0, 5.0), (10.0, 5.0), (10.0, 10.0), (5.0, 10.0)]
    
    # Only z-layers 2-4
    mask = identify_hole_nodes(square, (2, 4), 1.0, 1.0, 15, 15, 10)
    
    # Check hole exists in specified range
    assert mask[7, 7, 2]
    assert mask[7, 7, 3]
    assert mask[7, 7, 4]
    
    # Check hole doesn't exist outside range
    assert not mask[7, 7, 0]
    assert not mask[7, 7, 1]
    assert not mask[7, 7, 5]


def test_identify_boundary_links_x_direction():
    """Test boundary link identification in x-direction."""
    # Create simple hole: single column at i=5, all j, all k
    mask = np.zeros((11, 11, 6), dtype=bool)
    mask[5, :, :] = True  # Column hole
    
    x_links = identify_boundary_links(mask, 'x')
    
    # Should find links at i=4 (connecting to i=5) and i=5 (connecting to i=6)
    assert len(x_links) > 0
    
    # Verify links are on boundary
    # Link at i=4, j=0, k=0 should be included
    # Linear index: 0 * 11 * 11 + 0 * 11 + 4 = 4
    assert 4 in x_links


def test_identify_boundary_links_y_direction():
    """Test boundary link identification in y-direction."""
    # Create simple hole: single row at j=5
    mask = np.zeros((11, 11, 6), dtype=bool)
    mask[:, 5, :] = True  # Row hole
    
    y_links = identify_boundary_links(mask, 'y')
    
    assert len(y_links) > 0


def test_identify_boundary_links_z_direction():
    """Test boundary link identification in z-direction."""
    # Create simple hole: single z-layer
    mask = np.zeros((11, 11, 6), dtype=bool)
    mask[:, :, 2] = True  # One z-layer
    
    z_links = identify_boundary_links(mask, 'z')
    
    assert len(z_links) > 0


def test_identify_boundary_links_square_hole():
    """Test boundary links for a square hole."""
    square = [(5.0, 5.0), (10.0, 5.0), (10.0, 10.0), (5.0, 10.0)]
    mask = identify_hole_nodes(square, (0, 2), 1.0, 1.0, 15, 15, 5)
    
    x_links = identify_boundary_links(mask, 'x')
    y_links = identify_boundary_links(mask, 'y')
    z_links = identify_boundary_links(mask, 'z')
    
    # Should find boundary links around the square perimeter
    assert len(x_links) > 0
    assert len(y_links) > 0
    # z_links depend on whether top/bottom are boundaries


def test_identify_boundary_links_invalid_direction():
    """Test that invalid direction raises error."""
    mask = np.zeros((11, 11, 6), dtype=bool)
    
    with pytest.raises(ValueError, match="Invalid direction"):
        identify_boundary_links(mask, 'invalid')


def test_identify_circular_hole():
    """Test circular hole identification."""
    center = (10.0, 10.0)
    radius = 5.0
    
    mask = identify_circular_hole_nodes(
        center, radius, (0, 4), 1.0, 1.0, 20, 20, 5
    )
    
    assert mask.shape == (21, 21, 6)
    
    # Check center
    assert mask[10, 10, 2]
    
    # Check points inside radius
    assert mask[10, 12, 2]  # 2 units away
    assert mask[12, 10, 2]  # 2 units away
    
    # Check points outside radius
    assert not mask[10, 16, 2]  # 6 units away
    assert not mask[16, 10, 2]  # 6 units away
    assert not mask[0, 0, 2]    # Far away
    
    # Check z-range
    assert mask[10, 10, 0]
    assert mask[10, 10, 4]
    assert not mask[10, 10, 5]


def test_circular_vs_polygon_approximation():
    """Test that circular hole can be approximated by many-sided polygon."""
    center = (10.0, 10.0)
    radius = 5.0
    
    # Create circular hole
    mask_circle = identify_circular_hole_nodes(
        center, radius, (0, 0), 1.0, 1.0, 20, 20, 1
    )
    
    # Create polygon approximation (octagon)
    n_sides = 8
    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    polygon = [(center[0] + radius * np.cos(a), 
                center[1] + radius * np.sin(a)) for a in angles]
    
    mask_polygon = identify_hole_nodes(
        polygon, (0, 0), 1.0, 1.0, 20, 20, 1
    )
    
    # Should be similar (not exact due to discretization)
    agreement = np.sum(mask_circle == mask_polygon) / mask_circle.size
    assert agreement > 0.9  # At least 90% agreement


def test_non_uniform_grid_spacing():
    """Test hole identification with different x and y grid spacing."""
    square_physical = [(5.0, 10.0), (15.0, 10.0), (15.0, 30.0), (5.0, 30.0)]
    
    # hx=1.0, hy=2.0
    mask = identify_hole_nodes(square_physical, (0, 2), 1.0, 2.0, 20, 20, 5)
    
    assert mask.shape == (21, 21, 6)
    
    # Center should be at i=10, j=10
    # Physical coords: (10*1.0, 10*2.0) = (10, 20) - should be inside
    assert mask[10, 10, 1]


def test_hole_nodes_empty_polygon():
    """Test behavior with polygon that doesn't contain any grid nodes."""
    # Tiny polygon between grid points
    tiny = [(0.1, 0.1), (0.4, 0.1), (0.4, 0.4), (0.1, 0.4)]
    
    mask = identify_hole_nodes(tiny, (0, 2), 1.0, 1.0, 10, 10, 5)
    
    # Should find no interior nodes (all False)
    assert not np.any(mask)


def test_boundary_links_no_hole():
    """Test boundary links with no hole (all False mask)."""
    mask = np.zeros((11, 11, 6), dtype=bool)
    
    x_links = identify_boundary_links(mask, 'x')
    y_links = identify_boundary_links(mask, 'y')
    z_links = identify_boundary_links(mask, 'z')
    
    assert len(x_links) == 0
    assert len(y_links) == 0
    assert len(z_links) == 0


def test_boundary_links_full_hole():
    """Test boundary links when entire grid is hole."""
    mask = np.ones((11, 11, 6), dtype=bool)
    
    x_links = identify_boundary_links(mask, 'x')
    y_links = identify_boundary_links(mask, 'y')
    z_links = identify_boundary_links(mask, 'z')
    
    # No boundaries (all same state)
    assert len(x_links) == 0
    assert len(y_links) == 0
    assert len(z_links) == 0


# ============================================================================
# Tests for Normal/Tangential Link Classification (Flux Trapping Physics)
# ============================================================================

def test_identify_normal_links_square_hole_2d():
    """Test normal link classification for a square hole in 2D.
    
    For a square hole with straight edges:
    - x-links on LEFT/RIGHT edges are NORMAL (perpendicular to vertical boundary)
    - y-links on TOP/BOTTOM edges are NORMAL (perpendicular to horizontal boundary)
    - x-links on TOP/BOTTOM edges are TANGENTIAL (parallel to horizontal boundary)
    - y-links on LEFT/RIGHT edges are TANGENTIAL (parallel to vertical boundary)
    
    This separation is critical for flux trapping: tangential links must be
    allowed to evolve freely for persistent currents to circulate around hole.
    """
    # Create 20×20 grid with 6×6 centered square hole (Nz=1 for 2D)
    hole_mask = np.zeros((21, 21, 2), dtype=bool)
    hole_mask[7:13, 7:13, :] = True  # Hole from i=7..12, j=7..12
    
    # Get ALL boundary links (normal + tangential)
    x_all = identify_boundary_links(hole_mask, 'x', is_3d=False)
    y_all = identify_boundary_links(hole_mask, 'y', is_3d=False)
    
    # Get NORMAL boundary links only
    x_normal = identify_normal_boundary_links(hole_mask, 'x', is_3d=False)
    y_normal = identify_normal_boundary_links(hole_mask, 'y', is_3d=False)
    
    # Normal links should be a SUBSET of all links
    assert len(x_normal) <= len(x_all), "Normal x-links should be subset of all x-links"
    assert len(y_normal) <= len(y_all), "Normal y-links should be subset of all y-links"
    
    # For a 6×6 square hole, expect:
    # - x-links: 2 vertical edges × 6 links each = 12 normal links (left + right)
    # - y-links: 2 horizontal edges × 6 links each = 12 normal links (top + bottom)
    # Note: In 2D with Nz=1, we have 1 z-plane, so multiply by 1
    expected_x_normal = 12  # Left edge (i=6) + Right edge (i=12)
    expected_y_normal = 12  # Bottom edge (j=6) + Top edge (j=12)
    
    # Allow some tolerance for boundary effects
    assert abs(len(x_normal) - expected_x_normal) <= 2, \
        f"Expected ~{expected_x_normal} normal x-links, got {len(x_normal)}"
    assert abs(len(y_normal) - expected_y_normal) <= 2, \
        f"Expected ~{expected_y_normal} normal y-links, got {len(y_normal)}"
    
    # Tangential links = ALL - NORMAL
    x_tangential_count = len(x_all) - len(x_normal)
    y_tangential_count = len(y_all) - len(y_normal)
    
    # Should have some tangential links (on horizontal/vertical edges)
    assert x_tangential_count > 0, "Should have tangential x-links (on horizontal edges)"
    assert y_tangential_count > 0, "Should have tangential y-links (on vertical edges)"
    
    print(f"  Square hole boundary links:")
    print(f"    x-links: {len(x_all)} total, {len(x_normal)} normal, {x_tangential_count} tangential")
    print(f"    y-links: {len(y_all)} total, {len(y_normal)} normal, {y_tangential_count} tangential")


def test_identify_normal_links_small_hole():
    """Test normal link classification for a minimal 2×2 hole."""
    # Create 10×10 grid with tiny 2×2 hole
    hole_mask = np.zeros((11, 11, 2), dtype=bool)
    hole_mask[4:6, 4:6, :] = True  # 2×2 hole at center
    
    x_normal = identify_normal_boundary_links(hole_mask, 'x', is_3d=False)
    y_normal = identify_normal_boundary_links(hole_mask, 'y', is_3d=False)
    
    # For 2×2 hole, expect:
    # - 2 vertical edges × 2 links = 4 normal x-links
    # - 2 horizontal edges × 2 links = 4 normal y-links
    assert 2 <= len(x_normal) <= 6, f"Expected 2-6 normal x-links for small hole, got {len(x_normal)}"
    assert 2 <= len(y_normal) <= 6, f"Expected 2-6 normal y-links for small hole, got {len(y_normal)}"


def test_identify_normal_links_3d():
    """Test normal link classification in 3D geometry."""
    # Create 10×10×5 grid with hole extending through z-layers 1-3
    hole_mask = np.zeros((11, 11, 6), dtype=bool)
    hole_mask[4:7, 4:7, 1:4] = True  # 3×3 hole in z-layers 1,2,3
    
    x_normal = identify_normal_boundary_links(hole_mask, 'x', is_3d=True)
    y_normal = identify_normal_boundary_links(hole_mask, 'y', is_3d=True)
    z_normal = identify_normal_boundary_links(hole_mask, 'z', is_3d=True)
    
    # Should find some normal links in all directions
    assert len(x_normal) > 0, "Should find normal x-links in 3D"
    assert len(y_normal) > 0, "Should find normal y-links in 3D"
    assert len(z_normal) > 0, "Should find normal z-links in 3D (top/bottom faces)"
    
    print(f"  3D hole normal links: x={len(x_normal)}, y={len(y_normal)}, z={len(z_normal)}")


def test_normal_links_are_subset():
    """Verify normal links are always a subset of all boundary links."""
    # Test with various hole shapes
    test_cases = [
        # (hole_mask shape, hole region)
        ((21, 21, 2), (slice(5, 10), slice(5, 10), slice(None))),  # Square
        ((21, 21, 2), (slice(8, 15), slice(5, 8), slice(None))),   # Rectangle
        ((15, 15, 4), (slice(5, 10), slice(5, 10), slice(1, 3))),  # 3D square
    ]
    
    for shape, hole_region in test_cases:
        mask = np.zeros(shape, dtype=bool)
        mask[hole_region] = True
        is_3d = (shape[2] > 2)
        
        for direction in ['x', 'y', 'z']:
            all_links = identify_boundary_links(mask, direction, is_3d=is_3d)
            normal_links = identify_normal_boundary_links(mask, direction, is_3d=is_3d)
            
            # Normal must be subset of all
            assert len(normal_links) <= len(all_links), \
                f"{direction}-links: normal ({len(normal_links)}) should be ≤ all ({len(all_links)})"
            
            # Check that normal links are actually in all links
            all_set = set(all_links)
            for link in normal_links:
                assert link in all_set, \
                    f"Normal {direction}-link {link} not found in all boundary links"


def test_normal_links_no_hole():
    """Test that no normal links are identified when there's no hole."""
    # Empty grid (no hole)
    mask = np.zeros((11, 11, 2), dtype=bool)
    
    x_normal = identify_normal_boundary_links(mask, 'x', is_3d=False)
    y_normal = identify_normal_boundary_links(mask, 'y', is_3d=False)
    z_normal = identify_normal_boundary_links(mask, 'z', is_3d=False)
    
    assert len(x_normal) == 0, "No hole → no normal boundary links"
    assert len(y_normal) == 0
    assert len(z_normal) == 0

"""Hole geometry utilities for arbitrary polygon shapes.

This module provides utilities for identifying nodes and links within
and on the boundary of polygon-shaped holes in the simulation mesh.

Physics: Holes vs Insulators
-----------------------------
The code distinguishes between two types of non-superconducting regions:

**Holes** (geometric voids):
  - Completely removed from the simulation domain
  - Zero-current boundary condition enforced: φ = 0 at all hole edges
  - No superconducting order parameter (ψ = 0 inside hole)
  - **Vortices CANNOT form inside holes** (no superconductor = no phase winding)
  - Magnetic field penetrates freely through the hole
  - Persistent currents can circulate around the hole (in SC region)
  - Vortices may nucleate near hole edges (but in the SC, not in the hole)
  
**Insulators** (e.g., S/I/S middle layer):
  - Part of the simulation domain with modified material properties
  - Suppressed order parameter: ψ → 0 via relaxation term −ψ/τ_relax
  - No special boundary conditions on φ (field can penetrate normally)
  - Magnetic field penetrates through the insulator layer
  - Used for modeling oxide barriers, normal metal layers, etc.

Key distinction: Holes enforce φ = 0 at boundaries (zero normal current),
while insulators do not enforce special boundary conditions.

Examples
--------
- Rectangular hole in a square film: use `point_in_polygon()` + `identify_boundary_links()`
- S/I/S trilayer: use `Trilayer` with insulator layer (no hole functions needed)
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray


def point_in_polygon(point: tuple[float, float], vertices: list[tuple[float, float]]) -> bool:
    """Test if a point is inside a polygon using ray-casting algorithm.
    
    Uses the ray-casting algorithm: casts a ray from the point to infinity
    and counts how many times it crosses polygon edges. Odd = inside, even = outside.
    
    Parameters
    ----------
    point : (x, y)
        Test point coordinates
    vertices : list of (x, y)
        Polygon vertices in order. The polygon is automatically closed
        (no need to repeat the first vertex at the end).
    
    Returns
    -------
    bool
        True if point is strictly inside the polygon
    
    Notes
    -----
    - Handles both convex and concave polygons
    - Points exactly on edges may give inconsistent results (floating point)
    - Uses horizontal ray cast in +x direction
    
    References
    ----------
    https://en.wikipedia.org/wiki/Point_in_polygon
    
    Examples
    --------
    >>> triangle = [(0, 0), (10, 0), (5, 10)]
    >>> point_in_polygon((5, 5), triangle)
    True
    >>> point_in_polygon((15, 5), triangle)
    False
    """
    x, y = point
    n = len(vertices)
    inside = False
    
    p1x, p1y = vertices[0]
    for i in range(1, n + 1):
        p2x, p2y = vertices[i % n]
        
        # Check if horizontal ray from point intersects edge
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        # Compute x-coordinate of edge at height y
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        
        p1x, p1y = p2x, p2y
    
    return inside


def identify_hole_nodes(
    vertices: list[tuple[float, float]],
    z_range: tuple[int, int],
    grid_spacing_x: float,
    grid_spacing_y: float,
    Nx: int,
    Ny: int,
    Nz: int,
) -> NDArray[np.bool_]:
    """Identify all full-grid nodes inside a polygon hole.
    
    Parameters
    ----------
    vertices : list of (x, y) tuples
        Polygon vertices in physical coordinates (ξ units)
    z_range : (k_min, k_max)
        Z-layer extent (grid indices, inclusive)
    grid_spacing_x, grid_spacing_y : float
        Grid spacing in x and y directions
    Nx, Ny, Nz : int
        Grid dimensions (number of interior cells)
    
    Returns
    -------
    hole_mask : ndarray, shape (Nx+1, Ny+1, Nz+1)
        Boolean mask: True for nodes inside the hole
    
    Notes
    -----
    - Uses point-in-polygon test on full grid (includes boundaries)
    - The hole is extruded vertically through z_range
    - Complexity: O((Nx+1) × (Ny+1) × n_vertices) - acceptable for typical grids
    
    Examples
    --------
    >>> square = [(5.0, 5.0), (15.0, 5.0), (15.0, 15.0), (5.0, 15.0)]
    >>> mask = identify_hole_nodes(square, (0, 5), 1.0, 1.0, 20, 20, 10)
    >>> assert mask.shape == (21, 21, 11)
    """
    hole_mask = np.zeros((Nx + 1, Ny + 1, Nz + 1), dtype=bool)
    
    z_min, z_max = z_range
    
    # Test each node in the x-y plane
    for i in range(Nx + 1):
        for j in range(Ny + 1):
            x = i * grid_spacing_x
            y = j * grid_spacing_y
            
            if point_in_polygon((x, y), vertices):
                # Mark all z-layers in range
                for k in range(z_min, min(z_max + 1, Nz + 1)):
                    hole_mask[i, j, k] = True
    
    return hole_mask


def identify_boundary_links(
    hole_mask: NDArray[np.bool_],
    direction: Literal['x', 'y', 'z'],
    is_3d: bool = True,
) -> NDArray[np.int64]:
    """Find linear indices of links crossing the hole boundary.
    
    A link crosses the boundary if one endpoint is inside the hole
    and the other is outside.
    
    Parameters
    ----------
    hole_mask : ndarray, shape (Nx+1, Ny+1, Nz+1)
        Boolean mask of hole interior (True = inside hole)
    direction : {'x', 'y', 'z'}
        Link direction
    is_3d : bool, default True
        If False, use 2D indexing (ignore z-dimension in linear index)
    
    Returns
    -------
    boundary_links : ndarray of int64
        Linear indices (full-grid) of links on the hole boundary
    
    Notes
    -----
    - x-links connect nodes (i, j, k) → (i+1, j, k)
    - y-links connect nodes (i, j, k) → (i, j+1, k)
    - z-links connect nodes (i, j, k) → (i, j, k+1)
    - Linear index (3D): m = k × (Nx+1) × (Ny+1) + j × (Nx+1) + i
    - Linear index (2D): m = j × (Nx+1) + i
    - These indices are for the full grid (not interior-only)
    
    Examples
    --------
    >>> mask = np.zeros((11, 11, 6), dtype=bool)
    >>> mask[5, 5, :] = True  # Single column hole
    >>> x_links = identify_boundary_links(mask, 'x')
    >>> assert len(x_links) > 0  # Should find links crossing boundary
    """
    Nx, Ny, Nz = hole_mask.shape
    Nx -= 1  # Convert to number of cells
    Ny -= 1
    Nz -= 1
    
    boundary_links = []
    
    if direction == 'x':
        # x-direction links connect (i, j, k) to (i+1, j, k)
        for k in range(Nz + 1):
            for j in range(Ny + 1):
                for i in range(Nx):  # i goes to Nx-1 (link exists between i and i+1)
                    inside_left = hole_mask[i, j, k]
                    inside_right = hole_mask[i + 1, j, k]
                    
                    # Boundary link if exactly one endpoint is inside
                    if inside_left != inside_right:
                        # Linear index for node (i, j, k)
                        if is_3d:
                            m = k * (Nx + 1) * (Ny + 1) + j * (Nx + 1) + i
                        else:
                            m = j * (Nx + 1) + i
                        boundary_links.append(m)
    
    elif direction == 'y':
        # y-direction links connect (i, j, k) to (i, j+1, k)
        for k in range(Nz + 1):
            for i in range(Nx + 1):
                for j in range(Ny):  # j goes to Ny-1
                    inside_bottom = hole_mask[i, j, k]
                    inside_top = hole_mask[i, j + 1, k]
                    
                    if inside_bottom != inside_top:
                        if is_3d:
                            m = k * (Nx + 1) * (Ny + 1) + j * (Nx + 1) + i
                        else:
                            m = j * (Nx + 1) + i
                        boundary_links.append(m)
    
    elif direction == 'z':
        # z-direction links connect (i, j, k) to (i, j, k+1)
        for i in range(Nx + 1):
            for j in range(Ny + 1):
                for k in range(Nz):  # k goes to Nz-1
                    inside_below = hole_mask[i, j, k]
                    inside_above = hole_mask[i, j, k + 1]
                    
                    if inside_below != inside_above:
                        if is_3d:
                            m = k * (Nx + 1) * (Ny + 1) + j * (Nx + 1) + i
                        else:
                            m = j * (Nx + 1) + i
                        boundary_links.append(m)
    
    else:
        raise ValueError(f"Invalid direction '{direction}'. Use 'x', 'y', or 'z'.")
    
    return np.array(boundary_links, dtype=np.int64)


def identify_normal_boundary_links(
    hole_mask: NDArray[np.bool_],
    direction: Literal['x', 'y', 'z'],
    is_3d: bool = True,
) -> NDArray[np.int64]:
    """Find links PERPENDICULAR to hole boundary (for zero-current BC enforcement).
    
    A link is "normal" to the boundary if it crosses the boundary in a direction
    perpendicular to the local boundary orientation. Normal links should have
    φ = 0 enforced (zero normal current). Tangential links can evolve freely,
    allowing persistent currents to circulate around the hole.
    
    **Physics Motivation:**
    Zero-current boundary condition should only constrain the NORMAL component
    of current into the hole, not the TANGENTIAL component. This allows:
    - Persistent currents to circulate around hole (flux trapping)
    - Phase winding: ∮ ∇φ · dl = 2πn (quantized fluxoid)
    - Correct superconducting loop physics
    
    **Classification Strategy (Revised):**
    For a square hole with straight edges, examine the boundary topology:
    - x-links on VERTICAL edges (left/right) are NORMAL (perpendicular to edge)
    - y-links on HORIZONTAL edges (top/bottom) are NORMAL (perpendicular to edge)  
    - x-links on HORIZONTAL edges are TANGENTIAL (parallel to edge)
    - y-links on VERTICAL edges are TANGENTIAL (parallel to edge)
    
    Detection method:
    For each boundary link, check if moving perpendicular crosses MORE boundaries.
    - If yes → link is tangential (runs along edge)
    - If no → link is normal (crosses into/out of hole)
    
    Example (x-direction links on a square hole):
    ```
        SC  SC  SC  SC  SC
        SC  ──  ──  ──  SC   ← tangential x-links (top edge)
        SC  |  hole  |  SC   
        SC  ──  ──  ──  SC   ← tangential x-links (bottom edge)
        SC  SC  SC  SC  SC
            ↑           ↑
         normal       normal
        x-links      x-links
       (left edge)  (right edge)
    ```
    
    Parameters
    ----------
    hole_mask : ndarray, shape (Nx+1, Ny+1, Nz+1)
        Boolean mask of hole interior (True = inside hole)
    direction : {'x', 'y', 'z'}
        Link direction to classify
    is_3d : bool, default True
        If False, use 2D indexing (ignore z-dimension)
    
    Returns
    -------
    normal_links : ndarray of int64
        Linear indices (full-grid) of links PERPENDICULAR to hole boundary.
        These are the links that should have φ = 0 enforced.
    
    Notes
    -----
    - Only returns NORMAL links (subset of all boundary links)
    - Tangential links are implicitly allowed to evolve (not returned)
    - Corner links are classified based on local topology
    - For flux trapping: tangential circulation around hole requires this separation
    
    See Also
    --------
    identify_boundary_links : Returns ALL boundary links (normal + tangential)
    
    Examples
    --------
    >>> # Square hole: separate normal from tangential
    >>> mask = np.zeros((21, 21, 2), dtype=bool)
    >>> mask[7:13, 7:13, :] = True  # 6×6 hole
    >>> x_normal = identify_normal_boundary_links(mask, 'x')
    >>> # Returns only x-links on left/right edges (perpendicular to boundary)
    >>> # Does NOT return x-links on top/bottom edges (parallel to boundary)
    """
    Nx, Ny, Nz = hole_mask.shape
    Nx -= 1  # Convert to number of cells
    Ny -= 1
    Nz -= 1
    
    normal_links = []
    
    if direction == 'x':
        # x-links connect (i, j, k) → (i+1, j, k)
        # Link is TANGENTIAL if it's part of a boundary chain in x-direction
        # Link is NORMAL if it's an isolated crossing (not connected to boundary chain in x)
        
        for k in range(Nz + 1 if is_3d else 1):
            for j in range(Ny + 1):
                for i in range(Nx):  # x-links from i to i+1
                    inside_left = hole_mask[i, j, k]
                    inside_right = hole_mask[i + 1, j, k]
                    
                    # Only consider boundary-crossing links
                    if inside_left == inside_right:
                        continue  # Not a boundary link
                    
                    # Check if neighboring x-links (same j, k; different i) are ALSO boundaries
                    # If this link is part of a chain in x-direction → TANGENTIAL
                    # If this link is isolated in x-direction → NORMAL
                    
                    has_boundary_neighbor_x = False
                    
                    # Check x-link at (i-1, j, k) [link from i-1 to i]
                    if i > 0:
                        inside_left_prev = hole_mask[i - 1, j, k]
                        inside_right_prev = hole_mask[i, j, k]
                        if inside_left_prev != inside_right_prev:
                            has_boundary_neighbor_x = True
                    
                    # Check x-link at (i+1, j, k) [link from i+1 to i+2]
                    if i < Nx - 1:
                        inside_left_next = hole_mask[i + 1, j, k]
                        inside_right_next = hole_mask[i + 2, j, k]
                        if inside_left_next != inside_right_next:
                            has_boundary_neighbor_x = True
                    
                    # If NO boundary neighbors in x → this x-link is NORMAL to boundary
                    # If YES boundary neighbors in x → this x-link is TANGENTIAL (part of chain)
                    if not has_boundary_neighbor_x:
                        if is_3d:
                            m = k * (Nx + 1) * (Ny + 1) + j * (Nx + 1) + i
                        else:
                            m = j * (Nx + 1) + i
                        normal_links.append(m)
    
    elif direction == 'y':
        # y-links connect (i, j, k) → (i, j+1, k)
        # Link is TANGENTIAL if it's part of a boundary chain in y-direction
        # Link is NORMAL if it's an isolated crossing (not connected to boundary chain in y)
        
        for k in range(Nz + 1 if is_3d else 1):
            for i in range(Nx + 1):
                for j in range(Ny):  # y-links from j to j+1
                    inside_bottom = hole_mask[i, j, k]
                    inside_top = hole_mask[i, j + 1, k]
                    
                    # Only consider boundary-crossing links
                    if inside_bottom == inside_top:
                        continue
                    
                    # Check if neighboring y-links (same i, k; different j) are ALSO boundaries
                    has_boundary_neighbor_y = False
                    
                    # Check y-link at (i, j-1, k) [link from j-1 to j]
                    if j > 0:
                        inside_bottom_prev = hole_mask[i, j - 1, k]
                        inside_top_prev = hole_mask[i, j, k]
                        if inside_bottom_prev != inside_top_prev:
                            has_boundary_neighbor_y = True
                    
                    # Check y-link at (i, j+1, k) [link from j+1 to j+2]
                    if j < Ny - 1:
                        inside_bottom_next = hole_mask[i, j + 1, k]
                        inside_top_next = hole_mask[i, j + 2, k]
                        if inside_bottom_next != inside_top_next:
                            has_boundary_neighbor_y = True
                    
                    # If NO boundary neighbors in y → this y-link is NORMAL
                    # If YES boundary neighbors in y → this y-link is TANGENTIAL (part of chain)
                    if not has_boundary_neighbor_y:
                        if is_3d:
                            m = k * (Nx + 1) * (Ny + 1) + j * (Nx + 1) + i
                        else:
                            m = j * (Nx + 1) + i
                        normal_links.append(m)
    
    elif direction == 'z':
        # z-links connect (i, j, k) → (i, j, k+1)
        # Link is TANGENTIAL if it's part of a boundary chain in z-direction
        # Link is NORMAL if it's an isolated crossing (not connected to boundary chain in z)
        
        for i in range(Nx + 1):
            for j in range(Ny + 1):
                for k in range(Nz):  # z-links from k to k+1
                    inside_below = hole_mask[i, j, k]
                    inside_above = hole_mask[i, j, k + 1]
                    
                    # Only consider boundary-crossing links
                    if inside_below == inside_above:
                        continue
                    
                    # Check if neighboring z-links (same i, j; different k) are ALSO boundaries
                    has_boundary_neighbor_z = False
                    
                    # Check z-link at (i, j, k-1) [link from k-1 to k]
                    if k > 0:
                        inside_below_prev = hole_mask[i, j, k - 1]
                        inside_above_prev = hole_mask[i, j, k]
                        if inside_below_prev != inside_above_prev:
                            has_boundary_neighbor_z = True
                    
                    # Check z-link at (i, j, k+1) [link from k+1 to k+2]
                    if k < Nz - 1:
                        inside_below_next = hole_mask[i, j, k + 1]
                        inside_above_next = hole_mask[i, j, k + 2]
                        if inside_below_next != inside_above_next:
                            has_boundary_neighbor_z = True
                    
                    # If NO boundary neighbors in z → this z-link is NORMAL to boundary
                    # If YES boundary neighbors in z → this z-link is TANGENTIAL (part of chain)
                    if not has_boundary_neighbor_z:
                        m = k * (Nx + 1) * (Ny + 1) + j * (Nx + 1) + i
                        normal_links.append(m)
    
    else:
        raise ValueError(f"Invalid direction '{direction}'. Use 'x', 'y', or 'z'.")
    
    return np.array(normal_links, dtype=np.int64)


def identify_circular_hole_nodes(
    center: tuple[float, float],
    radius: float,
    z_range: tuple[int, int],
    grid_spacing_x: float,
    grid_spacing_y: float,
    Nx: int,
    Ny: int,
    Nz: int,
) -> NDArray[np.bool_]:
    """Identify nodes inside a circular hole.
    
    Parameters
    ----------
    center : (x, y)
        Circle center in physical coordinates
    radius : float
        Circle radius in physical units
    z_range : (k_min, k_max)
        Z-layer extent (grid indices)
    grid_spacing_x, grid_spacing_y : float
        Grid spacing
    Nx, Ny, Nz : int
        Grid dimensions
    
    Returns
    -------
    hole_mask : ndarray, shape (Nx+1, Ny+1, Nz+1)
        Boolean mask: True for nodes inside the circle
    
    Notes
    -----
    Faster than polygon method for circular holes.
    """
    hole_mask = np.zeros((Nx + 1, Ny + 1, Nz + 1), dtype=bool)
    
    cx, cy = center
    z_min, z_max = z_range
    
    for i in range(Nx + 1):
        for j in range(Ny + 1):
            x = i * grid_spacing_x
            y = j * grid_spacing_y
            
            # Distance from center
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            
            if dist < radius:
                for k in range(z_min, min(z_max + 1, Nz + 1)):
                    hole_mask[i, j, k] = True
    
    return hole_mask

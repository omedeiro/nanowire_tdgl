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

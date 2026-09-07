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

from typing import Literal, Optional

import numpy as np
from numpy.typing import NDArray


def _distance_to_segment(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    """Shortest distance from *point* to the segment ``start``–``end``."""
    px, py = point
    ax, ay = start
    bx, by = end
    dx, dy = bx - ax, by - ay
    length_squared = dx * dx + dy * dy
    if length_squared == 0.0:
        return float(np.hypot(px - ax, py - ay))
    t = ((px - ax) * dx + (py - ay) * dy) / length_squared
    t = min(1.0, max(0.0, t))
    return float(np.hypot(px - (ax + t * dx), py - (ay + t * dy)))


def point_in_polygon(
    point: tuple[float, float],
    vertices: list[tuple[float, float]],
    edge_tolerance: float = 0.0,
) -> bool:
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
    edge_tolerance : float, default 0.0
        Points within this distance of the polygon boundary count as inside.
        Pass a small positive value to get the *closed* region.

    Returns
    -------
    bool
        True if point is inside the polygon (or on its boundary when
        ``edge_tolerance > 0``)

    Notes
    -----
    - Handles both convex and concave polygons
    - Uses horizontal ray cast in +x direction

    .. warning::
       With ``edge_tolerance = 0`` the ray-casting rule is **half-open**: a
       point exactly on the low-x/low-y edge counts as outside while one on the
       high-x/high-y edge counts as inside.  That is a consistent tiling rule
       but it is **not mirror-symmetric**, so a polygon whose edges land exactly
       on grid nodes — the usual case, since holes are specified at round
       coordinates — comes out shifted by half a cell.  Callers that carve
       geometry should pass a small positive tolerance;
       :func:`identify_hole_nodes` does so by default.

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
    >>> square = [(3.0, 3.0), (7.0, 3.0), (7.0, 7.0), (3.0, 7.0)]
    >>> point_in_polygon((3.0, 5.0), square)      # on the low edge: excluded
    False
    >>> point_in_polygon((3.0, 5.0), square, 1e-9)  # closed region: included
    True
    """
    if edge_tolerance > 0.0:
        n_vertices = len(vertices)
        for index in range(n_vertices):
            start = vertices[index]
            end = vertices[(index + 1) % n_vertices]
            if _distance_to_segment(point, start, end) <= edge_tolerance:
                return True

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
    edge_tolerance: Optional[float] = None,
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
    edge_tolerance : float, optional
        Nodes within this distance of the polygon boundary are carved out.
        Defaults to ``1e-9 × min(grid_spacing_x, grid_spacing_y)``, which takes
        the **closed** region: a hole given as ``[3, 7]`` removes the nodes at
        ``x = 3`` and ``x = 7`` and everything between.

    Returns
    -------
    hole_mask : ndarray, shape (Nx+1, Ny+1, Nz+1)
        Boolean mask: True for nodes inside the hole

    Notes
    -----
    - Uses point-in-polygon test on full grid (includes boundaries)
    - The hole is extruded vertically through z_range
    - Complexity: O((Nx+1) × (Ny+1) × n_vertices) - acceptable for typical grids

    The default tolerance exists to make the carved geometry **mirror
    symmetric**.  Bare ray casting is half-open — the low edges fall outside and
    the high edges inside — so a hole centred in the film comes out displaced by
    half a cell, and every symmetry of the device is broken by that much.  Pass
    ``edge_tolerance=0.0`` to recover the raw half-open behaviour.

    Examples
    --------
    >>> square = [(5.0, 5.0), (15.0, 5.0), (15.0, 15.0), (5.0, 15.0)]
    >>> mask = identify_hole_nodes(square, (0, 5), 1.0, 1.0, 20, 20, 10)
    >>> assert mask.shape == (21, 21, 11)
    >>> bool(mask[5, 10, 0]) and bool(mask[15, 10, 0])  # both edges carved
    True
    """
    hole_mask = np.zeros((Nx + 1, Ny + 1, Nz + 1), dtype=bool)

    if edge_tolerance is None:
        edge_tolerance = 1e-9 * min(grid_spacing_x, grid_spacing_y)

    z_min, z_max = z_range

    xs = np.arange(Nx + 1, dtype=np.float64) * grid_spacing_x
    ys = np.arange(Ny + 1, dtype=np.float64) * grid_spacing_y
    plane = _points_in_polygon_grid(xs, ys, vertices, edge_tolerance)

    k_hi = min(z_max + 1, Nz + 1)
    if k_hi > z_min:
        hole_mask[:, :, z_min:k_hi] = plane[:, :, None]

    return hole_mask


def _points_in_polygon_grid(
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    vertices: list[tuple[float, float]],
    edge_tolerance: float,
) -> NDArray[np.bool_]:
    """Vectorised :func:`point_in_polygon` over the outer product ``xs × ys``.

    Returns a ``(len(xs), len(ys))`` boolean mask.  Same ray-casting rule and
    same edge tolerance as the scalar function, evaluated one polygon edge at a
    time over the whole grid instead of one grid node at a time over the whole
    polygon — which is what made carving a hole into a large film cost minutes.
    """
    x = xs[:, None]
    y = ys[None, :]
    inside = np.zeros((xs.size, ys.size), dtype=bool)

    n = len(vertices)
    for index in range(n):
        p1x, p1y = vertices[index]
        p2x, p2y = vertices[(index + 1) % n]

        crosses = (
            (y > min(p1y, p2y)) & (y <= max(p1y, p2y)) & (x <= max(p1x, p2x))
        )
        if p1y != p2y:
            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
            crosses &= (p1x == p2x) | (x <= xinters)
        else:
            # A horizontal edge can never satisfy both y-tests, so this branch
            # contributes nothing; keeping it explicit mirrors the scalar code.
            crosses &= p1x == p2x
        inside ^= crosses

    if edge_tolerance > 0.0:
        for index in range(n):
            ax, ay = vertices[index]
            bx, by = vertices[(index + 1) % n]
            dx, dy = bx - ax, by - ay
            length_squared = dx * dx + dy * dy
            if length_squared == 0.0:
                dist = np.hypot(x - ax, y - ay)
            else:
                t = ((x - ax) * dx + (y - ay) * dy) / length_squared
                t = np.clip(t, 0.0, 1.0)
                dist = np.hypot(x - (ax + t * dx), y - (ay + t * dy))
            inside |= dist <= edge_tolerance

    return inside


def _links_from_mask(
    crossing: NDArray[np.bool_],
    axis_order: tuple[int, int, int],
    mj: int,
    mk: int,
    is_3d: bool,
) -> NDArray[np.int64]:
    """Linear indices of the ``True`` entries of *crossing*, in loop order.

    *crossing* is indexed ``[i, j, k]``.  ``axis_order`` gives the nesting of
    the loops the scalar implementation used (outermost first), so that the
    returned order matches it exactly.
    """
    if not crossing.any():
        return np.array([], dtype=np.int64)
    found = np.nonzero(crossing.transpose(axis_order))
    coords = [None, None, None]
    for position, axis in enumerate(axis_order):
        coords[axis] = found[position]
    i, j, k = coords
    m = j.astype(np.int64) * mj + i
    if is_3d:
        m += k.astype(np.int64) * mk
    return m


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
    mj = Nx + 1
    mk = (Nx + 1) * (Ny + 1)

    # A link crosses the boundary when its two endpoints disagree, which is one
    # shifted XOR of the mask against itself.  The scalar loops these replace
    # ran over every node of the grid in Python.
    if direction == 'x':
        crossing = np.zeros_like(hole_mask)
        crossing[:Nx, :, :] = hole_mask[:Nx, :, :] ^ hole_mask[1:, :, :]
        # scalar loop nesting was k, j, i
        return _links_from_mask(crossing, (2, 1, 0), mj, mk, is_3d)

    if direction == 'y':
        crossing = np.zeros_like(hole_mask)
        crossing[:, :Ny, :] = hole_mask[:, :Ny, :] ^ hole_mask[:, 1:, :]
        # scalar loop nesting was k, i, j
        return _links_from_mask(crossing, (2, 0, 1), mj, mk, is_3d)

    if direction == 'z':
        crossing = np.zeros_like(hole_mask)
        crossing[:, :, :Nz] = hole_mask[:, :, :Nz] ^ hole_mask[:, :, 1:]
        # scalar loop nesting was i, j, k
        return _links_from_mask(crossing, (0, 1, 2), mj, mk, is_3d)

    raise ValueError(f"Invalid direction '{direction}'. Use 'x', 'y', or 'z'.")


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
    mj = Nx + 1
    mk = (Nx + 1) * (Ny + 1)

    def _normal(axis: int, n_links: int) -> NDArray[np.bool_]:
        """Boundary-crossing links along *axis* with no crossing neighbour.

        ``crossing[..., i, ...]`` is the link from index ``i`` to ``i+1`` along
        *axis*; valid links are ``i < n_links``.  A link is tangential when the
        link before or after it along the same axis also crosses, so the normal
        ones are those whose two axis-neighbours are both non-crossing.
        """
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[axis] = slice(0, n_links)
        hi[axis] = slice(1, n_links + 1)
        crossing = np.zeros_like(hole_mask)
        crossing[tuple(lo)] = hole_mask[tuple(lo)] ^ hole_mask[tuple(hi)]

        prev_lo = [slice(None)] * 3
        prev_hi = [slice(None)] * 3
        prev_lo[axis] = slice(1, n_links)      # link i, for i >= 1
        prev_hi[axis] = slice(0, n_links - 1)  # its neighbour, link i-1
        next_lo = [slice(None)] * 3
        next_hi = [slice(None)] * 3
        next_lo[axis] = slice(0, n_links - 1)  # link i, for i <= n_links - 2
        next_hi[axis] = slice(1, n_links)      # its neighbour, link i+1

        has_neighbour = np.zeros_like(hole_mask)
        if n_links > 1:
            has_neighbour[tuple(prev_lo)] |= crossing[tuple(prev_hi)]
            has_neighbour[tuple(next_lo)] |= crossing[tuple(next_hi)]
        return crossing & ~has_neighbour

    if direction == 'x':
        normal = _normal(0, Nx)
        if not is_3d:
            normal[:, :, 1:] = False
        # scalar loop nesting was k, j, i
        return _links_from_mask(normal, (2, 1, 0), mj, mk, is_3d)

    if direction == 'y':
        normal = _normal(1, Ny)
        if not is_3d:
            normal[:, :, 1:] = False
        # scalar loop nesting was k, i, j
        return _links_from_mask(normal, (2, 0, 1), mj, mk, is_3d)

    if direction == 'z':
        normal = _normal(2, Nz)
        # the scalar z branch always used 3-D linear indices
        # scalar loop nesting was i, j, k
        return _links_from_mask(normal, (0, 1, 2), mj, mk, True)

    raise ValueError(f"Invalid direction '{direction}'. Use 'x', 'y', or 'z'.")


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

    x = np.arange(Nx + 1, dtype=np.float64)[:, None] * grid_spacing_x
    y = np.arange(Ny + 1, dtype=np.float64)[None, :] * grid_spacing_y

    # Distance from center.  The comparison is inclusive so that nodes
    # landing exactly on the rim are carved on every side alike; a
    # strict inequality is symmetric for a node-centred circle but not
    # for one centred between nodes.
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    plane = dist <= radius * (1.0 + 1e-9)

    k_hi = min(z_max + 1, Nz + 1)
    if k_hi > z_min:
        hole_mask[:, :, z_min:k_hi] = plane[:, :, None]

    return hole_mask

"""Grid index construction — vectorised rewrite of ``contruct_indices.m``.

Builds all the index arrays that map between the interior nodes and the
full ``(Nx+1) × (Ny+1) × (Nz+1)`` grid, including boundary-face and
edge masks.

Naming conventions
------------------
Each axis (x, y, z) has four face-index arrays that come in two flavours:

*   **"all"** variants span every ``(j, k)`` pair on that face (including
    edge/corner nodes).  They are named ``x_face_lo``, ``x_first``,
    ``x_last``, ``x_face_hi``.
*   **"inner"** variants only include nodes whose *other two* indices are
    strictly interior (no edges).  They are named ``x_face_lo_inner``,
    ``x_first_inner``, etc.

``interior_to_full``
    1-D array of full-grid linear indices for the interior nodes (the map
    from the compact interior numbering to the full grid).
``bfield_interior``
    Subset of *interior* indices (in interior numbering) that are one more
    layer inward — used for evaluating *B* = curl(*A*) where neighbours are
    needed.
``x_normal_bc_mask`` / ``y_…`` / ``z_…``
    Full-grid indices of every node on the two faces perpendicular to that
    axis.  Used to zero out the normal link-variable component on the
    boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ..core.parameters import SimulationParameters


# ---------------------------------------------------------------------------
# Helper: linear index on the full (Nx+1)×(Ny+1)×(Nz+1) grid
# ---------------------------------------------------------------------------

def _linear_index(
    i: NDArray[np.intp],
    j: NDArray[np.intp],
    k: NDArray[np.intp],
    mj: int,
    mk: int,
) -> NDArray[np.intp]:
    """Convert (i, j, k) grid coordinates to flat C-order linear indices."""
    return i + mj * j + mk * k


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class GridIndices:
    """Pre-computed index arrays for the structured Cartesian grid.

    ``*_face_lo``  — low-side ghost face  (i = 0 / j = 0 / k = 0)
    ``*_first``    — first interior layer (i = 1 / j = 1 / k = 1)
    ``*_last``     — last interior layer  (i = Nx-1 / j = Ny-1 / k = Nz-1)
    ``*_face_hi``  — high-side ghost face (i = Nx / j = Ny / k = Nz)

    "inner" variants restrict the *other* two indices to strictly interior
    values (edges excluded).

    ``*_normal_bc_mask`` — union of both boundary faces perpendicular to that
    axis, used to zero out the normal link-variable component.
    """

    # -- Interior-to-full mapping -------------------------------------------
    interior_to_full: NDArray[np.intp] = field(
        default_factory=lambda: np.array([], dtype=np.intp)
    )

    # -- B-field interior (one extra layer inward, in interior numbering) ---
    bfield_interior: NDArray[np.intp] = field(
        default_factory=lambda: np.array([], dtype=np.intp)
    )

    # -- x-axis face indices (all j, k) ------------------------------------
    x_face_lo: NDArray[np.intp] = field(default_factory=lambda: np.array([], dtype=np.intp))
    x_first: NDArray[np.intp] = field(default_factory=lambda: np.array([], dtype=np.intp))
    x_last: NDArray[np.intp] = field(default_factory=lambda: np.array([], dtype=np.intp))
    x_face_hi: NDArray[np.intp] = field(default_factory=lambda: np.array([], dtype=np.intp))

    # -- x-axis face indices (inner j, k only) -----------------------------
    x_face_lo_inner: NDArray[np.intp] = field(default_factory=lambda: np.array([], dtype=np.intp))
    x_first_inner: NDArray[np.intp] = field(default_factory=lambda: np.array([], dtype=np.intp))
    x_last_inner: NDArray[np.intp] = field(default_factory=lambda: np.array([], dtype=np.intp))
    x_face_hi_inner: NDArray[np.intp] = field(default_factory=lambda: np.array([], dtype=np.intp))

    # -- y-axis face indices -----------------------------------------------
    y_face_lo: NDArray[np.intp] = field(default_factory=lambda: np.array([], dtype=np.intp))
    y_first: NDArray[np.intp] = field(default_factory=lambda: np.array([], dtype=np.intp))
    y_last: NDArray[np.intp] = field(default_factory=lambda: np.array([], dtype=np.intp))
    y_face_hi: NDArray[np.intp] = field(default_factory=lambda: np.array([], dtype=np.intp))

    y_face_lo_inner: NDArray[np.intp] = field(default_factory=lambda: np.array([], dtype=np.intp))
    y_first_inner: NDArray[np.intp] = field(default_factory=lambda: np.array([], dtype=np.intp))
    y_last_inner: NDArray[np.intp] = field(default_factory=lambda: np.array([], dtype=np.intp))
    y_face_hi_inner: NDArray[np.intp] = field(default_factory=lambda: np.array([], dtype=np.intp))

    # -- z-axis face indices (empty for 2-D) --------------------------------
    z_face_lo: NDArray[np.intp] = field(default_factory=lambda: np.array([], dtype=np.intp))
    z_first: NDArray[np.intp] = field(default_factory=lambda: np.array([], dtype=np.intp))
    z_last: NDArray[np.intp] = field(default_factory=lambda: np.array([], dtype=np.intp))
    z_face_hi: NDArray[np.intp] = field(default_factory=lambda: np.array([], dtype=np.intp))

    z_face_lo_inner: NDArray[np.intp] = field(default_factory=lambda: np.array([], dtype=np.intp))
    z_first_inner: NDArray[np.intp] = field(default_factory=lambda: np.array([], dtype=np.intp))
    z_last_inner: NDArray[np.intp] = field(default_factory=lambda: np.array([], dtype=np.intp))
    z_face_hi_inner: NDArray[np.intp] = field(default_factory=lambda: np.array([], dtype=np.intp))

    # -- Boundary-normal masks (union of both faces ⊥ to that axis) ---------
    x_normal_bc_mask: NDArray[np.intp] = field(default_factory=lambda: np.array([], dtype=np.intp))
    y_normal_bc_mask: NDArray[np.intp] = field(default_factory=lambda: np.array([], dtype=np.intp))
    z_normal_bc_mask: NDArray[np.intp] = field(default_factory=lambda: np.array([], dtype=np.intp))


# ---------------------------------------------------------------------------
# Vectorised constructor
# ---------------------------------------------------------------------------

def construct_indices(params: SimulationParameters) -> GridIndices:
    """Build all index arrays for the given grid parameters.

    All indices are **0-based**.  The construction is fully vectorised
    (no Python loops) for clarity and speed.

    Parameters
    ----------
    params : SimulationParameters

    Returns
    -------
    GridIndices
    """
    Nx, Ny, Nz = params.Nx, params.Ny, params.Nz
    mj = Nx + 1          # stride in j (y) direction
    mk = (Nx + 1) * (Ny + 1)  # stride in k (z) direction
    is_3d = Nz > 1

    # Full ranges for each axis
    all_i = np.arange(Nx + 1, dtype=np.intp)
    all_j = np.arange(Ny + 1, dtype=np.intp)
    all_k = np.arange(Nz + 1, dtype=np.intp) if is_3d else np.array([0], dtype=np.intp)

    # Interior ranges (strictly inside the boundary)
    int_i = np.arange(1, Nx, dtype=np.intp)
    int_j = np.arange(1, Ny, dtype=np.intp)
    int_k = np.arange(1, Nz, dtype=np.intp) if is_3d else np.array([0], dtype=np.intp)

    # =================================================================
    # interior_to_full  (formerly M2)
    # =================================================================
    if is_3d:
        gi, gj, gk = np.meshgrid(int_i, int_j, int_k, indexing="ij")
    else:
        gi, gj = np.meshgrid(int_i, int_j, indexing="ij")
        gk = np.zeros_like(gi)
    interior_to_full = _linear_index(
        gi.ravel(), gj.ravel(), gk.ravel(), mj, mk
    )

    # =================================================================
    # bfield_interior  (formerly M2B) — in *interior* numbering
    # =================================================================
    int_Nx_m1 = Nx - 1  # interior x-extent
    int_Ny_m1 = Ny - 1
    # In interior numbering, index = (i-1) + int_Nx_m1*(j-1) + ...
    # The B-field interior skips the outermost layer of interior nodes,
    # so (i-1) ranges from 0 to Nx-3, i.e., arange(0, Nx-2).
    bi = np.arange(0, Nx - 2, dtype=np.intp)
    bj = np.arange(0, Ny - 2, dtype=np.intp)
    if is_3d:
        bk = np.arange(0, Nz - 2, dtype=np.intp)
        bgi, bgj, bgk = np.meshgrid(bi, bj, bk, indexing="ij")
        bfield_interior = (
            bgi.ravel()
            + int_Nx_m1 * bgj.ravel()
            + int_Nx_m1 * int_Ny_m1 * bgk.ravel()
        )
    else:
        bgi, bgj = np.meshgrid(bi, bj, indexing="ij")
        bfield_interior = bgi.ravel() + int_Nx_m1 * bgj.ravel()

    # =================================================================
    # Helper: build the 4-tuple of face indices for one axis
    # =================================================================
    def _face_indices_for_axis(
        axis: str,
    ) -> tuple[
        NDArray, NDArray, NDArray, NDArray,
        NDArray, NDArray, NDArray, NDArray,
    ]:
        """Return (face_lo, first, last, face_hi,
                   face_lo_inner, first_inner, last_inner, face_hi_inner)
        as flat linear-index arrays for the given axis.
        """
        if axis == "x":
            i_lo, i_first, i_last, i_hi = 0, 1, Nx - 1, Nx
            other1_all, other2_all = all_j, all_k
            other1_int, other2_int = int_j, int_k
            def lin(i_val, o1, o2):
                return _linear_index(
                    np.full_like(o1, i_val), o1, o2, mj, mk
                )
            def lin_mesh(i_val, r1, r2):
                g1, g2 = np.meshgrid(r1, r2, indexing="ij")
                return _linear_index(
                    np.full(g1.size, i_val, dtype=np.intp),
                    g1.ravel(), g2.ravel(), mj, mk,
                )
        elif axis == "y":
            i_lo, i_first, i_last, i_hi = 0, 1, Ny - 1, Ny
            other1_all, other2_all = all_i, all_k
            other1_int, other2_int = int_i, int_k
            def lin(j_val, o1, o2):
                return _linear_index(o1, np.full_like(o1, j_val), o2, mj, mk)
            def lin_mesh(j_val, r1, r2):
                g1, g2 = np.meshgrid(r1, r2, indexing="ij")
                return _linear_index(
                    g1.ravel(),
                    np.full(g1.size, j_val, dtype=np.intp),
                    g2.ravel(), mj, mk,
                )
        else:  # z
            if not is_3d:
                empty = np.array([], dtype=np.intp)
                return empty, empty, empty, empty, empty, empty, empty, empty
            i_lo, i_first, i_last, i_hi = 0, 1, Nz - 1, Nz
            other1_all, other2_all = all_i, all_j
            other1_int, other2_int = int_i, int_j
            def lin(k_val, o1, o2):
                return _linear_index(o1, o2, np.full_like(o1, k_val), mj, mk)
            def lin_mesh(k_val, r1, r2):
                g1, g2 = np.meshgrid(r1, r2, indexing="ij")
                return _linear_index(
                    g1.ravel(), g2.ravel(),
                    np.full(g1.size, k_val, dtype=np.intp),
                    mj, mk,
                )

        face_lo = lin_mesh(i_lo, other1_all, other2_all)
        first   = lin_mesh(i_first, other1_all, other2_all)
        last    = lin_mesh(i_last, other1_all, other2_all)
        face_hi = lin_mesh(i_hi, other1_all, other2_all)

        face_lo_inner = lin_mesh(i_lo, other1_int, other2_int)
        first_inner   = lin_mesh(i_first, other1_int, other2_int)
        last_inner    = lin_mesh(i_last, other1_int, other2_int)
        face_hi_inner = lin_mesh(i_hi, other1_int, other2_int)

        return (
            face_lo, first, last, face_hi,
            face_lo_inner, first_inner, last_inner, face_hi_inner,
        )

    (
        x_face_lo, x_first, x_last, x_face_hi,
        x_face_lo_inner, x_first_inner, x_last_inner, x_face_hi_inner,
    ) = _face_indices_for_axis("x")

    (
        y_face_lo, y_first, y_last, y_face_hi,
        y_face_lo_inner, y_first_inner, y_last_inner, y_face_hi_inner,
    ) = _face_indices_for_axis("y")

    (
        z_face_lo, z_first, z_last, z_face_hi,
        z_face_lo_inner, z_first_inner, z_last_inner, z_face_hi_inner,
    ) = _face_indices_for_axis("z")

    # =================================================================
    # Normal-component BC masks
    # =================================================================
    def _normal_mask(axis: str) -> NDArray[np.intp]:
        """Full-grid indices on both faces perpendicular to *axis*."""
        if axis == "x":
            vals = [0, Nx - 1]
            oj, ok = all_j, all_k
            def _lin(v, o1, o2):
                g1, g2 = np.meshgrid(o1, o2, indexing="ij")
                return _linear_index(
                    np.full(g1.size, v, dtype=np.intp),
                    g1.ravel(), g2.ravel(), mj, mk,
                )
        elif axis == "y":
            vals = [0, Ny - 1]
            oj, ok = all_i, all_k
            def _lin(v, o1, o2):
                g1, g2 = np.meshgrid(o1, o2, indexing="ij")
                return _linear_index(
                    g1.ravel(),
                    np.full(g1.size, v, dtype=np.intp),
                    g2.ravel(), mj, mk,
                )
        else:
            if not is_3d:
                return np.array([], dtype=np.intp)
            vals = [0, Nz - 1]
            oj, ok = all_i, all_j
            def _lin(v, o1, o2):
                g1, g2 = np.meshgrid(o1, o2, indexing="ij")
                return _linear_index(
                    g1.ravel(), g2.ravel(),
                    np.full(g1.size, v, dtype=np.intp),
                    mj, mk,
                )
        parts = [_lin(v, oj, ok) for v in vals]
        return np.concatenate(parts)

    x_normal_bc_mask = _normal_mask("x")
    y_normal_bc_mask = _normal_mask("y")
    z_normal_bc_mask = _normal_mask("z")

    return GridIndices(
        interior_to_full=interior_to_full,
        bfield_interior=bfield_interior,
        x_face_lo=x_face_lo, x_first=x_first,
        x_last=x_last, x_face_hi=x_face_hi,
        x_face_lo_inner=x_face_lo_inner, x_first_inner=x_first_inner,
        x_last_inner=x_last_inner, x_face_hi_inner=x_face_hi_inner,
        y_face_lo=y_face_lo, y_first=y_first,
        y_last=y_last, y_face_hi=y_face_hi,
        y_face_lo_inner=y_face_lo_inner, y_first_inner=y_first_inner,
        y_last_inner=y_last_inner, y_face_hi_inner=y_face_hi_inner,
        z_face_lo=z_face_lo, z_first=z_first,
        z_last=z_last, z_face_hi=z_face_hi,
        z_face_lo_inner=z_face_lo_inner, z_first_inner=z_first_inner,
        z_last_inner=z_last_inner, z_face_hi_inner=z_face_hi_inner,
        x_normal_bc_mask=x_normal_bc_mask,
        y_normal_bc_mask=y_normal_bc_mask,
        z_normal_bc_mask=z_normal_bc_mask,
    )

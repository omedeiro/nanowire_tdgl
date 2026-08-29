"""Right-hand side evaluation — Python port of ``eval_f.m``.

Given the state vector X = [ψ; φ_x; φ_y; φ_z] and the boundary-condition
information (applied field, periodic flags) this module returns dX/dt.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from ..core.material import MaterialMap
from ..core.parallel import chunk_count, run_chunks
from ..core.parameters import SimulationParameters
from ..mesh.indices import GridIndices
from ..operators.sparse_operators import chunk_planes, rhs_rows


class BoundaryVectors:
    """Holds the (Bx, By, Bz) boundary vectors for a single evaluation."""

    __slots__ = ("Bx", "By", "Bz")

    def __init__(self, Bx: NDArray, By: NDArray, Bz: NDArray):
        self.Bx = Bx
        self.By = By
        self.Bz = Bz


def _expand_interior_to_full(
    interior_vals: NDArray[np.complexfloating],
    params: SimulationParameters,
    idx: GridIndices,
    out: NDArray[np.complex128] | None = None,
) -> NDArray[np.complex128]:
    """Scatter interior values into a full-grid vector (0 elsewhere).

    *out* may be a reused buffer; it is zeroed first, which the boundary
    conditions rely on because they accumulate into the ghost faces.
    """
    if out is None:
        full = np.zeros(params.dim_x, dtype=interior_vals.dtype)
    else:
        full = out
        full[:] = 0.0
    full[idx.interior_to_full] = interior_vals
    return full


def _node_coords(
    nodes: NDArray[np.intp], params: SimulationParameters
) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.intp]]:
    """Return ``(i, j, k)`` grid coordinates of full-grid linear indices."""
    i = nodes % params.mj
    j = (nodes // params.mj) % (params.Ny + 1)
    k = nodes // params.mk if params.is_3d else np.zeros_like(nodes)
    return i, j, k


def _shared_edge_weight(
    nodes: NDArray[np.intp],
    params: SimulationParameters,
    axis: int,
    limit: int,
) -> NDArray[np.float64]:
    """Weight of the applied-field term on a *hi* boundary face.

    Every plaquette on a boundary face is given the applied flux by offsetting
    the single ghost link that closes it.  Where two *hi* faces meet, the same
    plaquette is closed by two ghost links — one from each face — and a full
    offset on both would give it twice the applied field (and, because that
    plaquette is a live interior plaquette, an unbalanced curl-curl force that
    makes its link variables drift without bound).  Splitting the offset evenly
    between the two ghost links gives the plaquette the correct flux while
    keeping the treatment of the two axes symmetric.

    Parameters
    ----------
    nodes : ndarray
        Full-grid indices of the boundary face being written to.
    params : SimulationParameters
    axis : int
        Axis (0=x, 1=y, 2=z) of the *other* hi face that shares the plaquette.
    limit : int
        Index of the last interior node along ``axis``.
    """
    coords = _node_coords(nodes, params)
    weight = np.ones(len(nodes), dtype=np.float64)
    weight[coords[axis] == limit] = 0.5
    return weight


def _shared_edge_weight_cached(
    idx: GridIndices,
    nodes: NDArray[np.intp],
    params: SimulationParameters,
    axis: int,
    limit: int,
    key: str,
) -> NDArray[np.float64]:
    """:func:`_shared_edge_weight`, computed once per device.

    The weights depend only on the grid, and the boundary conditions ask for
    six of them on every right-hand-side evaluation.
    """
    st = idx.neighbours(params)
    cached = st.get(key)
    if cached is None:
        cached = _shared_edge_weight(nodes, params, axis, limit)
        st[key] = cached
    return cached


def _apply_boundary_conditions(
    x: NDArray[np.complex128],
    y1: NDArray[np.complex128],
    y2: NDArray[np.complex128],
    y3: NDArray[np.complex128],
    params: SimulationParameters,
    idx: GridIndices,
    u: BoundaryVectors,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Apply periodic or zero-current + magnetic-field BCs to full-grid vectors.

    This is a direct translation of the boundary-condition blocks in ``eval_f.m``.
    The vectors are modified **in place** and also returned.

    Every block writes to a *ghost face* (i = 0 or Nx, j = 0 or Ny, k = 0 or Nz)
    and reads from a *first/last interior layer* whose other two indices are
    strictly interior, so no block can read a value another block has already
    written.  The MATLAB original kept ``x00``/``y100``/``y200``/``y300`` copies
    to guarantee that; on this grid they are four full-grid copies per
    evaluation that never change an answer, so the reads are taken from the
    live arrays instead.  ``test_boundary_conditions_read_disjoint_indices``
    pins the disjointness the shortcut rests on.
    """
    hx, hy, hz = params.hx, params.hy, params.hz

    # Zero out normal-component link variables at boundary faces.
    # This must happen BEFORE we make the copies so that y100/y200/y300
    # also carry the zeroed boundary values (matching the MATLAB code).
    y1[idx.x_normal_bc_mask] = 0.0
    y2[idx.y_normal_bc_mask] = 0.0
    if params.is_3d:
        y3[idx.z_normal_bc_mask] = 0.0

    # The pre-update values the BC blocks reference.  Aliases, not copies —
    # see the disjointness argument in the docstring.
    x00, y100, y200, y300 = x, y1, y2, y3

    # --- x boundaries -------------------------------------------------------
    if params.periodic_x:
        x[idx.x_face_lo_inner] += x00[idx.x_last_inner]
        x[idx.x_face_hi_inner] += x00[idx.x_first_inner]
        y1[idx.x_face_lo_inner] += y100[idx.x_last_inner]
        y1[idx.x_face_hi_inner] += y100[idx.x_first_inner]
    else:
        # Zero-current on x
        x[idx.x_face_lo_inner] += x00[idx.x_first_inner] * np.exp(-1j * y100[idx.x_face_lo_inner])
        x[idx.x_face_hi_inner] += x00[idx.x_last_inner] * np.exp(1j * y100[idx.x_last_inner])
        # Magnetic-field x BCs (eq. 37 in report)
        wz = _shared_edge_weight_cached(
            idx, idx.x_face_hi_inner, params, 1, params.Ny - 1, "_w_x_hi_z")
        y2[idx.x_face_lo_inner] += -u.Bz[idx.x_face_lo_inner] * hx * hy + y200[idx.x_first_inner]
        y2[idx.x_face_hi_inner] += (
            wz * u.Bz[idx.x_face_hi_inner] * hx * hy + y200[idx.x_last_inner]
        )
        y3[idx.x_face_lo_inner] += u.By[idx.x_face_lo_inner] * hz * hx + y300[idx.x_first_inner]
        if params.is_3d:
            wy = _shared_edge_weight_cached(
                idx, idx.x_face_hi_inner, params, 2, params.Nz - 1, "_w_x_hi_y")
        else:
            wy = 1.0
        y3[idx.x_face_hi_inner] += (
            -wy * u.By[idx.x_face_hi_inner] * hz * hx + y300[idx.x_last_inner]
        )

    # --- y boundaries -------------------------------------------------------
    if params.periodic_y:
        x[idx.y_face_lo_inner] += x00[idx.y_last_inner]
        x[idx.y_face_hi_inner] += x00[idx.y_first_inner]
        y2[idx.y_face_lo_inner] += y200[idx.y_last_inner]
        y2[idx.y_face_hi_inner] += y200[idx.y_first_inner]
    else:
        x[idx.y_face_lo_inner] += x00[idx.y_first_inner] * np.exp(-1j * y200[idx.y_face_lo_inner])
        x[idx.y_face_hi_inner] += x00[idx.y_last_inner] * np.exp(1j * y200[idx.y_last_inner])
        wz = _shared_edge_weight_cached(
            idx, idx.y_face_hi_inner, params, 0, params.Nx - 1, "_w_y_hi_z")
        y1[idx.y_face_lo_inner] += u.Bz[idx.y_face_lo_inner] * hx * hy + y100[idx.y_first_inner]
        y1[idx.y_face_hi_inner] += (
            -wz * u.Bz[idx.y_face_hi_inner] * hx * hy + y100[idx.y_last_inner]
        )
        y3[idx.y_face_lo_inner] += -u.Bx[idx.y_face_lo_inner] * hy * hz + y300[idx.y_first_inner]
        if params.is_3d:
            wx = _shared_edge_weight_cached(
                idx, idx.y_face_hi_inner, params, 2, params.Nz - 1, "_w_y_hi_x")
        else:
            wx = 1.0
        y3[idx.y_face_hi_inner] += (
            wx * u.Bx[idx.y_face_hi_inner] * hy * hz + y300[idx.y_last_inner]
        )

    # --- z boundaries -------------------------------------------------------
    if params.is_3d:
        if params.periodic_z:
            x[idx.z_face_lo_inner] += x00[idx.z_last_inner]
            x[idx.z_face_hi_inner] += x00[idx.z_first_inner]
            y3[idx.z_face_lo_inner] += y300[idx.z_last_inner]
            y3[idx.z_face_hi_inner] += y300[idx.z_first_inner]
        else:
            x[idx.z_face_lo_inner] += x00[idx.z_first_inner] * np.exp(
                -1j * y300[idx.z_face_lo_inner]
            )
            x[idx.z_face_hi_inner] += x00[idx.z_last_inner] * np.exp(1j * y300[idx.z_last_inner])
            wy = _shared_edge_weight_cached(
                idx, idx.z_face_hi_inner, params, 0, params.Nx - 1, "_w_z_hi_y")
            wx = _shared_edge_weight_cached(
                idx, idx.z_face_hi_inner, params, 1, params.Ny - 1, "_w_z_hi_x")
            y1[idx.z_face_lo_inner] += (
                -u.By[idx.z_face_lo_inner] * hz * hx + y100[idx.z_first_inner]
            )
            y1[idx.z_face_hi_inner] += (
                wy * u.By[idx.z_face_hi_inner] * hz * hx + y100[idx.z_last_inner]
            )
            y2[idx.z_face_lo_inner] += u.Bx[idx.z_face_lo_inner] * hy * hz + y200[idx.z_first_inner]
            y2[idx.z_face_hi_inner] += (
                -wx * u.Bx[idx.z_face_hi_inner] * hy * hz + y200[idx.z_last_inner]
            )

    # NOTE: We do NOT enforce φ=0 on hole boundaries (unlike external boundaries).
    # Physical reasoning:
    # - Holes represent vacuum/insulator regions where ψ=0 (enforced by material mask)
    # - The TDGL equations naturally prevent current flow into regions where ψ=0
    # - Enforcing φ=0 at hole boundaries artificially prevents flux trapping by
    #   forcing ∮∇φ·dl = 0 (no phase winding around hole)
    # - For flux quantization, we need φ free to vary around hole boundary
    # - The zero-current BC (J_n = 0) emerges naturally from ψ=0, not from φ=0
    #
    # This is fundamentally different from external simulation boundaries, where
    # φ=0 is needed to prevent numerical artifacts at infinity.

    return x, y1, y2, y3


def _strip_boundary_rows(L: sp.csr_matrix, idx: GridIndices) -> NDArray:
    """Extract only the rows corresponding to interior nodes."""
    return L[idx.interior_to_full, :]


def eval_f(
    X: NDArray[np.complexfloating],
    params: SimulationParameters,
    idx: GridIndices,
    u: BoundaryVectors,
    material: Optional[MaterialMap] = None,
) -> NDArray[np.complex128]:
    """Evaluate the full TDGL right-hand side F(X).

    Parameters
    ----------
    X : ndarray, shape (n_state,)
        Flat state vector [ψ; φ_x; φ_y; φ_z].
    params : SimulationParameters
    idx : GridIndices
    u : BoundaryVectors
        Boundary magnetic-field vectors for the current time step.
    material : MaterialMap, optional
        Per-node material properties.  When ``None`` the uniform
        ``params.kappa`` is used everywhere and all nodes are
        superconducting.

    Returns
    -------
    F : ndarray, shape (n_state,)
        Time derivative dX/dt.
    """
    n = params.n_interior
    # Everything downstream follows the state's precision: a complex64 run
    # gets complex64 scratch, output and material coefficients, and never
    # silently promotes back to double part-way through.
    dtype = np.dtype(X.dtype)
    if dtype not in (np.complex64, np.complex128):
        dtype = np.dtype(np.complex128)

    # Unpack interior values
    psi_int = X[:n]
    phi_x_int = X[n : 2 * n]
    phi_y_int = X[2 * n : 3 * n]
    phi_z_int = X[3 * n : 4 * n] if params.is_3d else np.zeros(n, dtype=dtype)

    # Expand to full grid, into buffers the device lends us.  Allocating and
    # first-touching four full-grid arrays per evaluation is hundreds of
    # megabytes of page faults on a large mesh, repeated tens of thousands of
    # times in a run.
    work = idx.workspace(params, 4, dtype=dtype)
    try:
        x = _expand_interior_to_full(psi_int, params, idx, work[0])
        y1 = _expand_interior_to_full(phi_x_int, params, idx, work[1])
        y2 = _expand_interior_to_full(phi_y_int, params, idx, work[2])
        y3 = _expand_interior_to_full(phi_z_int, params, idx, work[3])

        # Apply BCs (modifies in place)
        x, y1, y2, y3 = _apply_boundary_conditions(x, y1, y2, y3, params, idx, u)

        # Evaluate the interior stencil.  ``rhs_rows`` is the matrix-free
        # equivalent of building ``construct_LPSI_*`` / ``construct_LPHI_*``,
        # slicing out the interior rows and multiplying — the assembly and the
        # row slice dominated the cost and produced the same sparsity pattern
        # on every call.  It is split across threads by interior node; the
        # kernel is memory-bound, so the cores buy bandwidth.
        n_blocks = 4 if params.is_3d else 3
        F = np.empty(n_blocks * n, dtype=dtype)
        blocks = tuple(F[i * n : (i + 1) * n] for i in range(n_blocks))
        if not params.is_3d:
            blocks = blocks + (np.empty(0, dtype=dtype),)

        # Split by slab of interior planes: a chunk has to stay a rectangular
        # box for the stencil to be slices rather than gathers, and a slab of
        # the slowest axis is contiguous in memory.
        n_planes = chunk_planes(params)
        run_chunks(
            lambda pl: rhs_rows(
                x, y1, y2, y3, params, idx, material, pl, blocks
            ),
            n_planes,
            min(chunk_count(n), n_planes),
        )
    finally:
        idx.release_workspace(params, work)

    # NOTE: We do NOT enforce dφ/dt=0 on hole boundaries.
    # Physical reasoning (same as in _apply_boundary_conditions):
    # - The ψ=0 material mask inside holes naturally prevents current flow
    # - The TDGL equation dφ/dt ∝ J·∇ψ* automatically gives dφ/dt→0 where ψ=0
    # - Explicit freezing of dφ/dt would prevent flux trapping by blocking phase winding
    # - We need φ free to evolve around hole boundaries for ∮∇φ·dl = n·2π (flux quantization)
    return F

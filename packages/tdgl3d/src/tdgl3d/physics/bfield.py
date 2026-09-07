"""Evaluate the magnetic field from the link variables — port of ``eval_Bfield.m``."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..core.parameters import SimulationParameters
from ..mesh.indices import GridIndices
from .rhs import _expand_interior_to_full


def eval_bfield_full(
    phi_x_full: NDArray[np.complexfloating],
    phi_y_full: NDArray[np.complexfloating],
    phi_z_full: NDArray[np.complexfloating],
    params: SimulationParameters,
    idx: GridIndices,
) -> tuple[NDArray, NDArray, NDArray]:
    """Compute B = curl(A) at ALL interior nodes using full-grid link variables.

    This function computes the magnetic field at all interior nodes (including
    holes/insulators) by using full-grid arrays that include boundary ghost nodes.
    This allows the curl stencil to access neighbors safely.

    Parameters
    ----------
    phi_x_full : ndarray, shape (n_full,)
        x-component of link variables on full grid (including boundaries)
    phi_y_full : ndarray, shape (n_full,)
        y-component of link variables on full grid
    phi_z_full : ndarray, shape (n_full,)
        z-component of link variables on full grid (zeros for 2D)
    params : SimulationParameters
    idx : GridIndices

    Returns
    -------
    Bx, By, Bz : ndarray, each shape (n_interior,)
        Magnetic field at ALL interior nodes (including holes).

    Notes
    -----
    This function enables B-field visualization in holes by computing curl
    everywhere. The input arrays must be full-grid (not interior-only) so
    that boundary neighbors exist for the curl stencil.

    For typical use, call this after expanding interior state to full grid
    and applying boundary conditions. See examples/sis_square_with_hole.py.
    """
    # Use full-grid indices for interior nodes
    m = idx.interior_to_full
    mj = params.mj  # Full-grid stride in j direction
    mk = params.mk  # Full-grid stride in k direction

    if params.is_3d:
        # B = ∇×A using finite differences on full grid
        # Bx = ∂Az/∂y - ∂Ay/∂z
        Bx = (1.0 / (params.hy * params.hz)) * (
            phi_y_full[m] - phi_y_full[m + mk] - phi_z_full[m] + phi_z_full[m + mj]
        )
        # By = ∂Ax/∂z - ∂Az/∂x
        By = (1.0 / (params.hz * params.hx)) * (
            phi_z_full[m] - phi_z_full[m + 1] - phi_x_full[m] + phi_x_full[m + mk]
        )
        # Bz = ∂Ay/∂x - ∂Ax/∂y
        Bz = (1.0 / (params.hx * params.hy)) * (
            phi_x_full[m] - phi_x_full[m + mj] - phi_y_full[m] + phi_y_full[m + 1]
        )
    else:
        # 2D: only Bz non-zero
        n_int = params.n_interior
        Bx = np.zeros(n_int, dtype=np.float64)
        By = np.zeros(n_int, dtype=np.float64)
        Bz = (1.0 / (params.hx * params.hy)) * (
            phi_x_full[m] - phi_x_full[m + mj] - phi_y_full[m] + phi_y_full[m + 1]
        )

    return np.real(Bx), np.real(By), np.real(Bz)


def eval_bfield(
    state_data: NDArray[np.complexfloating],
    params: SimulationParameters,
    idx: GridIndices,
    full_interior: bool = False,
) -> tuple[NDArray, NDArray, NDArray]:
    """Compute B = curl(A) on the interior grid.

    Parameters
    ----------
    state_data : ndarray, shape (n_state,)
        Flat state vector [ψ, φ_x, φ_y, φ_z].
    params : SimulationParameters
    idx : GridIndices
    full_interior : bool, default False
        If True, compute B at ALL interior nodes (requires expanded full-grid state).
        If False, compute B only at bfield_interior subset (safe for interior-only state).

    Returns
    -------
    Bx, By, Bz : ndarray
        Discrete curl of the link variables.
        If full_interior=False: shape (len(bfield_interior),)
        If full_interior=True: shape (n_interior,)

    Notes
    -----
    The interior link variables are scattered onto the full grid and the curl is
    evaluated by :func:`eval_bfield_full`, so there is exactly one curl stencil in
    the code base.  Boundary link variables are left at zero here, which means the
    applied field is *not* included; ``Solution.bfield`` applies the boundary
    conditions first when the applied field matters.

    ``full_interior=False`` restricts the result to ``idx.bfield_interior`` (one
    layer inward), where every stencil neighbour is itself an interior node and the
    result is therefore independent of the boundary treatment.
    """
    n = params.n_interior

    phi_x = _expand_interior_to_full(state_data[n : 2 * n], params, idx)
    phi_y = _expand_interior_to_full(state_data[2 * n : 3 * n], params, idx)
    if params.is_3d:
        phi_z = _expand_interior_to_full(state_data[3 * n : 4 * n], params, idx)
    else:
        phi_z = np.zeros(params.dim_x, dtype=np.complex128)

    Bx, By, Bz = eval_bfield_full(phi_x, phi_y, phi_z, params, idx)

    if not full_interior:
        Bx, By, Bz = Bx[idx.bfield_interior], By[idx.bfield_interior], Bz[idx.bfield_interior]

    return np.real(Bx), np.real(By), np.real(Bz)

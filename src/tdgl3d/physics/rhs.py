"""Right-hand side evaluation — Python port of ``eval_f.m``.

Given the state vector X = [ψ; φ_x; φ_y; φ_z] and the boundary-condition
information (applied field, periodic flags) this module returns dX/dt.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from ..core.parameters import SimulationParameters
from ..core.material import MaterialMap
from ..mesh.indices import GridIndices
from ..operators.sparse_operators import (
    construct_FPHI_x,
    construct_FPHI_y,
    construct_FPHI_z,
    construct_FPSI,
    construct_LPHI_x,
    construct_LPHI_y,
    construct_LPHI_z,
    construct_LPSI_x,
    construct_LPSI_y,
    construct_LPSI_z,
)


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
) -> NDArray[np.complex128]:
    """Scatter interior values into a full-grid vector (0 elsewhere)."""
    full = np.zeros(params.dim_x, dtype=np.complex128)
    full[idx.interior_to_full] = interior_vals
    return full


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
    """
    hx, hy, hz = params.hx, params.hy, params.hz

    # Zero out normal-component link variables at boundary faces.
    # This must happen BEFORE we make the copies so that y100/y200/y300
    # also carry the zeroed boundary values (matching the MATLAB code).
    y1[idx.x_normal_bc_mask] = 0.0
    y2[idx.y_normal_bc_mask] = 0.0
    if params.is_3d:
        y3[idx.z_normal_bc_mask] = 0.0

    # Keep copies of the (already boundary-zeroed) values for BC referencing
    x00 = x.copy()
    y100 = y1.copy()
    y200 = y2.copy()
    y300 = y3.copy()

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
        y2[idx.x_face_lo_inner] += -u.Bz[idx.x_face_lo_inner] * hx * hy + y200[idx.x_first_inner]
        y2[idx.x_face_hi_inner] += u.Bz[idx.x_face_hi_inner] * hx * hy + y200[idx.x_last_inner]
        y3[idx.x_face_lo_inner] += u.By[idx.x_face_lo_inner] * hz * hx + y300[idx.x_first_inner]
        y3[idx.x_face_hi_inner] += -u.By[idx.x_face_hi_inner] * hz * hx + y300[idx.x_last_inner]

    # --- y boundaries -------------------------------------------------------
    if params.periodic_y:
        x[idx.y_face_lo_inner] += x00[idx.y_last_inner]
        x[idx.y_face_hi_inner] += x00[idx.y_first_inner]
        y2[idx.y_face_lo_inner] += y200[idx.y_last_inner]
        y2[idx.y_face_hi_inner] += y200[idx.y_first_inner]
    else:
        x[idx.y_face_lo_inner] += x00[idx.y_first_inner] * np.exp(-1j * y200[idx.y_face_lo_inner])
        x[idx.y_face_hi_inner] += x00[idx.y_last_inner] * np.exp(1j * y200[idx.y_last_inner])
        y1[idx.y_face_lo_inner] += u.Bz[idx.y_face_lo_inner] * hx * hy + y100[idx.y_first_inner]
        y1[idx.y_face_hi_inner] += -u.Bz[idx.y_face_hi_inner] * hx * hy + y100[idx.y_last_inner]
        y3[idx.y_face_lo_inner] += -u.Bx[idx.y_face_lo_inner] * hy * hz + y300[idx.y_first_inner]
        y3[idx.y_face_hi_inner] += u.Bx[idx.y_face_hi_inner] * hy * hz + y300[idx.y_last_inner]

    # --- z boundaries -------------------------------------------------------
    if params.is_3d:
        if params.periodic_z:
            x[idx.z_face_lo_inner] += x00[idx.z_last_inner]
            x[idx.z_face_hi_inner] += x00[idx.z_first_inner]
            y3[idx.z_face_lo_inner] += y300[idx.z_last_inner]
            y3[idx.z_face_hi_inner] += y300[idx.z_first_inner]
        else:
            x[idx.z_face_lo_inner] += x00[idx.z_first_inner] * np.exp(-1j * y300[idx.z_face_lo_inner])
            x[idx.z_face_hi_inner] += x00[idx.z_last_inner] * np.exp(1j * y300[idx.z_last_inner])
            y1[idx.z_face_lo_inner] += -u.By[idx.z_face_lo_inner] * hz * hx + y100[idx.z_first_inner]
            y1[idx.z_face_hi_inner] += u.By[idx.z_face_hi_inner] * hz * hx + y100[idx.z_last_inner]
            y2[idx.z_face_lo_inner] += u.Bx[idx.z_face_lo_inner] * hy * hz + y200[idx.z_first_inner]
            y2[idx.z_face_hi_inner] += -u.Bx[idx.z_face_hi_inner] * hy * hz + y200[idx.z_last_inner]

    # Zero out normal-component link variables at hole boundaries.
    # This must happen AFTER the outer boundary conditions so that hole BCs
    # take precedence and are not overwritten.
    if idx.hole_x_bc_mask.size > 0:
        y1[idx.hole_x_bc_mask] = 0.0
    if idx.hole_y_bc_mask.size > 0:
        y2[idx.hole_y_bc_mask] = 0.0
    if params.is_3d and idx.hole_z_bc_mask.size > 0:
        y3[idx.hole_z_bc_mask] = 0.0

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
    hx, hy, hz = params.hx, params.hy, params.hz

    # Unpack interior values
    psi_int = X[:n]
    phi_x_int = X[n : 2 * n]
    phi_y_int = X[2 * n : 3 * n]
    phi_z_int = X[3 * n : 4 * n] if params.is_3d else np.zeros(n, dtype=np.complex128)

    # Expand to full grid
    x = _expand_interior_to_full(psi_int, params, idx)
    y1 = _expand_interior_to_full(phi_x_int, params, idx)
    y2 = _expand_interior_to_full(phi_y_int, params, idx)
    y3 = _expand_interior_to_full(phi_z_int, params, idx)

    # Apply BCs (modifies in place)
    x, y1, y2, y3 = _apply_boundary_conditions(x, y1, y2, y3, params, idx, u)

    # Build operators
    LPSIX = construct_LPSI_x(y1, params, idx)
    LPSIY = construct_LPSI_y(y2, params, idx)
    LPSIZ = construct_LPSI_z(y3, params, idx)

    LPHIX = construct_LPHI_x(params, idx, material)
    LPHIY = construct_LPHI_y(params, idx, material)
    LPHIZ = construct_LPHI_z(params, idx, material)

    FPSI = construct_FPSI(x, params, idx, material)
    FPHIX = construct_FPHI_x(x, y1, y2, y3, params, idx, material)
    FPHIY = construct_FPHI_y(x, y1, y2, y3, params, idx, material)
    FPHIZ = construct_FPHI_z(x, y1, y2, y3, params, idx, material)

    # Extract interior rows from the Laplacians
    LPSIX_int = LPSIX[idx.interior_to_full, :]
    LPSIY_int = LPSIY[idx.interior_to_full, :]
    LPSIZ_int = LPSIZ[idx.interior_to_full, :]

    LPHIX_int = LPHIX[idx.interior_to_full, :]
    LPHIY_int = LPHIY[idx.interior_to_full, :]
    LPHIZ_int = LPHIZ[idx.interior_to_full, :]

    # Remove all-zero rows (boundary equations)
    # In the Python version we already extracted interior rows, so skip this step
    # (the MATLAB code removes zero rows because the full matrix has them)

    # dψ/dt
    dPsidt = (LPSIX_int / hx**2 + LPSIY_int / hy**2 + LPSIZ_int / hz**2) @ x + FPSI

    # dφ/dt
    dPhidtX = (LPHIY_int + LPHIZ_int) @ y1 + FPHIX
    dPhidtY = (LPHIX_int + LPHIZ_int) @ y2 + FPHIY
    dPhidtZ = (LPHIX_int + LPHIY_int) @ y3 + FPHIZ
    
    # Enforce zero time derivative at hole boundaries (keeps φ = 0 there)
    # This ensures link variables remain at zero throughout the simulation
    if idx.hole_x_bc_interior.size > 0:
        dPhidtX[idx.hole_x_bc_interior] = 0.0
    if idx.hole_y_bc_interior.size > 0:
        dPhidtY[idx.hole_y_bc_interior] = 0.0
    if params.is_3d and idx.hole_z_bc_interior.size > 0:
        dPhidtZ[idx.hole_z_bc_interior] = 0.0

    if params.is_3d:
        return np.concatenate([dPsidt, dPhidtX, dPhidtY, dPhidtZ])
    else:
        return np.concatenate([dPsidt, dPhidtX, dPhidtY])

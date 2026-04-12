"""Evaluate the magnetic field from the link variables — port of ``eval_Bfield.m``."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..core.parameters import SimulationParameters
from ..mesh.indices import GridIndices


def eval_bfield(
    state_data: NDArray[np.complexfloating],
    params: SimulationParameters,
    idx: GridIndices,
) -> tuple[NDArray, NDArray, NDArray]:
    """Compute B = curl(A) on the interior B-field grid.

    Parameters
    ----------
    state_data : ndarray, shape (n_state,)
        Flat state vector [ψ, φ_x, φ_y, φ_z].
    params : SimulationParameters
    idx : GridIndices

    Returns
    -------
    Bx, By, Bz : ndarray, each shape (len(M2B),)
        Discrete curl of the link variables.
    """
    n = params.n_interior
    mj_int = params.Nx - 1
    mk_int = (params.Nx - 1) * (params.Ny - 1)

    phi_x = state_data[n : 2 * n]
    phi_y = state_data[2 * n : 3 * n]

    m = idx.bfield_interior

    if params.is_3d:
        phi_z = state_data[3 * n : 4 * n]

        Bx = (1.0 / (params.hy * params.hz)) * (
            phi_y[m] - phi_y[m + mk_int] - phi_z[m] + phi_z[m + mj_int]
        )
        By = (1.0 / (params.hz * params.hx)) * (
            phi_z[m] - phi_z[m + 1] - phi_x[m] + phi_x[m + mk_int]
        )
        Bz = (1.0 / (params.hx * params.hy)) * (
            phi_x[m] - phi_x[m + mj_int] - phi_y[m] + phi_y[m + 1]
        )
    else:
        Bx = np.zeros(len(m), dtype=np.float64)
        By = np.zeros(len(m), dtype=np.float64)
        Bz = (1.0 / (params.hx * params.hy)) * (
            phi_x[m] - phi_x[m + mj_int] - phi_y[m] + phi_y[m + 1]
        )

    return np.real(Bx), np.real(By), np.real(Bz)

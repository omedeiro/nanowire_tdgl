"""Sparse operator construction — Python port of the MATLAB ``construct_*`` files.

Each function returns a scipy sparse matrix (CSR).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from ..core.parameters import SimulationParameters
from ..core.material import MaterialMap
from ..mesh.indices import GridIndices


def _kappa_at(m: NDArray[np.intp],
              params: SimulationParameters,
              material: Optional[MaterialMap] = None) -> NDArray[np.float64]:
    """Return κ values at full-grid indices *m*.

    If *material* is ``None`` the uniform ``params.kappa`` is used.
    """
    if material is not None:
        return material.kappa[m]
    return np.full(len(m), params.kappa, dtype=np.float64)


# ---------------------------------------------------------------------------
# LPSI operators  (gauge-covariant Laplacian for ψ)
# ---------------------------------------------------------------------------

def construct_LPSI_x(
    y: NDArray[np.complexfloating],
    params: SimulationParameters,
    idx: GridIndices,
) -> sp.csr_matrix:
    """Laplacian in x for ψ:  e^{-iφ_x} ψ_{i-1} - 2ψ_i + e^{iφ_x} ψ_{i+1}.

    Corresponds to ``construct_LPSIXm.m``.
    """
    N = params.dim_x
    m = idx.interior_to_full

    data_diag = np.full(len(m), -2.0, dtype=np.complex128)
    data_m1 = np.exp(-1j * y[m - 1])
    data_p1 = np.exp(1j * y[m])

    L = sp.csr_matrix((data_diag, (m, m)), shape=(N, N), dtype=np.complex128)
    L += sp.csr_matrix((data_m1, (m, m - 1)), shape=(N, N), dtype=np.complex128)
    L += sp.csr_matrix((data_p1, (m, m + 1)), shape=(N, N), dtype=np.complex128)
    return L


def construct_LPSI_y(
    y: NDArray[np.complexfloating],
    params: SimulationParameters,
    idx: GridIndices,
) -> sp.csr_matrix:
    """Laplacian in y for ψ.  Corresponds to ``construct_LPSIYm.m``."""
    N = params.dim_x
    mj = params.mj
    m = idx.interior_to_full

    data_diag = np.full(len(m), -2.0, dtype=np.complex128)
    L = sp.csr_matrix((data_diag, (m, m)), shape=(N, N), dtype=np.complex128)

    if params.Ny > 1:
        data_pj = np.exp(1j * y[m])
        data_mj = np.exp(-1j * y[m - mj])
        L += sp.csr_matrix((data_pj, (m, m + mj)), shape=(N, N), dtype=np.complex128)
        L += sp.csr_matrix((data_mj, (m, m - mj)), shape=(N, N), dtype=np.complex128)
    return L


def construct_LPSI_z(
    y: NDArray[np.complexfloating],
    params: SimulationParameters,
    idx: GridIndices,
) -> sp.csr_matrix:
    """Laplacian in z for ψ.  Corresponds to ``construct_LPSIZm.m``.

    Returns a zero matrix for 2-D (Nz == 1).
    """
    N = params.dim_x
    mk = params.mk
    m = idx.interior_to_full

    if not params.is_3d:
        return sp.csr_matrix((N, N), dtype=np.complex128)

    data_diag = np.full(len(m), -2.0, dtype=np.complex128)
    L = sp.csr_matrix((data_diag, (m, m)), shape=(N, N), dtype=np.complex128)

    data_pk = np.exp(1j * y[m])
    data_mk = np.exp(-1j * y[m - mk])
    L += sp.csr_matrix((data_pk, (m, m + mk)), shape=(N, N), dtype=np.complex128)
    L += sp.csr_matrix((data_mk, (m, m - mk)), shape=(N, N), dtype=np.complex128)
    return L


# ---------------------------------------------------------------------------
# LPHI operators  (Laplacian for link variables φ)
# ---------------------------------------------------------------------------

def construct_LPHI_x(params: SimulationParameters, idx: GridIndices,
                     material: Optional[MaterialMap] = None) -> sp.csr_matrix:
    """Laplacian cross-terms for φ_x (y and z derivatives).

    Corresponds to ``construct_LPHIXm.m``.
    """
    N = params.dim_x
    mj = params.mj
    mk = params.mk
    m = idx.interior_to_full
    hy, hz = params.hy, params.hz

    kappa_m = _kappa_at(m, params, material)
    coeff_y = kappa_m**2 / hy**2
    coeff_z = kappa_m**2 / hz**2 if params.is_3d else np.zeros_like(kappa_m)

    data_diag = -2.0 * (coeff_y + coeff_z)
    L = sp.csr_matrix((data_diag.astype(np.complex128), (m, m)), shape=(N, N), dtype=np.complex128)

    if params.Ny > 1:
        L += sp.csr_matrix((coeff_y.astype(np.complex128), (m, m + mj)), shape=(N, N))
        L += sp.csr_matrix((coeff_y.astype(np.complex128), (m, m - mj)), shape=(N, N))

    if params.is_3d:
        L += sp.csr_matrix((coeff_z.astype(np.complex128), (m, m + mk)), shape=(N, N))
        L += sp.csr_matrix((coeff_z.astype(np.complex128), (m, m - mk)), shape=(N, N))

    return L


def construct_LPHI_y(params: SimulationParameters, idx: GridIndices,
                     material: Optional[MaterialMap] = None) -> sp.csr_matrix:
    """Laplacian cross-terms for φ_y (x and z derivatives).

    Corresponds to ``construct_LPHIYm.m``.
    """
    N = params.dim_x
    mj = params.mj
    mk = params.mk
    m = idx.interior_to_full
    hx, hz = params.hx, params.hz

    kappa_m = _kappa_at(m, params, material)
    coeff_x = kappa_m**2 / hx**2
    coeff_z = kappa_m**2 / hz**2 if params.is_3d else np.zeros_like(kappa_m)

    data_diag = -2.0 * (coeff_x + coeff_z)
    L = sp.csr_matrix((data_diag.astype(np.complex128), (m, m)), shape=(N, N), dtype=np.complex128)

    if params.Nx > 1:
        L += sp.csr_matrix((coeff_x.astype(np.complex128), (m, m + 1)), shape=(N, N))
        L += sp.csr_matrix((coeff_x.astype(np.complex128), (m, m - 1)), shape=(N, N))

    if params.is_3d:
        L += sp.csr_matrix((coeff_z.astype(np.complex128), (m, m + mk)), shape=(N, N))
        L += sp.csr_matrix((coeff_z.astype(np.complex128), (m, m - mk)), shape=(N, N))

    return L


def construct_LPHI_z(params: SimulationParameters, idx: GridIndices,
                     material: Optional[MaterialMap] = None) -> sp.csr_matrix:
    """Laplacian cross-terms for φ_z (x and y derivatives).

    Corresponds to ``construct_LPHIZm.m``.
    """
    N = params.dim_x
    mj = params.mj
    mk = params.mk
    m = idx.interior_to_full
    hx, hy = params.hx, params.hy

    kappa_m = _kappa_at(m, params, material)
    coeff_x = kappa_m**2 / hx**2
    coeff_y = kappa_m**2 / hy**2

    data_diag = -2.0 * (coeff_x + coeff_y)
    L = sp.csr_matrix((data_diag.astype(np.complex128), (m, m)), shape=(N, N), dtype=np.complex128)

    if params.Nx > 1:
        L += sp.csr_matrix((coeff_x.astype(np.complex128), (m, m + 1)), shape=(N, N))
        L += sp.csr_matrix((coeff_x.astype(np.complex128), (m, m - 1)), shape=(N, N))

    if params.Ny > 1:
        L += sp.csr_matrix((coeff_y.astype(np.complex128), (m, m + mj)), shape=(N, N))
        L += sp.csr_matrix((coeff_y.astype(np.complex128), (m, m - mj)), shape=(N, N))

    return L


# ---------------------------------------------------------------------------
# FPSI / FPHI  forcing (nonlinear source terms)
# ---------------------------------------------------------------------------

def construct_FPSI(
    x: NDArray[np.complexfloating],
    params: SimulationParameters,
    idx: GridIndices,
    material: Optional[MaterialMap] = None,
) -> NDArray[np.complex128]:
    """Nonlinear forcing for ψ:  sc_mask * (1 - |ψ|²) ψ  −  (1-sc_mask) * ψ/τ.

    In superconductor nodes this is the usual ``(1 - |ψ|²)ψ``.
    In insulator nodes the term drives ψ → 0 on a fast relaxation time-scale
    τ_relax (hard-coded to 0.1 for now).

    Corresponds to ``construct_FPSIm.m``.  Returns a dense vector of length
    ``n_interior``.
    """
    m = idx.interior_to_full
    psi_m = x[m]
    gl_term = (1.0 - np.conj(psi_m) * psi_m) * psi_m

    if material is not None:
        sc = material.interior_sc_mask
        tau_relax = 0.1  # fast decay in insulator
        return (sc * gl_term - (1.0 - sc) * psi_m / tau_relax).astype(np.complex128)

    return gl_term.astype(np.complex128)


def construct_FPHI_x(
    x: NDArray[np.complexfloating],
    y1: NDArray[np.complexfloating],
    y2: NDArray[np.complexfloating],
    y3: NDArray[np.complexfloating],
    params: SimulationParameters,
    idx: GridIndices,
    material: Optional[MaterialMap] = None,
) -> NDArray[np.complex128]:
    """Forcing for φ_x.  Corresponds to ``construct_FPHIXm.m``."""
    m = idx.interior_to_full
    mj = params.mj
    mk = params.mk

    kappa_m = _kappa_at(m, params, material)

    supercurrent = np.imag(np.exp(-1j * y1[m]) * np.conj(x[m]) * x[m + 1])

    curl_yz = (kappa_m**2 / params.hy**2) * (
        -y2[m + 1] + y2[m] + y2[m + 1 - mj] - y2[m - mj]
    )

    if params.is_3d:
        curl_yz += (kappa_m**2 / params.hz**2) * (
            -y3[m + 1] + y3[m] + y3[m + 1 - mk] - y3[m - mk]
        )

    return (curl_yz + supercurrent).astype(np.complex128)


def construct_FPHI_y(
    x: NDArray[np.complexfloating],
    y1: NDArray[np.complexfloating],
    y2: NDArray[np.complexfloating],
    y3: NDArray[np.complexfloating],
    params: SimulationParameters,
    idx: GridIndices,
    material: Optional[MaterialMap] = None,
) -> NDArray[np.complex128]:
    """Forcing for φ_y.  Corresponds to ``construct_FPHIYm.m``."""
    m = idx.interior_to_full
    mj = params.mj
    mk = params.mk

    kappa_m = _kappa_at(m, params, material)

    supercurrent = np.imag(np.exp(-1j * y2[m]) * np.conj(x[m]) * x[m + mj])

    curl_xz = (kappa_m**2 / params.hx**2) * (
        -y1[m + mj] + y1[m] + y1[m + mj - 1] - y1[m - 1]
    )

    if params.is_3d:
        curl_xz += (kappa_m**2 / params.hz**2) * (
            -y3[m + mj] + y3[m] + y3[m + mj - mk] - y3[m - mk]
        )

    return (curl_xz + supercurrent).astype(np.complex128)


def construct_FPHI_z(
    x: NDArray[np.complexfloating],
    y1: NDArray[np.complexfloating],
    y2: NDArray[np.complexfloating],
    y3: NDArray[np.complexfloating],
    params: SimulationParameters,
    idx: GridIndices,
    material: Optional[MaterialMap] = None,
) -> NDArray[np.complex128]:
    """Forcing for φ_z.  Corresponds to ``construct_FPHIZm.m``."""
    if not params.is_3d:
        return np.zeros(params.n_interior, dtype=np.complex128)

    m = idx.interior_to_full
    mj = params.mj
    mk = params.mk

    kappa_m = _kappa_at(m, params, material)

    supercurrent = np.imag(np.exp(-1j * y3[m]) * np.conj(x[m]) * x[m + mk])

    curl_xy = (kappa_m**2 / params.hx**2) * (
        -y1[m + mk] + y1[m] + y1[m + mk - 1] - y1[m - 1]
    )
    curl_xy += (kappa_m**2 / params.hy**2) * (
        -y2[m + mk] + y2[m] + y2[m + mk - mj] - y2[m - mj]
    )

    return (curl_xy + supercurrent).astype(np.complex128)

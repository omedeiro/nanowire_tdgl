"""Sparse operator construction — Python port of the MATLAB ``construct_*`` files.

Each function returns a scipy sparse matrix (CSR).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from ..core.material import MaterialMap
from ..core.parameters import SimulationParameters
from ..mesh.indices import GridIndices

#: Relaxation time with which ψ is driven to zero on non-superconducting nodes.
#: Chosen well below the Ginzburg-Landau time scale so that the insulator reaches
#: ψ ≈ 0 quickly without tightening the explicit stability limit, which is set by
#: the much stiffer κ²∇×∇× term.
INSULATOR_RELAXATION_TIME = 0.1


def _kappa_at(m: NDArray[np.intp],
              params: SimulationParameters,
              material: Optional[MaterialMap] = None) -> NDArray[np.float64]:
    """Return the Maxwell-term κ at full-grid indices *m*.

    This is :attr:`MaterialMap.magnetic_kappa`, **not**
    :attr:`MaterialMap.kappa`: the coefficient multiplying
    ``κ²∇×(∇×A)`` is the vacuum field energy and does not vary between
    materials.  When no layer overrode it — the usual case — the
    uniform ``params.kappa`` is used everywhere, insulators, holes and
    vacuum included.  See :class:`~tdgl3d.core.material.MaterialMap`.
    """
    if material is not None and material.magnetic_kappa is not None:
        return material.magnetic_kappa[m]
    return np.full(len(m), params.kappa, dtype=np.float64)


def _forward(a: NDArray, axis: int) -> NDArray:
    """Shift *a* one node forward along *axis*, repeating the last plane."""
    i = np.arange(a.shape[axis])
    return np.take(a, np.minimum(i + 1, a.shape[axis] - 1), axis=axis)


def plaquette_kappa2(
    params: SimulationParameters,
    material: Optional[MaterialMap] = None,
) -> Optional[tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]]:
    """κ² averaged over each plaquette, or ``None`` when it is uniform.

    Returns ``(nu_x, nu_y, nu_z)``, full-grid arrays indexed by the
    plaquette's lower-corner node: ``nu_z[m]`` belongs to the plaquette
    spanning nodes ``m, m+1, m+mj, m+1+mj`` — the one whose flux is
    ``Bz[m]`` — and likewise for the other two normals.

    Why plaquettes.  The magnetic term is the gradient of
    ``Σ_p ν_p B_p²``, and each link borders *two* plaquettes of a given
    normal.  Reading ν once, at the node the link starts from, gives
    both plaquettes the same coefficient, and the result is not the
    gradient of any energy: the operator loses self-adjointness at a
    material interface (measured at 19% of the operator norm for a
    κ = 2/4/2 stack) and the free energy stops being a Lyapunov
    functional.  Evaluating ν per plaquette restores both.

    ``None`` means the coefficient is uniform, where the two forms
    coincide exactly and callers can use the scalar ``params.kappa**2``.
    """
    if material is None or material.magnetic_kappa is None:
        return None

    k2 = material.magnetic_kappa ** 2
    if params.is_3d:
        a = k2.reshape(params.Nz + 1, params.Ny + 1, params.Nx + 1)  # (k, j, i)
        i_ax, j_ax, k_ax = 2, 1, 0
        nu_x = 0.25 * (a + _forward(a, j_ax) + _forward(a, k_ax)
                       + _forward(_forward(a, k_ax), j_ax))
        nu_y = 0.25 * (a + _forward(a, i_ax) + _forward(a, k_ax)
                       + _forward(_forward(a, k_ax), i_ax))
        nu_z = 0.25 * (a + _forward(a, i_ax) + _forward(a, j_ax)
                       + _forward(_forward(a, j_ax), i_ax))
    else:
        a = k2.reshape(params.Ny + 1, params.Nx + 1)                 # (j, i)
        i_ax, j_ax = 1, 0
        nu_z = 0.25 * (a + _forward(a, i_ax) + _forward(a, j_ax)
                       + _forward(_forward(a, j_ax), i_ax))
        nu_x = nu_y = a  # φ_z is absent in 2-D; these are never read

    return nu_x.ravel(), nu_y.ravel(), nu_z.ravel()


def _nu_pair(
    nu: Optional[tuple[NDArray, NDArray, NDArray]],
    normal: int,
    m: NDArray[np.intp],
    stride: int,
    params: SimulationParameters,
    material: Optional[MaterialMap],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The two plaquette coefficients a link at *m* sits between.

    Returns ``(nu_hi, nu_lo)`` = ``(ν[m], ν[m - stride])`` for the
    plaquette family with the given *normal* (0=x, 1=y, 2=z).  Falls
    back to the uniform ``κ²`` when *nu* is ``None``.
    """
    if nu is None:
        k2 = _kappa_at(m, params, material) ** 2
        return k2, k2
    arr = nu[normal]
    return arr[m], arr[m - stride]


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
    data_m1 = np.exp(1j * y[m - 1])
    data_p1 = np.exp(-1j * y[m])

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
        data_pj = np.exp(-1j * y[m])
        data_mj = np.exp(1j * y[m - mj])
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

    data_pk = np.exp(-1j * y[m])
    data_mk = np.exp(1j * y[m - mk])
    L += sp.csr_matrix((data_pk, (m, m + mk)), shape=(N, N), dtype=np.complex128)
    L += sp.csr_matrix((data_mk, (m, m - mk)), shape=(N, N), dtype=np.complex128)
    return L


# ---------------------------------------------------------------------------
# LPHI operators  (Laplacian for link variables φ)
# ---------------------------------------------------------------------------

def construct_LPHI_x(params: SimulationParameters, idx: GridIndices,
                     material: Optional[MaterialMap] = None) -> sp.csr_matrix:
    """Laplacian cross-terms for phi_x (y and z derivatives).

    Together with :func:`construct_FPHI_x` this forms
    ``-(curl (nu curl A))_x``, assembled plaquette by plaquette:

    ::

        dphi_x[m]/dt = -(hx/hy)(nu_z[m] Bz[m] - nu_z[m-mj] Bz[m-mj])
                       +(hx/hz)(nu_y[m] By[m] - nu_y[m-mk] By[m-mk])

    This function carries the phi_x part of those four plaquette
    fluxes; :func:`construct_FPHI_x` carries the phi_y and phi_z parts.

    Corresponds to ``construct_LPHIXm.m``.
    """
    N = params.dim_x
    mj = params.mj
    mk = params.mk
    m = idx.interior_to_full
    hy, hz = params.hy, params.hz

    nu = plaquette_kappa2(params, material)
    # The y-derivative of phi_x closes the two Bz plaquettes above and
    # below the link; the z-derivative closes the two By plaquettes.
    nu_z_hi, nu_z_lo = _nu_pair(nu, 2, m, mj, params, material)
    if params.is_3d:
        nu_y_hi, nu_y_lo = _nu_pair(nu, 1, m, mk, params, material)
    else:
        nu_y_hi = nu_y_lo = np.zeros_like(nu_z_hi)

    coeff_y_hi = nu_z_hi / hy**2
    coeff_y_lo = nu_z_lo / hy**2
    coeff_z_hi = nu_y_hi / hz**2
    coeff_z_lo = nu_y_lo / hz**2

    data_diag = -(coeff_y_hi + coeff_y_lo + coeff_z_hi + coeff_z_lo)
    L = sp.csr_matrix((data_diag.astype(np.complex128), (m, m)), shape=(N, N), dtype=np.complex128)

    if params.Ny > 1:
        L += sp.csr_matrix((coeff_y_hi.astype(np.complex128), (m, m + mj)), shape=(N, N))
        L += sp.csr_matrix((coeff_y_lo.astype(np.complex128), (m, m - mj)), shape=(N, N))

    if params.is_3d:
        L += sp.csr_matrix((coeff_z_hi.astype(np.complex128), (m, m + mk)), shape=(N, N))
        L += sp.csr_matrix((coeff_z_lo.astype(np.complex128), (m, m - mk)), shape=(N, N))

    return L


def construct_LPHI_y(params: SimulationParameters, idx: GridIndices,
                     material: Optional[MaterialMap] = None) -> sp.csr_matrix:
    """Laplacian cross-terms for phi_y (x and z derivatives).

    The phi_y part of

    ::

        dphi_y[m]/dt = -(hy/hz)(nu_x[m] Bx[m] - nu_x[m-mk] Bx[m-mk])
                       +(hy/hx)(nu_z[m] Bz[m] - nu_z[m-1]  Bz[m-1])

    Corresponds to ``construct_LPHIYm.m``.
    """
    N = params.dim_x
    mk = params.mk
    m = idx.interior_to_full
    hx, hz = params.hx, params.hz

    nu = plaquette_kappa2(params, material)
    nu_z_hi, nu_z_lo = _nu_pair(nu, 2, m, 1, params, material)
    if params.is_3d:
        nu_x_hi, nu_x_lo = _nu_pair(nu, 0, m, mk, params, material)
    else:
        nu_x_hi = nu_x_lo = np.zeros_like(nu_z_hi)

    coeff_x_hi = nu_z_hi / hx**2
    coeff_x_lo = nu_z_lo / hx**2
    coeff_z_hi = nu_x_hi / hz**2
    coeff_z_lo = nu_x_lo / hz**2

    data_diag = -(coeff_x_hi + coeff_x_lo + coeff_z_hi + coeff_z_lo)
    L = sp.csr_matrix((data_diag.astype(np.complex128), (m, m)), shape=(N, N), dtype=np.complex128)

    if params.Nx > 1:
        L += sp.csr_matrix((coeff_x_hi.astype(np.complex128), (m, m + 1)), shape=(N, N))
        L += sp.csr_matrix((coeff_x_lo.astype(np.complex128), (m, m - 1)), shape=(N, N))

    if params.is_3d:
        L += sp.csr_matrix((coeff_z_hi.astype(np.complex128), (m, m + mk)), shape=(N, N))
        L += sp.csr_matrix((coeff_z_lo.astype(np.complex128), (m, m - mk)), shape=(N, N))

    return L


def construct_LPHI_z(params: SimulationParameters, idx: GridIndices,
                     material: Optional[MaterialMap] = None) -> sp.csr_matrix:
    """Laplacian cross-terms for phi_z (x and y derivatives).

    The phi_z part of

    ::

        dphi_z[m]/dt = -(hz/hx)(nu_y[m] By[m] - nu_y[m-1]  By[m-1])
                       +(hz/hy)(nu_x[m] Bx[m] - nu_x[m-mj] Bx[m-mj])

    Corresponds to ``construct_LPHIZm.m``.
    """
    N = params.dim_x
    mj = params.mj
    m = idx.interior_to_full
    hx, hy = params.hx, params.hy

    nu = plaquette_kappa2(params, material)
    nu_y_hi, nu_y_lo = _nu_pair(nu, 1, m, 1, params, material)
    nu_x_hi, nu_x_lo = _nu_pair(nu, 0, m, mj, params, material)

    coeff_x_hi = nu_y_hi / hx**2
    coeff_x_lo = nu_y_lo / hx**2
    coeff_y_hi = nu_x_hi / hy**2
    coeff_y_lo = nu_x_lo / hy**2

    data_diag = -(coeff_x_hi + coeff_x_lo + coeff_y_hi + coeff_y_lo)
    L = sp.csr_matrix((data_diag.astype(np.complex128), (m, m)), shape=(N, N), dtype=np.complex128)

    if params.Nx > 1:
        L += sp.csr_matrix((coeff_x_hi.astype(np.complex128), (m, m + 1)), shape=(N, N))
        L += sp.csr_matrix((coeff_x_lo.astype(np.complex128), (m, m - 1)), shape=(N, N))

    if params.Ny > 1:
        L += sp.csr_matrix((coeff_y_hi.astype(np.complex128), (m, m + mj)), shape=(N, N))
        L += sp.csr_matrix((coeff_y_lo.astype(np.complex128), (m, m - mj)), shape=(N, N))

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
    In insulator nodes the term drives ψ → 0 on the relaxation time-scale
    :data:`INSULATOR_RELAXATION_TIME`.

    Corresponds to ``construct_FPSIm.m``.  Returns a dense vector of length
    ``n_interior``.
    """
    m = idx.interior_to_full
    psi_m = x[m]
    gl_term = (1.0 - np.conj(psi_m) * psi_m) * psi_m

    if material is not None:
        sc = material.interior_sc_mask
        return (
            sc * gl_term - (1.0 - sc) * psi_m / INSULATOR_RELAXATION_TIME
        ).astype(np.complex128)

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
    """Forcing for phi_x: supercurrent plus the phi_y/phi_z plaquette terms.

    The magnetic part is the phi_y and phi_z half of

    ::

        -(hx/hy)(nu_z[m] Bz[m] - nu_z[m-mj] Bz[m-mj])
        +(hx/hz)(nu_y[m] By[m] - nu_y[m-mk] By[m-mk])

    with each plaquette carrying its own coefficient; see
    :func:`plaquette_kappa2`.  Corresponds to ``construct_FPHIXm.m``.
    """
    m = idx.interior_to_full
    mj = params.mj
    mk = params.mk

    nu = plaquette_kappa2(params, material)
    nu_z_hi, nu_z_lo = _nu_pair(nu, 2, m, mj, params, material)

    supercurrent = np.imag(np.exp(-1j * y1[m]) * np.conj(x[m]) * x[m + 1])

    curl_yz = (
        nu_z_hi * (y2[m] - y2[m + 1]) + nu_z_lo * (y2[m + 1 - mj] - y2[m - mj])
    ) / params.hy**2

    if params.is_3d:
        nu_y_hi, nu_y_lo = _nu_pair(nu, 1, m, mk, params, material)
        curl_yz = curl_yz + (
            nu_y_hi * (y3[m] - y3[m + 1]) - nu_y_lo * (y3[m - mk] - y3[m + 1 - mk])
        ) / params.hz**2

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
    """Forcing for phi_y.  Corresponds to ``construct_FPHIYm.m``."""
    m = idx.interior_to_full
    mj = params.mj
    mk = params.mk

    nu = plaquette_kappa2(params, material)
    nu_z_hi, nu_z_lo = _nu_pair(nu, 2, m, 1, params, material)

    supercurrent = np.imag(np.exp(-1j * y2[m]) * np.conj(x[m]) * x[m + mj])

    curl_xz = (
        nu_z_hi * (y1[m] - y1[m + mj]) - nu_z_lo * (y1[m - 1] - y1[m + mj - 1])
    ) / params.hx**2

    if params.is_3d:
        nu_x_hi, nu_x_lo = _nu_pair(nu, 0, m, mk, params, material)
        curl_xz = curl_xz + (
            nu_x_hi * (y3[m] - y3[m + mj]) - nu_x_lo * (y3[m - mk] - y3[m + mj - mk])
        ) / params.hz**2

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
    """Forcing for phi_z.  Corresponds to ``construct_FPHIZm.m``."""
    if not params.is_3d:
        return np.zeros(params.n_interior, dtype=np.complex128)

    m = idx.interior_to_full
    mj = params.mj
    mk = params.mk

    nu = plaquette_kappa2(params, material)
    nu_y_hi, nu_y_lo = _nu_pair(nu, 1, m, 1, params, material)
    nu_x_hi, nu_x_lo = _nu_pair(nu, 0, m, mj, params, material)

    supercurrent = np.imag(np.exp(-1j * y3[m]) * np.conj(x[m]) * x[m + mk])

    curl_xy = (
        nu_y_hi * (y1[m] - y1[m + mk]) - nu_y_lo * (y1[m - 1] - y1[m + mk - 1])
    ) / params.hx**2
    curl_xy = curl_xy + (
        nu_x_hi * (y2[m] - y2[m + mk]) - nu_x_lo * (y2[m - mj] - y2[m + mk - mj])
    ) / params.hy**2

    return (curl_xy + supercurrent).astype(np.complex128)

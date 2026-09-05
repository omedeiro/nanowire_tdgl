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


def kappa_sq_interior(
    params: SimulationParameters,
    idx: GridIndices,
    material: Optional[MaterialMap] = None,
) -> NDArray[np.float64]:
    """κ² on the interior nodes, cached alongside the neighbour stencil.

    Six of the operators need this on every right-hand-side evaluation and it
    never changes during a run, so it is gathered once and kept.  The cache is
    keyed on the material map it was built from *and* on ``params.kappa`` —
    the two things the answer depends on — so changing either rebuilds it
    rather than returning a stale array.  The second half of that key matters:
    :meth:`GridIndices.neighbours` does not look at the parameters it is
    handed, so without it a second solve on the same grid and material at a
    different κ silently keeps the first one's κ.  That is not a small error
    but the wrong penetration depth, and it shows up as a device that screens
    as though λ were whatever κ ran first.

    This is the *node* form, correct only where the coefficient is
    uniform — which is the default, since ``magnetic_kappa`` is unset
    unless a layer asks for it.  Where it does vary, the operators use
    :func:`plaquette_kappa2` instead; see there for why.
    """
    st = idx.neighbours(params)
    cached = st.get("_kappa_sq")
    if cached is not None and cached[0] is material and cached[2] == params.kappa:
        return cached[1]
    kappa_sq = _kappa_at(st["m"], params, material) ** 2
    st["_kappa_sq"] = (material, kappa_sq, params.kappa)
    return kappa_sq


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
    coincide exactly and callers can use the node form above.
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


def nu_pairs_interior(
    params: SimulationParameters,
    idx: GridIndices,
    material: Optional[MaterialMap] = None,
) -> Optional[tuple]:
    """Plaquette coefficient pairs on the interior nodes, cached.

    ``None`` when the coefficient is uniform, which is what lets the
    right-hand side keep the cheaper node form on the common path.
    Otherwise returns the six arrays the φ-equations need, in the order
    ``(z_hi, z_lo, y_hi, y_lo, x_hi, x_lo)``, each over all interior
    nodes so a caller can slice them by row block.

    Cached on the neighbour stencil and keyed on the material map, in
    the same way as :func:`kappa_sq_interior`.
    """
    if material is None or material.magnetic_kappa is None:
        return None

    st = idx.neighbours(params)
    cached = st.get("_nu_pairs")
    if cached is not None and cached[0] is material:
        return cached[1]

    m = st["m"]
    mj, mk = params.mj, params.mk
    nu = plaquette_kappa2(params, material)
    # φ_x reads the Bz pair across mj and the By pair across mk; φ_y the
    # Bz pair across 1 and the Bx pair across mk; φ_z the By pair across
    # 1 and the Bx pair across mj.  Six arrays cover all three.
    z_hi_j, z_lo_j = _nu_pair(nu, 2, m, mj, params, material)
    z_hi_i, z_lo_i = _nu_pair(nu, 2, m, 1, params, material)
    y_hi_k, y_lo_k = _nu_pair(nu, 1, m, mk, params, material)
    y_hi_i, y_lo_i = _nu_pair(nu, 1, m, 1, params, material)
    x_hi_k, x_lo_k = _nu_pair(nu, 0, m, mk, params, material)
    x_hi_j, x_lo_j = _nu_pair(nu, 0, m, mj, params, material)
    pairs = (z_hi_j, z_lo_j, z_hi_i, z_lo_i,
             y_hi_k, y_lo_k, y_hi_i, y_lo_i,
             x_hi_k, x_lo_k, x_hi_j, x_lo_j)
    st["_nu_pairs"] = (material, pairs)
    return pairs


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
    link_factor: Optional[NDArray[np.complex128]] = None,
) -> NDArray[np.complex128]:
    """Forcing for φ_x: supercurrent plus the φ_y/φ_z plaquette terms.

    The magnetic part is the φ_y and φ_z half of

    ::

        dphi_x[m]/dt = -(hx/hy)(nu_z[m] Bz[m] - nu_z[m-mj] Bz[m-mj])
                       +(hx/hz)(nu_y[m] By[m] - nu_y[m-mk] By[m-mk])

    with each plaquette carrying its own coefficient; see
    :func:`plaquette_kappa2`.  Corresponds to ``construct_FPHIXm.m``.

    *link_factor* is ``exp(-1j * y1[m])``; :func:`~tdgl3d.physics.rhs.eval_f`
    passes it in because :func:`apply_LPSI` needs the same array, and a complex
    exponential over the whole interior is one of the more expensive things in
    the evaluation.
    """
    m = idx.interior_to_full
    mj = params.mj
    mk = params.mk

    nu = plaquette_kappa2(params, material)
    nu_z_hi, nu_z_lo = _nu_pair(nu, 2, m, mj, params, material)
    if link_factor is None:
        link_factor = np.exp(-1j * y1[m])

    supercurrent = np.imag(link_factor * np.conj(x[m]) * x[m + 1])

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
    link_factor: Optional[NDArray[np.complex128]] = None,
) -> NDArray[np.complex128]:
    """Forcing for φ_y.  Corresponds to ``construct_FPHIYm.m``.

    *link_factor* is ``exp(-1j * y2[m])``; see :func:`construct_FPHI_x`.
    """
    m = idx.interior_to_full
    mj = params.mj
    mk = params.mk

    nu = plaquette_kappa2(params, material)
    nu_z_hi, nu_z_lo = _nu_pair(nu, 2, m, 1, params, material)
    if link_factor is None:
        link_factor = np.exp(-1j * y2[m])

    supercurrent = np.imag(link_factor * np.conj(x[m]) * x[m + mj])

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
    link_factor: Optional[NDArray[np.complex128]] = None,
) -> NDArray[np.complex128]:
    """Forcing for φ_z.  Corresponds to ``construct_FPHIZm.m``.

    *link_factor* is ``exp(-1j * y3[m])``; see :func:`construct_FPHI_x`.
    """
    if not params.is_3d:
        return np.zeros(params.n_interior, dtype=np.complex128)

    m = idx.interior_to_full
    mj = params.mj
    mk = params.mk

    nu = plaquette_kappa2(params, material)
    nu_y_hi, nu_y_lo = _nu_pair(nu, 1, m, 1, params, material)
    nu_x_hi, nu_x_lo = _nu_pair(nu, 0, m, mj, params, material)
    if link_factor is None:
        link_factor = np.exp(-1j * y3[m])

    supercurrent = np.imag(link_factor * np.conj(x[m]) * x[m + mk])

    curl_xy = (
        nu_y_hi * (y1[m] - y1[m + mk]) - nu_y_lo * (y1[m - 1] - y1[m + mk - 1])
    ) / params.hx**2
    curl_xy = curl_xy + (
        nu_x_hi * (y2[m] - y2[m + mk]) - nu_x_lo * (y2[m - mj] - y2[m + mk - mj])
    ) / params.hy**2

    return (curl_xy + supercurrent).astype(np.complex128)


# ---------------------------------------------------------------------------
# Matrix-free operator application
# ---------------------------------------------------------------------------
#
# The ``construct_*`` functions above assemble each operator as a sparse
# matrix.  That is the readable form, and the tests check the matrices
# entry-by-entry against it, but it is not how the right-hand side should be
# evaluated: every call rebuilds three COO matrices per operator, sums them
# into CSR, slices out the interior rows and only then multiplies — and the
# multiply is under a tenth of the cost.  The structure is identical on every
# call; only the values of ``y`` change.
#
# The ``apply_*`` functions below compute the same interior rows of
# ``L @ vec`` directly from the neighbour stencil, with no matrix in between.
# They are the path :func:`tdgl3d.physics.rhs.eval_f` takes; the assembled
# matrices remain available for tests, analysis and anything that needs the
# operator itself rather than its action.


def _rows(idx: GridIndices, params: SimulationParameters, rows: slice | None):
    """Interior full-grid indices for *rows*, and the two grid strides."""
    m = idx.neighbours(params)["m"]
    return (m if rows is None else m[rows]), params.mj, params.mk


def apply_LPSI(
    x: NDArray[np.complexfloating],
    y1: NDArray[np.complexfloating],
    y2: NDArray[np.complexfloating],
    y3: NDArray[np.complexfloating],
    params: SimulationParameters,
    idx: GridIndices,
    rows: slice | None = None,
) -> tuple[NDArray[np.complex128], tuple]:
    """Interior rows of ``(LPSIX/hx² + LPSIY/hy² + LPSIZ/hz²) @ x``.

    Matrix-free equivalent of building :func:`construct_LPSI_x`,
    :func:`construct_LPSI_y` and :func:`construct_LPSI_z`, scaling each by its
    grid spacing, and multiplying by *x*.  *rows* restricts the computation to
    a contiguous block of interior nodes, which is how the right-hand side
    splits the work across threads.

    Returns
    -------
    dpsi : ndarray
        The operator applied to *x*, on the selected interior rows.
    link_factors : tuple of ndarray
        ``(exp(-1j*y1[m]), exp(-1j*y2[m]), exp(-1j*y3[m]))`` — the on-site
        Peierls factors, which :func:`construct_FPHI_x` and its siblings need
        as well.  ``y3``'s entry is ``None`` in 2-D.
    """
    m, mj, mk = _rows(idx, params, rows)

    x_m = x[m]
    fx = np.exp(-1j * y1[m])
    fy = np.exp(-1j * y2[m])
    out = (np.exp(1j * y1[m - 1]) * x[m - 1]
           + fx * x[m + 1]
           - 2.0 * x_m) / params.hx**2
    out += (np.exp(1j * y2[m - mj]) * x[m - mj]
            + fy * x[m + mj]
            - 2.0 * x_m) / params.hy**2
    fz = None
    if params.is_3d:
        fz = np.exp(-1j * y3[m])
        out += (np.exp(1j * y3[m - mk]) * x[m - mk]
                + fz * x[m + mk]
                - 2.0 * x_m) / params.hz**2
    return out, (fx, fy, fz)


def _pair_laplacian(
    y: NDArray[np.complexfloating],
    m: NDArray[np.intp],
    stride: int,
    hi: NDArray[np.float64],
    lo: NDArray[np.float64],
    h2: float,
) -> NDArray[np.complex128]:
    """``(hi (y[m+s] - y[m]) - lo (y[m] - y[m-s])) / h²``.

    The variable-coefficient second difference the φ-equations use when
    the two plaquettes either side of a link carry different ν.  With
    ``hi == lo`` it is the constant-coefficient Laplacian.
    """
    return (hi * (y[m + stride] - y[m]) - lo * (y[m] - y[m - stride])) / h2


def apply_LPHI_x(
    y1: NDArray[np.complexfloating],
    params: SimulationParameters,
    idx: GridIndices,
    material: Optional[MaterialMap] = None,
    rows: slice | None = None,
) -> NDArray[np.complex128]:
    """Interior rows of ``construct_LPHI_x(...) @ y1``, matrix-free."""
    m, mj, mk = _rows(idx, params, rows)
    sl = rows if rows else slice(None)
    pairs = nu_pairs_interior(params, idx, material)

    if pairs is None:
        kappa2 = kappa_sq_interior(params, idx, material)[sl]
        out = (kappa2 / params.hy**2) * (y1[m + mj] + y1[m - mj] - 2.0 * y1[m])
        if params.is_3d:
            out += (kappa2 / params.hz**2) * (y1[m + mk] + y1[m - mk] - 2.0 * y1[m])
        return out

    out = _pair_laplacian(y1, m, mj, pairs[0][sl], pairs[1][sl], params.hy**2)
    if params.is_3d:
        out = out + _pair_laplacian(y1, m, mk, pairs[4][sl], pairs[5][sl], params.hz**2)
    return out


def apply_LPHI_y(
    y2: NDArray[np.complexfloating],
    params: SimulationParameters,
    idx: GridIndices,
    material: Optional[MaterialMap] = None,
    rows: slice | None = None,
) -> NDArray[np.complex128]:
    """Interior rows of ``construct_LPHI_y(...) @ y2``, matrix-free."""
    m, _mj, mk = _rows(idx, params, rows)
    sl = rows if rows else slice(None)
    pairs = nu_pairs_interior(params, idx, material)

    if pairs is None:
        kappa2 = kappa_sq_interior(params, idx, material)[sl]
        out = (kappa2 / params.hx**2) * (y2[m + 1] + y2[m - 1] - 2.0 * y2[m])
        if params.is_3d:
            out += (kappa2 / params.hz**2) * (y2[m + mk] + y2[m - mk] - 2.0 * y2[m])
        return out

    out = _pair_laplacian(y2, m, 1, pairs[2][sl], pairs[3][sl], params.hx**2)
    if params.is_3d:
        out = out + _pair_laplacian(y2, m, mk, pairs[8][sl], pairs[9][sl], params.hz**2)
    return out


def apply_LPHI_z(
    y3: NDArray[np.complexfloating],
    params: SimulationParameters,
    idx: GridIndices,
    material: Optional[MaterialMap] = None,
    rows: slice | None = None,
) -> NDArray[np.complex128]:
    """Interior rows of ``construct_LPHI_z(...) @ y3``, matrix-free."""
    m, mj, _mk = _rows(idx, params, rows)
    sl = rows if rows else slice(None)
    pairs = nu_pairs_interior(params, idx, material)

    if pairs is None:
        kappa2 = kappa_sq_interior(params, idx, material)[sl]
        out = (kappa2 / params.hx**2) * (y3[m + 1] + y3[m - 1] - 2.0 * y3[m])
        out += (kappa2 / params.hy**2) * (y3[m + mj] + y3[m - mj] - 2.0 * y3[m])
        return out

    out = _pair_laplacian(y3, m, 1, pairs[6][sl], pairs[7][sl], params.hx**2)
    out = out + _pair_laplacian(y3, m, mj, pairs[10][sl], pairs[11][sl], params.hy**2)
    return out


# ---------------------------------------------------------------------------
# The right-hand side kernel
# ---------------------------------------------------------------------------
#
# The interior nodes are a contiguous box inside the full grid, so every
# stencil neighbour is a *strided view* of the state, not a gather.  Walking
# the box with fancy indexing — ``x[m]``, ``x[m + 1]``, ``x[m + mj]`` … — costs
# an index array the width of the data (``intp`` is 8 bytes against 16 for a
# complex128) on every one of the twenty-odd reads, and materialises a fresh
# ``m + offset`` array for most of them.  Slicing costs neither.  The two paths
# compute the same numbers bit for bit; ``test_matrix_free_operators`` holds
# them together against the assembled matrices.


def _grid_view(a: NDArray, params: SimulationParameters) -> NDArray:
    """View a full-grid vector as ``[k, j, i]``.

    The full grid is ``i + mj*j + mk*k`` with ``mj = Nx+1`` and
    ``mk = (Nx+1)(Ny+1)``, so i is the fastest axis.  A 2-D grid gets a
    length-1 k axis, which lets one kernel serve both.
    """
    nk = params.Nz + 1 if params.is_3d else 1
    return a.reshape(nk, params.Ny + 1, params.Nx + 1)


def chunk_planes(params: SimulationParameters) -> int:
    """Number of independent planes the interior can be split into.

    The chunk unit is a plane of the slowest interior axis — k in 3-D, j in
    2-D — because a chunk has to stay a rectangular box for the stencil to be
    expressible as slices.
    """
    return (params.Nz - 1) if params.is_3d else (params.Ny - 1)


def _plane_bounds(params: SimulationParameters, planes: slice) -> tuple:
    """Grid-index bounds ``(k0, k1, j0, j1)`` of the interior box for *planes*."""
    start, stop, _ = planes.indices(chunk_planes(params))
    if params.is_3d:
        return 1 + start, 1 + stop, 1, params.Ny
    return 0, 1, 1 + start, 1 + stop


def _interior_shape(params: SimulationParameters) -> tuple:
    """Shape of an interior-numbered array, in its own (i, j, k) order."""
    return (params.Nx - 1, params.Ny - 1, max(params.Nz - 1, 1))


def _to_grid_order(v: NDArray, params: SimulationParameters, real_dtype) -> NDArray:
    """An interior-numbered array as a contiguous ``[k, j, i]`` one.

    The interior numbering is i-slowest, so the permutation is a transpose and
    needs no sort.  Made contiguous once, so the kernel multiplies by a
    contiguous array rather than a strided one, and kept in *real_dtype* so a
    single-precision state is not silently promoted back to double the first
    time it is scaled by a coefficient.
    """
    return np.ascontiguousarray(
        v.reshape(_interior_shape(params)).transpose(2, 1, 0).astype(
            real_dtype, copy=False
        )
    )


def interior_material(
    params: SimulationParameters,
    idx: GridIndices,
    material: Optional[MaterialMap],
    real_dtype=np.float64,
) -> tuple:
    """``(κ², superconducting mask)`` as ``[k, j, i]`` arrays, cached per device.

    Keyed on the material map, the dtype **and** ``params.kappa``:
    :meth:`GridIndices.neighbours` does not look at the parameters it is
    handed, so without the last of those a second solve on the same grid and
    material at a different κ would silently keep the first one's.
    """
    st = idx.neighbours(params)
    real_dtype = np.dtype(real_dtype)
    cached = st.get("_interior_material")
    if (
        cached is not None
        and cached[0] is material
        and cached[3] == real_dtype
        and cached[4] == params.kappa
    ):
        return cached[1], cached[2]

    kappa_sq = _to_grid_order(
        kappa_sq_interior(params, idx, material), params, real_dtype
    )
    sc = (
        None if material is None
        else _to_grid_order(material.interior_sc_mask, params, real_dtype)
    )
    st["_interior_material"] = (material, kappa_sq, sc, real_dtype, params.kappa)
    return kappa_sq, sc


def interior_nu_pairs(
    params: SimulationParameters,
    idx: GridIndices,
    material: Optional[MaterialMap],
    real_dtype=np.float64,
) -> Optional[tuple]:
    """:func:`nu_pairs_interior` as ``[k, j, i]`` arrays, cached per device.

    ``None`` when the coefficient is uniform, which is the common case and the
    one :func:`rhs_rows` has a dedicated path for.
    """
    pairs = nu_pairs_interior(params, idx, material)
    if pairs is None:
        return None

    st = idx.neighbours(params)
    real_dtype = np.dtype(real_dtype)
    cached = st.get("_interior_nu_pairs")
    if cached is not None and cached[0] is material and cached[2] == real_dtype:
        return cached[1]

    permuted = tuple(_to_grid_order(a, params, real_dtype) for a in pairs)
    st["_interior_nu_pairs"] = (material, permuted, real_dtype)
    return permuted


def _out_view(block: NDArray, params: SimulationParameters) -> NDArray:
    """View an output block (interior numbering) as ``[k, j, i]``."""
    return block.reshape(_interior_shape(params)).transpose(2, 1, 0)


def _expi(y: NDArray, sign: int) -> NDArray:
    """``exp(sign * 1j * y)`` — the on-link Peierls factor.

    The complex exponential is the most expensive thing in the kernel: NumPy
    evaluates it with a scalar ``cexp`` per element, and — against
    intuition — its ``complex64`` loop is *slower* than its ``complex128`` one
    (measured 13.4 ms against 8.9 ms per 300 k elements), which is why running
    the state narrow used to buy nothing even though every other operation in
    the kernel halved.

    Splitting it into real transcendentals fixes that, because NumPy's float32
    ``sin``/``cos`` are SIMD loops where its float64 ones are not::

        exp(s·i·(a + i·b)) = exp(−s·b) · (cos a + i·s·sin a)

    measured 1.6 ms against 13.4 ms at ``complex64``.  At ``complex128`` the
    real loops are scalar as well, so the split saves nothing (8.5 against 8.7
    ms, inside the noise) and would move results in the last bit — so double
    precision keeps the exact complex exponential and only the narrow path
    takes the split.
    """
    if y.dtype != np.complex64:
        return np.exp((sign * 1j) * y)

    out = np.empty(y.shape, dtype=np.complex64)
    a = y.real
    np.cos(a, out=out.real)
    np.sin(a, out=out.imag)
    if sign < 0:
        np.negative(out.imag, out=out.imag)
    b = y.imag
    if b.any():
        out *= np.exp(np.float32(-sign) * b)
    return out


def rhs_rows(
    x: NDArray[np.complexfloating],
    y1: NDArray[np.complexfloating],
    y2: NDArray[np.complexfloating],
    y3: NDArray[np.complexfloating],
    params: SimulationParameters,
    idx: GridIndices,
    material: Optional[MaterialMap],
    planes: slice,
    out: tuple[NDArray, NDArray, NDArray, NDArray],
) -> None:
    """Write dψ/dt and dφ/dt for a slab of interior *planes* into *out*.

    *planes* selects a contiguous range along the slowest interior axis — k in
    3-D, j in 2-D — of the :func:`chunk_planes` available; that is the unit the
    right-hand side splits across threads.  Each thread therefore owns a slab
    that is contiguous in memory and writes only its own rows.

    Doing all four output blocks for one slab, rather than one output block for
    the whole grid, keeps a thread reading the same neighbourhood of ``x``,
    ``y1``, ``y2`` and ``y3`` throughout.  The four writes go through the
    transpose into the interior numbering, which is where the strided access is
    paid; there are four of those against twenty reads.

    It is the same arithmetic as :func:`apply_LPSI`, :func:`apply_LPHI_x` and
    the ``construct_F*`` forcings; ``test_matrix_free_operators`` holds the two
    paths together.
    """
    k0, k1, j0, j1 = _plane_bounds(params, planes)
    if k1 <= k0 or j1 <= j0:
        return

    Nx = params.Nx
    is_3d = params.is_3d
    x3, y13, y23, y33 = (_grid_view(a, params) for a in (x, y1, y2, y3))

    def S(a3, di=0, dj=0, dk=0):
        """The chunk's interior box, shifted by one stencil offset."""
        return a3[k0 + dk:k1 + dk, j0 + dj:j1 + dj, 1 + di:Nx + di]

    # The chunk's slice of the interior-numbered arrays, in [k, j, i] order.
    chunk = (planes, slice(None), slice(None)) if is_3d else (
        slice(None), planes, slice(None)
    )
    real_dtype = out[0].real.dtype
    kappa_sq_all, sc_all = interior_material(
        params, idx, material, real_dtype=real_dtype
    )
    nu_all = interior_nu_pairs(params, idx, material, real_dtype=real_dtype)
    kappa2 = kappa_sq_all[chunk]
    hx2, hy2, hz2 = params.hx**2, params.hy**2, params.hz**2

    x_m = S(x3)
    fx = _expi(S(y13), -1)
    fy = _expi(S(y23), -1)

    # --- dψ/dt: covariant Laplacian plus the Ginzburg-Landau forcing --------
    dpsi = (_expi(S(y13, di=-1), 1) * S(x3, di=-1) + fx * S(x3, di=1)
            - 2.0 * x_m) / hx2
    dpsi += (_expi(S(y23, dj=-1), 1) * S(x3, dj=-1) + fy * S(x3, dj=1)
             - 2.0 * x_m) / hy2
    fz = None
    if is_3d:
        fz = _expi(S(y33), -1)
        dpsi += (_expi(S(y33, dk=-1), 1) * S(x3, dk=-1) + fz * S(x3, dk=1)
                 - 2.0 * x_m) / hz2

    gl_term = (1.0 - np.conj(x_m) * x_m) * x_m
    if sc_all is not None:
        sc = sc_all[chunk]
        dpsi += sc * gl_term - (1.0 - sc) * x_m / INSULATOR_RELAXATION_TIME
    else:
        dpsi += gl_term
    _out_view(out[0], params)[chunk] = dpsi

    if nu_all is None:
        # Uniform coefficient: one κ² per node factors out of both the
        # Laplacian and the cross terms.
        kx, ky, kz = kappa2 / hx2, kappa2 / hy2, kappa2 / hz2

        # --- dφ_x/dt: transverse Laplacian, curl-curl cross terms, current --
        dpx = ky * (S(y13, dj=1) + S(y13, dj=-1) - 2.0 * S(y13))
        dpx += ky * (-S(y23, di=1) + S(y23) + S(y23, di=1, dj=-1) - S(y23, dj=-1))
        if is_3d:
            dpx += kz * (S(y13, dk=1) + S(y13, dk=-1) - 2.0 * S(y13))
            dpx += kz * (-S(y33, di=1) + S(y33) + S(y33, di=1, dk=-1) - S(y33, dk=-1))
        dpx += np.imag(fx * np.conj(x_m) * S(x3, di=1))
        _out_view(out[1], params)[chunk] = dpx

        # --- dφ_y/dt --------------------------------------------------------
        dpy = kx * (S(y23, di=1) + S(y23, di=-1) - 2.0 * S(y23))
        dpy += kx * (-S(y13, dj=1) + S(y13) + S(y13, dj=1, di=-1) - S(y13, di=-1))
        if is_3d:
            dpy += kz * (S(y23, dk=1) + S(y23, dk=-1) - 2.0 * S(y23))
            dpy += kz * (-S(y33, dj=1) + S(y33) + S(y33, dj=1, dk=-1) - S(y33, dk=-1))
        dpy += np.imag(fy * np.conj(x_m) * S(x3, dj=1))
        _out_view(out[2], params)[chunk] = dpy

        # --- dφ_z/dt --------------------------------------------------------
        if is_3d:
            dpz = kx * (S(y33, di=1) + S(y33, di=-1) - 2.0 * S(y33))
            dpz += ky * (S(y33, dj=1) + S(y33, dj=-1) - 2.0 * S(y33))
            dpz += kx * (-S(y13, dk=1) + S(y13) + S(y13, dk=1, di=-1) - S(y13, di=-1))
            dpz += ky * (-S(y23, dk=1) + S(y23) + S(y23, dk=1, dj=-1) - S(y23, dj=-1))
            dpz += np.imag(fz * np.conj(x_m) * S(x3, dk=1))
            _out_view(out[3], params)[chunk] = dpz
        return

    # Coefficient varies between plaquettes, so each of the four fluxes
    # a link closes carries its own ν and none of them factors out.  Same
    # assembly as construct_LPHI_* / construct_FPHI_*, which is what
    # test_matrix_free_operators holds this against.
    zhj, zlj = nu_all[0][chunk], nu_all[1][chunk]
    zhi, zli = nu_all[2][chunk], nu_all[3][chunk]
    yhk, ylk = nu_all[4][chunk], nu_all[5][chunk]
    yhi, yli = nu_all[6][chunk], nu_all[7][chunk]
    xhk, xlk = nu_all[8][chunk], nu_all[9][chunk]
    xhj, xlj = nu_all[10][chunk], nu_all[11][chunk]

    # --- dφ_x/dt ------------------------------------------------------------
    dpx = (zhj * (S(y13, dj=1) - S(y13)) - zlj * (S(y13) - S(y13, dj=-1))) / hy2
    dpx += (zhj * (S(y23) - S(y23, di=1))
            + zlj * (S(y23, di=1, dj=-1) - S(y23, dj=-1))) / hy2
    if is_3d:
        dpx += (yhk * (S(y13, dk=1) - S(y13)) - ylk * (S(y13) - S(y13, dk=-1))) / hz2
        dpx += (yhk * (S(y33) - S(y33, di=1))
                - ylk * (S(y33, dk=-1) - S(y33, di=1, dk=-1))) / hz2
    dpx += np.imag(fx * np.conj(x_m) * S(x3, di=1))
    _out_view(out[1], params)[chunk] = dpx

    # --- dφ_y/dt ------------------------------------------------------------
    dpy = (zhi * (S(y23, di=1) - S(y23)) - zli * (S(y23) - S(y23, di=-1))) / hx2
    dpy += (zhi * (S(y13) - S(y13, dj=1))
            - zli * (S(y13, di=-1) - S(y13, dj=1, di=-1))) / hx2
    if is_3d:
        dpy += (xhk * (S(y23, dk=1) - S(y23)) - xlk * (S(y23) - S(y23, dk=-1))) / hz2
        dpy += (xhk * (S(y33) - S(y33, dj=1))
                - xlk * (S(y33, dk=-1) - S(y33, dj=1, dk=-1))) / hz2
    dpy += np.imag(fy * np.conj(x_m) * S(x3, dj=1))
    _out_view(out[2], params)[chunk] = dpy

    # --- dφ_z/dt ------------------------------------------------------------
    if is_3d:
        dpz = (yhi * (S(y33, di=1) - S(y33)) - yli * (S(y33) - S(y33, di=-1))) / hx2
        dpz += (xhj * (S(y33, dj=1) - S(y33)) - xlj * (S(y33) - S(y33, dj=-1))) / hy2
        dpz += (yhi * (S(y13) - S(y13, dk=1))
                - yli * (S(y13, di=-1) - S(y13, dk=1, di=-1))) / hx2
        dpz += (xhj * (S(y23) - S(y23, dk=1))
                - xlj * (S(y23, dj=-1) - S(y23, dk=1, dj=-1))) / hy2
        dpz += np.imag(fz * np.conj(x_m) * S(x3, dk=1))
        _out_view(out[3], params)[chunk] = dpz

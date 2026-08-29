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
    """Return κ values at full-grid indices *m*.

    If *material* is ``None`` the uniform ``params.kappa`` is used.
    """
    if material is not None:
        return material.kappa[m]
    return np.full(len(m), params.kappa, dtype=np.float64)


def kappa_sq_interior(
    params: SimulationParameters,
    idx: GridIndices,
    material: Optional[MaterialMap] = None,
) -> NDArray[np.float64]:
    """κ² on the interior nodes, cached alongside the neighbour stencil.

    Six of the operators need this on every right-hand-side evaluation and it
    never changes during a run, so it is gathered once and kept.  The cache is
    keyed on the material map it was built from, so swapping materials on a
    device rebuilds it rather than returning a stale array.
    """
    st = idx.neighbours(params)
    cached = st.get("_kappa_sq")
    if cached is not None and cached[0] is material:
        return cached[1]
    kappa_sq = _kappa_at(st["m"], params, material) ** 2
    st["_kappa_sq"] = (material, kappa_sq)
    return kappa_sq


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
    """Forcing for φ_x.  Corresponds to ``construct_FPHIXm.m``.

    *link_factor* is ``exp(-1j * y1[m])``; :func:`~tdgl3d.physics.rhs.eval_f`
    passes it in because :func:`apply_LPSI` needs the same array, and a complex
    exponential over the whole interior is one of the more expensive things in
    the evaluation.
    """
    m = idx.interior_to_full
    mj = params.mj
    mk = params.mk

    kappa_m2 = kappa_sq_interior(params, idx, material)
    if link_factor is None:
        link_factor = np.exp(-1j * y1[m])

    supercurrent = np.imag(link_factor * np.conj(x[m]) * x[m + 1])

    curl_yz = (kappa_m2 / params.hy**2) * (
        -y2[m + 1] + y2[m] + y2[m + 1 - mj] - y2[m - mj]
    )

    if params.is_3d:
        curl_yz += (kappa_m2 / params.hz**2) * (
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
    link_factor: Optional[NDArray[np.complex128]] = None,
) -> NDArray[np.complex128]:
    """Forcing for φ_y.  Corresponds to ``construct_FPHIYm.m``.

    *link_factor* is ``exp(-1j * y2[m])``; see :func:`construct_FPHI_x`.
    """
    m = idx.interior_to_full
    mj = params.mj
    mk = params.mk

    kappa_m2 = kappa_sq_interior(params, idx, material)
    if link_factor is None:
        link_factor = np.exp(-1j * y2[m])

    supercurrent = np.imag(link_factor * np.conj(x[m]) * x[m + mj])

    curl_xz = (kappa_m2 / params.hx**2) * (
        -y1[m + mj] + y1[m] + y1[m + mj - 1] - y1[m - 1]
    )

    if params.is_3d:
        curl_xz += (kappa_m2 / params.hz**2) * (
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

    kappa_m2 = kappa_sq_interior(params, idx, material)
    if link_factor is None:
        link_factor = np.exp(-1j * y3[m])

    supercurrent = np.imag(link_factor * np.conj(x[m]) * x[m + mk])

    curl_xy = (kappa_m2 / params.hx**2) * (
        -y1[m + mk] + y1[m] + y1[m + mk - 1] - y1[m - 1]
    )
    curl_xy += (kappa_m2 / params.hy**2) * (
        -y2[m + mk] + y2[m] + y2[m + mk - mj] - y2[m - mj]
    )

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


def grid_order(params: SimulationParameters, idx: GridIndices) -> tuple:
    """Interior nodes renumbered into full-grid order, cached per device.

    The two numberings run opposite ways — the full grid is i-fastest, the
    interior numbering is i-slowest — so consecutive interior nodes land tens
    of thousands of elements apart on the full grid.  Walking the stencil in
    interior order therefore fetches a fresh cache line for almost every one of
    its twenty-odd gathers: measured 6.7 ms against 2.4 ms for the same gather
    taken in ascending order, on an 800 k-node grid.

    Sorting ``interior_to_full`` is just reading the interior array in (k, j, i)
    order instead of (i, j, k), so the permutation is a transpose and needs no
    sort.  The right-hand side walks the nodes this way and pays the scattered
    access on its four writes instead of on all its reads.

    Returns
    -------
    order : ndarray
        ``order[p]`` is the interior index of the p-th node in full-grid order.
    m_sorted : ndarray
        ``interior_to_full[order]`` — ascending.
    """
    st = idx.neighbours(params)
    cached = st.get("_grid_order")
    if cached is not None:
        return cached

    shape = (params.Nx - 1, params.Ny - 1, max(params.Nz - 1, 1))
    order = np.arange(params.n_interior).reshape(shape).transpose(2, 1, 0).ravel()
    order = np.ascontiguousarray(order)
    m_sorted = idx.interior_to_full[order]
    st["_grid_order"] = (order, m_sorted)
    return st["_grid_order"]


def _material_in_grid_order(
    params: SimulationParameters,
    idx: GridIndices,
    material: Optional[MaterialMap],
    real_dtype=np.float64,
) -> tuple:
    """``(κ², superconducting mask)`` permuted into full-grid order, cached.

    Kept in *real_dtype* so a single-precision state is not silently promoted
    back to double the first time it is multiplied by a coefficient.
    """
    st = idx.neighbours(params)
    real_dtype = np.dtype(real_dtype)
    cached = st.get("_material_grid_order")
    if cached is not None and cached[0] is material and cached[3] == real_dtype:
        return cached[1], cached[2]

    order, _ = grid_order(params, idx)
    kappa_sq = kappa_sq_interior(params, idx, material)[order].astype(
        real_dtype, copy=False
    )
    sc = (
        None if material is None
        else material.interior_sc_mask[order].astype(real_dtype, copy=False)
    )
    st["_material_grid_order"] = (material, kappa_sq, sc, real_dtype)
    return kappa_sq, sc


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


def apply_LPHI_x(
    y1: NDArray[np.complexfloating],
    params: SimulationParameters,
    idx: GridIndices,
    material: Optional[MaterialMap] = None,
    rows: slice | None = None,
) -> NDArray[np.complex128]:
    """Interior rows of ``construct_LPHI_x(...) @ y1``, matrix-free."""
    m, mj, mk = _rows(idx, params, rows)
    kappa2 = kappa_sq_interior(params, idx, material)[rows if rows else slice(None)]

    out = (kappa2 / params.hy**2) * (y1[m + mj] + y1[m - mj] - 2.0 * y1[m])
    if params.is_3d:
        out += (kappa2 / params.hz**2) * (y1[m + mk] + y1[m - mk] - 2.0 * y1[m])
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
    kappa2 = kappa_sq_interior(params, idx, material)[rows if rows else slice(None)]

    out = (kappa2 / params.hx**2) * (y2[m + 1] + y2[m - 1] - 2.0 * y2[m])
    if params.is_3d:
        out += (kappa2 / params.hz**2) * (y2[m + mk] + y2[m - mk] - 2.0 * y2[m])
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
    kappa2 = kappa_sq_interior(params, idx, material)[rows if rows else slice(None)]

    out = (kappa2 / params.hx**2) * (y3[m + 1] + y3[m - 1] - 2.0 * y3[m])
    out += (kappa2 / params.hy**2) * (y3[m + mj] + y3[m - mj] - 2.0 * y3[m])
    return out


def rhs_rows(
    x: NDArray[np.complexfloating],
    y1: NDArray[np.complexfloating],
    y2: NDArray[np.complexfloating],
    y3: NDArray[np.complexfloating],
    params: SimulationParameters,
    idx: GridIndices,
    material: Optional[MaterialMap],
    rows: slice,
    out: tuple[NDArray, NDArray, NDArray, NDArray],
) -> None:
    """Write dψ/dt and dφ/dt for interior rows *rows* into *out*.

    *rows* selects a contiguous block of the **grid-ordered** nodes — see
    :func:`grid_order` — so a thread's gathers walk the full grid in order
    rather than jumping a plane at a time.  Doing all four output blocks for
    one block of nodes, rather than one output block for all nodes, keeps that
    thread reading the same neighbourhood of ``x``, ``y1``, ``y2`` and ``y3``
    throughout.  The four writes go back through the permutation, which is
    where the scattered access is paid; there are four of those against twenty
    reads.

    It is the same arithmetic as :func:`apply_LPSI`, :func:`apply_LPHI_x` and
    the ``construct_F*`` forcings; ``test_matrix_free_operators`` holds the two
    paths together.
    """
    order, m_sorted = grid_order(params, idx)
    real_dtype = out[0].real.dtype
    kappa_sq_all, sc_all = _material_in_grid_order(
        params, idx, material, real_dtype=real_dtype
    )
    m = m_sorted[rows]
    write = order[rows]
    mj, mk = params.mj, params.mk
    kappa2 = kappa_sq_all[rows]
    out_psi, out_px, out_py, out_pz = out
    hx2, hy2, hz2 = params.hx**2, params.hy**2, params.hz**2
    is_3d = params.is_3d

    x_m = x[m]
    fx = np.exp(-1j * y1[m])
    fy = np.exp(-1j * y2[m])

    # --- dψ/dt: covariant Laplacian plus the Ginzburg-Landau forcing --------
    dpsi = (np.exp(1j * y1[m - 1]) * x[m - 1] + fx * x[m + 1] - 2.0 * x_m) / hx2
    dpsi += (np.exp(1j * y2[m - mj]) * x[m - mj] + fy * x[m + mj] - 2.0 * x_m) / hy2
    if is_3d:
        fz = np.exp(-1j * y3[m])
        dpsi += (np.exp(1j * y3[m - mk]) * x[m - mk] + fz * x[m + mk] - 2.0 * x_m) / hz2

    gl_term = (1.0 - np.conj(x_m) * x_m) * x_m
    if sc_all is not None:
        sc = sc_all[rows]
        dpsi += sc * gl_term - (1.0 - sc) * x_m / INSULATOR_RELAXATION_TIME
    else:
        dpsi += gl_term
    out_psi[write] = dpsi

    # --- dφ_x/dt: transverse Laplacian, curl-curl cross terms, supercurrent -
    kx, ky, kz = kappa2 / hx2, kappa2 / hy2, kappa2 / hz2
    xp = m + 1

    dpx = ky * (y1[m + mj] + y1[m - mj] - 2.0 * y1[m])
    dpx += ky * (-y2[xp] + y2[m] + y2[xp - mj] - y2[m - mj])
    if is_3d:
        dpx += kz * (y1[m + mk] + y1[m - mk] - 2.0 * y1[m])
        dpx += kz * (-y3[xp] + y3[m] + y3[xp - mk] - y3[m - mk])
    dpx += np.imag(fx * np.conj(x_m) * x[xp])
    out_px[write] = dpx

    # --- dφ_y/dt ------------------------------------------------------------
    yp = m + mj
    dpy = kx * (y2[xp] + y2[m - 1] - 2.0 * y2[m])
    dpy += kx * (-y1[yp] + y1[m] + y1[yp - 1] - y1[m - 1])
    if is_3d:
        dpy += kz * (y2[m + mk] + y2[m - mk] - 2.0 * y2[m])
        dpy += kz * (-y3[yp] + y3[m] + y3[yp - mk] - y3[m - mk])
    dpy += np.imag(fy * np.conj(x_m) * x[yp])
    out_py[write] = dpy

    # --- dφ_z/dt ------------------------------------------------------------
    if is_3d:
        zp = m + mk
        dpz = kx * (y3[xp] + y3[m - 1] - 2.0 * y3[m])
        dpz += ky * (y3[yp] + y3[m - mj] - 2.0 * y3[m])
        dpz += kx * (-y1[zp] + y1[m] + y1[zp - 1] - y1[m - 1])
        dpz += ky * (-y2[zp] + y2[m] + y2[zp - mj] - y2[m - mj])
        dpz += np.imag(fz * np.conj(x_m) * x[zp])
        out_pz[write] = dpz

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


# ---------------------------------------------------------------------------
# The right-hand side kernel
# ---------------------------------------------------------------------------
#
# The interior nodes are a contiguous box inside the full grid, so every
# stencil neighbour is a *strided view* of the state, not a gather.  Walking the
# box with fancy indexing — ``x[m]``, ``x[m + 1]``, ``x[m + mj]`` … — costs an
# index array the width of the data (``intp`` is 8 bytes against 16 for a
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
    n_planes = chunk_planes(params)
    start, stop, _ = planes.indices(n_planes)
    if params.is_3d:
        return 1 + start, 1 + stop, 1, params.Ny
    return 0, 1, 1 + start, 1 + stop


def interior_material(
    params: SimulationParameters,
    idx: GridIndices,
    material: Optional[MaterialMap],
    real_dtype=np.float64,
) -> tuple:
    """``(κ², superconducting mask)`` as ``[k, j, i]`` arrays, cached per device.

    The interior numbering is i-slowest (C order over ``(Nx-1, Ny-1, Nz-1)``),
    so the transpose to grid order is a view; it is made contiguous once so the
    kernel multiplies by a contiguous array rather than a strided one.  Kept in
    *real_dtype* so a single-precision state is not silently promoted back to
    double the first time it is scaled by a coefficient.
    """
    st = idx.neighbours(params)
    real_dtype = np.dtype(real_dtype)
    cached = st.get("_interior_material")
    if cached is not None and cached[0] is material and cached[3] == real_dtype:
        return cached[1], cached[2]

    shape = (params.Nx - 1, params.Ny - 1, max(params.Nz - 1, 1))

    def to_grid(v):
        return np.ascontiguousarray(
            v.reshape(shape).transpose(2, 1, 0).astype(real_dtype, copy=False)
        )

    kappa_sq = to_grid(kappa_sq_interior(params, idx, material))
    sc = None if material is None else to_grid(material.interior_sc_mask)
    st["_interior_material"] = (material, kappa_sq, sc, real_dtype)
    return kappa_sq, sc


def _out_view(block: NDArray, params: SimulationParameters) -> NDArray:
    """View an output block (interior numbering) as ``[k, j, i]``."""
    shape = (params.Nx - 1, params.Ny - 1, max(params.Nz - 1, 1))
    return block.reshape(shape).transpose(2, 1, 0)


def _expi(y: NDArray, sign: int) -> NDArray:
    """``exp(sign * 1j * y)`` — the on-link Peierls factor.

    The complex exponential is the most expensive thing in the kernel: NumPy
    evaluates it with a scalar ``cexp`` per element, and — against
    intuition — its ``complex64`` loop is *slower* than its ``complex128`` one
    (measured 13.4 ms against 8.9 ms per 300 k elements), which is why running
    the state narrow used to buy nothing even though every other operation in
    the kernel halved.

    Splitting it into real transcendentals fixes that, because NumPy's float32
    ``sin``/``cos`` are SIMD loops where its float64 ones are not:

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

    # --- dφ_x/dt: transverse Laplacian, curl-curl cross terms, supercurrent -
    kx, ky, kz = kappa2 / hx2, kappa2 / hy2, kappa2 / hz2

    dpx = ky * (S(y13, dj=1) + S(y13, dj=-1) - 2.0 * S(y13))
    dpx += ky * (-S(y23, di=1) + S(y23) + S(y23, di=1, dj=-1) - S(y23, dj=-1))
    if is_3d:
        dpx += kz * (S(y13, dk=1) + S(y13, dk=-1) - 2.0 * S(y13))
        dpx += kz * (-S(y33, di=1) + S(y33) + S(y33, di=1, dk=-1) - S(y33, dk=-1))
    dpx += np.imag(fx * np.conj(x_m) * S(x3, di=1))
    _out_view(out[1], params)[chunk] = dpx

    # --- dφ_y/dt ------------------------------------------------------------
    dpy = kx * (S(y23, di=1) + S(y23, di=-1) - 2.0 * S(y23))
    dpy += kx * (-S(y13, dj=1) + S(y13) + S(y13, dj=1, di=-1) - S(y13, di=-1))
    if is_3d:
        dpy += kz * (S(y23, dk=1) + S(y23, dk=-1) - 2.0 * S(y23))
        dpy += kz * (-S(y33, dj=1) + S(y33) + S(y33, dj=1, dk=-1) - S(y33, dk=-1))
    dpy += np.imag(fy * np.conj(x_m) * S(x3, dj=1))
    _out_view(out[2], params)[chunk] = dpy

    # --- dφ_z/dt ------------------------------------------------------------
    if is_3d:
        dpz = kx * (S(y33, di=1) + S(y33, di=-1) - 2.0 * S(y33))
        dpz += ky * (S(y33, dj=1) + S(y33, dj=-1) - 2.0 * S(y33))
        dpz += kx * (-S(y13, dk=1) + S(y13) + S(y13, dk=1, di=-1) - S(y13, di=-1))
        dpz += ky * (-S(y23, dk=1) + S(y23) + S(y23, dk=1, dj=-1) - S(y23, dj=-1))
        dpz += np.imag(fz * np.conj(x_m) * S(x3, dk=1))
        _out_view(out[3], params)[chunk] = dpz

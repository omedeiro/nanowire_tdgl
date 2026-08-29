"""Truncated Generalized Conjugate Residual (TGCR) — matrix-free variant.

Python port of ``tgcr_MatrixFree.m`` and ``tgcr_MatrixFreetrap.m``.

The truncation the name promises is :data:`DEFAULT_MAX_KRYLOV`.  Every
iteration stores two vectors the length of the state, and on a large 3-D mesh
that is most of a gigabyte each: at 15 M interior nodes an untruncated solve
that ran twenty iterations would hold 38 GB of search directions.  Keeping a
bounded window of the most recent directions caps that, at the cost of
orthogonalising against fewer of them.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

#: Search directions kept.  Measured solves in this code converge in 1-8
#: iterations, so this bound does not bind in practice — it exists so that a
#: solve that fails to converge cannot exhaust memory on a large mesh.
DEFAULT_MAX_KRYLOV = 30


def _truncate(p_list: list, Ap_list: list, max_krylov: int) -> None:
    """Drop the oldest directions once the window is full."""
    excess = len(p_list) - max_krylov
    if excess > 0:
        del p_list[:excess]
        del Ap_list[:excess]


def tgcr_matrix_free(
    eval_f: Callable[[NDArray], NDArray],
    x_lin: NDArray[np.complexfloating],
    b: NDArray[np.complexfloating],
    tol: float = 1e-4,
    max_iter: int | None = None,
    eps_mf: float = 1e-4,
    f_base: NDArray[np.complexfloating] | None = None,
    max_krylov: int = DEFAULT_MAX_KRYLOV,
) -> NDArray[np.complex128]:
    """Solve ``J δx = b`` where ``J = ∂f/∂x`` using matrix-free directional
    derivatives of *eval_f* evaluated at *x_lin*.

    Parameters
    ----------
    eval_f : callable
        ``eval_f(x)`` → f(x).  Must accept and return complex arrays.
    x_lin : ndarray
        Linearisation point (state vector where we evaluate the Jacobian).
    b : ndarray
        Right-hand side.
    tol : float
        Relative residual tolerance.
    max_iter : int, optional
        Maximum Krylov iterations (defaults to ``max(N, 0.2*N)``).
    eps_mf : float
        Perturbation scale for finite-difference directional derivative.
    f_base : ndarray, optional
        ``eval_f(x_lin)``, if the caller already has it.  The linearisation
        point does not move during the Krylov solve, so this value is the same
        in every iteration; passing it in removes one right-hand-side
        evaluation per iteration, which is half of them.
    max_krylov : int
        Search directions kept; see :data:`DEFAULT_MAX_KRYLOV`.

    Returns
    -------
    x : ndarray
        Approximate solution, or empty array on failure.
    """
    N = len(b)
    if f_base is None:
        f_base = eval_f(x_lin)
    if max_iter is None:
        max_iter = max(N, int(round(0.2 * N)))

    x = np.zeros_like(b)
    r = b.copy()
    r_norms = [np.linalg.norm(r, 2)]

    if r_norms[0] == 0.0:
        return x

    p_list: list[NDArray] = []
    Ap_list: list[NDArray] = []

    k = 0
    while r_norms[-1] / r_norms[0] > tol and k < max_iter:
        k += 1
        pk = r.copy()

        # Matrix-free A*p via finite differences
        epsilon = eps_mf * (1.0 + np.linalg.norm(x_lin)) / np.linalg.norm(pk)
        f_pert = eval_f(x_lin + epsilon * pk)
        Apk = (f_pert - f_base) / epsilon

        # Orthogonalise against previous directions
        for j in range(len(Ap_list)):
            beta = np.vdot(Apk, Ap_list[j]).real
            pk = pk - beta * p_list[j]
            Apk = Apk - beta * Ap_list[j]

        # Normalise
        norm_Ap = np.linalg.norm(Apk, 2)
        if norm_Ap < 1e-14:
            break
        Apk /= norm_Ap
        pk /= norm_Ap

        p_list.append(pk)
        Ap_list.append(Apk)
        _truncate(p_list, Ap_list, max_krylov)

        alpha = np.vdot(r, Apk).real
        x = x + alpha * pk
        r = r - alpha * Apk
        r_norms.append(np.linalg.norm(r, 2))

    if r_norms[-1] > tol * r_norms[0]:
        return np.array([], dtype=np.complex128)  # did not converge
    return x


def tgcr_matrix_free_trap(
    eval_f: Callable[[NDArray], NDArray],
    x_lin: NDArray[np.complexfloating],
    b: NDArray[np.complexfloating],
    dt: float,
    tol: float = 1e-4,
    max_iter: int | None = None,
    eps_mf: float = 1e-4,
    f_base: NDArray[np.complexfloating] | None = None,
    max_krylov: int = DEFAULT_MAX_KRYLOV,
    scaling: NDArray[np.floating] | None = None,
) -> NDArray[np.complex128]:
    """TGCR for the trapezoidal implicit system ``(I - dt/2 J) δx = b``.

    The matrix-vector product is approximated as:
        A·p ≈ p - (dt/2) * (f(x+εp) - f(x)) / ε

    ``f(x)`` at the linearisation point is constant for the whole solve; pass
    it as *f_base* (Newton already has it) to avoid recomputing it once per
    Krylov iteration.

    *scaling* right-preconditions the solve: the search direction becomes
    ``M⁻¹ r`` with ``M⁻¹`` the elementwise *scaling*.  Right rather than left,
    so the residual this loop monitors and stops on stays the residual of the
    original system and the tolerance keeps its meaning.

    .. note::
       A **diagonal (Jacobi)** scaling is the obvious thing to put here and it
       does not work — measured 0.92x to 1.10x against no preconditioner across
       κ = 2, 5, 10, h = 1 and 0.5, and step sizes from 2x to 32x the explicit
       limit.  The reason is that this operator is nearly constant-coefficient:
       within each block the diagonal is the same number at every node, so
       scaling by it is close to a uniform rescale and changes no eigenvalue
       ratio.  The conditioning comes from the Laplacian's spread across
       spatial frequencies, which only a method that treats frequencies
       differently touches — multigrid, or a sine-transform solve, which the
       structured Cartesian grid here would suit.  This parameter is the seam
       such a preconditioner plugs into.
    """
    N = len(b)
    if f_base is None:
        f_base = eval_f(x_lin)
    if max_iter is None:
        max_iter = max(N, int(round(0.2 * N)))

    x = np.zeros_like(b)
    r = b.copy()
    r_norms = [np.linalg.norm(r, 2)]

    if r_norms[0] == 0.0:
        return x

    p_list: list[NDArray] = []
    Ap_list: list[NDArray] = []

    k = 0
    while r_norms[-1] / r_norms[0] > tol and k < max_iter:
        k += 1
        # The search direction lives in the preconditioned space; ``pk`` below
        # is the direction actually added to x, so it carries the scaling and
        # nothing has to be undone at the end.
        pk = r * scaling if scaling is not None else r.copy()

        epsilon = eps_mf * (1.0 + np.linalg.norm(x_lin)) / np.linalg.norm(pk)
        f_pert = eval_f(x_lin + epsilon * pk)
        Jv = (f_pert - f_base) / epsilon
        Apk = pk - (dt / 2.0) * Jv  # (I - dt/2 J) * p

        for j in range(len(Ap_list)):
            beta = np.vdot(Apk, Ap_list[j]).real
            pk = pk - beta * p_list[j]
            Apk = Apk - beta * Ap_list[j]

        norm_Ap = np.linalg.norm(Apk, 2)
        if norm_Ap < 1e-14:
            break
        Apk /= norm_Ap
        pk /= norm_Ap

        p_list.append(pk)
        Ap_list.append(Apk)
        _truncate(p_list, Ap_list, max_krylov)

        alpha = np.vdot(r, Apk).real
        x = x + alpha * pk
        r = r - alpha * Apk
        r_norms.append(np.linalg.norm(r, 2))

    if r_norms[-1] > tol * r_norms[0]:
        return np.array([], dtype=np.complex128)
    return x

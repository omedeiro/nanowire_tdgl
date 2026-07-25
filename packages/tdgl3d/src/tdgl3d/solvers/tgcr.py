"""Truncated Generalized Conjugate Residual (TGCR) — matrix-free variant.

Python port of ``tgcr_MatrixFree.m`` and ``tgcr_MatrixFreetrap.m``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray


def tgcr_matrix_free(
    eval_f: Callable[[NDArray], NDArray],
    x_lin: NDArray[np.complexfloating],
    b: NDArray[np.complexfloating],
    tol: float = 1e-4,
    max_iter: int | None = None,
    eps_mf: float = 1e-4,
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

    Returns
    -------
    x : ndarray
        Approximate solution, or empty array on failure.
    """
    N = len(b)
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
        f_base = eval_f(x_lin)
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
) -> NDArray[np.complex128]:
    """TGCR for the trapezoidal implicit system ``(I - dt/2 J) δx = b``.

    The matrix-vector product is approximated as:
        A·p ≈ p - (dt/2) * (f(x+εp) - f(x)) / ε
    """
    N = len(b)
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

        epsilon = eps_mf * (1.0 + np.linalg.norm(x_lin)) / np.linalg.norm(pk)
        f_pert = eval_f(x_lin + epsilon * pk)
        f_base = eval_f(x_lin)
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

        alpha = np.vdot(r, Apk).real
        x = x + alpha * pk
        r = r - alpha * Apk
        r_norms.append(np.linalg.norm(r, 2))

    if r_norms[-1] > tol * r_norms[0]:
        return np.array([], dtype=np.complex128)
    return x

"""Newton-GCR solver — Python port of ``NewtonGCR.m`` and ``NewtonGCRtrap.m``."""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from .tgcr import tgcr_matrix_free, tgcr_matrix_free_trap


def newton_gcr(
    eval_f: Callable[[NDArray], NDArray],
    x0: NDArray[np.complexfloating],
    *,
    tol_f: float = 1e-3,
    tol_dx: float = 1e-3,
    max_iter: int = 20,
    tol_gcr: float = 1e-4,
    eps_mf: float = 1e-4,
    verbose: bool = False,
) -> tuple[NDArray[np.complex128], bool, int]:
    """Newton iteration with matrix-free GCR inner solve.

    Solves ``f(x) = 0``.

    Parameters
    ----------
    eval_f : callable
        ``eval_f(x)`` → f(x).
    x0 : ndarray
        Initial guess.
    tol_f, tol_dx : float
        Convergence tolerances on ||f||∞ and ||Δx||∞.
    max_iter : int
        Maximum Newton iterations.
    tol_gcr, eps_mf : float
        GCR tolerance and FD perturbation.
    verbose : bool
        Print iteration info.

    Returns
    -------
    x : ndarray
        Solution (last iterate).
    converged : bool
    iterations : int
    """
    x = np.array(x0, dtype=np.complex128)
    f = eval_f(x)
    errf = np.linalg.norm(f, np.inf)
    err_dx = 0.0

    for k in range(1, max_iter + 1):
        dx = tgcr_matrix_free(eval_f, x, -f, tol=tol_gcr, eps_mf=eps_mf)
        if dx.size == 0:
            if verbose:
                print(f"  Newton iter {k}: GCR did not converge")
            return x, False, k

        x = x + dx
        f = eval_f(x)
        errf = np.linalg.norm(f, np.inf)
        err_dx = np.linalg.norm(dx, np.inf)

        if verbose:
            print(f"  Newton iter {k}: ||f||∞ = {errf:.3e}, ||Δx||∞ = {err_dx:.3e}")

        if errf <= tol_f and err_dx <= tol_dx:
            return x, True, k

    return x, False, max_iter


def newton_gcr_trap(
    eval_f: Callable[[NDArray], NDArray],
    x0: NDArray[np.complexfloating],
    gamma: NDArray[np.complexfloating],
    dt: float,
    *,
    tol_f: float = 1e-3,
    tol_dx: float = 1e-3,
    max_iter: int = 20,
    tol_gcr: float = 1e-4,
    eps_mf: float = 1e-4,
    verbose: bool = False,
) -> tuple[NDArray[np.complex128], bool, int]:
    """Newton iteration for the **trapezoidal** implicit system:

        g(x) = x - (dt/2) f(x) - γ = 0

    where γ = x_n + (dt/2) f(x_n).
    """
    x = np.array(x0, dtype=np.complex128)

    for k in range(1, max_iter + 1):
        f = eval_f(x)
        g = x - (dt / 2.0) * f - gamma
        errf = np.linalg.norm(g, np.inf)

        dx = tgcr_matrix_free_trap(eval_f, x, -g, dt, tol=tol_gcr, eps_mf=eps_mf)
        if dx.size == 0:
            if verbose:
                print(f"  Newton-Trap iter {k}: GCR did not converge")
            return x, False, k

        x = x + dx
        f = eval_f(x)
        g = x - (dt / 2.0) * f - gamma
        errf = np.linalg.norm(g, np.inf)
        err_dx = np.linalg.norm(dx, np.inf)

        if verbose:
            print(f"  Newton-Trap iter {k}: ||g||∞ = {errf:.3e}, ||Δx||∞ = {err_dx:.3e}")

        if errf <= tol_f and err_dx <= tol_dx:
            return x, True, k

    return x, False, max_iter

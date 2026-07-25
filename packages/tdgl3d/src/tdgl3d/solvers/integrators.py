"""Time-stepping schemes — Forward Euler and Trapezoidal.

Python ports of ``ForwardEuler.m`` and ``Trapezoidal.m``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from typing import Optional

from ..core.material import MaterialMap
from ..core.parameters import SimulationParameters
from ..mesh.indices import GridIndices
from ..physics.rhs import BoundaryVectors, eval_f
from .newton import newton_gcr_trap


def _make_eval_f(
    params: SimulationParameters,
    idx: GridIndices,
    u: BoundaryVectors,
    material: Optional[MaterialMap] = None,
) -> Callable[[NDArray], NDArray]:
    """Return a closure ``f(X) -> dX/dt`` for the current boundary conditions."""
    def f(X: NDArray) -> NDArray:
        return eval_f(X, params, idx, u, material=material)
    return f


def forward_euler(
    x0: NDArray[np.complexfloating],
    params: SimulationParameters,
    idx: GridIndices,
    eval_u: Callable[[float, NDArray], BoundaryVectors],
    t_start: float,
    t_stop: float,
    dt: float,
    *,
    save_every: int = 1,
    progress: bool = True,
    material: Optional[MaterialMap] = None,
) -> tuple[NDArray, NDArray]:
    """Explicit Forward-Euler time integration.

    Parameters
    ----------
    x0 : ndarray
        Initial state vector.
    params : SimulationParameters
    idx : GridIndices
    eval_u : callable
        ``eval_u(t, X)`` → BoundaryVectors for time *t*.
    t_start, t_stop, dt : float
        Integration window and time step.
    save_every : int
        Save every *n*-th step to the history.
    progress : bool
        Show a tqdm progress bar.

    Returns
    -------
    times : ndarray, shape (n_saved,)
    X_history : ndarray, shape (n_state, n_saved)
    """
    n_steps = int(np.ceil((t_stop - t_start) / dt))
    X = np.array(x0, dtype=np.complex128)

    times_list = [t_start]
    history_list = [X.copy()]

    t = t_start
    rng = tqdm(range(n_steps), desc="Forward Euler", disable=not progress)
    for step in rng:
        dt_actual = min(dt, t_stop - t)
        u = eval_u(t, X)
        f = eval_f(X, params, idx, u, material=material)
        X = X + dt_actual * f
        t += dt_actual

        if (step + 1) % save_every == 0 or step == n_steps - 1:
            times_list.append(t)
            history_list.append(X.copy())

    return np.array(times_list), np.column_stack(history_list)


def trapezoidal(
    x0: NDArray[np.complexfloating],
    params: SimulationParameters,
    idx: GridIndices,
    eval_u: Callable[[float, NDArray], BoundaryVectors],
    t_start: float,
    t_stop: float,
    dt: float,
    *,
    newton_tol_f: float = 1e-3,
    newton_tol_dx: float = 1e-3,
    newton_max_iter: int = 20,
    tol_gcr: float = 1e-4,
    eps_mf: float = 1e-4,
    save_every: int = 1,
    adaptive: bool = True,
    dt_min: float = 1e-8,
    progress: bool = True,
    verbose: bool = False,
    material: Optional[MaterialMap] = None,
) -> tuple[NDArray, NDArray]:
    """Implicit Trapezoidal time integration with Newton-GCR.

    Uses adaptive time-stepping: if Newton fails to converge the step size
    is reduced by a factor of 10 (down to *dt_min*).

    Parameters
    ----------
    x0 : ndarray
        Initial state vector.
    params, idx : grid info
    eval_u : callable
        ``eval_u(t, X)`` → BoundaryVectors.
    t_start, t_stop, dt : float
    newton_tol_f, newton_tol_dx, newton_max_iter : Newton params
    tol_gcr, eps_mf : GCR params
    save_every : int
    adaptive : bool
        Allow dt reduction on Newton failure.
    dt_min : float
        Smallest allowed dt before raising.
    progress : bool
    verbose : bool

    Returns
    -------
    times, X_history
    """
    X = np.array(x0, dtype=np.complex128)
    t = t_start
    current_dt = dt

    times_list = [t]
    history_list = [X.copy()]
    step = 0

    pbar = tqdm(total=t_stop - t_start, desc="Trapezoidal", disable=not progress, unit="t")

    while t < t_stop - 1e-14:
        dt_actual = min(current_dt, t_stop - t)
        u = eval_u(t, X)
        f_func = _make_eval_f(params, idx, u, material=material)

        # Explicit predictor
        f_n = f_func(X)
        X_pred = X + dt_actual * f_n
        gamma = X + (dt_actual / 2.0) * f_n

        # Newton solve for implicit correction
        x_new, converged, iters = newton_gcr_trap(
            f_func,
            X_pred,
            gamma,
            dt_actual,
            tol_f=newton_tol_f,
            tol_dx=newton_tol_dx,
            max_iter=newton_max_iter,
            tol_gcr=tol_gcr,
            eps_mf=eps_mf,
            verbose=verbose,
        )

        if converged:
            X = x_new
            t += dt_actual
            step += 1
            pbar.update(dt_actual)

            if step % save_every == 0:
                times_list.append(t)
                history_list.append(X.copy())

            # Restore dt if it was reduced
            current_dt = dt
        else:
            if not adaptive:
                raise RuntimeError(
                    f"Newton did not converge at t={t:.6e} with dt={dt_actual:.3e}."
                )
            current_dt /= 10.0
            if current_dt < dt_min:
                raise RuntimeError(
                    f"dt reduced below dt_min={dt_min:.1e} at t={t:.6e}; aborting."
                )
            if verbose:
                print(f"  dt reduced to {current_dt:.3e}")

    pbar.close()

    # Always include final state
    if times_list[-1] < t:
        times_list.append(t)
        history_list.append(X.copy())

    return np.array(times_list), np.column_stack(history_list)

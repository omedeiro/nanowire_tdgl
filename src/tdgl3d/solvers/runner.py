"""High-level ``solve()`` entry point — the main user-facing function."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ..core.device import Device
from ..core.parameters import SimulationParameters
from ..core.solution import Solution
from ..core.state import StateVector
from ..mesh.indices import GridIndices
from ..physics.applied_field import AppliedField, build_boundary_field_vectors
from ..physics.rhs import BoundaryVectors
from .integrators import forward_euler, trapezoidal


def _make_eval_u(
    applied_field: AppliedField,
    params: SimulationParameters,
    idx: GridIndices,
    t_stop: float,
):
    """Return a callable ``eval_u(t, X) -> BoundaryVectors``."""

    def eval_u(t: float, X: NDArray) -> BoundaryVectors:
        bx, by, bz = applied_field.evaluate(t, t_stop)
        Bx_vec, By_vec, Bz_vec = build_boundary_field_vectors(bx, by, bz, params, idx)
        return BoundaryVectors(Bx_vec, By_vec, Bz_vec)

    return eval_u


def solve(
    device: Device,
    t_start: float = 0.0,
    t_stop: float = 10.0,
    dt: float = 0.05,
    method: Literal["euler", "trapezoidal"] = "trapezoidal",
    x0: NDArray | StateVector | None = None,
    *,
    save_every: int = 1,
    progress: bool = True,
    verbose: bool = False,
    # Newton / GCR options (trapezoidal only)
    newton_tol_f: float = 1e-3,
    newton_tol_dx: float = 1e-3,
    newton_max_iter: int = 20,
    tol_gcr: float = 1e-4,
    eps_mf: float = 1e-4,
    adaptive: bool = True,
) -> Solution:
    """Run a TDGL simulation.

    Parameters
    ----------
    device : Device
        The device to simulate (contains parameters, indices, applied field).
    t_start, t_stop : float
        Simulation time window.
    dt : float
        Time step.
    method : ``"euler"`` or ``"trapezoidal"``
        Time integration scheme.
    x0 : ndarray or StateVector, optional
        Initial state.  Defaults to uniform superconducting (|ψ|=1, φ=0).
    save_every : int
        Save every n-th time step.
    progress : bool
        Show progress bar.
    verbose : bool
        Print Newton iteration info.
    newton_tol_f, newton_tol_dx, newton_max_iter : Newton params.
    tol_gcr, eps_mf : GCR params.
    adaptive : bool
        Allow adaptive dt reduction on Newton failure.

    Returns
    -------
    Solution
        Contains time array, state history, and post-processing methods.
    """
    params = device.params
    idx = device.idx
    material = getattr(device, 'material', None)

    # Initial state
    if x0 is None:
        x0_arr = StateVector.uniform_superconducting(params).data
        # If device has a trilayer, suppress ψ in the insulator
        if material is not None:
            sv = StateVector(x0_arr, params)
            sv.psi[:] *= material.interior_sc_mask
        x0_arr = x0_arr if material is None else sv.data
    elif isinstance(x0, StateVector):
        x0_arr = x0.data
    else:
        x0_arr = np.asarray(x0, dtype=np.complex128)

    eval_u = _make_eval_u(device.applied_field, params, idx, t_stop)

    if method == "euler":
        times, X_hist = forward_euler(
            x0_arr, params, idx, eval_u, t_start, t_stop, dt,
            save_every=save_every, progress=progress,
            material=material,
        )
    elif method == "trapezoidal":
        times, X_hist = trapezoidal(
            x0_arr, params, idx, eval_u, t_start, t_stop, dt,
            newton_tol_f=newton_tol_f,
            newton_tol_dx=newton_tol_dx,
            newton_max_iter=newton_max_iter,
            tol_gcr=tol_gcr,
            eps_mf=eps_mf,
            save_every=save_every,
            adaptive=adaptive,
            progress=progress,
            verbose=verbose,
            material=material,
        )
    else:
        raise ValueError(f"Unknown method {method!r}. Use 'euler' or 'trapezoidal'.")

    return Solution(times=times, states=X_hist, params=params, idx=idx)

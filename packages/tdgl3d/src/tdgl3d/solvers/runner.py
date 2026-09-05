"""High-level ``solve()`` entry point — the main user-facing function."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ..core.device import Device
from ..core.parameters import SimulationParameters
from ..core.solution import Solution
from ..core.state import StateVector
from ..io.logging import TimingContext, create_run_metadata
from ..mesh.indices import GridIndices
from ..physics.applied_field import AppliedField, build_boundary_field_vectors
from ..physics.rhs import BoundaryVectors
from .history import make_history
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
    # Initial state noise
    noise_amplitude: float = 0.01,
    noise_seed: int | None = None,
    # Logging options
    log_metadata: bool = True,
    log_dir: str | Path = "logs",
    # Frame storage
    stream_path: str | Path | None = None,
    precision: Literal["double", "single"] = "double",
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
        Initial state.  Defaults to uniform superconducting with symmetry-breaking
        noise (|ψ|≈1, φ=0, small random perturbation).
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
    noise_amplitude : float, default 0.01
        Amplitude of complex Gaussian noise added to ψ in superconducting regions
        when generating the initial state (only used when ``x0 is None``).
        Set to 0.0 for a perfectly uniform state.
    noise_seed : int, optional
        Random seed for initial-state noise reproducibility.
    log_metadata : bool
        If True, create run metadata and auto-save to JSON.
    log_dir : str or Path
        Directory for log files (default: "logs").
    stream_path : str or Path, optional
        Write saved frames to this HDF5 file as they are produced, instead of
        holding them all in memory.  A frame is the whole state vector — 116 MB
        on a 1.8 M-node grid, 960 MB at 15 M — so on a large mesh the history is
        what limits how long a run can be, not the solver.  The file is a
        complete artifact: ``Solution.load`` reads it directly, and the returned
        solution slices it lazily rather than reading it back.  Call
        ``Solution.close()`` (or use it as a context manager) when done.
    precision : ``"double"`` or ``"single"``
        Width of the state.  ``"single"`` runs in complex64, which halves both
        the memory a mesh needs and the bandwidth the evaluation is limited by
        — worth roughly 1.5x in time and a 2x larger mesh for the same RAM.

        It is not free accuracy-wise and is not the default.  Forward Euler is
        first-order and takes tens of thousands of steps, so round-off
        accumulates; ``test_precision.py`` measures the divergence against
        double on a real relaxation.  Use it for exploring large geometries,
        and confirm any number you intend to publish at double precision.
    Returns
    -------
    Solution
        Contains time array, state history, post-processing methods, and metadata.
    """
    params = device.params
    idx = device.idx
    material = getattr(device, 'material', None)

    # Initial state
    if x0 is None:
        # Use device.initial_state() with symmetry-breaking noise
        x0_arr = device.initial_state(
            noise_amplitude=noise_amplitude, seed=noise_seed,
        ).data
    elif isinstance(x0, StateVector):
        x0_arr = x0.data
    else:
        x0_arr = np.asarray(x0, dtype=np.complex128)

    if precision not in ("double", "single"):
        raise ValueError(
            f"Unknown precision {precision!r}. Use 'double' or 'single'."
        )
    state_dtype = np.complex64 if precision == "single" else np.complex128
    x0_arr = np.asarray(x0_arr, dtype=state_dtype)

    eval_u = _make_eval_u(device.applied_field, params, idx, t_stop)

    if method not in ("euler", "trapezoidal"):
        raise ValueError(f"Unknown method {method!r}. Use 'euler' or 'trapezoidal'.")

    n_steps_est = int(np.ceil((t_stop - t_start) / dt)) if dt > 0 else 1
    history = make_history(
        x0_arr.size, n_steps_est // max(save_every, 1) + 2, stream_path,
        dtype=state_dtype,
    )

    # Wrap integration with timing
    try:
        with TimingContext() as timer:
            if method == "euler":
                times, X_hist = forward_euler(
                    x0_arr, params, idx, eval_u, t_start, t_stop, dt,
                    save_every=save_every, progress=progress,
                    material=material, history=history,
                )
            else:
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
                    history=history,
                )
    except BaseException:
        history.close()
        raise

    # Create metadata
    metadata_dict = None
    if log_metadata:
        metadata = create_run_metadata(
            params=params,
            device=device,
            method=method,
            dt=dt,
            t_final=t_stop,
            wall_time=timer.elapsed,
            atol=newton_tol_f if method == "trapezoidal" else None,
            rtol=newton_tol_dx if method == "trapezoidal" else None,
            total_steps=len(times),
        )
        metadata_dict = metadata.to_dict()

        # Auto-save metadata to JSON
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_file = log_path / f"run_{timestamp}.json"
        metadata.save_json(json_file)

    if stream_path is not None:
        # Finish the file so it stands on its own: frames alone are not
        # interpretable without the grid they were computed on.
        history.write_solution_metadata(params, idx, metadata_dict)

    return Solution(
        times=times, states=X_hist, params=params, idx=idx, device=device,
        metadata=metadata_dict,
        _history=history if stream_path is not None else None,
    )

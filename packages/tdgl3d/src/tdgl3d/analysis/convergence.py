"""Steady-state detection and convergence monitoring for TDGL simulations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..core.device import Device
    from ..core.solution import Solution


def compute_convergence_metrics(
    solution: Solution,
    device: Optional[Device] = None,
    step: int = -1,
    window_size: int = 10,
) -> dict[str, float]:
    """Compute convergence metrics for a solution at a given step.

    Calculates relative changes in |ψ|² and supercurrent density over
    a rolling window to assess if the system has reached steady state.

    Parameters
    ----------
    solution : Solution
        The simulation result
    device : Device, optional
        The device (needed for supercurrent calculation)
    step : int, default -1
        Which saved step to analyze (negative indices from end)
    window_size : int, default 10
        Number of saved steps to look back for comparison

    Returns
    -------
    metrics : dict
        Dictionary with keys:
        - 'psi2_mean_current': mean |ψ|² at current step
        - 'psi2_mean_past': mean |ψ|² at past step
        - 'psi2_rel_change': relative change in mean |ψ|²
        - 'current_mean_current': mean |J_s| at current step (if device provided)
        - 'current_mean_past': mean |J_s| at past step (if device provided)
        - 'current_rel_change': relative change in mean |J_s| (if device provided)
    """
    metrics = {}

    # Adjust step index for negative indexing
    if step < 0:
        step = solution.n_steps + step

    # Ensure we have enough history
    past_step = max(0, step - window_size)

    # --- |ψ|² convergence ---
    psi_current = solution.psi(step=step)
    psi_past = solution.psi(step=past_step)

    psi2_current = np.abs(psi_current) ** 2
    psi2_past = np.abs(psi_past) ** 2

    # Compute mean only over superconducting regions if device available
    if device is not None and device.material is not None:
        sc_mask = device.material.interior_sc_mask
        psi2_mean_current = float(np.mean(psi2_current[sc_mask > 0]))
        psi2_mean_past = float(np.mean(psi2_past[sc_mask > 0]))
    else:
        psi2_mean_current = float(np.mean(psi2_current))
        psi2_mean_past = float(np.mean(psi2_past))

    # Relative change
    if psi2_mean_current > 1e-10:
        psi2_rel_change = abs(psi2_mean_current - psi2_mean_past) / psi2_mean_current
    else:
        psi2_rel_change = float('inf')

    metrics['psi2_mean_current'] = psi2_mean_current
    metrics['psi2_mean_past'] = psi2_mean_past
    metrics['psi2_rel_change'] = psi2_rel_change

    # --- Supercurrent convergence (if device available) ---
    if device is not None:
        try:
            # Use the Solution method which correctly expands interior state
            Jx_cur, Jy_cur, Jz_cur = solution.supercurrent_density(step=step)
            J_mag_current = np.sqrt(
                Jx_cur**2 + Jy_cur**2 + (Jz_cur**2 if Jz_cur is not None else 0)
            )

            Jx_past, Jy_past, Jz_past = solution.supercurrent_density(step=past_step)
            J_mag_past = np.sqrt(
                Jx_past**2 + Jy_past**2 + (Jz_past**2 if Jz_past is not None else 0)
            )

            # Mean current magnitude
            if device.material is not None:
                sc_mask = device.material.interior_sc_mask
                current_mean_current = float(np.mean(J_mag_current[sc_mask > 0]))
                current_mean_past = float(np.mean(J_mag_past[sc_mask > 0]))
            else:
                current_mean_current = float(np.mean(J_mag_current))
                current_mean_past = float(np.mean(J_mag_past))

            # Relative change
            if current_mean_current > 1e-10:
                current_rel_change = (
                    abs(current_mean_current - current_mean_past) / current_mean_current
                )
            else:
                current_rel_change = float('inf')

            metrics['current_mean_current'] = current_mean_current
            metrics['current_mean_past'] = current_mean_past
            metrics['current_rel_change'] = current_rel_change

        except Exception as e:
            # If supercurrent calculation fails, just skip it
            metrics['current_error'] = str(e)

    return metrics


def convergence_history(
    solution: Solution,
    device: Optional[Device] = None,
    window_size: int = 10,
    start_step: int = 20,
    stride: int = 1,
) -> tuple[NDArray, NDArray, NDArray]:
    """Sample the convergence metrics across a run.

    :func:`compute_convergence_metrics` costs two supercurrent evaluations per
    call, so a long run is usually sampled rather than walked step by step --
    hence ``stride``.

    Parameters
    ----------
    solution : Solution
        The simulation result
    device : Device, optional
        The device (needed for supercurrent-based convergence check)
    window_size : int, default 10
        Number of saved steps to look back for comparison
    start_step : int, default 20
        First saved step to sample (allow the initial transient to pass)
    stride : int, default 1
        Sample every ``stride``-th saved step

    Returns
    -------
    steps : ndarray of int
        The sampled saved-step indices.
    psi2_rel_change, current_rel_change : ndarray of float
        Relative change at each sampled step.  ``current_rel_change`` is NaN
        wherever the supercurrent was unavailable (no device, or the
        evaluation raised), which the steady-state scan reads as "no opinion"
        rather than as "not converged".
    """
    steps = np.arange(start_step, solution.n_steps, stride, dtype=int)
    psi2_rel_change = np.full(steps.size, np.nan)
    current_rel_change = np.full(steps.size, np.nan)

    for i, step in enumerate(steps):
        metrics = compute_convergence_metrics(
            solution, device, step=int(step), window_size=window_size
        )
        psi2_rel_change[i] = metrics["psi2_rel_change"]
        if "current_rel_change" in metrics:
            current_rel_change[i] = metrics["current_rel_change"]

    return steps, psi2_rel_change, current_rel_change


def first_sustained_sample(
    psi2_rel_change: NDArray,
    current_rel_change: NDArray,
    psi_threshold: float = 1e-4,
    current_threshold: float = 1e-4,
    min_sustained: int = 1,
) -> int:
    """Index of the first sample that begins a sustained converged run.

    Scans the traces from :func:`convergence_history` for the first run of
    ``min_sustained`` consecutive samples with both relative changes below
    their thresholds, and returns the index of the first sample in that run
    (-1 if there is no such run).

    A NaN in ``current_rel_change`` means the supercurrent was unavailable at
    that sample, which counts as "no opinion" rather than as "not converged";
    a NaN in ``psi2_rel_change`` breaks the run.

    Parameters
    ----------
    psi2_rel_change, current_rel_change : ndarray
        Relative-change traces, as returned by :func:`convergence_history`
    psi_threshold : float, default 1e-4
        Max relative change in mean |ψ|² to consider steady
    current_threshold : float, default 1e-4
        Max relative change in mean |J_s| to consider steady
    min_sustained : int, default 1
        How many consecutive samples must be below threshold.  The default of
        1 accepts the first sample that dips below, which a single transient
        is enough to produce.
    """
    consecutive = 0
    for i in range(psi2_rel_change.size):
        psi_value = psi2_rel_change[i]
        psi_converged = not np.isnan(psi_value) and psi_value < psi_threshold

        current_value = current_rel_change[i]
        current_converged = np.isnan(current_value) or current_value < current_threshold

        if psi_converged and current_converged:
            consecutive += 1
            if consecutive >= min_sustained:
                return i - min_sustained + 1
        else:
            consecutive = 0

    return -1


def check_steady_state(
    solution: Solution,
    device: Optional[Device] = None,
    window_size: int = 10,
    psi_threshold: float = 1e-4,
    current_threshold: float = 1e-4,
    start_step: int = 20,
    min_sustained: int = 1,
    stride: int = 1,
) -> tuple[bool, int, dict]:
    """Detect if simulation has reached steady state.

    Checks if relative change in |ψ|² and |J_s| (if device provided) over
    a rolling window falls below threshold. Scans through all saved steps
    to find when steady state was first achieved.

    Parameters
    ----------
    solution : Solution
        The simulation result
    device : Device, optional
        The device (needed for supercurrent-based convergence check)
    window_size : int, default 10
        Number of saved steps to average over for comparison
    psi_threshold : float, default 1e-4
        Max relative change in mean |ψ|² to consider steady
    current_threshold : float, default 1e-4
        Max relative change in mean |J_s| to consider steady (if device provided)
    start_step : int, default 20
        Don't check for steady state before this step (allow initial transient)
    min_sustained : int, default 1
        How many *consecutive samples* must be below threshold before the run
        counts as steady, and the reported step is the first of them.  The
        default of 1 accepts the first sample that dips below, which a single
        transient -- one relaxation that happens to stall between two vortex
        entries -- is enough to produce.  Raise it to require that the run
        stays down.
    stride : int, default 1
        Sample every ``stride``-th saved step.  Note that ``min_sustained``
        counts samples, so a run of ``min_sustained`` samples spans
        ``min_sustained * stride`` saved steps.

    Returns
    -------
    is_steady : bool
        True if steady state was reached by final step
    steady_step : int
        First step where steady state was achieved (-1 if never reached)
    metrics : dict
        Diagnostic info from the final step:
        - 'psi2_rel_change': relative change in |ψ|² at final step
        - 'current_rel_change': relative change in |J_s| at final step (if device)
        - 'steady_time': simulation time when steady state reached (if applicable)
    """
    n_steps = solution.n_steps
    final_metrics = compute_convergence_metrics(
        solution, device, step=-1, window_size=window_size
    )

    if n_steps < start_step + window_size:
        # Not enough data to assess convergence
        return False, -1, final_metrics

    steps, psi2_rel_change, current_rel_change = convergence_history(
        solution, device, window_size=window_size, start_step=start_step, stride=stride
    )
    sample = first_sustained_sample(
        psi2_rel_change, current_rel_change, psi_threshold=psi_threshold,
        current_threshold=current_threshold, min_sustained=min_sustained,
    )
    steady_step = int(steps[sample]) if sample >= 0 else -1

    if steady_step >= 0:
        final_metrics['steady_time'] = float(solution.times[steady_step])
        final_metrics['steady_step'] = steady_step
        is_steady = True
    else:
        is_steady = False

    return is_steady, steady_step, final_metrics

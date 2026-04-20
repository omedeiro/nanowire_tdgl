"""Steady-state detection and convergence monitoring for TDGL simulations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..core.solution import Solution
    from ..core.device import Device


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
            from ..physics.current_density import eval_supercurrent_density
            
            # Compute supercurrent magnitude at both steps
            Jx_cur, Jy_cur, Jz_cur = eval_supercurrent_density(solution, device, step=step)
            J_mag_current = np.sqrt(Jx_cur**2 + Jy_cur**2 + (Jz_cur**2 if Jz_cur is not None else 0))
            
            Jx_past, Jy_past, Jz_past = eval_supercurrent_density(solution, device, step=past_step)
            J_mag_past = np.sqrt(Jx_past**2 + Jy_past**2 + (Jz_past**2 if Jz_past is not None else 0))
            
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
                current_rel_change = abs(current_mean_current - current_mean_past) / current_mean_current
            else:
                current_rel_change = float('inf')
            
            metrics['current_mean_current'] = current_mean_current
            metrics['current_mean_past'] = current_mean_past
            metrics['current_rel_change'] = current_rel_change
            
        except Exception as e:
            # If supercurrent calculation fails, just skip it
            metrics['current_error'] = str(e)
    
    return metrics


def check_steady_state(
    solution: Solution,
    device: Optional[Device] = None,
    window_size: int = 10,
    psi_threshold: float = 1e-4,
    current_threshold: float = 1e-4,
    start_step: int = 20,
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
    
    if n_steps < start_step + window_size:
        # Not enough data to assess convergence
        final_metrics = compute_convergence_metrics(solution, device, step=-1, window_size=window_size)
        return False, -1, final_metrics
    
    # Scan through steps to find when steady state first achieved
    steady_step = -1
    
    for step in range(start_step, n_steps):
        metrics = compute_convergence_metrics(solution, device, step=step, window_size=window_size)
        
        # Check if converged
        psi_converged = metrics['psi2_rel_change'] < psi_threshold
        
        if device is not None and 'current_rel_change' in metrics:
            current_converged = metrics['current_rel_change'] < current_threshold
            is_converged = psi_converged and current_converged
        else:
            is_converged = psi_converged
        
        if is_converged:
            steady_step = step
            break
    
    # Get final metrics
    final_metrics = compute_convergence_metrics(solution, device, step=-1, window_size=window_size)
    
    if steady_step >= 0:
        final_metrics['steady_time'] = float(solution.times[steady_step])
        final_metrics['steady_step'] = steady_step
        is_steady = True
    else:
        is_steady = False
    
    return is_steady, steady_step, final_metrics

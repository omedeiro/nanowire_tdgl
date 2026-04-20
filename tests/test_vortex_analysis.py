"""Tests for vortex counting and convergence analysis."""

from __future__ import annotations

import numpy as np
import pytest

import tdgl3d
from tdgl3d.analysis.vortex_counting import (
    count_vortices_plaquette,
    count_hole_flux_quanta,
    find_vortex_cores,
)
from tdgl3d.analysis.convergence import check_steady_state, compute_convergence_metrics


def test_convergence_metrics_uniform_state():
    """Test convergence metrics on a uniform steady state."""
    # Create a simple 2D simulation
    params = tdgl3d.SimulationParameters(Nx=10, Ny=10, hx=1.0, hy=1.0, hz=1.0, kappa=2.0)
    device = tdgl3d.Device(params, applied_field=tdgl3d.AppliedField(Bz=0.0))
    
    # Create solution with uniform |ψ| = 1 at all times
    n_steps = 20
    times = np.linspace(0, 10, n_steps)
    states = np.ones((params.n_state, n_steps), dtype=np.complex128)
    
    solution = tdgl3d.core.solution.Solution(
        times=times,
        states=states,
        params=params,
        idx=device.idx,
        device=device,
    )
    
    # Check convergence at final step
    metrics = compute_convergence_metrics(solution, device, step=-1, window_size=5)
    
    # Should have zero relative change (uniform state)
    assert metrics['psi2_rel_change'] < 1e-10
    assert 'psi2_mean_current' in metrics
    assert abs(metrics['psi2_mean_current'] - 1.0) < 1e-6


def test_check_steady_state_constant():
    """Test steady-state detection on constant state."""
    params = tdgl3d.SimulationParameters(Nx=10, Ny=10, hx=1.0, hy=1.0, hz=1.0, kappa=2.0)
    device = tdgl3d.Device(params, applied_field=tdgl3d.AppliedField(Bz=0.0))
    
    # Constant state
    n_steps = 50
    times = np.linspace(0, 25, n_steps)
    states = np.ones((params.n_state, n_steps), dtype=np.complex128)
    
    solution = tdgl3d.core.solution.Solution(
        times=times,
        states=states,
        params=params,
        idx=device.idx,
        device=device,
    )
    
    # Should detect steady state quickly
    is_steady, steady_step, metrics = check_steady_state(
        solution, device, window_size=5, psi_threshold=1e-4, start_step=10
    )
    
    assert is_steady
    assert steady_step >= 10  # After start_step
    assert 'steady_time' in metrics


def test_check_steady_state_evolving():
    """Test that evolving state is not marked as steady."""
    params = tdgl3d.SimulationParameters(Nx=10, Ny=10, hx=1.0, hy=1.0, hz=1.0, kappa=2.0)
    device = tdgl3d.Device(params, applied_field=tdgl3d.AppliedField(Bz=0.0))
    
    # Linearly increasing state (never reaches steady)
    n_steps = 30
    times = np.linspace(0, 15, n_steps)
    states = np.zeros((params.n_state, n_steps), dtype=np.complex128)
    
    for i in range(n_steps):
        # Linearly increase amplitude
        amplitude = 0.5 + 0.5 * (i / n_steps)
        states[:params.n_interior, i] = amplitude * (1 + 0j)
    
    solution = tdgl3d.core.solution.Solution(
        times=times,
        states=states,
        params=params,
        idx=device.idx,
        device=device,
    )
    
    # Should NOT detect steady state
    is_steady, steady_step, metrics = check_steady_state(
        solution, device, window_size=5, psi_threshold=1e-4, start_step=10
    )
    
    # Expect not steady (state is continuously changing)
    assert not is_steady
    assert steady_step == -1


def test_vortex_counting_no_vortex():
    """Test vortex counting on uniform state (no vortices)."""
    params = tdgl3d.SimulationParameters(Nx=20, Ny=20, hx=1.0, hy=1.0, hz=1.0, kappa=2.0)
    device = tdgl3d.Device(params, applied_field=tdgl3d.AppliedField(Bz=0.0))
    
    # Uniform ψ = 1 (no phase winding)
    times = np.array([0.0])
    states = np.ones((params.n_state, 1), dtype=np.complex128)
    
    solution = tdgl3d.core.solution.Solution(
        times=times,
        states=states,
        params=params,
        idx=device.idx,
        device=device,
    )
    
    # Count vortices
    n_vort, vort_pos, winding = count_vortices_plaquette(solution, device, slice_z=0, step=0)
    
    assert n_vort == 0
    assert len(vort_pos) == 0
    assert len(winding) == 0


def test_vortex_counting_single_vortex():
    """Test vortex counting on a state with single vortex at center."""
    params = tdgl3d.SimulationParameters(Nx=30, Ny=30, hx=1.0, hy=1.0, hz=1.0, kappa=2.0)
    device = tdgl3d.Device(params, applied_field=tdgl3d.AppliedField(Bz=0.0))
    
    # Create state with vortex at center
    times = np.array([0.0])
    states = np.zeros((params.n_state, 1), dtype=np.complex128)
    
    # Set ψ with radial phase winding around center
    nx, ny = params.Nx - 1, params.Ny - 1
    cx, cy = nx / 2, ny / 2  # center
    
    for i in range(nx):
        for j in range(ny):
            idx_interior = i + j * nx
            
            # Distance from center
            dx = i - cx
            dy = j - cy
            r = np.sqrt(dx**2 + dy**2)
            
            # Phase winds by 2π around center
            theta = np.arctan2(dy, dx)
            
            # Amplitude suppressed near core
            amplitude = np.tanh(r / 2.0)  # Smooth vortex profile
            
            states[idx_interior, 0] = amplitude * np.exp(1j * theta)
    
    solution = tdgl3d.core.solution.Solution(
        times=times,
        states=states,
        params=params,
        idx=device.idx,
        device=device,
    )
    
    # Count vortices
    n_vort, vort_pos, winding = count_vortices_plaquette(
        solution, device, slice_z=0, step=0, winding_threshold=0.5
    )
    
    # Should detect one vortex near center
    assert n_vort >= 1  # At least one vortex
    # Winding should be close to +1
    if n_vort > 0:
        max_winding = np.max(np.abs(winding))
        assert 0.5 < max_winding < 1.5  # Approximately ±1


def test_hole_flux_uniform_field():
    """Test flux through hole with uniform applied field."""
    params = tdgl3d.SimulationParameters(Nx=20, Ny=20, hx=1.0, hy=1.0, hz=1.0, kappa=2.0)
    field = tdgl3d.AppliedField(Bz=0.5)
    device = tdgl3d.Device(params, applied_field=field)
    
    # Run longer simulation to allow field penetration
    x0 = device.initial_state()
    solution = tdgl3d.solve(device, x0=x0, dt=0.01, t_stop=5.0, method='euler', 
                           save_every=10, progress=False)
    
    # Define hole region (center 10×10)
    hole_bounds = (5, 15, 5, 15)
    
    # Count flux through hole
    n_flux = count_hole_flux_quanta(solution, device, hole_bounds, slice_z=0, step=-1)
    
    # Expected flux: Bz * Area / (2π)
    # Area = 10 * 10 * hx * hy = 100
    # Flux = 0.5 * 100 / (2π) ≈ 7.96 Φ₀
    expected_flux = 0.5 * 100 / (2 * np.pi)
    
    # In a uniform SC film, flux will be partially screened
    # Just check that we measure some reasonable flux
    assert n_flux > 0  # Should have some flux
    assert n_flux < expected_flux * 1.2  # But not more than applied (with some tolerance)


def test_find_vortex_cores():
    """Test vortex core finding."""
    params = tdgl3d.SimulationParameters(Nx=20, Ny=20, hx=1.0, hy=1.0, hz=1.0, kappa=2.0)
    device = tdgl3d.Device(params, applied_field=tdgl3d.AppliedField(Bz=0.0))
    
    # Create state with low |ψ| at center
    times = np.array([0.0])
    states = np.ones((params.n_state, 1), dtype=np.complex128)
    
    nx, ny = params.Nx - 1, params.Ny - 1
    cx, cy = nx / 2, ny / 2
    
    for i in range(nx):
        for j in range(ny):
            idx_interior = i + j * nx
            dx = i - cx
            dy = j - cy
            r = np.sqrt(dx**2 + dy**2)
            
            # Gaussian dip at center
            amplitude = 1.0 - 0.95 * np.exp(-r**2 / 4.0)
            states[idx_interior, 0] = amplitude
    
    solution = tdgl3d.core.solution.Solution(
        times=times,
        states=states,
        params=params,
        idx=device.idx,
        device=device,
    )
    
    # Find cores
    cores = find_vortex_cores(solution, device, slice_z=0, step=0, threshold=0.2)
    
    # Should find one core near center
    assert len(cores) >= 1
    if len(cores) > 0:
        # Closest core should be near center
        distances = np.sqrt((cores[:, 0] - cx)**2 + (cores[:, 1] - cy)**2)
        min_dist = np.min(distances)
        assert min_dist < 3.0  # Within 3 grid points of center

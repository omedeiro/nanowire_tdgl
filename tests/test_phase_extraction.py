"""Tests for phase extraction with NaN masking.

Verifies that:
1. Phase is correctly computed from order parameter
2. Phase is NaN where |ψ|² < threshold (vortex cores, insulators, holes)
3. Phase is well-defined in superconducting regions
4. Threshold parameter works correctly
"""

from __future__ import annotations

import numpy as np
import pytest

import tdgl3d


def test_phase_basic():
    """Verify basic phase extraction from uniform superconducting state."""
    params = tdgl3d.SimulationParameters(Nx=8, Ny=8, Nz=1, hx=1.0, hy=1.0, hz=1.0, kappa=2.0)
    device = tdgl3d.Device(params, applied_field=tdgl3d.AppliedField(Bz=0.0))
    
    # Start with uniform |ψ|=1, φ=0 (no applied field)
    dt = 0.01
    x0 = device.initial_state()
    solution = tdgl3d.solve(device, x0=x0, dt=dt, t_stop=0.5, method="euler", save_every=10, progress=False)
    
    # Get phase
    phase = solution.phase(step=-1)
    
    # Should return array of length n_interior
    assert phase.shape[0] == params.n_interior
    
    # Phase should be mostly well-defined (not NaN) in bulk SC
    n_nan = np.sum(np.isnan(phase))
    n_total = phase.shape[0]
    nan_fraction = n_nan / n_total
    
    print(f"NaN fraction: {nan_fraction:.2%} ({n_nan}/{n_total})")
    
    # Most of the phase should be defined (allow some NaN at edges or fluctuations)
    assert nan_fraction < 0.1, f"Expected < 10% NaN, got {nan_fraction:.2%}"
    
    # Where phase is defined, it should be in [-π, π]
    phase_valid = phase[~np.isnan(phase)]
    assert np.all(phase_valid >= -np.pi) and np.all(phase_valid <= np.pi), \
        "Phase should be in [-π, π]"


def test_phase_with_field():
    """Verify phase with applied magnetic field (induces spatial variation)."""
    params = tdgl3d.SimulationParameters(Nx=10, Ny=10, Nz=1, hx=1.0, hy=1.0, hz=1.0, kappa=2.0)
    field = tdgl3d.AppliedField(Bz=0.3)
    device = tdgl3d.Device(params, applied_field=field)
    
    dt = 0.01
    x0 = device.initial_state()
    solution = tdgl3d.solve(device, x0=x0, dt=dt, t_stop=2.0, method="euler", save_every=20, progress=False)
    
    # Get phase
    phase = solution.phase(step=-1)
    
    # Phase should show spatial variation due to applied field
    phase_valid = phase[~np.isnan(phase)]
    phase_std = np.std(phase_valid)
    
    print(f"Phase std dev: {phase_std:.4f}")
    
    # With applied field, expect some phase variation
    # (In equilibrium with screening, phase varies to create supercurrents)
    assert phase_std > 0.005, f"Expected phase variation with applied field, got std={phase_std:.4f}"


def test_phase_threshold():
    """Verify that threshold parameter controls NaN masking."""
    params = tdgl3d.SimulationParameters(Nx=8, Ny=8, Nz=1, hx=1.0, hy=1.0, hz=1.0, kappa=2.0)
    field = tdgl3d.AppliedField(Bz=0.2)
    device = tdgl3d.Device(params, applied_field=field)
    
    dt = 0.01
    x0 = device.initial_state()
    solution = tdgl3d.solve(device, x0=x0, dt=dt, t_stop=1.0, method="euler", save_every=10, progress=False)
    
    # Get phase with different thresholds
    phase_strict = solution.phase(step=-1, mask_threshold=0.5)   # High threshold → more NaNs
    phase_relaxed = solution.phase(step=-1, mask_threshold=0.01) # Low threshold → fewer NaNs
    
    n_nan_strict = np.sum(np.isnan(phase_strict))
    n_nan_relaxed = np.sum(np.isnan(phase_relaxed))
    
    print(f"NaN count (threshold=0.5): {n_nan_strict}")
    print(f"NaN count (threshold=0.01): {n_nan_relaxed}")
    
    # Higher threshold should produce more NaNs
    assert n_nan_strict >= n_nan_relaxed, \
        f"Higher threshold should produce more NaNs: {n_nan_strict} vs {n_nan_relaxed}"


def test_phase_in_insulator():
    """Verify that phase is NaN in insulator regions (where |ψ| → 0)."""
    # Create S/I/S trilayer
    Nx, Ny, Nz = 8, 8, 9  # 3 SC + 3 I + 3 SC
    params = tdgl3d.SimulationParameters(Nx=Nx, Ny=Ny, Nz=Nz, hx=1.0, hy=1.0, hz=1.0, kappa=2.0)
    
    trilayer = tdgl3d.Trilayer(
        bottom=tdgl3d.Layer(thickness_z=3, kappa=2.0),
        insulator=tdgl3d.Layer(thickness_z=3, kappa=2.0, is_superconductor=False),
        top=tdgl3d.Layer(thickness_z=3, kappa=2.0),
    )
    
    field = tdgl3d.AppliedField(Bz=0.0)
    device = tdgl3d.Device(params, applied_field=field, trilayer=trilayer)
    
    dt = 0.01
    x0 = device.initial_state()
    solution = tdgl3d.solve(device, x0=x0, dt=dt, t_stop=1.0, method="euler", save_every=10, progress=False)
    
    # Get phase and |ψ|²
    phase = solution.phase(step=-1, mask_threshold=0.02)
    psi = solution.psi(step=-1)
    psi2 = np.abs(psi)**2
    
    # Reshape to 3D grid
    Nx_int, Ny_int, Nz_int = Nx - 1, Ny - 1, Nz - 1
    phase_3d = phase.reshape(Nx_int, Ny_int, Nz_int)
    psi2_3d = psi2.reshape(Nx_int, Ny_int, Nz_int)
    
    # Get z-ranges
    z_ranges = trilayer.z_ranges()
    k_ins_mid = (z_ranges["insulator"][0] + z_ranges["insulator"][1]) // 2
    sz_ins = max(k_ins_mid - 1, 0)
    
    # In insulator slice, phase should be mostly NaN
    phase_ins = phase_3d[:, :, sz_ins]
    psi2_ins = psi2_3d[:, :, sz_ins]
    
    n_nan_ins = np.sum(np.isnan(phase_ins))
    n_total_ins = phase_ins.size
    nan_fraction_ins = n_nan_ins / n_total_ins
    
    print(f"Insulator slice z={sz_ins}:")
    print(f"  |ψ|² max: {psi2_ins.max():.6f}")
    print(f"  Phase NaN fraction: {nan_fraction_ins:.2%}")
    
    # Most of insulator should have NaN phase (since |ψ| → 0)
    assert nan_fraction_ins > 0.8, \
        f"Expected > 80% NaN in insulator, got {nan_fraction_ins:.2%}"
    
    # In SC slice, phase should be mostly defined
    k_sc_mid = (z_ranges["bottom"][0] + z_ranges["bottom"][1]) // 2
    sz_sc = max(k_sc_mid - 1, 0)
    phase_sc = phase_3d[:, :, sz_sc]
    psi2_sc = psi2_3d[:, :, sz_sc]
    
    n_nan_sc = np.sum(np.isnan(phase_sc))
    n_total_sc = phase_sc.size
    nan_fraction_sc = n_nan_sc / n_total_sc
    
    print(f"SC slice z={sz_sc}:")
    print(f"  |ψ|² mean: {psi2_sc.mean():.6f}, min: {psi2_sc.min():.6f}")
    print(f"  Phase NaN fraction: {nan_fraction_sc:.2%}")
    
    # Most of SC should have defined phase
    assert nan_fraction_sc < 0.2, \
        f"Expected < 20% NaN in SC, got {nan_fraction_sc:.2%}"


def test_phase_nan_not_sentinel():
    """Verify that undefined phase uses NaN (not sentinel values like 999)."""
    params = tdgl3d.SimulationParameters(Nx=8, Ny=8, Nz=1, hx=1.0, hy=1.0, hz=1.0, kappa=2.0)
    
    trilayer = tdgl3d.Trilayer(
        bottom=tdgl3d.Layer(thickness_z=1, kappa=2.0),
        insulator=tdgl3d.Layer(thickness_z=0, kappa=2.0, is_superconductor=False),
        top=tdgl3d.Layer(thickness_z=0, kappa=2.0),
    )
    
    device = tdgl3d.Device(params, applied_field=tdgl3d.AppliedField(Bz=0.0), trilayer=trilayer)
    
    # Carve a small hole to suppress |ψ|
    Nx, Ny = params.Nx, params.Ny
    i_center, j_center = Nx // 2, Ny // 2
    m_center = i_center + (Nx + 1) * j_center
    device.material.sc_mask[m_center] = 0.0
    device.material.interior_sc_mask[:] = device.material.sc_mask[device.idx.interior_to_full]
    
    dt = 0.01
    x0 = device.initial_state()
    solution = tdgl3d.solve(device, x0=x0, dt=dt, t_stop=3.0, method="euler", save_every=30, progress=False)
    
    phase = solution.phase(step=-1, mask_threshold=0.05)
    psi = solution.psi(step=-1)
    psi2 = np.abs(psi)**2
    
    # Check |ψ|² in the hole region
    print(f"|ψ|² stats: min={psi2.min():.6f}, max={psi2.max():.6f}, mean={psi2.mean():.6f}")
    
    # Check that no sentinel values (like 999) are used
    phase_finite = phase[np.isfinite(phase)]
    assert np.all(np.abs(phase_finite) <= np.pi), \
        "All finite phase values should be in [-π, π] (no sentinels like 999)"
    
    # Check that NaN is actually used for masking (at least somewhere where |ψ|² is low)
    n_nan = np.sum(np.isnan(phase))
    low_psi2_count = np.sum(psi2 < 0.05)
    
    print(f"NaN count: {n_nan}, Low |ψ|² count (< 0.05): {low_psi2_count}")
    
    # If there are regions with low |ψ|², they should be masked with NaN
    if low_psi2_count > 0:
        assert n_nan > 0, "Expected NaN values where |ψ|² < threshold"
    
    print(f"Using NaN for masking: {n_nan} nodes have phase = NaN")

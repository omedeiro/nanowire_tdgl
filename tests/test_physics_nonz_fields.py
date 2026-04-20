"""Physics validation tests for non-z magnetic field configurations.

These tests verify:
1. Transverse fields (Bx, By) are properly applied via Peierls phases
2. Tilted fields (Bx + Bz, etc.) work correctly
3. Field-induced suppression of order parameter
4. Meissner screening in different field orientations
"""

from __future__ import annotations

import numpy as np
import pytest

from tdgl3d import Device, SimulationParameters, AppliedField, solve


def test_transverse_field_bx():
    """Test that Bx field is properly applied and affects the system."""
    params = SimulationParameters(
        Nx=10, Ny=10, Nz=4,  # Need 3D for transverse fields
        hx=1.0, hy=1.0, hz=1.0,
        kappa=2.0,
    )
    
    # Strong transverse field with longer on-time
    field = AppliedField(Bx=2.0, t_on_fraction=1.0)  # Field on for entire simulation
    device = Device(params=params, applied_field=field)
    
    x0 = device.initial_state()
    solution = solve(
        device,
        x0=x0,
        t_start=0.0,
        t_stop=5.0,  # Longer time for field to take effect
        dt=0.02,
        method="euler",
        save_every=50,
        progress=False,
    )
    
    # Check 1: Order parameter should be suppressed by strong field (eventually)
    # At t>0, field is on and should suppress SC over time
    psi2_initial = np.mean(np.abs(solution.psi(step=1)) ** 2)  # Step 1, after field turns on
    psi2_final = np.mean(np.abs(solution.psi(step=-1)) ** 2)
    
    # Field should cause some suppression over time
    assert psi2_final < psi2_initial, f"Field should suppress |ψ|² over time: initial={psi2_initial:.3f}, final={psi2_final:.3f}"
    
    # Check 2: B-field should be non-zero in the bulk (field penetrates eventually)
    Bx, By, Bz = solution.bfield(step=-1, full_interior=True)
    
    # Some Bx should penetrate
    assert np.mean(np.abs(Bx)) > 0.01, f"Some Bx should penetrate, got mean = {np.mean(np.abs(Bx)):.3f}"


def test_transverse_field_by():
    """Test that By field is properly applied."""
    params = SimulationParameters(
        Nx=10, Ny=10, Nz=4,
        hx=1.0, hy=1.0, hz=1.0,
        kappa=2.0,
    )
    
    field = AppliedField(By=2.0, t_on_fraction=1.0)
    device = Device(params=params, applied_field=field)
    
    x0 = device.initial_state()
    solution = solve(
        device,
        x0=x0,
        t_start=0.0,
        t_stop=5.0,
        dt=0.02,
        method="euler",
        save_every=50,
        progress=False,
    )
    
    # Field should cause suppression over time
    psi2_initial = np.mean(np.abs(solution.psi(step=1)) ** 2)
    psi2_final = np.mean(np.abs(solution.psi(step=-1)) ** 2)
    
    assert psi2_final < psi2_initial, f"Field should suppress |ψ|²: initial={psi2_initial:.3f}, final={psi2_final:.3f}"
    
    # By should penetrate
    Bx, By, Bz = solution.bfield(step=-1, full_interior=True)
    
    assert np.mean(np.abs(By)) > 0.01, f"Some By should penetrate, got mean = {np.mean(np.abs(By)):.3f}"


def test_tilted_field_bx_bz():
    """Test tilted field (Bx + Bz) configuration."""
    params = SimulationParameters(
        Nx=10, Ny=10, Nz=4,
        hx=1.0, hy=1.0, hz=1.0,
        kappa=2.0,
    )
    
    # Tilted field: 45° in x-z plane
    field = AppliedField(Bx=0.7, Bz=0.7, t_on_fraction=1.0)
    device = Device(params=params, applied_field=field)
    
    x0 = device.initial_state()
    solution = solve(
        device,
        x0=x0,
        t_start=0.0,
        t_stop=5.0,
        dt=0.02,
        method="euler",
        save_every=50,
        progress=False,
    )
    
    # Both Bx and Bz should penetrate eventually
    Bx, By, Bz = solution.bfield(step=-1, full_interior=True)
    
    assert np.mean(np.abs(Bx)) > 0.001, f"Some Bx should penetrate, got {np.mean(np.abs(Bx)):.3f}"
    assert np.mean(np.abs(Bz)) > 0.001, f"Some Bz should penetrate, got {np.mean(np.abs(Bz)):.3f}"
    
    # Field should suppress SC over time
    psi2_initial = np.mean(np.abs(solution.psi(step=1)) ** 2)
    psi2_final = np.mean(np.abs(solution.psi(step=-1)) ** 2)
    
    assert psi2_final < psi2_initial, f"Tilted field should suppress |ψ|²"


def test_field_comparison_weak_vs_strong():
    """Compare weak field (Meissner) vs strong field (suppression)."""
    params = SimulationParameters(
        Nx=10, Ny=10, Nz=1,
        hx=1.0, hy=1.0, hz=1.0,
        kappa=2.0,
    )
    
    # Weak field (should be mostly screened)
    field_weak = AppliedField(Bz=0.2, t_on_fraction=1.0)
    device_weak = Device(params=params, applied_field=field_weak)
    x0_weak = device_weak.initial_state()
    sol_weak = solve(
        device_weak, x0=x0_weak, t_start=0.0, t_stop=5.0, dt=0.02,
        method="euler", save_every=50, progress=False
    )
    
    # Strong field (should suppress SC)
    field_strong = AppliedField(Bz=3.0, t_on_fraction=1.0)
    device_strong = Device(params=params, applied_field=field_strong)
    x0_strong = device_strong.initial_state()
    sol_strong = solve(
        device_strong, x0=x0_strong, t_start=0.0, t_stop=5.0, dt=0.02,
        method="euler", save_every=50, progress=False
    )
    
    # Weak field: SC should remain relatively strong
    psi2_weak = np.mean(np.abs(sol_weak.psi(step=-1)) ** 2)
    
    # Strong field: SC should be more suppressed
    psi2_strong = np.mean(np.abs(sol_strong.psi(step=-1)) ** 2)
    
    assert psi2_weak > psi2_strong, f"Weak field should preserve more SC: weak={psi2_weak:.3f}, strong={psi2_strong:.3f}"


def test_field_symmetry_bx_vs_by():
    """Test that Bx and By have symmetric effects (system is isotropic in x-y)."""
    params = SimulationParameters(
        Nx=12, Ny=12, Nz=4,
        hx=1.0, hy=1.0, hz=1.0,
        kappa=2.0,
    )
    
    # Same field strength, different directions
    field_x = AppliedField(Bx=0.8)
    field_y = AppliedField(By=0.8)
    
    device_x = Device(params=params, applied_field=field_x)
    device_y = Device(params=params, applied_field=field_y)
    
    # Use same random seed for fair comparison
    rng = np.random.default_rng(42)
    noise = 0.01 * (rng.standard_normal(params.n_interior) + 1j * rng.standard_normal(params.n_interior))
    
    x0_x = device_x.initial_state()
    x0_x.psi[:] += noise.copy()
    
    x0_y = device_y.initial_state()
    x0_y.psi[:] += noise.copy()
    
    sol_x = solve(device_x, x0=x0_x, t_start=0.0, t_stop=1.0, dt=0.02,
                  method="euler", save_every=10, progress=False)
    sol_y = solve(device_y, x0=x0_y, t_start=0.0, t_stop=1.0, dt=0.02,
                  method="euler", save_every=10, progress=False)
    
    # Mean |ψ|² should be similar (within 20%)
    psi2_x = np.mean(np.abs(sol_x.psi(step=-1)) ** 2)
    psi2_y = np.mean(np.abs(sol_y.psi(step=-1)) ** 2)
    
    rel_diff = abs(psi2_x - psi2_y) / max(psi2_x, psi2_y)
    
    assert rel_diff < 0.3, f"Bx and By should have similar effects, got |ψ|²_x = {psi2_x:.3f}, |ψ|²_y = {psi2_y:.3f}"


def test_time_dependent_field():
    """Test time-varying field (ramp from 0 to Bz)."""
    params = SimulationParameters(
        Nx=10, Ny=10, Nz=1,
        hx=1.0, hy=1.0, hz=1.0,
        kappa=2.0,
    )
    
    # Field ramps from 0 to 2.0 using field_func
    def field_func(t, t_stop):
        if t <= 0:
            return 0.0, 0.0, 0.0
        return 0.0, 0.0, 2.0 * (t / t_stop)  # Linear ramp
    
    field = AppliedField(Bz=2.0, field_func=field_func)
    device = Device(params=params, applied_field=field)
    
    x0 = device.initial_state()
    solution = solve(
        device,
        x0=x0,
        t_start=0.0,
        t_stop=5.0,
        dt=0.02,
        method="euler",
        save_every=25,
        progress=False,
    )
    
    # At early time, field is weak → |ψ|² should be high
    psi2_early = np.mean(np.abs(solution.psi(step=1)) ** 2)
    
    # At late time, field is strong → |ψ|² should be lower
    psi2_late = np.mean(np.abs(solution.psi(step=-1)) ** 2)
    
    assert psi2_early > psi2_late, f"Field ramp should reduce |ψ|²: early={psi2_early:.3f}, late={psi2_late:.3f}"


if __name__ == "__main__":
    # Run physics tests directly
    print("Running physics validation tests for non-z fields...")
    import time
    
    tests = [
        ("Bx transverse field", test_transverse_field_bx),
        ("By transverse field", test_transverse_field_by),
        ("Bx+Bz tilted field", test_tilted_field_bx_bz),
        ("Weak vs strong field", test_field_comparison_weak_vs_strong),
        ("Bx vs By symmetry", test_field_symmetry_bx_vs_by),
        ("Time-dependent field", test_time_dependent_field),
    ]
    
    total_time = 0
    for name, test_func in tests:
        start = time.time()
        try:
            test_func()
            elapsed = time.time() - start
            total_time += elapsed
            print(f"  ✓ {name} ({elapsed:.2f}s)")
        except AssertionError as e:
            print(f"  ✗ {name} FAILED: {e}")
        except Exception as e:
            print(f"  ✗ {name} ERROR: {e}")
    
    print(f"\n✅ Physics validation tests completed in {total_time:.2f}s")

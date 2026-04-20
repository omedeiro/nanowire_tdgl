"""Quick smoke test for hole boundary conditions (<5 seconds).

This test verifies that:
1. Holes can be added to a device
2. Zero-current BCs are enforced at hole boundaries
3. Simulation runs without errors
4. Basic physics is correct (flux penetration, vortex detection works)

Designed for fast CI/CD validation.
"""

from __future__ import annotations

import numpy as np
import pytest

from tdgl3d import Device, SimulationParameters, AppliedField, solve
from tdgl3d.physics.rhs import _expand_interior_to_full
from tdgl3d.analysis.vortex_counting import count_vortices_plaquette, count_hole_flux_quanta


def test_hole_bc_smoke_fast():
    """Quick smoke test: 10×10 grid, 20 time steps, verify hole BCs work."""
    # Small grid for speed
    params = SimulationParameters(
        Nx=10, Ny=10, Nz=1,
        hx=1.0, hy=1.0, hz=1.0,
        kappa=2.0,
    )
    
    # Strong field for quick vortex entry
    field = AppliedField(Bz=1.0)
    device = Device(params=params, applied_field=field)
    
    # Add small square hole in center (3×3)
    hole_vertices = [
        (4.0, 4.0),
        (6.0, 4.0),
        (6.0, 6.0),
        (4.0, 6.0),
    ]
    device.add_hole(hole_vertices)
    
    # Verify hole was added
    assert len(device.idx.hole_x_bc_mask) > 0, "Hole x-boundaries should exist"
    assert len(device.idx.hole_y_bc_mask) > 0, "Hole y-boundaries should exist"
    
    # Initial state with noise
    x0 = device.initial_state()
    rng = np.random.default_rng(42)
    noise = 0.01 * (rng.standard_normal(params.n_interior) 
                   + 1j * rng.standard_normal(params.n_interior))
    x0.psi[:] += noise * device.material.interior_sc_mask
    
    # Quick simulation (20 steps ≈ 0.5 seconds on laptop)
    solution = solve(
        device,
        x0=x0,
        t_start=0.0,
        t_stop=0.2,
        dt=0.01,
        method="euler",
        save_every=10,
        progress=False,
    )
    
    # Check 1: Link variables at hole boundaries should be zero
    n = params.n_interior
    state_final = solution.states[:, -1]
    
    phi_x_int = state_final[n:2*n]
    phi_y_int = state_final[2*n:3*n]
    
    # Expand to full grid
    phi_x_full = _expand_interior_to_full(phi_x_int, params, device.idx)
    phi_y_full = _expand_interior_to_full(phi_y_int, params, device.idx)
    
    # Check hole boundaries
    phi_x_hole = phi_x_full[device.idx.hole_x_bc_mask]
    phi_y_hole = phi_y_full[device.idx.hole_y_bc_mask]
    
    max_phi_x = np.max(np.abs(phi_x_hole))
    max_phi_y = np.max(np.abs(phi_y_hole))
    
    assert max_phi_x < 1e-12, f"φ_x at hole boundaries should be ~0, got {max_phi_x:.2e}"
    assert max_phi_y < 1e-12, f"φ_y at hole boundaries should be ~0, got {max_phi_y:.2e}"
    
    # Check 2: ψ in hole should be small
    psi_int = state_final[:n]
    hole_mask_int = device.material.interior_sc_mask == 0.0
    if np.sum(hole_mask_int) > 0:
        psi_hole = psi_int[hole_mask_int]
        max_psi_hole = np.max(np.abs(psi_hole))
        assert max_psi_hole < 0.2, f"|ψ| in hole should be small, got {max_psi_hole:.2e}"
    
    # Check 3: Vortex detection should work (may or may not find vortices, just shouldn't crash)
    n_vort, pos, winding = count_vortices_plaquette(solution, device, slice_z=0, step=-1)
    assert n_vort >= 0, "Vortex count should be non-negative"
    
    # Check 4: Flux calculation should work
    hole_bounds = (4.0, 6.0, 4.0, 6.0)
    flux = count_hole_flux_quanta(solution, device, hole_bounds, slice_z=0, step=-1)
    assert np.isfinite(flux), "Flux calculation should return finite value"


def test_hole_bc_3d_smoke():
    """Quick 3D smoke test with hole through z-layers."""
    params = SimulationParameters(
        Nx=8, Ny=8, Nz=4,
        hx=1.0, hy=1.0, hz=1.0,
        kappa=2.0,
    )
    
    field = AppliedField(Bz=0.5)
    device = Device(params=params, applied_field=field)
    
    # Add hole through middle z-layers
    hole_vertices = [
        (3.0, 3.0),
        (5.0, 3.0),
        (5.0, 5.0),
        (3.0, 5.0),
    ]
    device.add_hole(hole_vertices, z_range=(1, 2))
    
    # Verify hole was added (should have z-boundaries in 3D)
    assert len(device.idx.hole_x_bc_mask) > 0
    assert len(device.idx.hole_y_bc_mask) > 0
    assert len(device.idx.hole_z_bc_mask) > 0, "3D hole should have z-boundaries"
    
    # Quick simulation
    x0 = device.initial_state()
    solution = solve(
        device,
        x0=x0,
        t_start=0.0,
        t_stop=0.1,
        dt=0.01,
        method="euler",
        save_every=10,
        progress=False,
    )
    
    # Verify BCs are enforced
    n = params.n_interior
    state_final = solution.states[:, -1]
    
    phi_x_int = state_final[n:2*n]
    phi_y_int = state_final[2*n:3*n]
    phi_z_int = state_final[3*n:4*n]
    
    phi_x_full = _expand_interior_to_full(phi_x_int, params, device.idx)
    phi_y_full = _expand_interior_to_full(phi_y_int, params, device.idx)
    phi_z_full = _expand_interior_to_full(phi_z_int, params, device.idx)
    
    max_phi_x = np.max(np.abs(phi_x_full[device.idx.hole_x_bc_mask]))
    max_phi_y = np.max(np.abs(phi_y_full[device.idx.hole_y_bc_mask]))
    max_phi_z = np.max(np.abs(phi_z_full[device.idx.hole_z_bc_mask]))
    
    assert max_phi_x < 1e-12, f"φ_x at hole boundaries should be ~0"
    assert max_phi_y < 1e-12, f"φ_y at hole boundaries should be ~0"
    assert max_phi_z < 1e-12, f"φ_z at hole boundaries should be ~0"


def test_hole_save_load():
    """Verify that hole BC info is preserved through save/load."""
    from tdgl3d import Solution
    
    params = SimulationParameters(Nx=8, Ny=8, Nz=1)
    device = Device(params)
    
    # Add hole
    hole_vertices = [(3.0, 3.0), (5.0, 3.0), (5.0, 5.0), (3.0, 5.0)]
    device.add_hole(hole_vertices)
    
    x0 = device.initial_state()
    sol = solve(device, x0=x0, t_start=0.0, t_stop=0.05, dt=0.01, 
                method="euler", save_every=5, progress=False)
    
    # Save
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
        filename = f.name
    
    try:
        sol.save(filename)
        
        # Load
        sol2 = Solution.load(filename)
        
        # Verify hole BC masks were saved
        assert len(sol2.idx.hole_x_bc_mask) == len(sol.idx.hole_x_bc_mask)
        assert len(sol2.idx.hole_y_bc_mask) == len(sol.idx.hole_y_bc_mask)
        assert np.array_equal(sol2.idx.hole_x_bc_mask, sol.idx.hole_x_bc_mask)
        assert np.array_equal(sol2.idx.hole_y_bc_mask, sol.idx.hole_y_bc_mask)
        
    finally:
        if os.path.exists(filename):
            os.remove(filename)


if __name__ == "__main__":
    # Run smoke tests directly
    print("Running quick smoke tests for hole BCs...")
    import time
    
    start = time.time()
    test_hole_bc_smoke_fast()
    elapsed_1 = time.time() - start
    print(f"✓ 2D smoke test passed ({elapsed_1:.2f}s)")
    
    start = time.time()
    test_hole_bc_3d_smoke()
    elapsed_2 = time.time() - start
    print(f"✓ 3D smoke test passed ({elapsed_2:.2f}s)")
    
    start = time.time()
    test_hole_save_load()
    elapsed_3 = time.time() - start
    print(f"✓ Save/load test passed ({elapsed_3:.2f}s)")
    
    total = elapsed_1 + elapsed_2 + elapsed_3
    print(f"\n✅ All smoke tests passed in {total:.2f}s (target: <5s)")

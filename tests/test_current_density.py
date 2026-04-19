"""Tests for current density computation (supercurrent and normal current).

Verifies that:
1. Supercurrent density is computed correctly
2. Current magnitude methods work properly  
3. Current density handles 2D and 3D cases
4. Current is zero in insulator/hole regions
"""

from __future__ import annotations

import numpy as np
import pytest

import tdgl3d


def test_supercurrent_density_2d():
    """Verify supercurrent density computation in 2D."""
    params = tdgl3d.SimulationParameters(Nx=10, Ny=10, Nz=1, hx=1.0, hy=1.0, hz=1.0, kappa=2.0)
    field = tdgl3d.AppliedField(Bz=0.3)
    device = tdgl3d.Device(params, applied_field=field)
    
    dt = 0.01
    x0 = device.initial_state()
    solution = tdgl3d.solve(device, x0=x0, dt=dt, t_stop=2.0, method="euler", save_every=20, progress=False)
    
    # Get supercurrent density
    Jsx, Jsy, Jsz = solution.supercurrent_density(step=-1)
    
    # Should return vectors of length n_interior
    assert Jsx.shape[0] == params.n_interior
    assert Jsy.shape[0] == params.n_interior
    assert Jsz.shape[0] == params.n_interior
    
    # In 2D, Jsz should be zero
    assert np.allclose(Jsz, 0.0), "Jsz should be zero in 2D"
    
    # Supercurrent should be non-zero (we have applied field)
    J_s_mag = np.sqrt(Jsx**2 + Jsy**2)
    assert J_s_mag.max() > 0.001, f"Expected non-zero supercurrent, got max={J_s_mag.max():.6f}"


def test_supercurrent_density_3d():
    """Verify supercurrent density computation in 3D."""
    params = tdgl3d.SimulationParameters(Nx=6, Ny=6, Nz=4, hx=1.0, hy=1.0, hz=1.0, kappa=2.0)
    field = tdgl3d.AppliedField(Bz=0.3)
    device = tdgl3d.Device(params, applied_field=field)
    
    dt = 0.01
    x0 = device.initial_state()
    solution = tdgl3d.solve(device, x0=x0, dt=dt, t_stop=2.0, method="euler", save_every=20, progress=False)
    
    # Get supercurrent density
    Jsx, Jsy, Jsz = solution.supercurrent_density(step=-1)
    
    # Should return vectors of length n_interior
    assert Jsx.shape[0] == params.n_interior
    assert Jsy.shape[0] == params.n_interior
    assert Jsz.shape[0] == params.n_interior
    
    # In 3D with Bz applied, should have non-zero J_s in x-y plane
    J_s_mag = np.sqrt(Jsx**2 + Jsy**2 + Jsz**2)
    assert J_s_mag.max() > 0.001, f"Expected non-zero supercurrent, got max={J_s_mag.max():.6f}"


def test_current_magnitude():
    """Verify current magnitude helper method."""
    params = tdgl3d.SimulationParameters(Nx=8, Ny=8, Nz=1, hx=1.0, hy=1.0, hz=1.0, kappa=2.0)
    field = tdgl3d.AppliedField(Bz=0.2)
    device = tdgl3d.Device(params, applied_field=field)
    
    dt = 0.01
    x0 = device.initial_state()
    solution = tdgl3d.solve(device, x0=x0, dt=dt, t_stop=1.0, method="euler", save_every=10, progress=False)
    
    # Test supercurrent magnitude
    J_s_mag = solution.current_magnitude(step=-1, dataset="supercurrent")
    assert J_s_mag.shape[0] == params.n_interior
    assert J_s_mag.min() >= 0.0, "Current magnitude should be non-negative"
    
    # Test total current magnitude (should equal supercurrent since J_n is not implemented)
    J_tot_mag = solution.current_magnitude(step=-1, dataset=None)
    assert J_tot_mag.shape[0] == params.n_interior
    assert np.allclose(J_tot_mag, J_s_mag), "Total current should equal supercurrent (J_n not implemented)"


def test_normal_current_not_implemented():
    """Verify that normal current returns None (not yet implemented)."""
    params = tdgl3d.SimulationParameters(Nx=6, Ny=6, Nz=1, hx=1.0, hy=1.0, hz=1.0, kappa=2.0)
    device = tdgl3d.Device(params, applied_field=tdgl3d.AppliedField(Bz=0.0))
    
    dt = 0.01
    x0 = device.initial_state()
    solution = tdgl3d.solve(device, x0=x0, dt=dt, t_stop=0.5, method="euler", save_every=10, progress=False)
    
    # Normal current should return None
    J_n = solution.normal_current_density(step=-1)
    assert J_n is None, "Normal current should return None (not yet implemented)"


def test_current_density_total():
    """Verify total current density J = J_s + J_n."""
    params = tdgl3d.SimulationParameters(Nx=8, Ny=8, Nz=1, hx=1.0, hy=1.0, hz=1.0, kappa=2.0)
    field = tdgl3d.AppliedField(Bz=0.2)
    device = tdgl3d.Device(params, applied_field=field)
    
    dt = 0.01
    x0 = device.initial_state()
    solution = tdgl3d.solve(device, x0=x0, dt=dt, t_stop=1.0, method="euler", save_every=10, progress=False)
    
    # Get total current
    Jx, Jy, Jz = solution.current_density(step=-1)
    
    # Should return vectors of length n_interior
    assert Jx.shape[0] == params.n_interior
    assert Jy.shape[0] == params.n_interior
    assert Jz.shape[0] == params.n_interior
    
    # Since J_n is None, total current should equal supercurrent
    Jsx, Jsy, Jsz = solution.supercurrent_density(step=-1)
    assert np.allclose(Jx, Jsx)
    assert np.allclose(Jy, Jsy)
    assert np.allclose(Jz, Jsz)


def test_current_in_hole():
    """Verify that supercurrent is suppressed in holes (where |ψ| → 0)."""
    # Small 2D grid
    Nx, Ny = 16, 16
    params = tdgl3d.SimulationParameters(
        Nx=Nx, Ny=Ny, Nz=1,
        hx=1.0, hy=1.0, hz=1.0,
        kappa=2.0,
    )
    
    Bz_applied = 0.2
    field = tdgl3d.AppliedField(Bz=Bz_applied)
    
    # Use trilayer to get MaterialMap
    trilayer = tdgl3d.Trilayer(
        bottom=tdgl3d.Layer(thickness_z=1, kappa=2.0),
        insulator=tdgl3d.Layer(thickness_z=0, kappa=2.0, is_superconductor=False),
        top=tdgl3d.Layer(thickness_z=0, kappa=2.0),
    )
    device = tdgl3d.Device(params, applied_field=field, trilayer=trilayer)
    
    # Carve hole in center (4×4 grid cells)
    i_lo, i_hi = Nx // 2 - 2, Nx // 2 + 2
    j_lo, j_hi = Ny // 2 - 2, Ny // 2 + 2
    
    mj = Nx + 1
    mk = mj * (Ny + 1)
    k = 0  # 2D, only one z-plane
    
    for j in range(j_lo, j_hi + 1):
        for i in range(i_lo, i_hi + 1):
            m_full = i + mj * j + mk * k
            if m_full < len(device.material.sc_mask):
                device.material.sc_mask[m_full] = 0.0  # Mark as non-SC
    
    # Rebuild interior_sc_mask
    device.material.interior_sc_mask[:] = device.material.sc_mask[device.idx.interior_to_full]
    
    # Run simulation
    dt = 0.01
    x0 = device.initial_state()
    # Add small noise only in SC regions
    rng = np.random.default_rng(42)
    noise = 0.01 * (rng.standard_normal(params.n_interior) + 1j * rng.standard_normal(params.n_interior))
    x0.psi[:] += noise * device.material.interior_sc_mask
    
    solution = tdgl3d.solve(
        device,
        x0=x0,
        dt=dt,
        t_stop=5.0,
        method="euler",
        save_every=50,
        progress=False,
    )
    
    # Get supercurrent
    Jsx, Jsy, Jsz = solution.supercurrent_density(step=-1)
    J_s_mag = np.sqrt(Jsx**2 + Jsy**2)
    
    # Reshape to 2D
    Nx_int, Ny_int = Nx - 1, Ny - 1
    J_s_2d = J_s_mag.reshape(Nx_int, Ny_int)
    
    # Extract current in hole region (interior indices are shifted by 1)
    i_lo_int = max(i_lo - 1, 0)
    i_hi_int = min(i_hi - 1, Nx_int - 1)
    j_lo_int = max(j_lo - 1, 0)
    j_hi_int = min(j_hi - 1, Ny_int - 1)
    
    J_s_hole = J_s_2d[i_lo_int:i_hi_int+1, j_lo_int:j_hi_int+1]
    
    # Extract current in SC bulk (corners)
    J_s_sc_bulk = []
    J_s_sc_bulk.append(J_s_2d[1:4, 1:4])
    J_s_sc_bulk.append(J_s_2d[-4:-1, 1:4])
    J_s_sc = np.concatenate([j.ravel() for j in J_s_sc_bulk])
    
    J_s_hole_max = np.max(J_s_hole)
    J_s_sc_mean = np.mean(J_s_sc)
    
    print(f"|J_s| in hole: max={J_s_hole_max:.6f}")
    print(f"|J_s| in SC bulk: mean={J_s_sc_mean:.6f}")
    
    # Supercurrent in hole should be much weaker than in SC (where |ψ| → 0, J_s → 0)
    assert J_s_hole_max < J_s_sc_mean * 0.5, \
        f"Expected |J_s| in hole ({J_s_hole_max:.6f}) < 0.5× SC bulk ({J_s_sc_mean:.6f})"

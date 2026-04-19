"""Tests for magnetic field computation in holes (non-SC regions).

Verifies that:
1. B-field is correctly computed at all interior nodes (not just bfield_interior subset)
2. Applied field penetrates holes (non-SC regions) without Meissner screening
3. eval_bfield_full() returns B at all interior nodes
"""

from __future__ import annotations

import numpy as np
import pytest

import tdgl3d
from tdgl3d.physics.bfield import eval_bfield, eval_bfield_full
from tdgl3d.physics.rhs import _expand_interior_to_full, _apply_boundary_conditions, BoundaryVectors
from tdgl3d.physics.applied_field import build_boundary_field_vectors


def test_bfield_full_coverage_2d():
    """Verify eval_bfield_full returns B at all interior nodes in 2D."""
    params = tdgl3d.SimulationParameters(Nx=10, Ny=10, Nz=1, hx=1.0, hy=1.0, hz=1.0, kappa=2.0)
    field = tdgl3d.AppliedField(Bz=0.5)
    device = tdgl3d.Device(params, applied_field=field)
    
    # Run short simulation with CFL-stable dt
    dt = 0.01  # CFL limit for kappa=2, h=1 is 0.0625
    x0 = device.initial_state()
    solution = tdgl3d.solve(device, x0=x0, dt=dt, t_stop=0.5, method="euler", save_every=10)
    
    # Get B-field using both methods
    Bx_old, By_old, Bz_old = solution.bfield(step=-1, full_interior=False)  # Old method (subset)
    Bx_full, By_full, Bz_full = solution.bfield(step=-1, full_interior=True)  # New method (all interior)
    
    # Full method should return more nodes
    n_interior = params.n_interior
    assert Bx_full.shape[0] == n_interior, f"Expected {n_interior} nodes, got {Bx_full.shape[0]}"
    assert By_full.shape[0] == n_interior
    assert Bz_full.shape[0] == n_interior
    
    # Old method should return fewer nodes
    assert Bx_old.shape[0] < n_interior, "Old method should return subset of interior nodes"


def test_bfield_full_coverage_3d():
    """Verify eval_bfield_full returns B at all interior nodes in 3D."""
    params = tdgl3d.SimulationParameters(Nx=6, Ny=6, Nz=4, hx=1.0, hy=1.0, hz=1.0, kappa=2.0)
    field = tdgl3d.AppliedField(Bz=0.5)
    device = tdgl3d.Device(params, applied_field=field)
    
    # Run short simulation with CFL-stable dt
    dt = 0.01
    x0 = device.initial_state()
    solution = tdgl3d.solve(device, x0=x0, dt=dt, t_stop=0.5, method="euler", save_every=10)
    
    # Get B-field using full method
    Bx_full, By_full, Bz_full = solution.bfield(step=-1, full_interior=True)
    
    # Should return all interior nodes
    n_interior = params.n_interior
    assert Bx_full.shape[0] == n_interior
    assert By_full.shape[0] == n_interior
    assert Bz_full.shape[0] == n_interior
    
    # Old method should return fewer nodes
    assert Bx_old.shape[0] < n_interior, "Old method should return subset of interior nodes"


def test_bfield_full_coverage_3d():
    """Verify eval_bfield_full returns B at all interior nodes in 3D."""
    params = tdgl3d.SimulationParameters(Nx=6, Ny=6, Nz=4, hx=1.0, hy=1.0, hz=1.0, kappa=2.0)
    field = tdgl3d.AppliedField(Bz=0.5)
    device = tdgl3d.Device(params, applied_field=field)
    
    # Run short simulation
    x0 = device.initial_state()
    solution = tdgl3d.solve(device, x0=x0, dt=0.01, t_stop=0.5, method="euler", save_every=10)
    
    # Get B-field using full method
    Bx_full, By_full, Bz_full = solution.bfield(step=-1, full_interior=True)
    
    # Should return all interior nodes
    n_interior = params.n_interior
    assert Bx_full.shape[0] == n_interior
    assert By_full.shape[0] == n_interior
    assert Bz_full.shape[0] == n_interior


def test_applied_field_in_hole():
    """Verify that applied B-field penetrates a hole (non-SC region).
    
    Setup:
    - Create small 2D system with uniform applied Bz
    - Carve a hole in the center (set sc_mask=0, keep kappa non-zero)
    - Run to equilibrium
    - Check that B-field in hole is significantly higher than in SC bulk
      (hole has no Meissner screening, so field penetrates more easily)
    """
    # Small 2D grid
    Nx, Ny = 20, 20
    params = tdgl3d.SimulationParameters(
        Nx=Nx, Ny=Ny, Nz=1,
        hx=1.0, hy=1.0, hz=1.0,
        kappa=2.0,
    )
    
    Bz_applied = 0.3  # Moderate field
    field = tdgl3d.AppliedField(Bz=Bz_applied)
    
    # Use trilayer to get MaterialMap
    trilayer = tdgl3d.Trilayer(
        bottom=tdgl3d.Layer(thickness_z=1, kappa=2.0),
        insulator=tdgl3d.Layer(thickness_z=0, kappa=2.0, is_superconductor=False),
        top=tdgl3d.Layer(thickness_z=0, kappa=2.0),
    )
    device = tdgl3d.Device(params, applied_field=field, trilayer=trilayer)
    
    # Carve hole in center (6×6 grid cells)
    i_lo, i_hi = Nx // 2 - 3, Nx // 2 + 3
    j_lo, j_hi = Ny // 2 - 3, Ny // 2 + 3
    
    mj = Nx + 1
    mk = mj * (Ny + 1)
    k = 0  # 2D, only one z-plane
    
    hole_nodes = []
    for j in range(j_lo, j_hi + 1):
        for i in range(i_lo, i_hi + 1):
            m_full = i + mj * j + mk * k
            if m_full < len(device.material.sc_mask):
                device.material.sc_mask[m_full] = 0.0  # Mark as non-SC
                hole_nodes.append(m_full)
    
    # Rebuild interior_sc_mask
    device.material.interior_sc_mask[:] = device.material.sc_mask[device.idx.interior_to_full]
    
    n_hole_interior = int(np.sum(device.material.interior_sc_mask == 0.0))
    print(f"Created hole with {n_hole_interior} interior nodes")
    
    # Initial state
    x0 = device.initial_state()
    # Add small noise only in SC regions
    rng = np.random.default_rng(42)
    noise = 0.01 * (rng.standard_normal(params.n_interior) + 1j * rng.standard_normal(params.n_interior))
    x0.psi[:] += noise * device.material.interior_sc_mask
    
    # Run to equilibrium
    # Note: Field needs time to diffuse in from boundaries
    solution = tdgl3d.solve(
        device,
        x0=x0,
        dt=0.01,
        t_stop=10.0,  # Long enough for equilibration
        method="euler",
        save_every=100,
        progress=False,
    )
    
    # Get B-field at all interior nodes
    Bx, By, Bz = solution.bfield(step=-1, full_interior=True)
    
    # Reshape to 2D grid
    Nx_int, Ny_int = Nx - 1, Ny - 1
    Bz_2d = Bz.reshape(Nx_int, Ny_int)
    
    # Extract B-field in hole region (interior indices are shifted by 1)
    i_lo_int = max(i_lo - 1, 0)
    i_hi_int = min(i_hi - 1, Nx_int - 1)
    j_lo_int = max(j_lo - 1, 0)
    j_hi_int = min(j_hi - 1, Ny_int - 1)
    
    Bz_hole = Bz_2d[i_lo_int:i_hi_int+1, j_lo_int:j_hi_int+1]
    
    # Extract B-field in SC bulk (corners, far from hole)
    Bz_sc_bulk = []
    # Top-left corner
    Bz_sc_bulk.append(Bz_2d[1:4, 1:4])
    # Top-right corner
    Bz_sc_bulk.append(Bz_2d[-4:-1, 1:4])
    # Bottom-left corner
    Bz_sc_bulk.append(Bz_2d[1:4, -4:-1])
    # Bottom-right corner
    Bz_sc_bulk.append(Bz_2d[-4:-1, -4:-1])
    Bz_sc = np.concatenate([b.ravel() for b in Bz_sc_bulk])
    
    # In the hole, expect B ≈ applied field (no screening)
    Bz_hole_mean = np.mean(Bz_hole)
    Bz_hole_std = np.std(Bz_hole)
    Bz_sc_mean = np.mean(Bz_sc)
    
    print(f"B-field in hole: mean={Bz_hole_mean:.4f}, std={Bz_hole_std:.4f}")
    print(f"B-field in SC bulk: mean={Bz_sc_mean:.4f}")
    print(f"Applied field: {Bz_applied}")
    print(f"Ratio Bz_hole/Bz_sc: {Bz_hole_mean/Bz_sc_mean:.2f}")
    
    # Key test: B-field in hole should be significantly higher (in magnitude) than in SC bulk
    # (The hole has no Meissner screening, so field penetrates more)
    # We don't require B_hole = B_applied exactly because field diffusion is slow,
    # but we do require |B_hole| > |B_sc| significantly
    assert abs(Bz_hole_mean) > abs(Bz_sc_mean) * 2.0, \
        f"Expected |B| in hole ({abs(Bz_hole_mean):.4f}) to be > 2× SC bulk ({abs(Bz_sc_mean):.4f})"
    
    # Also check that |ψ|² is nearly zero in the hole
    psi = solution.psi(step=-1)
    psi_2d = psi.reshape(Nx_int, Ny_int)
    psi2_hole = np.abs(psi_2d[i_lo_int:i_hi_int+1, j_lo_int:j_hi_int+1])**2
    psi2_hole_max = np.max(psi2_hole)
    
    print(f"|ψ|² in hole: max={psi2_hole_max:.6f}")
    assert psi2_hole_max < 0.05, f"Expected |ψ|² ≈ 0 in hole, got max={psi2_hole_max:.6f}"


def test_bfield_default_is_full_interior():
    """Verify that Solution.bfield() defaults to full_interior=True."""
    params = tdgl3d.SimulationParameters(Nx=10, Ny=10, Nz=1, hx=1.0, hy=1.0, hz=1.0, kappa=2.0)
    field = tdgl3d.AppliedField(Bz=0.5)
    device = tdgl3d.Device(params, applied_field=field)
    
    dt = 0.01
    x0 = device.initial_state()
    solution = tdgl3d.solve(device, x0=x0, dt=dt, t_stop=0.5, method="euler", save_every=10)
    
    # Default call should use full_interior=True
    Bx_default, By_default, Bz_default = solution.bfield(step=-1)
    
    # Should return all interior nodes
    assert Bx_default.shape[0] == params.n_interior
    assert By_default.shape[0] == params.n_interior
    assert Bz_default.shape[0] == params.n_interior

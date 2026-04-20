"""Tests for hole boundary condition enforcement.

This module verifies that:
1. GridIndices.define_hole_polygon() correctly identifies boundary links
2. MaterialMap.carve_hole_polygon() marks hole interior as non-SC
3. Device.add_hole() integrates both mechanisms
4. _apply_boundary_conditions() zeros link variables at hole boundaries
"""

from __future__ import annotations

import numpy as np
import pytest

from tdgl3d import Device, SimulationParameters, AppliedField
from tdgl3d.core.material import MaterialMap
from tdgl3d.mesh.indices import construct_indices
from tdgl3d.physics.rhs import _apply_boundary_conditions, BoundaryVectors


def test_gridindices_define_hole_polygon_square():
    """Test that define_hole_polygon registers boundary links for a square hole."""
    params = SimulationParameters(Nx=20, Ny=20, Nz=5, hx=1.0, hy=1.0, hz=1.0, kappa=2.0)
    idx = construct_indices(params)
    
    # Square hole in center (5x5 to 15x15)
    square = [(5.0, 5.0), (15.0, 5.0), (15.0, 15.0), (5.0, 15.0)]
    idx.define_hole_polygon(square, z_range=(0, 5), params=params)
    
    # Should have non-empty boundary masks
    assert idx.hole_x_bc_mask.size > 0, "x-boundary links should be registered"
    assert idx.hole_y_bc_mask.size > 0, "y-boundary links should be registered"
    # z-boundary links may be empty for 2D hole (depends on z_range)


def test_gridindices_multiple_holes():
    """Test that multiple holes can be added and masks are concatenated."""
    params = SimulationParameters(Nx=30, Ny=30, Nz=5, hx=1.0, hy=1.0, hz=1.0, kappa=2.0)
    idx = construct_indices(params)
    
    # First hole
    hole1 = [(5.0, 5.0), (10.0, 5.0), (10.0, 10.0), (5.0, 10.0)]
    idx.define_hole_polygon(hole1, z_range=(0, 5), params=params)
    n_x_links_1 = idx.hole_x_bc_mask.size
    
    # Second hole
    hole2 = [(20.0, 20.0), (25.0, 20.0), (25.0, 25.0), (20.0, 25.0)]
    idx.define_hole_polygon(hole2, z_range=(0, 5), params=params)
    n_x_links_2 = idx.hole_x_bc_mask.size
    
    # Second call should append (not replace)
    assert n_x_links_2 > n_x_links_1, "Adding second hole should increase link count"


def test_materialmap_carve_hole_polygon():
    """Test that carve_hole_polygon marks hole interior as non-superconducting."""
    params = SimulationParameters(Nx=20, Ny=20, Nz=5, hx=1.0, hy=1.0, hz=1.0, kappa=2.0)
    idx = construct_indices(params)
    
    # Create uniform SC material map
    material = MaterialMap(
        kappa=np.full(params.dim_x, 2.0),
        sc_mask=np.ones(params.dim_x),
        interior_sc_mask=np.ones(params.n_interior),
    )
    
    # Carve a square hole
    square = [(5.0, 5.0), (15.0, 5.0), (15.0, 15.0), (5.0, 15.0)]
    material.carve_hole_polygon(square, z_range=(0, 5), params=params, idx=idx)
    
    # Check that some nodes are now non-SC
    assert np.sum(material.sc_mask == 0.0) > 0, "Hole should mark nodes as non-SC"
    assert np.sum(material.interior_sc_mask == 0.0) > 0, "Interior mask should reflect hole"


def test_device_add_hole_basic():
    """Test Device.add_hole() API integration."""
    params = SimulationParameters(Nx=20, Ny=20, Nz=5, hx=1.0, hy=1.0, hz=1.0, kappa=2.0)
    device = Device(params=params)
    
    # Add a square hole
    square = [(5.0, 5.0), (15.0, 5.0), (15.0, 15.0), (5.0, 15.0)]
    device.add_hole(square, z_range=(0, 5))
    
    # Check that GridIndices has boundary links
    assert device.idx.hole_x_bc_mask.size > 0, "Boundary links should be registered"
    
    # Check that MaterialMap was created and hole is carved
    assert device.material is not None, "MaterialMap should be created"
    assert np.sum(device.material.sc_mask == 0.0) > 0, "Hole should be carved in material"


def test_device_add_hole_default_z_range():
    """Test that add_hole uses full z-range by default."""
    params = SimulationParameters(Nx=20, Ny=20, Nz=10, hx=1.0, hy=1.0, hz=1.0, kappa=2.0)
    device = Device(params=params)
    
    square = [(5.0, 5.0), (15.0, 5.0), (15.0, 15.0), (5.0, 15.0)]
    device.add_hole(square)  # No z_range specified
    
    # Should still work
    assert device.idx.hole_x_bc_mask.size > 0


def test_device_add_hole_multiple():
    """Test adding multiple holes to a device."""
    params = SimulationParameters(Nx=40, Ny=40, Nz=5, hx=1.0, hy=1.0, hz=1.0, kappa=2.0)
    device = Device(params=params)
    
    hole1 = [(5.0, 5.0), (10.0, 5.0), (10.0, 10.0), (5.0, 10.0)]
    hole2 = [(25.0, 25.0), (30.0, 25.0), (30.0, 30.0), (25.0, 30.0)]
    
    device.add_hole(hole1)
    n_links_1 = device.idx.hole_x_bc_mask.size
    
    device.add_hole(hole2)
    n_links_2 = device.idx.hole_x_bc_mask.size
    
    assert n_links_2 > n_links_1, "Second hole should add more boundary links"


def test_apply_boundary_conditions_zeros_hole_links():
    """Test that _apply_boundary_conditions() zeros link variables at hole boundaries."""
    params = SimulationParameters(Nx=20, Ny=20, Nz=5, hx=1.0, hy=1.0, hz=1.0, kappa=2.0)
    device = Device(params=params)
    
    # Add a hole
    square = [(5.0, 5.0), (15.0, 5.0), (15.0, 15.0), (5.0, 15.0)]
    device.add_hole(square, z_range=(0, 5))
    
    # Create full-grid state vectors with non-zero values
    x = np.ones(params.dim_x, dtype=np.complex128)
    y1 = np.ones(params.dim_x, dtype=np.complex128) * (1.0 + 0.5j)
    y2 = np.ones(params.dim_x, dtype=np.complex128) * (0.5 + 1.0j)
    y3 = np.ones(params.dim_x, dtype=np.complex128) * (0.3 + 0.7j)
    
    # Create zero boundary field vectors
    u = BoundaryVectors(
        Bx=np.zeros(params.dim_x),
        By=np.zeros(params.dim_x),
        Bz=np.zeros(params.dim_x),
    )
    
    # Apply boundary conditions
    x_bc, y1_bc, y2_bc, y3_bc = _apply_boundary_conditions(
        x, y1, y2, y3, params, device.idx, u
    )
    
    # Check that hole boundary links are zeroed
    if device.idx.hole_x_bc_mask.size > 0:
        assert np.allclose(
            y1_bc[device.idx.hole_x_bc_mask], 0.0
        ), "x-links at hole boundary should be zero"
    
    if device.idx.hole_y_bc_mask.size > 0:
        assert np.allclose(
            y2_bc[device.idx.hole_y_bc_mask], 0.0
        ), "y-links at hole boundary should be zero"
    
    if device.idx.hole_z_bc_mask.size > 0:
        assert np.allclose(
            y3_bc[device.idx.hole_z_bc_mask], 0.0
        ), "z-links at hole boundary should be zero"


def test_hole_boundary_conditions_concave_polygon():
    """Test that hole BCs work with concave (non-convex) polygons."""
    params = SimulationParameters(Nx=30, Ny=30, Nz=5, hx=1.0, hy=1.0, hz=1.0, kappa=2.0)
    device = Device(params=params)
    
    # L-shaped hole (concave)
    l_shape = [
        (5.0, 5.0), (15.0, 5.0), (15.0, 10.0),
        (10.0, 10.0), (10.0, 15.0), (5.0, 15.0)
    ]
    device.add_hole(l_shape, z_range=(0, 5))
    
    # Should successfully register boundary links
    assert device.idx.hole_x_bc_mask.size > 0
    assert device.idx.hole_y_bc_mask.size > 0


def test_hole_in_trilayer_device():
    """Test that holes work correctly with trilayer devices."""
    from tdgl3d import Trilayer, Layer
    
    params = SimulationParameters(Nx=20, Ny=20, Nz=10, hx=1.0, hy=1.0, hz=1.0, kappa=2.0)
    trilayer = Trilayer(
        bottom=Layer(thickness_z=3, kappa=2.0, is_superconductor=True),
        insulator=Layer(thickness_z=4, kappa=0.0, is_superconductor=False),
        top=Layer(thickness_z=3, kappa=2.0, is_superconductor=True),
    )
    device = Device(params=params, trilayer=trilayer)
    
    # Add hole through bottom layer only
    square = [(5.0, 5.0), (15.0, 5.0), (15.0, 15.0), (5.0, 15.0)]
    device.add_hole(square, z_range=(0, 2))
    
    # Should have both trilayer material map AND hole carved
    assert device.material is not None
    # Check that hole is carved in bottom layer
    assert np.sum(device.material.sc_mask == 0.0) > 0


def test_initial_state_with_hole():
    """Test that Device.initial_state() correctly zeros psi in hole region."""
    params = SimulationParameters(Nx=20, Ny=20, Nz=5, hx=1.0, hy=1.0, hz=1.0, kappa=2.0)
    device = Device(params=params)
    
    square = [(5.0, 5.0), (15.0, 5.0), (15.0, 15.0), (5.0, 15.0)]
    device.add_hole(square)
    
    state = device.initial_state()
    
    # Psi should be zero where material is non-SC (hole interior)
    # This is enforced by interior_sc_mask in Device.initial_state()
    assert np.sum(np.abs(state.psi) == 0.0) > 0, "Psi should be zero in hole region"


def test_hole_boundary_no_duplicate_links():
    """Test that boundary links are not duplicated at corners."""
    params = SimulationParameters(Nx=20, Ny=20, Nz=5, hx=1.0, hy=1.0, hz=1.0, kappa=2.0)
    idx = construct_indices(params)
    
    square = [(5.0, 5.0), (15.0, 5.0), (15.0, 15.0), (5.0, 15.0)]
    idx.define_hole_polygon(square, z_range=(0, 5), params=params)
    
    # Check for unique links
    assert len(np.unique(idx.hole_x_bc_mask)) == len(idx.hole_x_bc_mask), \
        "x-boundary links should be unique"
    assert len(np.unique(idx.hole_y_bc_mask)) == len(idx.hole_y_bc_mask), \
        "y-boundary links should be unique"


def test_small_triangle_hole():
    """Test a small triangular hole to verify polygon algorithm."""
    params = SimulationParameters(Nx=20, Ny=20, Nz=5, hx=1.0, hy=1.0, hz=1.0, kappa=2.0)
    device = Device(params=params)
    
    triangle = [(10.0, 10.0), (12.0, 10.0), (11.0, 12.0)]
    device.add_hole(triangle)
    
    # Should register some boundary links even for small hole
    assert device.idx.hole_x_bc_mask.size > 0 or device.idx.hole_y_bc_mask.size > 0


def test_hole_bc_enforced_during_simulation():
    """Test that φ remains at zero at hole boundaries throughout a simulation."""
    from tdgl3d import solve
    from tdgl3d.physics.rhs import _expand_interior_to_full
    
    # Small 2D test
    params = SimulationParameters(Nx=20, Ny=20, Nz=1, hx=1.0, hy=1.0, hz=1.0, kappa=2.0)
    device = Device(params=params)
    
    # Square hole
    hole = [(7.0, 7.0), (13.0, 7.0), (13.0, 13.0), (7.0, 13.0)]
    device.add_hole(hole)
    
    # Initial state
    x0 = device.initial_state()
    
    # Run short simulation
    solution = solve(
        device,
        x0=x0,
        dt=0.01,
        t_stop=1.0,
        method="euler",
        save_every=5,
        progress=False,
    )
    
    # Check φ at hole boundaries at every saved time step
    n = params.n_interior
    
    for step in range(solution.n_steps):
        state = solution.states[:, step]
        phi_x_int = state[n:2*n]
        phi_y_int = state[2*n:3*n]
        
        # Expand to full grid
        phi_x_full = _expand_interior_to_full(phi_x_int, params, device.idx)
        phi_y_full = _expand_interior_to_full(phi_y_int, params, device.idx)
        
        # Check hole boundaries
        if len(device.idx.hole_x_bc_mask) > 0:
            phi_x_hole = phi_x_full[device.idx.hole_x_bc_mask]
            assert np.max(np.abs(phi_x_hole)) < 1e-12, \
                f"φ_x not zero at hole boundary at step {step}"
        
        if len(device.idx.hole_y_bc_mask) > 0:
            phi_y_hole = phi_y_full[device.idx.hole_y_bc_mask]
            assert np.max(np.abs(phi_y_hole)) < 1e-12, \
                f"φ_y not zero at hole boundary at step {step}"

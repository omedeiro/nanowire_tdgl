"""Verification script for hole boundary conditions.

This script:
1. Creates a simple 2D device with a square hole
2. Runs a short simulation with applied B-field
3. Visualizes currents, fields, and psi to verify BC enforcement
4. Checks that hole boundaries have zero link variables
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from tdgl3d import Device, SimulationParameters, AppliedField, solve
from tdgl3d.visualization.plotting import plot_current_density
from tdgl3d.physics.rhs import _expand_interior_to_full
from tdgl3d.analysis.vortex_counting import (
    count_vortices_plaquette,
    count_hole_flux_quanta,
    count_vortices_polygon,
)


def main():
    # Small 2D grid for quick testing
    params = SimulationParameters(
        Nx=40, Ny=40, Nz=1,
        hx=1.0, hy=1.0, hz=1.0,
        kappa=2.0,
    )
    
    # Applied field ramped from 0 to 1.0 (stronger field)
    def field_ramp(t, t_stop):
        if t < 0.2 * t_stop:
            return 0.0, 0.0, 0.0
        elif t < 0.4 * t_stop:
            scale = (t - 0.2*t_stop) / (0.2*t_stop)
            return 0.0, 0.0, 1.0 * scale
        else:
            return 0.0, 0.0, 1.0
    
    field = AppliedField(Bz=1.0, field_func=field_ramp)
    device = Device(params=params, applied_field=field)
    
    # Add square hole in center (10x10 in a 40x40 grid)
    hole_vertices = [
        (15.0, 15.0),
        (25.0, 15.0),
        (25.0, 25.0),
        (15.0, 25.0),
    ]
    device.add_hole(hole_vertices)
    
    print(f"Device: {params.Nx} × {params.Ny}")
    print(f"Hole: {len(hole_vertices)} vertices")
    print(f"Hole BC links: x={len(device.idx.hole_x_bc_mask)}, y={len(device.idx.hole_y_bc_mask)}")
    print(f"Interior nodes with sc_mask=0: {int(np.sum(device.material.interior_sc_mask == 0))}")
    
    # Initial state with small noise
    x0 = device.initial_state()
    rng = np.random.default_rng(42)
    noise = 0.01 * (rng.standard_normal(params.n_interior) 
                   + 1j * rng.standard_normal(params.n_interior))
    x0.psi[:] += noise * device.material.interior_sc_mask
    
    # Longer simulation for vortex entry
    print("\nRunning simulation...")
    solution = solve(
        device,
        x0=x0,
        dt=0.01,
        t_stop=30.0,  # Longer time
        method="euler",
        save_every=30,  # Still save ~50 steps
        progress=True,
    )
    
    print(f"Simulation complete: {solution.n_steps} steps saved")
    
    # ===== Verification checks =====
    print("\n" + "="*70)
    print("BOUNDARY CONDITION VERIFICATION")
    print("="*70)
    
    # Check 1: Link variables at hole boundaries should be zero
    n = params.n_interior
    state_final = solution.states[:, -1]
    
    phi_x_int = state_final[n:2*n]
    phi_y_int = state_final[2*n:3*n]
    
    # Expand to full grid
    phi_x_full = _expand_interior_to_full(phi_x_int, params, device.idx)
    phi_y_full = _expand_interior_to_full(phi_y_int, params, device.idx)
    
    # Check hole boundaries
    if len(device.idx.hole_x_bc_mask) > 0:
        phi_x_hole = phi_x_full[device.idx.hole_x_bc_mask]
        max_phi_x = np.max(np.abs(phi_x_hole))
        print(f"\n✓ CHECK 1: φ_x at hole x-boundaries")
        print(f"  Links checked: {len(phi_x_hole)}")
        print(f"  Max |φ_x|: {max_phi_x:.2e}")
        print(f"  Status: {'PASS ✓' if max_phi_x < 1e-12 else 'FAIL ✗ (should be ~0)'}")
    
    if len(device.idx.hole_y_bc_mask) > 0:
        phi_y_hole = phi_y_full[device.idx.hole_y_bc_mask]
        max_phi_y = np.max(np.abs(phi_y_hole))
        print(f"\n✓ CHECK 2: φ_y at hole y-boundaries")
        print(f"  Links checked: {len(phi_y_hole)}")
        print(f"  Max |φ_y|: {max_phi_y:.2e}")
        print(f"  Status: {'PASS ✓' if max_phi_y < 1e-12 else 'FAIL ✗ (should be ~0)'}")
    
    # Check 2: psi in hole should be ~0
    psi_int = state_final[:n]
    hole_mask_int = device.material.interior_sc_mask == 0.0
    if np.sum(hole_mask_int) > 0:
        psi_hole = psi_int[hole_mask_int]
        max_psi_hole = np.max(np.abs(psi_hole))
        print(f"\n✓ CHECK 3: ψ inside hole")
        print(f"  Nodes checked: {len(psi_hole)}")
        print(f"  Max |ψ|: {max_psi_hole:.2e}")
        print(f"  Status: {'PASS ✓' if max_psi_hole < 0.1 else 'FAIL ✗ (should be ~0)'}")
    
    # Check 3: psi in SC region should be ~1
    sc_mask_int = device.material.interior_sc_mask > 0.0
    if np.sum(sc_mask_int) > 0:
        psi_sc = psi_int[sc_mask_int]
        mean_psi2_sc = np.mean(np.abs(psi_sc)**2)
        print(f"\n✓ CHECK 4: ψ in superconductor")
        print(f"  Nodes checked: {len(psi_sc)}")
        print(f"  Mean |ψ|²: {mean_psi2_sc:.4f}")
        print(f"  Status: {'PASS ✓' if mean_psi2_sc > 0.5 else 'FAIL ✗ (should be ~1)'}")
    
    print("="*70)
    
    # ===== Vortex and Flux Analysis Over Time =====
    print("\n" + "="*70)
    print("VORTEX & FLUX ANALYSIS OVER TIME")
    print("="*70)
    
    # Track flux and vortices at each saved time step
    times = solution.times
    flux_in_hole = []
    vortices_in_sc = []
    flux_around_hole = []
    
    # Define hole bounds for flux calculation
    hole_bounds = (15.0, 25.0, 15.0, 25.0)  # (x_min, x_max, y_min, y_max)
    
    # Define polygon around hole (slightly larger than hole for vortex counting)
    margin = 1.0
    polygon_around_hole = np.array([
        [hole_bounds[0] - margin, hole_bounds[2] - margin],
        [hole_bounds[1] + margin, hole_bounds[2] - margin],
        [hole_bounds[1] + margin, hole_bounds[3] + margin],
        [hole_bounds[0] - margin, hole_bounds[3] + margin],
    ])
    
    print(f"\nAnalyzing {len(times)} time steps...")
    for step in range(len(times)):
        # Count flux quanta through hole
        flux = count_hole_flux_quanta(solution, device, hole_bounds, slice_z=0, step=step)
        flux_in_hole.append(flux)
        
        # Count vortices in SC region (whole domain)
        n_vortices, pos, winding = count_vortices_plaquette(
            solution, device, slice_z=0, step=step,
            winding_threshold=0.8, mask_threshold=0.02
        )
        vortices_in_sc.append(n_vortices)
        
        # Count flux around hole using polygon method
        flux_poly = count_vortices_polygon(solution, device, polygon_around_hole, slice_z=0, step=step)
        flux_around_hole.append(flux_poly)
    
    flux_in_hole = np.array(flux_in_hole)
    vortices_in_sc = np.array(vortices_in_sc)
    flux_around_hole = np.array(flux_around_hole)
    
    print(f"\nTime range: t ∈ [{times[0]:.2f}, {times[-1]:.2f}]")
    print(f"\n✓ Flux quanta through hole:")
    print(f"  Initial: {flux_in_hole[0]:.3f} Φ₀")
    print(f"  Final:   {flux_in_hole[-1]:.3f} Φ₀")
    print(f"  Max:     {np.max(flux_in_hole):.3f} Φ₀")
    
    print(f"\n✓ Vortices in SC region (plaquette method):")
    print(f"  Initial: {vortices_in_sc[0]}")
    print(f"  Final:   {vortices_in_sc[-1]}")
    print(f"  Max:     {np.max(vortices_in_sc)}")
    
    print(f"\n✓ Flux enclosed by polygon around hole:")
    print(f"  Initial: {flux_around_hole[0]:.3f} Φ₀")
    print(f"  Final:   {flux_around_hole[-1]:.3f} Φ₀")
    print(f"  Max:     {np.max(flux_around_hole):.3f} Φ₀")
    
    # Check if flux is non-zero (indicates field is penetrating)
    if np.max(np.abs(flux_in_hole)) < 0.01:
        print(f"\n⚠️  WARNING: Flux through hole is near zero for all time!")
        print(f"    This may indicate:")
        print(f"    - Applied B-field is too weak")
        print(f"    - Simulation time is too short for flux penetration")
        print(f"    - Meissner screening is preventing flux entry")
    else:
        print(f"\n✓ Flux is penetrating the hole (max {np.max(flux_in_hole):.3f} Φ₀)")
    
    # Check if vortices are being detected
    if np.max(vortices_in_sc) == 0:
        print(f"\n⚠️  WARNING: No vortices detected in SC region for all time!")
        print(f"    This may indicate:")
        print(f"    - Vortex detection is not working properly")
        print(f"    - Applied B-field is below H_c1 (no vortex entry)")
        print(f"    - Simulation time is too short for vortex nucleation")
        print(f"    - Winding threshold is too strict")
    else:
        print(f"\n✓ Vortices detected in SC (max {np.max(vortices_in_sc)} vortices)")
    
    print("="*70)
    
    # ===== Plot time evolution =====
    fig_time, axes = plt.subplots(3, 1, figsize=(10, 10))
    
    axes[0].plot(times, flux_in_hole, 'b-', linewidth=2, label="Flux through hole")
    axes[0].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Flux quanta (Φ₀)")
    axes[0].set_title("Magnetic Flux Through Hole")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    axes[1].plot(times, vortices_in_sc, 'r-', linewidth=2, marker='o', markersize=4, 
                 label="Vortices in SC")
    axes[1].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Number of vortices")
    axes[1].set_title("Vortex Count in Superconducting Region")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    axes[2].plot(times, flux_around_hole, 'g-', linewidth=2, label="Flux around hole (polygon)")
    axes[2].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("Flux quanta (Φ₀)")
    axes[2].set_title("Flux Enclosed by Polygon Around Hole")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    fig_time.suptitle("Time Evolution of Flux and Vortices", fontsize=14)
    fig_time.tight_layout()
    fig_time.savefig("hole_bc_verification_time_evolution.png", dpi=150, bbox_inches="tight")
    print("\n  ✓ Saved hole_bc_verification_time_evolution.png")
    plt.close(fig_time)
    
    # ===== Visualizations =====
    print("\nGenerating spatial visualizations...")
    
    # 1. Current density with hole outline
    fig_current, _ = plot_current_density(
        solution,
        step=-1,
        streamplot=True,
        stream_density=1.5,
        hole_polygon=hole_vertices,
        hole_color="red",
        hole_linestyle="--",
        hole_linewidth=2.0,
    )
    fig_current.suptitle(
        f"Current Density with Hole — t = {solution.times[-1]:.2f}\n"
        f"Red dashed line: hole boundary (zero-current BC)",
        fontsize=13,
    )
    fig_current.savefig("hole_bc_verification_current.png", dpi=150, bbox_inches="tight")
    print("  ✓ Saved hole_bc_verification_current.png")
    plt.close(fig_current)
    
    # 2. |psi|^2 with hole outline
    psi_2d = solution.psi_squared_2d(step=-1, slice_z=0)
    xs = np.arange(1, params.Nx) * params.hx
    ys = np.arange(1, params.Ny) * params.hy
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    im = ax.pcolormesh(xx, yy, psi_2d, cmap="inferno", vmin=0, vmax=1, shading="auto")
    
    # Draw hole outline
    poly_closed = list(hole_vertices) + [hole_vertices[0]]
    xs_hole = [p[0] for p in poly_closed]
    ys_hole = [p[1] for p in poly_closed]
    ax.plot(xs_hole, ys_hole, 'r--', linewidth=2, label="Hole boundary")
    
    ax.set_aspect("equal")
    ax.set_xlabel("x (ξ)")
    ax.set_ylabel("y (ξ)")
    ax.set_title(f"|ψ|² with Hole — t = {solution.times[-1]:.2f}\n"
                 f"Red dashed: hole boundary", fontsize=12)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="|ψ|²")
    ax.legend(loc="upper right")
    fig.savefig("hole_bc_verification_psi.png", dpi=150, bbox_inches="tight")
    print("  ✓ Saved hole_bc_verification_psi.png")
    plt.close(fig)
    
    # 3. Cross-section through hole center
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Horizontal slice through y=20 (middle of hole)
    j_mid = 19  # interior index for y=20
    psi_1d_h = psi_2d[:, j_mid]
    x_coords = xs
    
    axes[0].plot(x_coords, psi_1d_h, 'b-', linewidth=2)
    axes[0].axvline(15, color='r', linestyle='--', alpha=0.5, label="Hole boundary")
    axes[0].axvline(25, color='r', linestyle='--', alpha=0.5)
    axes[0].fill_between([15, 25], 0, 1, color='red', alpha=0.1, label="Hole region")
    axes[0].set_xlabel("x (ξ)")
    axes[0].set_ylabel("|ψ|²")
    axes[0].set_title(f"Horizontal cross-section at y=20")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_ylim(-0.05, 1.05)
    
    # Vertical slice through x=20
    i_mid = 19
    psi_1d_v = psi_2d[i_mid, :]
    y_coords = ys
    
    axes[1].plot(y_coords, psi_1d_v, 'b-', linewidth=2)
    axes[1].axvline(15, color='r', linestyle='--', alpha=0.5, label="Hole boundary")
    axes[1].axvline(25, color='r', linestyle='--', alpha=0.5)
    axes[1].fill_between([15, 25], 0, 1, color='red', alpha=0.1, label="Hole region")
    axes[1].set_xlabel("y (ξ)")
    axes[1].set_ylabel("|ψ|²")
    axes[1].set_title(f"Vertical cross-section at x=20")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_ylim(-0.05, 1.05)
    
    fig.suptitle("Order Parameter Cross-Sections Through Hole", fontsize=14)
    fig.tight_layout()
    fig.savefig("hole_bc_verification_crosssection.png", dpi=150, bbox_inches="tight")
    print("  ✓ Saved hole_bc_verification_crosssection.png")
    plt.close(fig)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("Vortices form in the SUPERCONDUCTOR, not in holes!")
    print("What you SHOULD see:")
    print("  ✓ Persistent currents circulating AROUND the hole")
    print("  ✓ Magnetic flux penetrating THROUGH the hole")
    print("  ✓ Zero order parameter (|ψ|² ≈ 0) INSIDE the hole")
    print("  ✓ Zero link variables (φ) AT the hole boundaries")
    print("\nWhat you SHOULD NOT expect:")
    print("  ✗ Vortices inside the hole (no SC = no vortices!)")
    print("  ✗ Large |ψ| inside the hole")
    print("="*70)


if __name__ == "__main__":
    main()

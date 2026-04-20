"""Quick test of extended simulation features with short runtime."""

from __future__ import annotations

import numpy as np
import tdgl3d
from tdgl3d.analysis import check_steady_state, count_vortices_plaquette, count_hole_flux_quanta

# Quick test parameters
params = tdgl3d.SimulationParameters(
    Nx=20, Ny=20,
    hx=1.0, hy=1.0, hz=1.0,
    kappa=2.0,
)

# S/I/S trilayer
trilayer = tdgl3d.Trilayer(
    bottom=tdgl3d.Layer(thickness_z=3, kappa=2.0),
    insulator=tdgl3d.Layer(thickness_z=2, kappa=2.0, is_superconductor=False),
    top=tdgl3d.Layer(thickness_z=3, kappa=2.0),
)

# Applied field
field = tdgl3d.AppliedField(Bz=0.3)
device = tdgl3d.Device(params, applied_field=field, trilayer=trilayer)

print(f"Quick test: Nx={params.Nx}, Ny={params.Ny}, Nz={params.Nz}")
print(f"Interior nodes: {params.n_interior}")
print(f"State vector length: {params.n_state}")

# Initial state
x0 = device.initial_state()

# Short simulation
print("\nRunning short simulation (t_stop=10.0)...")
solution = tdgl3d.solve(
    device,
    x0=x0,
    dt=0.01,
    t_stop=10.0,
    method="euler",
    save_every=20,
    progress=True,
)

print(f"Saved {solution.n_steps} snapshots")

# Test steady-state detection
print("\n" + "="*70)
print("Testing steady-state detection...")
print("="*70)

is_steady, steady_step, metrics = solution.check_steady_state(
    device=device,
    window_size=5,
    psi_threshold=1e-4,
    current_threshold=1e-4,
    start_step=10,
)

print(f"\nSteady State: {'YES ✓' if is_steady else 'NO ✗'}")
if is_steady:
    print(f"  Reached at t = {metrics['steady_time']:.2f}")
    print(f"  Δ|ψ|²: {metrics['psi2_rel_change']:.2e}")
else:
    print(f"  Still evolving at t = {solution.times[-1]:.2f}")
    print(f"  Final Δ|ψ|²: {metrics['psi2_rel_change']:.2e}")

# Test vortex counting
print("\n" + "="*70)
print("Testing vortex counting...")
print("="*70)

# Bottom SC layer
z_ranges = device.trilayer.z_ranges()
k_bot = (z_ranges["bottom"][0] + z_ranges["bottom"][1]) // 2
sz_bot = max(k_bot - 1, 0)

n_vort, vort_pos, winding = count_vortices_plaquette(
    solution, device, slice_z=sz_bot, step=-1
)

print(f"\nVortices at t = {solution.times[-1]:.2f}:")
print(f"  Count: {n_vort}")
if n_vort > 0:
    print(f"  Positions:")
    for i, (pos, w) in enumerate(zip(vort_pos, winding)):
        print(f"    {i+1}. ({pos[0]:.1f}, {pos[1]:.1f})  winding={w:+.2f}")

# Test hole flux (define small hole in center)
nx, ny = params.Nx - 1, params.Ny - 1
hole_bounds = (nx//2 - 2, nx//2 + 2, ny//2 - 2, ny//2 + 2)

n_flux = count_hole_flux_quanta(solution, device, hole_bounds, slice_z=sz_bot, step=-1)
print(f"\nFlux through center 4×4 region: {n_flux:.3f} Φ₀")

print("\n" + "="*70)
print("Quick test completed successfully! ✓")
print("="*70)

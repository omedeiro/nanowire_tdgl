"""Test flux trapping in superconducting holes.

This script tests whether magnetic flux becomes trapped in holes within a
superconductor when an external magnetic field is applied.

Physics parameters:
- κ (GL parameter) controls vortex size: ξ = 1/κ
- Smaller κ → larger vortices → easier to resolve on grid
- Hole must be significantly larger than vortex core for flux quantization
"""

from __future__ import annotations
import numpy as np
import sys
sys.path.insert(0, '/Users/owenmedeiros/nanowire_tdgl/src')

from tdgl3d import Device, SimulationParameters, solve, AppliedField
from tdgl3d.physics.rhs import _expand_interior_to_full


def compute_fluxoid(phi_x, phi_y, contour, params, idx):
    """Compute fluxoid ∮φ·dl around a closed contour in units of Φ₀."""
    phi_x_full = _expand_interior_to_full(phi_x, params, idx)
    phi_y_full = _expand_interior_to_full(phi_y, params, idx)
    
    nx, ny = params.Nx + 1, params.Ny + 1
    phi_x_grid = np.real(phi_x_full.reshape(nx, ny, order='F'))
    phi_y_grid = np.real(phi_y_full.reshape(nx, ny, order='F'))
    
    fluxoid = 0.0
    for i in range(len(contour) - 1):
        p1, p2 = contour[i], contour[i + 1]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        
        ix = int(np.clip((p1[0] + p2[0])/2, 0, nx-1))
        iy = int(np.clip((p1[1] + p2[1])/2, 0, ny-1))
        
        fluxoid += phi_x_grid[ix, iy] * dx + phi_y_grid[ix, iy] * dy
    
    return fluxoid / (2 * np.pi)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Large system with well-resolved vortices
KAPPA = 0.5        # GL parameter → vortex core ξ = 1/κ = 2 grid units
GRID_SIZE = 60     # 60×60 grid
HOLE_SIZE = 20     # 20×20 hole (10 vortex cores across)
H = 1.0            # Grid spacing

# Applied field to trap ~3-4 flux quanta
# Φ = Bz × Area, want Φ ≈ 3×2π ≈ 19
# Area = 20×20 = 400 → Bz ≈ 19/400 ≈ 0.05
BZ_APPLIED = 0.05

# Time stepping
DT = 0.01
T_STOP = 20.0
SAVE_EVERY = 20

# =============================================================================
# SETUP
# =============================================================================

print("=" * 80)
print("FLUX TRAPPING TEST - Larger Vortices & Geometries")
print("=" * 80)

params = SimulationParameters(
    Nx=GRID_SIZE, Ny=GRID_SIZE, Nz=1,
    hx=H, hy=H, hz=H,
    kappa=KAPPA
)

device = Device(params, applied_field=AppliedField(Bz=BZ_APPLIED))

# Center hole in domain
hole_center = GRID_SIZE // 2
hole_half = HOLE_SIZE // 2
hole_verts = [
    (hole_center - hole_half, hole_center - hole_half),
    (hole_center + hole_half, hole_center - hole_half),
    (hole_center + hole_half, hole_center + hole_half),
    (hole_center - hole_half, hole_center + hole_half),
]
device.add_hole(hole_verts)

# Contour just outside hole for fluxoid calculation
contour = np.array([
    [hole_center - hole_half - 1, hole_center - hole_half - 1],
    [hole_center + hole_half + 1, hole_center - hole_half - 1],
    [hole_center + hole_half + 1, hole_center + hole_half + 1],
    [hole_center - hole_half - 1, hole_center + hole_half + 1],
    [hole_center - hole_half - 1, hole_center - hole_half - 1],
], dtype=float)

hole_area = HOLE_SIZE * HOLE_SIZE
expected_flux = BZ_APPLIED * hole_area / (2 * np.pi)
vortex_core_size = 1.0 / KAPPA

print(f"\nPhysics parameters:")
print(f"  κ (GL parameter):     {KAPPA}")
print(f"  ξ (vortex core size): {vortex_core_size:.1f} grid units")
print(f"  Grid spacing h:       {H}")
print(f"  Points across core:   {vortex_core_size/H:.1f}")
print()
print(f"Geometry:")
print(f"  Grid:                 {GRID_SIZE}×{GRID_SIZE}")
print(f"  Hole:                 {HOLE_SIZE}×{HOLE_SIZE} (area={hole_area})")
print(f"  Hole size / ξ:        {HOLE_SIZE/vortex_core_size:.1f}")
print()
print(f"Applied field:")
print(f"  Bz:                   {BZ_APPLIED}")
print(f"  Expected flux:        {expected_flux:.2f} Φ₀")
print()

# CFL check
dt_max_cfl = H**2 / (4 * KAPPA**2)
print(f"Time stepping:")
print(f"  dt:                   {DT}")
print(f"  dt_max (CFL):         {dt_max_cfl:.4f}")
print(f"  CFL safe:             {'✓' if DT <= dt_max_cfl else '✗ WARNING'}")
print()

# =============================================================================
# RUN SIMULATION
# =============================================================================

print("Running simulation...")
sol = solve(
    device,
    dt=DT,
    t_stop=T_STOP,
    save_every=SAVE_EVERY,
    method="euler",
    progress=True,
)

print(f"✓ Completed: {len(sol.times)} timesteps saved")
print()

# =============================================================================
# ANALYZE FLUXOID
# =============================================================================

print("=" * 80)
print("FLUXOID EVOLUTION")
print("=" * 80)
print(f"{'Time':>8s}  {'Fluxoid (Φ₀)':>14s}  {'<|ψ|²>':>10s}")
print("-" * 80)

fluxoids = []
for i, t in enumerate(sol.times):
    phi_x = sol.phi_x(step=i)
    phi_y = sol.phi_y(step=i)
    psi = sol.psi(step=i)
    
    flux = compute_fluxoid(phi_x, phi_y, contour, params, device.idx)
    psi_sq_mean = np.mean(np.abs(psi)**2)
    
    fluxoids.append(flux)
    print(f"{t:8.2f}  {flux:14.6f}  {psi_sq_mean:10.6f}")

print("-" * 80)
print()

# =============================================================================
# SUMMARY
# =============================================================================

flux_peak = max(fluxoids)
flux_final = fluxoids[-1]
t_peak = sol.times[fluxoids.index(flux_peak)]

print("=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Expected flux:       {expected_flux:.2f} Φ₀")
print(f"Peak fluxoid:        {flux_peak:.3f} Φ₀ (at t={t_peak:.1f})")
print(f"Final fluxoid:       {flux_final:.3f} Φ₀ (at t={sol.times[-1]:.1f})")
print(f"Peak/Expected:       {flux_peak/expected_flux*100:.1f}%")
print()

# Check if flux is trapped
if flux_final > 0.8 * flux_peak:
    nearest_quantum = round(flux_final)
    if abs(flux_final - nearest_quantum) < 0.2:
        print(f"✓ FLUX TRAPPED: {flux_final:.2f} Φ₀ ≈ {nearest_quantum} Φ₀ (quantized)")
    else:
        print(f"⚠️  FLUX RETAINED: {flux_final:.2f} Φ₀ (not quantized)")
else:
    print(f"✗ FLUX ESCAPED: Dropped to {flux_final/flux_peak*100:.1f}% of peak")

print("=" * 80)

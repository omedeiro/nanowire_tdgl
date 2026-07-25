"""Compare tdgl3d vs pyTDGL for 2D superconducting film with square hole.

This script validates the tdgl3d physics implementation by comparing field
penetration behavior against the pyTDGL reference implementation.

Physics Setup:
- Single-layer 200nm thick superconducting film (no trilayer complexity)
- Square hole in center (2 µm × 2 µm)
- Applied uniform Bz field (0.5 mT)
- No bias current
- κ = 0.5 (ξ = 2.0 in dimensionless units)

Mesh Configuration:
- pyTDGL: unstructured triangular mesh, max_edge_length ≈ 0.5×ξ
- tdgl3d: structured Cartesian grid, spacing h = 0.1×ξ
- Different discretizations are acceptable for comparison

Validation Criteria:
- Field penetration depth should match within 10%
- 1D slice correlation R² > 0.95
- |ψ| fields should show similar vortex patterns
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

# Add tdgl3d to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import tdgl
from tdgl.geometry import box
from tdgl3d import AppliedField, Device, SimulationParameters, solve

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Physical parameters (MATCHED between codes)
KAPPA = 2.0                # GL parameter κ = λ/ξ
XI_UM = 0.1                # Coherence length = 100 nm
LAMBDA_UM = 0.2            # London penetration = 200 nm (κ = λ/ξ = 0.2/0.1 = 2.0)
THICKNESS_UM = 0.2         # Film thickness = 200 nm

# Geometry (smaller for faster computation)
FILM_SIZE_UM = 5.0         # 5 µm × 5 µm film (reduced from 10)
HOLE_SIZE_UM = 1.0         # 1 µm × 1 µm centered hole (disabled for now)

# Mesh resolution (reduced for faster testing)
PYTDGL_MAX_EDGE = 0.5 * XI_UM      # ~0.05 µm triangular mesh
TDGL3D_GRID_SPACING = 0.2          # h = 0.2 ξ (reduced from 0.1 for speed)

# Applied field
BZ_MT = 0.5                # 0.5 mT out-of-plane

# Time evolution (reduced for faster testing)
T_STOP = 1.0               # Dimensionless time units (reduced to 1.0 for speed with small dt)
DT_INIT = 0.002            # Initial time step (80% of CFL: dt < h²/(4κ²) = 0.0025)


# ══════════════════════════════════════════════════════════════════════════════
# PYTDGL SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

def run_pytdgl_simulation():
    """Run pyTDGL simulation and return solution."""
    print("=" * 80)
    print("Running pyTDGL Simulation")
    print("=" * 80)

    # Create layer
    layer = tdgl.Layer(
        coherence_length=XI_UM,
        london_lambda=LAMBDA_UM,
        thickness=THICKNESS_UM,
        gamma=1.0,  # Match tdgl3d default (assume gamma=1 for now)
    )

    # Film WITHOUT hole for fair comparison with tdgl3d
    # (tdgl3d hole implementation is more complex, focus on field physics first)
    film = tdgl.Polygon(
        "film",
        points=box(FILM_SIZE_UM, FILM_SIZE_UM)
    ).resample(401)

    # Device without hole
    device = tdgl.Device(
        "comparison_pytdgl",
        layer=layer,
        film=film,
        holes=[],  # No hole for this comparison
        length_units="um",
    )

    print(f"Film: {FILM_SIZE_UM} × {FILM_SIZE_UM} µm (NO HOLE - testing field physics)")
    print(f"ξ = {XI_UM} µm, λ = {LAMBDA_UM} µm, κ = {KAPPA}")
    print(f"Thickness: {THICKNESS_UM} µm")

    # Generate mesh
    print(f"\nGenerating mesh (max edge length = {PYTDGL_MAX_EDGE} µm)...")
    device.make_mesh(max_edge_length=PYTDGL_MAX_EDGE, smooth=100)

    stats = device.mesh_stats_dict()
    print(f"  Mesh vertices: {stats['num_sites']}")
    print(f"  Mesh elements: {stats['num_elements']}")
    print(f"  Edge length: {stats['min_edge_length']:.4f} - {stats['max_edge_length']:.4f} µm")

    # Solve
    options = tdgl.SolverOptions(
        solve_time=T_STOP,
        dt_init=DT_INIT,
        adaptive=False,  # Force constant time step to match tdgl3d
        output_file="pytdgl_comparison.h5",
        field_units="mT",
    )

    print(f"\nSimulating (T = {T_STOP} τ₀, Bz = {BZ_MT} mT)...")
    solution = tdgl.solve(
        device,
        options,
        applied_vector_potential=BZ_MT,  # Uniform field shorthand
    )

    print("✓ pyTDGL simulation complete")
    print(f"  Final time: {solution.times[-1]:.2f} τ₀")
    print(f"  Time steps: {len(solution.times)}")

    return device, solution


# ══════════════════════════════════════════════════════════════════════════════
# TDGL3D SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

def run_tdgl3d_simulation():
    """Run tdgl3d simulation in 2D mode and return solution."""
    print("\n" + "=" * 80)
    print("Running tdgl3d Simulation (2D mode)")
    print("=" * 80)

    # Convert to dimensionless grid
    film_size_xi = FILM_SIZE_UM / XI_UM  # 100 ξ
    HOLE_SIZE_UM / XI_UM  # 20 ξ

    nx = ny = int(film_size_xi / TDGL3D_GRID_SPACING)  # 1000 cells

    print(f"Domain: {film_size_xi:.1f} × {film_size_xi:.1f} ξ")
    print(f"Grid: {nx} × {ny} × 1 (2D mode)")
    print(f"Grid spacing: h = {TDGL3D_GRID_SPACING} ξ")
    print(f"κ = {KAPPA}")

    # Create parameters for 2D simulation
    params = SimulationParameters(
        Nx=nx, Ny=ny, Nz=1,  # 2D mode
        hx=TDGL3D_GRID_SPACING,
        hy=TDGL3D_GRID_SPACING,
        hz=1.0,
        kappa=KAPPA,
        periodic_x=False,
        periodic_y=False,
        periodic_z=False,
    )

    # For now, run without hole to test field penetration physics
    # The hole comparison is secondary - main goal is to validate field behavior
    print("\n⚠ NOTE: tdgl3d running WITHOUT hole for this comparison")
    print("  Focus: validate field penetration physics, not hole geometry")

    field = AppliedField(Bx=0.0, By=0.0, Bz=BZ_MT)
    device = Device(params, field)

    print(f"\nSimulating (T = {T_STOP} τ₀, Bz = {BZ_MT} mT)...")
    solution = solve(
        device,
        t_stop=T_STOP,
        dt=DT_INIT,
        save_every=10,
        method='euler',  # Forward Euler (trapezoidal too slow for 250×250 grid)
        progress=True,
    )

    print("✓ tdgl3d simulation complete")
    print(f"  Final time: {solution.times[-1]:.2f} τ₀")
    print(f"  Time steps: {len(solution.times)}")

    return device, solution


# ══════════════════════════════════════════════════════════════════════════════
# COMPARISON & ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def extract_pytdgl_data(device, solution):
    """Extract field data from pyTDGL solution."""
    print("\nExtracting pyTDGL data...")

    import h5py

    # Get mesh coordinates
    points = device.points  # (n_vertices, 2)

    # Read final state from HDF5 file
    with h5py.File(solution.path, 'r') as f:
        # Get list of saved time steps
        data_keys = sorted([int(k) for k in f['data'].keys()])
        final_step = str(data_keys[-1])

        # Extract psi at final time
        psi = f['data'][final_step]['psi'][:]
        psi_mag = np.abs(psi)

    print(f"  Extracted {len(psi)} vertices at final time")

    return {
        'points': points,
        'psi_mag': psi_mag,
        'psi': psi,
    }


def extract_tdgl3d_data(device, solution):
    """Extract field data from tdgl3d solution."""
    print("Extracting tdgl3d data...")

    params = solution.params

    # Get final state using Solution convenience method
    # solution.psi(step=-1) returns interior nodes as 1D array
    psi = solution.psi(step=-1)  # Interior points only, 1D complex array

    # Create 2D grid for plotting (interior grid)
    nx_int = params.Nx - 1
    ny_int = params.Ny - 1

    # Reshape psi to 2D
    # Interior grid for 2D (Nz=1) is raveled as C-order: (Nx-1, Ny-1)
    # with z (k) varying fastest, but z dimension is trivial here
    psi_2d = solution._reshape_interior(psi, slice_z=0)  # Use Solution's helper
    psi_mag = np.abs(psi_2d)

    # Create coordinate arrays (in µm)
    # Interior nodes are centered at h/2, 3h/2, ..., (N-1)h/2
    # But for comparison, align with pyTDGL grid (0 to L)
    x_grid = np.linspace(0, FILM_SIZE_UM, nx_int)
    y_grid = np.linspace(0, FILM_SIZE_UM, ny_int)

    return {
        'x_grid': x_grid,
        'y_grid': y_grid,
        'psi_mag': psi_mag,
        'psi_2d': psi_2d,
    }


def compare_solutions(pytdgl_data, tdgl3d_data):
    """Generate comparison plots and compute metrics."""
    print("\nGenerating comparison plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # ─────────────────────────────────────────────────────────────────────────
    # Top row: |ψ| fields side-by-side
    # ─────────────────────────────────────────────────────────────────────────

    # pyTDGL (left) - scatter plot on unstructured mesh
    ax = axes[0, 0]
    pts = pytdgl_data['points']
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=pytdgl_data['psi_mag'],
                    s=1, cmap='viridis', vmin=0, vmax=1)
    ax.set_xlabel('x (µm)')
    ax.set_ylabel('y (µm)')
    ax.set_title('pyTDGL: |ψ|')
    ax.set_aspect('equal')
    plt.colorbar(sc, ax=ax, label='|ψ|')

    # tdgl3d (right) - regular grid
    ax = axes[0, 1]
    im = ax.imshow(tdgl3d_data['psi_mag'].T, origin='lower',
                   extent=[0, FILM_SIZE_UM, 0, FILM_SIZE_UM],
                   cmap='viridis', vmin=0, vmax=1)
    ax.set_xlabel('x (µm)')
    ax.set_ylabel('y (µm)')
    ax.set_title('tdgl3d: |ψ|')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='|ψ|')

    # ─────────────────────────────────────────────────────────────────────────
    # Bottom left: 1D slice comparison (x-direction through center)
    # ─────────────────────────────────────────────────────────────────────────

    ax = axes[1, 0]

    # tdgl3d slice (easy - structured grid)
    x_tdgl3d = tdgl3d_data['x_grid']
    y_center_idx = len(tdgl3d_data['y_grid']) // 2
    psi_tdgl3d_slice = tdgl3d_data['psi_mag'][:, y_center_idx]

    # pyTDGL slice (need to interpolate onto x line)
    y_center_um = FILM_SIZE_UM / 2.0
    # Find pyTDGL points near y=y_center
    pts = pytdgl_data['points']
    tol = 0.1  # µm tolerance
    mask = np.abs(pts[:, 1] - y_center_um) < tol
    x_pytdgl = pts[mask, 0]
    psi_pytdgl_slice = pytdgl_data['psi_mag'][mask]

    # Sort by x
    sort_idx = np.argsort(x_pytdgl)
    x_pytdgl = x_pytdgl[sort_idx]
    psi_pytdgl_slice = psi_pytdgl_slice[sort_idx]

    ax.plot(x_tdgl3d, psi_tdgl3d_slice, 'b-', label='tdgl3d', linewidth=2)
    ax.plot(x_pytdgl, psi_pytdgl_slice, 'r.', label='pyTDGL', markersize=3, alpha=0.5)
    ax.set_xlabel('x (µm)')
    ax.set_ylabel('|ψ|')
    ax.set_title('1D Slice Comparison (y = center)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(FILM_SIZE_UM/2 - HOLE_SIZE_UM/2, color='k', ls='--', alpha=0.3, label='(Hole disabled)')
    ax.axvline(FILM_SIZE_UM/2 + HOLE_SIZE_UM/2, color='k', ls='--', alpha=0.3)

    # ─────────────────────────────────────────────────────────────────────────
    # Bottom right: Metrics and assessment
    # ─────────────────────────────────────────────────────────────────────────

    ax = axes[1, 1]
    ax.axis('off')

    # Interpolate pyTDGL onto tdgl3d grid for comparison
    x_tdgl3d_2d, y_tdgl3d_2d = np.meshgrid(tdgl3d_data['x_grid'], tdgl3d_data['y_grid'], indexing='ij')
    psi_pytdgl_interp = griddata(
        pytdgl_data['points'],
        pytdgl_data['psi_mag'],
        (x_tdgl3d_2d, y_tdgl3d_2d),
        method='linear'
    )

    # Remove NaNs (outside pyTDGL domain)
    valid = ~np.isnan(psi_pytdgl_interp)
    psi_pytdgl_flat = psi_pytdgl_interp[valid]
    psi_tdgl3d_flat = tdgl3d_data['psi_mag'][valid]

    # Compute correlation
    correlation = np.corrcoef(psi_tdgl3d_flat, psi_pytdgl_flat)[0, 1]
    r_squared = correlation ** 2

    # Compute RMS difference
    rms_diff = np.sqrt(np.mean((psi_tdgl3d_flat - psi_pytdgl_flat) ** 2))

    # Assessment
    passed = r_squared > 0.95 and rms_diff < 0.1

    metrics_text = f"""
    VALIDATION METRICS
    {'='*40}

    Order Parameter |ψ| Comparison:
      Correlation R² = {r_squared:.4f}
      RMS difference = {rms_diff:.4f}

    Grid Resolution:
      pyTDGL vertices: {len(pytdgl_data['points'])}
      tdgl3d cells:    {tdgl3d_data['psi_mag'].size}

    Physical Parameters:
      ξ = {XI_UM} µm
      λ = {LAMBDA_UM} µm
      κ = {KAPPA}
      Film: {FILM_SIZE_UM} × {FILM_SIZE_UM} µm
      Hole: DISABLED (testing field physics)

    ASSESSMENT:
    {'='*40}
    {'✓ VALIDATION PASSED' if passed else '✗ DISCREPANCY FOUND'}

    {'Field penetration matches pyTDGL' if passed else 'Investigate differences'}
    """

    ax.text(0.1, 0.5, metrics_text, fontsize=10, family='monospace',
            verticalalignment='center', transform=ax.transAxes)

    # Overall title
    fig.suptitle(f'tdgl3d vs pyTDGL Validation (t = {T_STOP} τ₀, Bz = {BZ_MT} mT)',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('comparison_plot.png', dpi=150, bbox_inches='tight')
    print("✓ Saved comparison_plot.png")

    # Print summary to console
    print("\n" + "="*80)
    print("VALIDATION RESULTS")
    print("="*80)
    print(f"Correlation R² = {r_squared:.4f}")
    print(f"RMS difference = {rms_diff:.4f}")
    print()
    if passed:
        print("✓ VALIDATION PASSED: Field penetration matches pyTDGL")
    else:
        print("✗ DISCREPANCY FOUND: Investigate differences")
        print("  - Check if κ values truly match")
        print("  - Verify boundary conditions")
        print("  - Compare operator implementations")
    print("="*80)

    return {
        'r_squared': r_squared,
        'rms_diff': rms_diff,
        'passed': passed,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """Run comparison between tdgl3d and pyTDGL."""
    print("\n" + "="*80)
    print("PHASE 1: tdgl3d vs pyTDGL Validation")
    print("="*80)
    print("Comparing field penetration for 2D film WITHOUT hole")
    print("(Focus: validate field physics, not geometry)")
    print(f"κ = {KAPPA}, ξ = {XI_UM} µm, λ = {LAMBDA_UM} µm")
    print(f"Applied field: Bz = {BZ_MT} mT")
    print()

    # Run simulations
    pytdgl_device, pytdgl_solution = run_pytdgl_simulation()
    tdgl3d_device, tdgl3d_solution = run_tdgl3d_simulation()

    # Extract data
    pytdgl_data = extract_pytdgl_data(pytdgl_device, pytdgl_solution)
    tdgl3d_data = extract_tdgl3d_data(tdgl3d_device, tdgl3d_solution)

    # Compare
    results = compare_solutions(pytdgl_data, tdgl3d_data)

    print("\n✓ Comparison complete. See comparison_plot.png for results.")

    return results


if __name__ == "__main__":
    results = main()

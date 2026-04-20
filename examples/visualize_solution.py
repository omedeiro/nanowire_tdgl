#!/usr/bin/env python3
"""Standalone visualization script for saved HDF5 solution files.

Usage
-----
# Generate all default plots
python visualize_solution.py simulation.h5

# Select specific plots
python visualize_solution.py simulation.h5 --plots psi current bfield

# Specify output directory
python visualize_solution.py simulation.h5 --output-dir ./figures

# Custom time step
python visualize_solution.py simulation.h5 --step 50 --plots psi

# Save high-resolution figures
python visualize_solution.py simulation.h5 --dpi 300

Available plots:
  psi       - Order parameter |ψ|²
  phase     - Phase of order parameter
  current   - Current density and streamlines
  bfield    - Magnetic field components
  vortices  - Vortex positions over time
  all       - Generate all plots (default)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from tdgl3d import Solution
from tdgl3d.visualization.plotting import plot_current_density


def plot_psi_2d(solution: Solution, step: int, slice_z: int, output_dir: Path, dpi: int):
    """Plot |ψ|² at a single time step."""
    params = solution.params
    
    psi_2d = solution.psi_squared_2d(step=step, slice_z=slice_z)
    xs = np.arange(1, params.Nx) * params.hx
    ys = np.arange(1, params.Ny) * params.hy
    xx, yy = np.meshgrid(xs, ys, indexing='ij')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.pcolormesh(xx, yy, psi_2d, cmap='inferno', vmin=0, vmax=1, shading='auto')
    
    ax.set_xlabel('x (ξ)', fontsize=12)
    ax.set_ylabel('y (ξ)', fontsize=12)
    ax.set_title(f'|ψ|² at t = {solution.times[step]:.2f}, z = {slice_z}', fontsize=14)
    ax.set_aspect('equal')
    
    plt.colorbar(im, ax=ax, label='|ψ|²', fraction=0.046, pad=0.04)
    
    filename = output_dir / f'psi_squared_step{step:04d}.png'
    fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {filename}")


def plot_phase_2d(solution: Solution, step: int, slice_z: int, output_dir: Path, dpi: int):
    """Plot phase of ψ at a single time step."""
    params = solution.params
    
    phase = solution.phase(step=step, mask_threshold=0.02)
    phase_2d = phase.reshape(params.Nx - 1, params.Ny - 1, max(params.Nz - 1, 1))[:, :, slice_z]
    
    xs = np.arange(1, params.Nx) * params.hx
    ys = np.arange(1, params.Ny) * params.hy
    xx, yy = np.meshgrid(xs, ys, indexing='ij')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.pcolormesh(xx, yy, phase_2d, cmap='twilight', vmin=-np.pi, vmax=np.pi, shading='auto')
    
    ax.set_xlabel('x (ξ)', fontsize=12)
    ax.set_ylabel('y (ξ)', fontsize=12)
    ax.set_title(f'Phase(ψ) at t = {solution.times[step]:.2f}, z = {slice_z}', fontsize=14)
    ax.set_aspect('equal')
    
    cbar = plt.colorbar(im, ax=ax, label='Phase (rad)', fraction=0.046, pad=0.04)
    cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cbar.set_ticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    
    filename = output_dir / f'phase_step{step:04d}.png'
    fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {filename}")


def plot_current_wrapper(solution: Solution, step: int, slice_z: int, output_dir: Path, dpi: int):
    """Plot current density using built-in function."""
    fig, _ = plot_current_density(
        solution,
        step=step,
        slice_z=slice_z,
        streamplot=True,
        stream_density=1.5,
    )
    fig.suptitle(f'Current density at t = {solution.times[step]:.2f}, z = {slice_z}', fontsize=14)
    
    filename = output_dir / f'current_step{step:04d}.png'
    fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {filename}")


def plot_bfield_components(solution: Solution, step: int, slice_z: int, output_dir: Path, dpi: int):
    """Plot B-field components."""
    params = solution.params
    
    Bx, By, Bz = solution.bfield(step=step, full_interior=True)
    
    # Reshape to 3D
    nx_int = params.Nx - 1
    ny_int = params.Ny - 1
    nz_int = max(params.Nz - 1, 1)
    
    Bx_3d = Bx.reshape(nx_int, ny_int, nz_int)
    By_3d = By.reshape(nx_int, ny_int, nz_int)
    Bz_3d = Bz.reshape(nx_int, ny_int, nz_int)
    
    # Extract slice
    Bx_slice = Bx_3d[:, :, slice_z]
    By_slice = By_3d[:, :, slice_z]
    Bz_slice = Bz_3d[:, :, slice_z]
    
    xs = np.arange(1, params.Nx) * params.hx
    ys = np.arange(1, params.Ny) * params.hy
    xx, yy = np.meshgrid(xs, ys, indexing='ij')
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Bx
    im0 = axes[0].pcolormesh(xx, yy, Bx_slice, cmap='RdBu_r', shading='auto')
    axes[0].set_xlabel('x (ξ)')
    axes[0].set_ylabel('y (ξ)')
    axes[0].set_title('Bx')
    axes[0].set_aspect('equal')
    plt.colorbar(im0, ax=axes[0], label='Bx')
    
    # By
    im1 = axes[1].pcolormesh(xx, yy, By_slice, cmap='RdBu_r', shading='auto')
    axes[1].set_xlabel('x (ξ)')
    axes[1].set_ylabel('y (ξ)')
    axes[1].set_title('By')
    axes[1].set_aspect('equal')
    plt.colorbar(im1, ax=axes[1], label='By')
    
    # Bz
    im2 = axes[2].pcolormesh(xx, yy, Bz_slice, cmap='RdBu_r', shading='auto')
    axes[2].set_xlabel('x (ξ)')
    axes[2].set_ylabel('y (ξ)')
    axes[2].set_title('Bz')
    axes[2].set_aspect('equal')
    plt.colorbar(im2, ax=axes[2], label='Bz')
    
    fig.suptitle(f'B-field components at t = {solution.times[step]:.2f}, z = {slice_z}', fontsize=14)
    fig.tight_layout()
    
    filename = output_dir / f'bfield_step{step:04d}.png'
    fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {filename}")


def plot_vortex_timeseries(solution: Solution, output_dir: Path, dpi: int):
    """Plot vortex count vs time (requires device reconstruction)."""
    print("  Warning: Vortex time series requires device info (not saved in HDF5)")
    print("  Skipping vortex plot. To generate, reconstruct device manually.")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize TDGL simulation results from HDF5 file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('input', type=str, help='Input HDF5 file (*.h5)')
    parser.add_argument('--plots', nargs='+', 
                        choices=['psi', 'phase', 'current', 'bfield', 'vortices', 'all'],
                        default=['all'],
                        help='Which plots to generate (default: all)')
    parser.add_argument('--step', type=int, default=-1,
                        help='Time step to plot (default: -1 = final step)')
    parser.add_argument('--slice-z', type=int, default=0,
                        help='Z-slice for 3D simulations (default: 0)')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Output directory for figures (default: current dir)')
    parser.add_argument('--dpi', type=int, default=150,
                        help='Figure resolution (default: 150)')
    
    args = parser.parse_args()
    
    # Load solution
    print(f"Loading solution from {args.input}...")
    try:
        solution = Solution.load(args.input)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)
    
    print(f"  Loaded {solution.n_steps} time steps")
    print(f"  Grid: {solution.params.Nx} × {solution.params.Ny} × {solution.params.Nz}")
    print(f"  Time range: [{solution.times[0]:.3f}, {solution.times[-1]:.3f}]")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Expand 'all' to specific plots
    if 'all' in args.plots:
        plot_types = ['psi', 'phase', 'current', 'bfield']
    else:
        plot_types = args.plots
    
    # Validate step
    step = args.step
    if step < 0:
        step = solution.n_steps + step  # Convert negative index
    if step < 0 or step >= solution.n_steps:
        print(f"Error: step {args.step} out of range [0, {solution.n_steps-1}]")
        sys.exit(1)
    
    # Validate slice_z
    nz_int = max(solution.params.Nz - 1, 1)
    if args.slice_z < 0 or args.slice_z >= nz_int:
        print(f"Error: slice_z {args.slice_z} out of range [0, {nz_int-1}]")
        sys.exit(1)
    
    print(f"\nGenerating plots for step {step} (t = {solution.times[step]:.3f})...")
    
    # Generate requested plots
    if 'psi' in plot_types:
        print("Plotting |ψ|²...")
        plot_psi_2d(solution, step, args.slice_z, output_dir, args.dpi)
    
    if 'phase' in plot_types:
        print("Plotting phase...")
        plot_phase_2d(solution, step, args.slice_z, output_dir, args.dpi)
    
    if 'current' in plot_types:
        print("Plotting current density...")
        plot_current_wrapper(solution, step, args.slice_z, output_dir, args.dpi)
    
    if 'bfield' in plot_types:
        print("Plotting B-field...")
        plot_bfield_components(solution, step, args.slice_z, output_dir, args.dpi)
    
    if 'vortices' in plot_types:
        print("Plotting vortices...")
        plot_vortex_timeseries(solution, output_dir, args.dpi)
    
    print(f"\n✅ Done! Figures saved to {output_dir.resolve()}")


if __name__ == '__main__':
    main()

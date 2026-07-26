"""Hole BC verification: vortex trapping and field penetration in a holed device.

Demonstrates the combined effects of a non-superconducting hole in a 2D film:
- Field penetration (B enhanced in hole)
- Vortex nucleation and trapping
- Zero-current BCs at hole boundaries (work-in-progress)

NOTE: Some aspects of hole BC enforcement are still WIP (see docs/notes/HOLE_BC_STATUS.md).
This figure may show incomplete physics.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tdgl3d import AppliedField, Device, SimulationParameters, solve
from tdgl3d.analysis.vortex_counting import (
    count_hole_flux_quanta,
    count_vortices_plaquette,
)
from tdgl3d.visualization.plotting import plot_current_density


def main(output_dir: Path = Path(__file__).parent, small: bool = False) -> list[Path]:
    if small:
        Nx, Ny = 20, 20
        t_stop = 1.0
        hole_lo, hole_hi = 8, 12
    else:
        Nx, Ny = 60, 60
        t_stop = 30.0
        hole_lo, hole_hi = 22, 38

    params = SimulationParameters(Nx=Nx, Ny=Ny, Nz=1, kappa=2.0)

    def field_ramp(t, t_stop):
        if t < 0.2 * t_stop:
            return 0.0, 0.0, 0.0
        elif t < 0.4 * t_stop:
            scale = (t - 0.2 * t_stop) / (0.2 * t_stop)
            return 0.0, 0.0, 1.0 * scale
        else:
            return 0.0, 0.0, 1.0

    field = AppliedField(Bz=1.0, field_func=field_ramp)
    device = Device(params=params, applied_field=field)

    hole_vertices = [
        (float(hole_lo), float(hole_lo)),
        (float(hole_hi), float(hole_lo)),
        (float(hole_hi), float(hole_hi)),
        (float(hole_lo), float(hole_hi)),
    ]
    device.add_hole(hole_vertices)

    x0 = device.initial_state()
    rng = np.random.default_rng(42)
    noise = 0.01 * (rng.standard_normal(params.n_interior)
                   + 1j * rng.standard_normal(params.n_interior))
    if device.material is not None:
        x0.psi[:] += noise * device.material.interior_sc_mask
    else:
        x0.psi[:] += noise

    save_every = max(1, int(t_stop / 0.5))
    sol = solve(device, x0=x0, dt=0.01, t_stop=t_stop, method="euler",
                save_every=save_every, progress=False)

    saved_paths = []

    # --- Figure 1: Time evolution ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    n_vort_list = []
    flux_hole_list = []
    times_list = []

    for step in range(sol.n_steps):
        try:
            n_v, _, _ = count_vortices_plaquette(sol, device, step=step)
        except Exception:
            n_v = 0
        try:
            flux = count_hole_flux_quanta(sol, device, step=step)
        except Exception:
            flux = 0.0
        n_vort_list.append(n_v)
        flux_hole_list.append(flux)
        times_list.append(sol.times[step])

    times_arr = np.array(times_list)
    axes[0].plot(times_arr, n_vort_list, "o-", color="C0", markersize=2)
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("Vortex count")
    axes[0].set_title("Vortex count vs time")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(times_arr, flux_hole_list, "o-", color="C1", markersize=2)
    # Expected: flux quantized to integer multiples of Φ₀
    axes[1].axhline(1.0, color="gray", ls=":", alpha=0.5, label="n = 1 Φ₀")
    axes[1].axhline(2.0, color="gray", ls=":", alpha=0.3, label="n = 2 Φ₀")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("Flux / Φ₀")
    axes[1].set_title("Flux through hole")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)

    # Bz at a point in the hole vs SC
    Bx, By, Bz = sol.bfield(step=-1, full_interior=True)
    Bz_2d = np.real(Bz.reshape(Nx - 1, Ny - 1))
    mid = (Nx - 1) // 2
    axes[2].plot(Bz_2d[mid, :], "o-", color="C2", markersize=2)
    # Mark hole region
    axes[2].axvspan(hole_lo, hole_hi, alpha=0.15, color="red", label="Hole")
    axes[2].set_xlabel("x index")
    axes[2].set_ylabel("Bz")
    axes[2].set_title("Bz cross-section at y-mid")
    axes[2].legend(fontsize=7)
    axes[2].grid(True, alpha=0.3)

    # Flux quantization error annotation
    final_flux = flux_hole_list[-1] if flux_hole_list else 0
    nearest_int = round(final_flux)
    flux_error = abs(final_flux - nearest_int)
    n_v_final = n_vort_list[-1] if n_vort_list else 0
    text = (
        f"Final flux:   {final_flux:.3f} Φ₀\n"
        f"Nearest int:  {nearest_int}\n"
        f"Quant. error: {flux_error:.4f}\n"
        f"Final vort:   {n_v_final}"
    )
    axes[1].text(
        0.97, 0.03, text, transform=axes[1].transAxes,
        fontsize=8, fontfamily="monospace",
        verticalalignment="bottom", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9),
    )

    fig.suptitle("Hole BC Verification — Time Evolution", fontsize=14, y=1.02)
    fig.tight_layout()
    p = output_dir / "hole_bc_verification_time_evolution.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(p)

    # --- Figure 2: Current density ---
    hole_polygon = [(float(hole_lo), float(hole_lo)),
                    (float(hole_hi), float(hole_lo)),
                    (float(hole_hi), float(hole_hi)),
                    (float(hole_lo), float(hole_hi))]
    fig, axes = plot_current_density(sol, step=-1, slice_z=0, streamplot=True,
                                     hole_polygon=hole_polygon, figsize=(18, 5))
    fig.suptitle("Hole BC Verification — Current Density", fontsize=14, y=1.02)
    fig.tight_layout()
    p = output_dir / "hole_bc_verification_current.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(p)

    # --- Figure 3: Order parameter ---
    psi2 = sol.psi_squared_2d(step=-1)
    fig, ax = plt.subplots(figsize=(6, 5))
    xs = np.arange(1, Nx) * params.hx
    ys = np.arange(1, Ny) * params.hy
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    im = ax.pcolormesh(xx, yy, psi2, cmap="inferno", vmin=0, vmax=1, shading="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="|ψ|²")
    hx_coords = [hole_lo, hole_hi, hole_hi, hole_lo, hole_lo]
    hy_coords = [hole_lo, hole_lo, hole_hi, hole_hi, hole_lo]
    ax.plot(hx_coords, hy_coords, "r--", linewidth=1.5)
    ax.set_xlabel("x (ξ)")
    ax.set_ylabel("y (ξ)")
    ax.set_title("Order parameter with hole")
    ax.set_aspect("equal")
    fig.tight_layout()
    p = output_dir / "hole_bc_verification_psi.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(p)

    # --- Figure 4: Cross-section ---
    fig, ax = plt.subplots(figsize=(8, 4))
    mid_y = (Ny - 1) // 2
    ax.plot(xs, psi2[:, mid_y], "o-", color="C0", markersize=2, label="|ψ|² at y-mid")
    ax.axvspan(hole_lo, hole_hi, alpha=0.2, color="red", label="Hole region")
    # Expected: |ψ|² → 0 inside hole, |ψ|² → 1 outside
    ax.axhline(0.0, color="gray", ls=":", alpha=0.3, label="Expected in hole: 0")
    ax.axhline(1.0, color="gray", ls=":", alpha=0.3, label="Expected in SC: 1")
    ax.set_xlabel("x (ξ)")
    ax.set_ylabel("|ψ|²")
    ax.set_title("Cross-section through hole center")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = output_dir / "hole_bc_verification_crosssection.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(p)

    return saved_paths


if __name__ == "__main__":
    main()

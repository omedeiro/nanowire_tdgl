"""Vortex entry: Abrikosov vortex nucleation above the lower critical field H_c1.

Demonstrates that when Bz exceeds H_c1, magnetic flux penetrates the superconductor
as quantized vortices with winding number ±1.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tdgl3d import AppliedField, Device, SimulationParameters, solve
from tdgl3d.analysis.vortex_counting import count_vortices_plaquette


def main(output_dir: Path = Path(__file__).parent, small: bool = False) -> list[Path]:
    if small:
        Nx, Ny, Nz = 12, 12, 1
        t_stop = 4.0
    else:
        Nx, Ny, Nz = 40, 40, 1
        t_stop = 40.0

    params = SimulationParameters(Nx=Nx, Ny=Ny, Nz=Nz, kappa=2.0)
    # H_c2 = 1 in these units (fields are measured in Φ₀/(2πξ²)), so the applied
    # field must stay below 1 or the sample is simply normal — at Bz = 2.5 the
    # order parameter is zero everywhere and the "vortices" reported are
    # detector noise in a non-superconducting region.  H_c1(κ=2) ≈ 0.15, so
    # Bz = 0.6 is comfortably inside the mixed state.
    Bz_applied = 0.6
    field = AppliedField(Bz=Bz_applied, t_on_fraction=1.0)
    device = Device(params, applied_field=field)

    sol = solve(
        device, dt=0.01, t_stop=t_stop, method="euler",
        save_every=max(1, int(t_stop)), progress=False,
    )

    psi2 = sol.psi_squared_2d(step=-1)
    n_vort, positions, windings = count_vortices_plaquette(sol, device, step=-1)

    # Expected vortex count: n ≈ B·A / Φ₀ = B·(Lx·Ly) / (2π)
    area = (params.Nx * params.hx) * (params.Ny * params.hy)
    expected_vortices = float(Bz_applied * area / (2 * np.pi))

    # Symmetry checks on |ψ|²
    Nx_int, Ny_int = Nx - 1, Ny - 1
    mid_x, mid_y = Nx_int // 2, Ny_int // 2

    # x-symmetry: compare left vs right halves
    psi2_left = psi2[:mid_x, :]
    psi2_right = psi2[Nx_int - mid_x:Nx_int, :][::-1, :]
    sym_x = float(np.max(np.abs(psi2_left - psi2_right))) if psi2_left.size > 0 else 0.0

    # y-symmetry: compare bottom vs top halves
    psi2_bottom = psi2[:, :mid_y]
    psi2_top = psi2[:, Ny_int - mid_y:Ny_int][:, ::-1]
    sym_y = float(np.max(np.abs(psi2_bottom - psi2_top))) if psi2_bottom.size > 0 else 0.0

    # Winding number check
    all_unit = bool(np.all(np.abs(np.abs(windings) - 1.0) < 0.3)) if len(windings) > 0 else True

    # --- Figure ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: |psi|^2 heatmap
    ax = axes[0]
    xs = np.arange(1, Nx) * params.hx
    ys = np.arange(1, Ny) * params.hy
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    im = ax.pcolormesh(xx, yy, psi2, cmap="inferno", vmin=0, vmax=1, shading="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="|ψ|²")

    # Mark vortex cores
    if n_vort > 0 and positions.size > 0:
        ax.plot(positions[:, 0], positions[:, 1], "x",
                color="cyan", markersize=10, markeredgewidth=2)
        for k, (vx, vy) in enumerate(positions):
            w = windings[k] if k < len(windings) else 0
            ax.annotate(f"w={w:.0f}", (vx, vy), textcoords="offset points",
                        xytext=(8, 8), fontsize=7, color="cyan")

    # Expected count annotation
    count_ratio = n_vort / expected_vortices if expected_vortices > 0 else 0
    text = (
        f"Detected:    {n_vort}\n"
        f"Expected:    {expected_vortices:.0f}\n"
        f"Ratio:       {count_ratio:.1%}\n"
        f"Sym-x:       {sym_x:.4f}\n"
        f"Sym-y:       {sym_y:.4f}\n"
        f"|w|=1:       {all_unit}"
    )
    ax.text(
        0.03, 0.03, text, transform=ax.transAxes,
        fontsize=8, fontfamily="monospace",
        verticalalignment="bottom", horizontalalignment="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9),
    )

    ax.set_xlabel("x (ξ)")
    ax.set_ylabel("y (ξ)")
    ax.set_title(f"|ψ|² — {n_vort} vortex(es) detected")
    ax.set_aspect("equal")

    # Right: Phase arg(psi)
    ax = axes[1]
    phase = sol.phase(step=-1, mask_threshold=0.02)
    phase_2d = sol._reshape_interior(phase, slice_z=0)
    im = ax.pcolormesh(xx, yy, phase_2d, cmap="twilight", vmin=-np.pi, vmax=np.pi, shading="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="arg(ψ) [rad]")

    if n_vort > 0 and positions.size > 0:
        ax.plot(positions[:, 0], positions[:, 1], "x",
                color="lime", markersize=10, markeredgewidth=2)

    ax.set_xlabel("x (ξ)")
    ax.set_ylabel("y (ξ)")
    ax.set_title("Phase winding")
    ax.set_aspect("equal")

    fig.suptitle(
        f"Vortex Nucleation — Bz = {Bz_applied} > H_c1, κ = {params.kappa}",
        fontsize=14, y=1.02,
    )
    fig.tight_layout()

    out = output_dir / "vortex_entry.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [out]


if __name__ == "__main__":
    main()

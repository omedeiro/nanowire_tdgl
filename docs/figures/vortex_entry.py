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
        Nx, Ny, Nz = 6, 6, 1
        t_stop = 2.0
    else:
        Nx, Ny, Nz = 20, 20, 1
        t_stop = 15.0

    params = SimulationParameters(Nx=Nx, Ny=Ny, Nz=Nz, kappa=2.0)
    field = AppliedField(Bz=2.5, t_on_fraction=1.0)
    device = Device(params, applied_field=field)

    sol = solve(
        device, dt=0.01, t_stop=t_stop, method="euler",
        save_every=max(1, int(t_stop)), progress=False,
    )

    psi2 = sol.psi_squared_2d(step=-1)
    n_vort, positions, windings = count_vortices_plaquette(sol, device, step=-1)

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
                        xytext=(8, 8), fontsize=8, color="cyan")

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

    fig.suptitle(f"Vortex Nucleation — Bz = 2.5 > H_c1, κ = {params.kappa}", fontsize=14, y=1.02)
    fig.tight_layout()

    out = output_dir / "vortex_entry.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [out]


if __name__ == "__main__":
    main()

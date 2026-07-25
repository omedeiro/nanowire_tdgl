"""Field penetration in holes: enhanced B-field in non-superconducting holes.

Demonstrates that a non-superconducting hole does not exhibit Meissner screening,
so the applied field penetrates much more strongly than in the surrounding SC bulk.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tdgl3d import AppliedField, Device, Layer, SimulationParameters, Trilayer, solve


def main(output_dir: Path = Path(__file__).parent, small: bool = False) -> list[Path]:
    if small:
        Nx, Ny = 8, 8
        t_stop = 2.0
        hole_lo, hole_hi = 3, 5
    else:
        Nx, Ny = 30, 30
        t_stop = 15.0
        hole_lo, hole_hi = 10, 20

    params = SimulationParameters(Nx=Nx, Ny=Ny, Nz=1, kappa=2.0)
    field = AppliedField(Bz=0.3, t_on_fraction=1.0)

    # Trilayer with zero-thickness insulator to get MaterialMap
    trilayer = Trilayer(
        bottom=Layer(thickness_z=1, kappa=2.0),
        insulator=Layer(thickness_z=0, kappa=2.0, is_superconductor=False),
        top=Layer(thickness_z=0, kappa=2.0),
    )
    device = Device(params, applied_field=field, trilayer=trilayer)

    # Add square hole in center
    hole_vertices = [
        (float(hole_lo), float(hole_lo)),
        (float(hole_hi), float(hole_lo)),
        (float(hole_hi), float(hole_hi)),
        (float(hole_lo), float(hole_hi)),
    ]
    device.add_hole(hole_vertices)

    sol = solve(
        device, dt=0.01, t_stop=t_stop, method="euler",
        save_every=max(1, int(t_stop)), progress=False,
    )

    psi2 = sol.psi_squared_2d(step=-1)
    Bx, By, Bz = sol.bfield(step=-1, full_interior=True)
    Bz_2d = Bz.reshape(Nx - 1, Ny - 1)

    # --- Figure ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Bz heatmap
    ax = axes[0]
    xs = np.arange(1, Nx) * params.hx
    ys = np.arange(1, Ny) * params.hy
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    im = ax.pcolormesh(xx, yy, Bz_2d, cmap="RdBu_r", shading="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Bz")
    # Draw hole outline
    hx_coords = [hole_lo, hole_hi, hole_hi, hole_lo, hole_lo]
    hy_coords = [hole_lo, hole_lo, hole_hi, hole_hi, hole_lo]
    ax.plot(hx_coords, hy_coords, "r--", linewidth=1.5, label="Hole")
    ax.set_xlabel("x (ξ)")
    ax.set_ylabel("y (ξ)")
    ax.set_title("Bz — enhanced in hole")
    ax.set_aspect("equal")
    ax.legend()

    # Right: |psi|^2 heatmap
    ax = axes[1]
    im = ax.pcolormesh(xx, yy, psi2, cmap="inferno", vmin=0, vmax=1, shading="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="|ψ|²")
    ax.plot(hx_coords, hy_coords, "r--", linewidth=1.5, label="Hole")
    ax.set_xlabel("x (ξ)")
    ax.set_ylabel("y (ξ)")
    ax.set_title("|ψ|² — suppressed in hole")
    ax.set_aspect("equal")
    ax.legend()

    fig.suptitle("Field Penetration in Holes — No Meissner Screening", fontsize=14, y=1.02)
    fig.tight_layout()

    out = output_dir / "hole_field_penetration.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [out]


if __name__ == "__main__":
    main()

"""Trilayer B-field penetration: S/I/S screening profile.

Demonstrates that in a Superconductor/Insulator/Superconductor trilayer,
the SC layers screen the magnetic field (Meissner effect) while the insulator
allows field penetration.
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
        Nx, Ny = 4, 4
        t_stop = 1.0
    else:
        Nx, Ny = 8, 8
        t_stop = 5.0

    trilayer = Trilayer(
        bottom=Layer(thickness_z=4, kappa=2.0),
        insulator=Layer(thickness_z=4, kappa=0.0, is_superconductor=False),
        top=Layer(thickness_z=4, kappa=2.0),
    )
    params = SimulationParameters(Nx=Nx, Ny=Ny, Nz=trilayer.Nz, kappa=2.0)
    field = AppliedField(Bz=0.3, t_on_fraction=1.0)
    device = Device(params, applied_field=field, trilayer=trilayer)

    save_every = max(1, int(t_stop / 0.5))
    sol = solve(
        device, dt=0.01, t_stop=t_stop, method="euler",
        save_every=save_every, progress=False,
    )

    # Extract Bz profile along z at center
    Bx, By, Bz = sol.bfield(step=-1, full_interior=True)
    Nz_int = max(trilayer.Nz - 1, 1)
    Bz_3d = Bz.reshape(Nx - 1, Ny - 1, Nz_int)
    mid_x, mid_y = (Nx - 1) // 2, (Ny - 1) // 2
    bz_z = Bz_3d[mid_x, mid_y, :]

    z_coords = np.arange(1, trilayer.Nz) * params.hz

    # Layer boundaries
    z_ranges = trilayer.z_ranges()

    # --- Figure ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Bz(z) profile with shaded layers
    ax = axes[0]
    ax.plot(z_coords, bz_z, "o-", color="C0", markersize=4, label="Bz(z)")

    # Shade layers
    z_bottom = z_ranges["bottom"]
    z_ins = z_ranges["insulator"]
    z_top = z_ranges["top"]
    def _shade_layer(ax, z_range, color, label):
        ax.axvspan(
            z_range[0] * params.hz, z_range[1] * params.hz,
            alpha=0.15, color=color, label=label,
        )

    _shade_layer(ax, z_bottom, "blue", "SC (bottom)")
    _shade_layer(ax, z_ins, "red", "Insulator")
    _shade_layer(ax, z_top, "blue", "SC (top)")
    ax.axhline(0.3, color="gray", ls=":", alpha=0.5, label="Bz_applied")
    ax.set_xlabel("z (ξ)")
    ax.set_ylabel("Bz")
    ax.set_title("B-field z-profile at center")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: |psi| z-profile
    ax = axes[1]
    psi = sol.psi(step=-1)
    psi_3d = psi.reshape(Nx - 1, Ny - 1, Nz_int)
    psi_z = np.abs(psi_3d[mid_x, mid_y, :]) ** 2
    ax.plot(z_coords, psi_z, "o-", color="C1", markersize=4, label="|ψ|²(z)")
    _shade_layer(ax, z_bottom, "blue", "SC (bottom)")
    _shade_layer(ax, z_ins, "red", "Insulator")
    _shade_layer(ax, z_top, "blue", "SC (top)")
    ax.set_xlabel("z (ξ)")
    ax.set_ylabel("|ψ|²")
    ax.set_title("Order parameter z-profile")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Trilayer S/I/S — B-field Penetration and Screening", fontsize=14, y=1.02)
    fig.tight_layout()

    out = output_dir / "trilayer_bfield.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [out]


if __name__ == "__main__":
    main()

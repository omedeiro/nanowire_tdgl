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
        Nx, Ny = 8, 8
        t_stop = 3.0
    else:
        Nx, Ny = 16, 16
        t_stop = 15.0

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

    Bz_applied = 0.3

    # Extract Bz profile along z at center
    Bx, By, Bz = sol.bfield(step=-1, full_interior=True)
    Nz_int = max(trilayer.Nz - 1, 1)
    Bz_3d = Bz.reshape(Nx - 1, Ny - 1, Nz_int)
    mid_x, mid_y = (Nx - 1) // 2, (Ny - 1) // 2
    bz_z = np.real(Bz_3d[mid_x, mid_y, :])

    z_coords = np.arange(1, trilayer.Nz) * params.hz

    # Layer boundaries (full-grid z-indices)
    z_ranges = trilayer.z_ranges()

    # Map full-grid z-ranges to interior z-indices for slicing
    def _interior_range(z0, z1):
        iz0 = max(z0, 1) - 1
        iz1 = min(z1, trilayer.Nz - 1)
        return iz0, iz1

    iz_bot = _interior_range(*z_ranges["bottom"])
    iz_ins = _interior_range(*z_ranges["insulator"])
    iz_top = _interior_range(*z_ranges["top"])

    bz_bottom_mean = float(np.mean(bz_z[iz_bot[0]:iz_bot[1]]))
    bz_insulator_mean = float(np.mean(bz_z[iz_ins[0]:iz_ins[1]]))
    bz_top_mean = float(np.mean(bz_z[iz_top[0]:iz_top[1]]))

    # Symmetry: bottom SC vs top SC (should be ≈ equal for symmetric stack)
    bz_sym_error = abs(bz_bottom_mean - bz_top_mean)
    bz_sym_error /= max(abs(bz_bottom_mean + bz_top_mean) / 2, 1e-12)

    # --- Expected values ---
    # SC layers: Bz → 0 (Meissner screening)
    # Insulator: Bz → Bz_applied (field penetration, though solver limitation gives ≈0)
    # |ψ|²: SC → 1, insulator → 0
    expected_bz_sc = 0.0
    expected_bz_ins = Bz_applied
    expected_psi_sc = 1.0
    expected_psi_ins = 0.0

    # --- Figure ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Bz(z) profile with shaded layers + expected
    ax = axes[0]
    ax.plot(z_coords, bz_z, "o-", color="C0", markersize=4, label="Bz(z)")

    # Expected reference lines
    ax.axhline(expected_bz_sc, color="C0", ls=":", alpha=0.4, linewidth=1.5,
               label=f"Expected SC: Bz → {expected_bz_sc}")
    ax.axhline(expected_bz_ins, color="C3", ls=":", alpha=0.4, linewidth=1.5,
               label=f"Expected ins.: Bz → {expected_bz_ins:.1f}")

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
    ax.axhline(Bz_applied, color="gray", ls=":", alpha=0.5, label="Bz_applied")

    # Error annotation box
    bz_sc_err = abs(bz_bottom_mean - expected_bz_sc)
    bz_ins_err = abs(bz_insulator_mean - expected_bz_ins)
    text = (
        f"SC screening error:  {bz_sc_err:.4f}\n"
        f"Ins. penetr. error:  {bz_ins_err:.4f}\n"
        f"SC symmetry error:   {bz_sym_error:.2%}"
    )
    ax.text(
        0.97, 0.97, text, transform=ax.transAxes,
        fontsize=8, fontfamily="monospace",
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9),
    )

    ax.set_xlabel("z (ξ)")
    ax.set_ylabel("Bz")
    ax.set_title("B-field z-profile at center")
    ax.legend(fontsize=7, loc="center right")
    ax.grid(True, alpha=0.3)

    # Right: |psi| z-profile + expected
    ax = axes[1]
    psi = sol.psi(step=-1)
    psi_3d = psi.reshape(Nx - 1, Ny - 1, Nz_int)
    psi_z = np.abs(psi_3d[mid_x, mid_y, :]) ** 2
    ax.plot(z_coords, psi_z, "o-", color="C1", markersize=4, label="|ψ|²(z)")

    # Expected reference lines
    ax.axhline(expected_psi_sc, color="C0", ls=":", alpha=0.4, linewidth=1.5,
               label=f"Expected SC: |ψ|² → {expected_psi_sc}")
    ax.axhline(expected_psi_ins, color="C3", ls=":", alpha=0.4, linewidth=1.5,
               label=f"Expected ins.: |ψ|² → {expected_psi_ins}")

    _shade_layer(ax, z_bottom, "blue", "SC (bottom)")
    _shade_layer(ax, z_ins, "red", "Insulator")
    _shade_layer(ax, z_top, "blue", "SC (top)")

    # Error annotation
    psi_bottom_mean = float(np.mean(psi_z[iz_bot[0]:iz_bot[1]]))
    psi_insulator_mean = float(np.mean(psi_z[iz_ins[0]:iz_ins[1]]))
    psi_top_mean = float(np.mean(psi_z[iz_top[0]:iz_top[1]]))
    psi_sym_error = abs(psi_bottom_mean - psi_top_mean)
    psi_sym_error /= max(abs(psi_bottom_mean + psi_top_mean) / 2, 1e-12)
    psi_sc_err = abs(psi_bottom_mean - expected_psi_sc)
    psi_ins_err = abs(psi_insulator_mean - expected_psi_ins)
    text2 = (
        f"SC |ψ|² error:      {psi_sc_err:.4f}\n"
        f"Ins. |ψ|² error:    {psi_ins_err:.4f}\n"
        f"SC symmetry error:   {psi_sym_error:.2%}"
    )
    ax.text(
        0.97, 0.97, text2, transform=ax.transAxes,
        fontsize=8, fontfamily="monospace",
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9),
    )

    ax.set_xlabel("z (ξ)")
    ax.set_ylabel("|ψ|²")
    ax.set_title("Order parameter z-profile")
    ax.legend(fontsize=7, loc="center right")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Trilayer S/I/S — B-field Penetration and Screening", fontsize=14, y=1.02)
    fig.tight_layout()

    out = output_dir / "trilayer_bfield.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [out]


if __name__ == "__main__":
    main()

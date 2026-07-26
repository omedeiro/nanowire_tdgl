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
        Nx, Ny = 16, 16
        t_stop = 2.0
        hole_lo, hole_hi = 5, 11
    else:
        Nx, Ny = 60, 60
        t_stop = 15.0
        hole_lo, hole_hi = 20, 40

    Bz_applied = 0.3
    params = SimulationParameters(Nx=Nx, Ny=Ny, Nz=1, kappa=2.0)
    field = AppliedField(Bz=Bz_applied, t_on_fraction=1.0)

    trilayer = Trilayer(
        bottom=Layer(thickness_z=1, kappa=2.0),
        insulator=Layer(thickness_z=0, kappa=2.0, is_superconductor=False),
        top=Layer(thickness_z=0, kappa=2.0),
    )
    device = Device(params, applied_field=field, trilayer=trilayer)

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
    Bz_2d = np.real(Bz.reshape(Nx - 1, Ny - 1))

    # Compute hole vs SC averages
    Nx_int, Ny_int = Nx - 1, Ny - 1
    hx_idx_lo = max(hole_lo - 1, 0)
    hx_idx_hi = min(hole_hi - 1, Nx_int)
    hy_idx_lo = max(hole_lo - 1, 0)
    hy_idx_hi = min(hole_hi - 1, Ny_int)

    bz_hole = float(np.mean(Bz_2d[hx_idx_lo:hx_idx_hi, hy_idx_lo:hy_idx_hi]))
    bz_sc_mask = np.ones((Nx_int, Ny_int), dtype=bool)
    bz_sc_mask[hx_idx_lo:hx_idx_hi, hy_idx_lo:hy_idx_hi] = False
    bz_sc = float(np.mean(Bz_2d[bz_sc_mask]))

    psi2_hole = float(np.mean(psi2[hx_idx_lo:hx_idx_hi, hy_idx_lo:hy_idx_hi]))
    psi2_sc = float(np.mean(psi2[bz_sc_mask]))

    # Symmetry: compare x-left vs x-right halves of SC region
    mid_x = Nx_int // 2
    psi2_left = psi2[:mid_x, :]
    psi2_right = psi2[Nx_int - mid_x:Nx_int, :][::-1, :]
    sym_x = float(np.max(np.abs(psi2_left - psi2_right)))

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
    # Expected reference lines
    ax.axhline(Bz_applied, color="gray", ls=":", alpha=0.3, linewidth=1)

    # Error annotation
    bz_hole_err = abs(bz_hole - Bz_applied)
    bz_sc_err = abs(bz_sc - 0.0)  # SC should screen to ~0
    text = (
        f"Bz(hole):  {bz_hole:.4f}\n"
        f"Bz(SC):    {bz_sc:.4f}\n"
        f"Bz(applied): {Bz_applied:.4f}\n"
        f"Hole error: {bz_hole_err:.4f}\n"
        f"SC error:   {bz_sc_err:.4f}"
    )
    ax.text(
        0.03, 0.03, text, transform=ax.transAxes,
        fontsize=8, fontfamily="monospace",
        verticalalignment="bottom", horizontalalignment="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9),
    )

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

    psi2_hole_err = abs(psi2_hole - 0.0)
    psi2_sc_err = abs(psi2_sc - 1.0)
    text2 = (
        f"|ψ|²(hole): {psi2_hole:.4f}\n"
        f"|ψ|²(SC):   {psi2_sc:.4f}\n"
        f"Hole error:  {psi2_hole_err:.4f}\n"
        f"SC error:    {psi2_sc_err:.4f}\n"
        f"Sym-x:       {sym_x:.4f}"
    )
    ax.text(
        0.03, 0.03, text2, transform=ax.transAxes,
        fontsize=8, fontfamily="monospace",
        verticalalignment="bottom", horizontalalignment="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9),
    )

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

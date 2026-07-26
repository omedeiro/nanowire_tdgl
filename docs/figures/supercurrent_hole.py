"""Supercurrent around a hole: screening currents diverted by a non-SC void.

Demonstrates that supercurrent J_s ∝ |ψ|² vanishes inside non-SC holes and
flows around them, consistent with zero-current boundary conditions.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tdgl3d import AppliedField, Device, Layer, SimulationParameters, Trilayer, solve
from tdgl3d.visualization.plotting import plot_current_density


def main(output_dir: Path = Path(__file__).parent, small: bool = False) -> list[Path]:
    if small:
        Nx, Ny = 16, 16
        t_stop = 2.0
        hole_lo, hole_hi = 5, 11
    else:
        Nx, Ny = 48, 48
        t_stop = 10.0
        hole_lo, hole_hi = 18, 30

    params = SimulationParameters(Nx=Nx, Ny=Ny, Nz=1, kappa=2.0)
    field = AppliedField(Bz=0.3, t_on_fraction=1.0)

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

    # Build mask for hole interior nodes using full-grid → interior mapping
    n = params.n_interior
    state = sol.states[:, -1]
    psi_int = state[:n]
    px_int = state[n:2*n]
    py_int = state[2*n:3*n]
    # J ∝ Im(ψ* (ψ·U - ψ)) where U = e^{iφ}
    jx = np.imag(np.conj(psi_int) * (psi_int * px_int - psi_int))
    jy = np.imag(np.conj(psi_int) * (psi_int * py_int - psi_int))
    j_mag = np.sqrt(jx**2 + jy**2)

    # Map interior indices to full-grid (i,j) to identify hole nodes
    from tdgl3d.mesh.indices import construct_indices
    idx = construct_indices(params)
    m = idx.interior_to_full
    mk = Nx + 1  # stride in full grid

    hole_mask = np.zeros(n, dtype=bool)
    for k_int in range(n):
        full_idx = m[k_int]
        i_full = full_idx % mk
        j_full = full_idx // mk
        # Check if (i_full, j_full) is inside the hole
        if (hole_lo <= i_full <= hole_hi and
                hole_lo <= j_full <= hole_hi):
            hole_mask[k_int] = True

    j_hole = float(np.mean(j_mag[hole_mask])) if np.any(hole_mask) else 0.0
    j_sc = float(np.mean(j_mag[~hole_mask])) if np.any(~hole_mask) else 0.0

    hole_polygon = [(float(hole_lo), float(hole_lo)),
                    (float(hole_hi), float(hole_lo)),
                    (float(hole_hi), float(hole_hi)),
                    (float(hole_lo), float(hole_hi))]

    fig, axes = plot_current_density(sol, step=-1, slice_z=0, streamplot=True,
                                     hole_polygon=hole_polygon, figsize=(18, 5))

    # Add annotation about current diversion
    text = (
        f"J(hole): {j_hole:.4f}\n"
        f"J(SC):   {j_sc:.4f}\n"
        f"Ratio:   {j_hole/max(j_sc, 1e-12):.2%}\n"
        f"Expected: J→0 in hole"
    )
    axes[0].text(
        0.03, 0.03, text, transform=axes[0].transAxes,
        fontsize=8, fontfamily="monospace",
        verticalalignment="bottom", horizontalalignment="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9),
    )

    fig.suptitle("Supercurrent Density — Diverted Around Hole", fontsize=14, y=1.02)
    fig.tight_layout()

    out = output_dir / "supercurrent_hole.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [out]


if __name__ == "__main__":
    main()

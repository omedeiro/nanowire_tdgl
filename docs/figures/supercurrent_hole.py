"""Supercurrent around a hole: screening currents diverted by a non-SC void.

Demonstrates that supercurrent J_s ∝ |ψ|² vanishes inside non-SC holes and
flows around them, consistent with zero-current boundary conditions.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
from tdgl3d import AppliedField, Device, Layer, SimulationParameters, Trilayer, solve
from tdgl3d.visualization.plotting import plot_current_density


def main(output_dir: Path = Path(__file__).parent, small: bool = False) -> list[Path]:
    if small:
        Nx, Ny = 8, 8
        t_stop = 2.0
        hole_lo, hole_hi = 3, 5
    else:
        Nx, Ny = 24, 24
        t_stop = 10.0
        hole_lo, hole_hi = 9, 15

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

    hole_polygon = [(float(hole_lo), float(hole_lo)),
                    (float(hole_hi), float(hole_lo)),
                    (float(hole_hi), float(hole_hi)),
                    (float(hole_lo), float(hole_hi))]

    fig, axes = plot_current_density(sol, step=-1, slice_z=0, streamplot=True,
                                     hole_polygon=hole_polygon, figsize=(18, 5))
    fig.suptitle("Supercurrent Density — Diverted Around Hole", fontsize=14, y=1.02)
    fig.tight_layout()

    out = output_dir / "supercurrent_hole.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [out]


if __name__ == "__main__":
    main()

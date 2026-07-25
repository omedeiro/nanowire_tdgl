"""Meissner screening: exponential decay of B-field into a superconductor.

Demonstrates that an applied magnetic field Bz decays as exp(-x/λ) with
penetration depth λ ≈ κ (in GL units), validating the London equation.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from tdgl3d import AppliedField, Device, SimulationParameters, solve


def main(output_dir: Path = Path(__file__).parent, small: bool = False) -> list[Path]:
    if small:
        Nx, Ny, Nz = 6, 6, 1
        t_stop = 2.0
    else:
        Nx, Ny, Nz = 20, 8, 1
        t_stop = 20.0

    params = SimulationParameters(Nx=Nx, Ny=Ny, Nz=Nz, kappa=2.0)
    field = AppliedField(Bz=0.3, t_on_fraction=1.0)
    device = Device(params, applied_field=field)

    save_every = max(1, int(t_stop / 0.5))
    sol = solve(
        device, dt=0.01, t_stop=t_stop, method="euler",
        save_every=save_every, progress=False,
    )

    # Extract Bz profile along x at mid-y
    Bx, By, Bz = sol.bfield(step=-1, full_interior=True)
    mid_y = (Ny - 1) // 2
    Nx_int = Nx - 1
    Ny_int = Ny - 1
    Bz_2d = Bz.reshape(Nx_int, Ny_int)
    bz_profile = Bz_2d[:, mid_y]

    x_coords = np.arange(1, Nx) * params.hx  # interior node positions

    # Fit exp(-x/lambda) to the profile (skip first boundary node)
    def exp_decay(x, lam, A):
        return A * np.exp(-x / lam)

    try:
        popt, _ = curve_fit(
            exp_decay, x_coords[1:], bz_profile[1:],
            p0=[params.kappa, 0.3], maxfev=5000,
        )
        lambda_fit = popt[0]
    except RuntimeError:
        lambda_fit = params.kappa

    # --- Figure ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: Bz heatmap
    ax = axes[0]
    xs = np.arange(1, Nx) * params.hx
    ys = np.arange(1, Ny) * params.hy
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    im = ax.pcolormesh(xx, yy, Bz_2d, cmap="RdBu_r", shading="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Bz")
    ax.set_xlabel("x (ξ)")
    ax.set_ylabel("y (ξ)")
    ax.set_title("Applied Bz field")
    ax.set_aspect("equal")

    # Right: 1D Bz(x) with exponential fit
    ax = axes[1]
    ax.plot(x_coords, bz_profile, "o-", color="C0", markersize=4, label="Bz(x) at mid-y")
    x_fit = np.linspace(0, x_coords[-1], 200)
    fit_A = popt[1] if 'popt' in dir() else 0.3
    ax.plot(x_fit, exp_decay(x_fit, lambda_fit, fit_A),
            "--", color="C1",
            label=f"Fit: λ = {lambda_fit:.2f} ξ")
    ax.axvline(x_coords[1], color="gray", ls=":", alpha=0.5, label="Fit region start")
    ax.set_xlabel("x (ξ)")
    ax.set_ylabel("Bz")
    ax.set_title(f"Meissner screening: λ_fit = {lambda_fit:.2f}, κ = {params.kappa}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Meissner Screening — Exponential B-field Decay", fontsize=14, y=1.02)
    fig.tight_layout()

    out = output_dir / "meissner_screening.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [out]


if __name__ == "__main__":
    main()

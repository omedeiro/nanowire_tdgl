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
        Nx, Ny, Nz = 12, 12, 1
        t_stop = 2.0
    else:
        Nx, Ny, Nz = 40, 16, 1
        t_stop = 20.0

    kappa = 2.0
    params = SimulationParameters(Nx=Nx, Ny=Ny, Nz=Nz, kappa=kappa)
    Bz_applied = 0.1
    field = AppliedField(Bz=Bz_applied, t_on_fraction=1.0)
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
    bz_profile = np.real(Bz_2d[:, mid_y])

    x_coords = np.arange(1, Nx) * params.hx  # interior node positions

    # Fit cosh model (matches test_physics_validation.py)
    L = (Nx_int) * params.hx
    x_center = L / 2.0
    n_fit = Nx_int - 1  # exclude last interior node (boundary leak)

    def cosh_model(x, A, lam, x0):
        return A * np.cosh((x - x0) / lam)

    lambda_fit = None
    fit_converged = False
    try:
        popt, pcov = curve_fit(
            cosh_model, x_coords[:n_fit], bz_profile[:n_fit],
            p0=[bz_profile[n_fit // 2], kappa, x_center],
            maxfev=10000,
        )
        lambda_fit = abs(popt[1])
        fit_converged = True
        perr = np.sqrt(np.diag(pcov))
        A_fit, lam_fit, x0_fit = popt
    except (RuntimeError, ValueError):
        popt = np.array([bz_profile[n_fit // 2], kappa, x_center])
        perr = np.array([0, 0, 0])

    # R² for fit quality
    if fit_converged:
        residuals = bz_profile[:n_fit] - cosh_model(x_coords[:n_fit], *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((bz_profile[:n_fit] - np.mean(bz_profile[:n_fit]))**2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    else:
        r_squared = 0.0

    # Symmetry: compare left half vs right half about center
    mid_idx = Nx_int // 2
    left_half = bz_profile[:mid_idx]
    right_half = bz_profile[Nx_int - mid_idx:Nx_int][::-1]  # reversed to align
    min_len = min(len(left_half), len(right_half))
    if min_len > 0:
        symmetry_error = float(np.max(np.abs(left_half[:min_len] - right_half[:min_len])))
        symmetry_relative = symmetry_error / max(float(np.max(bz_profile)), 1e-12)
    else:
        symmetry_error = 0.0
        symmetry_relative = 0.0

    # --- Figure ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: Bz heatmap
    ax = axes[0]
    xs = np.arange(1, Nx) * params.hx
    ys = np.arange(1, Ny) * params.hy
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    im = ax.pcolormesh(xx, yy, np.real(Bz_2d), cmap="RdBu_r", shading="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Bz")
    # Mark center line used for profile
    ax.axhline(ys[mid_y], color="white", ls="--", alpha=0.5, linewidth=1)
    ax.set_xlabel("x (ξ)")
    ax.set_ylabel("y (ξ)")
    ax.set_title("Applied Bz field")
    ax.set_aspect("equal")

    # Right: 1D Bz(x) with cosh fit + expected + symmetry
    ax = axes[1]
    ax.plot(x_coords, bz_profile, "o-", color="C0", markersize=3, label="Bz(x) at mid-y")

    # Fitted cosh curve
    if fit_converged:
        x_fit_line = np.linspace(0, x_coords[-1], 200)
        ax.plot(
            x_fit_line, cosh_model(x_fit_line, *popt),
            "--", color="C1", linewidth=1.5,
            label=f"Cosh fit: λ = {lambda_fit:.2f}",
        )

    # Expected symmetric profile (mirror left half to right)
    x_sym = x_coords[:n_fit]
    bz_sym = 0.5 * (bz_profile[:n_fit] + bz_profile[:n_fit][::-1])
    ax.plot(x_sym, bz_sym, ":", color="C2", linewidth=1.5,
            label="Symmetrized profile")

    # Reference line at Bz_applied
    ax.axhline(Bz_applied, color="gray", ls=":", alpha=0.5, label=f"Bz_applied = {Bz_applied}")

    # Annotation box
    text = (
        f"κ (set)    = {kappa:.2f}\n"
        f"λ (fit)     = {lambda_fit:.2f}\n"
        f"R²          = {r_squared:.4f}\n"
        f"Fit error   = {perr[1]:.3f}\n"
        f"Sym. error  = {symmetry_relative:.2%}"
    )
    ax.text(
        0.97, 0.97, text, transform=ax.transAxes,
        fontsize=8, fontfamily="monospace",
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9),
    )

    ax.set_xlabel("x (ξ)")
    ax.set_ylabel("Bz")
    ax.set_title(f"Meissner screening: λ_fit = {lambda_fit:.2f}, κ = {kappa}")
    ax.legend(loc="lower center", fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Meissner Screening — Exponential B-field Decay", fontsize=14, y=1.02)
    fig.tight_layout()

    out = output_dir / "meissner_screening.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [out]


if __name__ == "__main__":
    main()

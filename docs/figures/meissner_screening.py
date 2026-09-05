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

# ``Device.initial_state`` seeds ψ with 1% complex noise drawn from a
# non-deterministic RNG unless a seed is given, so an unseeded figure is a
# different realisation every time it is regenerated and cannot be compared
# against the one committed to the gallery.  Pin it.
NOISE_SEED = 7


def main(output_dir: Path = Path(__file__).parent, small: bool = False) -> list[Path]:
    kappa = 2.0
    # The grid must resolve λ = κ and the sample must be several λ across in
    # *both* in-plane directions.  At h = 1 there are only two cells per
    # penetration depth and the measured decay length is set by the stencil
    # rather than by the physics; h = κ/4 and L = 8κ recover λ ≈ κ.
    h = kappa / 4.0
    if small:
        Nx, Ny, Nz = 12, 12, 1
        t_stop = 2.0
    else:
        Nx = Ny = int(round(8.0 * kappa / h))
        Nz = 1
        t_stop = 12.0

    params = SimulationParameters(Nx=Nx, Ny=Ny, Nz=Nz, hx=h, hy=h, kappa=kappa)
    Bz_applied = 0.1
    field = AppliedField(Bz=Bz_applied, t_on_fraction=1.0)
    device = Device(params, applied_field=field)

    dt = 0.8 * h**2 / (4.0 * kappa**2)
    save_every = max(1, int(t_stop / dt / 20))
    sol = solve(
        device, dt=dt, t_stop=t_stop, method="euler",
        save_every=save_every, progress=False, noise_seed=NOISE_SEED,
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
        popt, _ = curve_fit(
            cosh_model, x_coords[:n_fit], bz_profile[:n_fit],
            p0=[bz_profile[n_fit // 2], kappa, x_center],
            maxfev=10000,
        )
        lambda_fit = abs(popt[1])
        fit_converged = True
    except (RuntimeError, ValueError):
        popt = np.array([bz_profile[n_fit // 2], kappa, x_center])

    # R² for fit quality
    if fit_converged:
        residuals = bz_profile[:n_fit] - cosh_model(x_coords[:n_fit], *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((bz_profile[:n_fit] - np.mean(bz_profile[:n_fit]))**2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    else:
        r_squared = 0.0

    # Direct measurement of the decay length from the edge: fit ln Bz over the
    # first two penetration depths.  This is the quantity the London equation
    # predicts (λ = κ); the cosh fit over the whole width also absorbs the
    # transverse screening of a square sample and reads a few percent high.
    edge_window = max(4, int(round(2.0 * kappa / params.hx)))
    edge_window = min(edge_window, n_fit)
    lambda_edge = -1.0 / np.polyfit(
        x_coords[:edge_window] - x_coords[0],
        np.log(np.abs(bz_profile[:edge_window])),
        1,
    )[0]
    lambda_edge_error = abs(lambda_edge - kappa) / kappa

    # Symmetry about the mid-plane.  Bz lives on plaquettes anchored at the
    # interior nodes 1…Nx-1, but the plaquette anchored at Nx-1 is the high-side
    # *boundary* plaquette whose mirror image (anchor 0) sits on the ghost ring
    # and is not part of the array — so it is dropped before reflecting.
    bz_mirrorable = bz_profile[:-1]
    symmetry_error = float(np.max(np.abs(bz_mirrorable - bz_mirrorable[::-1])))
    symmetry_relative = symmetry_error / max(float(np.max(bz_profile)), 1e-12)

    # --- Figure ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]

    # Left: Bz heatmap, on the same mirrorable plaquette block as the symmetry
    # check above.  Two things have to be right or a perfectly symmetric field
    # draws lopsided:
    #
    #   * the anchor-(N-1) row and column are the *pinned* boundary ring, and
    #     their mirror images — the ghost anchors at 0 — are not in the array.
    #     Drawing them puts the applied-field frame on the high sides only.
    #   * the plaquette anchored at node i is centred at (i + ½)h, not at i·h.
    #     Placing it on the node displaces the whole map by half a cell.
    bz_map = np.real(Bz_2d[:-1, :-1])
    xs = (np.arange(1, Nx - 1) + 0.5) * params.hx
    ys = (np.arange(1, Ny - 1) + 0.5) * params.hy
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    im = ax.pcolormesh(xx, yy, bz_map, cmap="RdBu_r", shading="auto")
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
    ax.plot(x_coords[:-1], 0.5 * (bz_mirrorable + bz_mirrorable[::-1]),
            ":", color="C2", linewidth=1.5, label="Symmetrized profile")

    # Reference line at Bz_applied
    ax.axhline(Bz_applied, color="gray", ls=":", alpha=0.5, label=f"Bz_applied = {Bz_applied}")

    # Annotation box
    text = (
        f"κ (set)      = {kappa:.2f}\n"
        f"λ (edge fit)  = {lambda_edge:.2f}  ({lambda_edge_error:.1%} vs κ)\n"
        f"λ (cosh fit)  = {lambda_fit:.2f}\n"
        f"R² (cosh)     = {r_squared:.4f}\n"
        f"Sym. error    = {symmetry_relative:.2%}"
    )
    ax.text(
        0.97, 0.97, text, transform=ax.transAxes,
        fontsize=8, fontfamily="monospace",
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9),
    )

    ax.set_xlabel("x (ξ)")
    ax.set_ylabel("Bz")
    ax.set_title(f"Meissner screening: λ = {lambda_edge:.2f} ξ, κ = {kappa}")
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

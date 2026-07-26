"""Insulator psi decay: exponential relaxation of order parameter in non-SC regions.

Demonstrates that in non-superconducting insulator layers, |ψ| decays
exponentially with time constant τ = τ_relax ≈ 0.1 (the built-in suppression timescale).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from tdgl3d import AppliedField, Device, Layer, SimulationParameters, Trilayer, solve


def main(output_dir: Path = Path(__file__).parent, small: bool = False) -> list[Path]:
    if small:
        Nx, Ny = 8, 8
        t_stop = 0.5
    else:
        Nx, Ny = 16, 16
        t_stop = 2.0

    kappa = 2.0
    trilayer = Trilayer(
        bottom=Layer(thickness_z=2, kappa=kappa),
        insulator=Layer(thickness_z=2, kappa=0.0, is_superconductor=False),
        top=Layer(thickness_z=2, kappa=kappa),
    )
    params = SimulationParameters(Nx=Nx, Ny=Ny, Nz=trilayer.Nz, kappa=kappa)
    field = AppliedField(Bz=0.0)
    device = Device(params, applied_field=field, trilayer=trilayer)

    # Uniform initial state (including insulator)
    x0 = device.initial_state()
    x0.psi[:] = 1.0  # Force uniform |psi|=1 everywhere

    sol = solve(device, x0=x0, dt=0.01, t_stop=t_stop, method="euler",
                save_every=max(1, int(t_stop / 0.05)), progress=False)

    # Extract mean |psi| in insulator over time
    z_ranges = trilayer.z_ranges()
    ins_start, ins_end = z_ranges["insulator"]
    Nz_int = max(trilayer.Nz - 1, 1)

    psi_ins_mean = []
    for step in range(sol.n_steps):
        psi = sol.psi(step=step)
        psi_3d = psi.reshape(Nx - 1, Ny - 1, Nz_int)
        ins_z = ins_start  # first insulator z-index
        if ins_z < Nz_int:
            psi_ins_mean.append(np.mean(np.abs(psi_3d[:, :, ins_z]) ** 2))
        else:
            psi_ins_mean.append(0.0)

    psi_ins_mean = np.array(psi_ins_mean)
    times = sol.times

    # Fit exponential decay
    def exp_decay(t, tau, A, C):
        return A * np.exp(-t / tau) + C

    tau_fit = 0.1
    popt = None
    try:
        popt, _ = curve_fit(exp_decay, times, psi_ins_mean, p0=[0.1, 0.9, 0.0], maxfev=5000)
        tau_fit = abs(popt[0])
    except RuntimeError:
        pass

    # --- Expected values ---
    # Insulator: |ψ|² → 0 (no superconductivity)
    # SC layers: |ψ|² → 1
    # Decay time: τ_fit should be on order of 1/κ = 0.5 (but solver uses τ_relax)
    expected_tau = 1.0 / kappa

    # Symmetry: bottom SC vs top SC in final z-profile
    psi_final = sol.psi(step=-1)
    psi_3d = psi_final.reshape(Nx - 1, Ny - 1, Nz_int)
    mid_x, mid_y = (Nx - 1) // 2, (Ny - 1) // 2
    psi_z = np.abs(psi_3d[mid_x, mid_y, :]) ** 2

    iz_bot_start = max(z_ranges["bottom"][0], 1) - 1
    iz_bot_end = min(z_ranges["bottom"][1], trilayer.Nz - 1)
    iz_top_start = max(z_ranges["top"][0], 1) - 1
    iz_top_end = min(z_ranges["top"][1], trilayer.Nz - 1)

    psi_bot_mean = float(np.mean(psi_z[iz_bot_start:iz_bot_end]))
    psi_top_mean = float(np.mean(psi_z[iz_top_start:iz_top_end]))
    psi_sym_error = abs(psi_bot_mean - psi_top_mean)
    psi_sym_error /= max(abs(psi_bot_mean + psi_top_mean) / 2, 1e-12)

    # --- Figure ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: |psi_insulator|(t) with fit + expected
    ax = axes[0]
    ax.plot(times, psi_ins_mean, "o-", color="C0", markersize=3, label="|ψ|²_insulator(t)")

    # Expected: |ψ|² → 0
    ax.axhline(0.0, color="gray", ls=":", alpha=0.5, label="Expected: |ψ|² → 0")

    t_fit = np.linspace(0, times[-1], 200)
    if popt is not None:
        ax.plot(t_fit, exp_decay(t_fit, *popt), "--", color="C1",
                label=f"Fit: τ = {tau_fit:.3f}")

    # Error annotation
    tau_error = abs(tau_fit - expected_tau) / expected_tau
    text = (
        f"τ (fit):     {tau_fit:.4f}\n"
        f"τ (expected): ~{expected_tau:.2f} (1/κ)\n"
        f"τ error:     {tau_error:.1%}\n"
        f"|ψ|²_final:  {psi_ins_mean[-1]:.6f}"
    )
    ax.text(
        0.97, 0.97, text, transform=ax.transAxes,
        fontsize=8, fontfamily="monospace",
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9),
    )

    ax.set_xlabel("t (ξ/v_F)")
    ax.set_ylabel("mean |ψ|² in insulator")
    ax.set_title(f"Insulator relaxation — τ_fit = {tau_fit:.3f}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: |psi| z-profile at final time + expected + symmetry
    ax = axes[1]
    z_coords = np.arange(1, trilayer.Nz) * params.hz

    # Color by layer
    ins_z_start = max(z_ranges["insulator"][0], 1) - 1
    ins_z_end = min(z_ranges["insulator"][1], trilayer.Nz - 1)
    bar_colors = []
    for i in range(Nz_int):
        if i < ins_z_start:
            bar_colors.append("C0")
        elif i < ins_z_end:
            bar_colors.append("C3")
        else:
            bar_colors.append("C0")

    ax.bar(z_coords, psi_z, width=params.hz * 0.8, color=bar_colors)

    # Expected reference lines
    ax.axhline(1.0, color="C0", ls=":", alpha=0.4, linewidth=1.5, label="Expected SC: |ψ|² → 1")
    ax.axhline(0.0, color="C3", ls=":", alpha=0.4, linewidth=1.5, label="Expected ins.: |ψ|² → 0")

    # Shade layers
    z_ranges_dict = trilayer.z_ranges()
    for layer_name, color in [("bottom", "blue"), ("insulator", "red"), ("top", "blue")]:
        lr = z_ranges_dict[layer_name]
        ax.axvspan(lr[0] * params.hz, lr[1] * params.hz, alpha=0.1, color=color)

    # Symmetry annotation
    psi_ins_mean_final = float(np.mean(psi_z[ins_z_start:ins_z_end]))
    psi_sc_err = abs(psi_bot_mean - 1.0)
    psi_ins_err = abs(psi_ins_mean_final - 0.0)
    text2 = (
        f"SC(bottom): {psi_bot_mean:.4f}\n"
        f"SC(top):    {psi_top_mean:.4f}\n"
        f"Insulator:  {psi_ins_mean_final:.4f}\n"
        f"SC error:   {psi_sc_err:.4f}\n"
        f"Ins error:  {psi_ins_err:.4f}\n"
        f"Sym error:  {psi_sym_error:.2%}"
    )
    ax.text(
        0.97, 0.97, text2, transform=ax.transAxes,
        fontsize=8, fontfamily="monospace",
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9),
    )

    ax.set_xlabel("z (ξ)")
    ax.set_ylabel("|ψ|²")
    ax.set_title("z-profile at final time")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Insulator Order Parameter Decay — Exponential Relaxation", fontsize=14, y=1.02)
    fig.tight_layout()

    out = output_dir / "insulator_psi_decay.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [out]


if __name__ == "__main__":
    main()

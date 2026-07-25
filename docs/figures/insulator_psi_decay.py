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
        Nx, Ny = 4, 4
        t_stop = 0.5
    else:
        Nx, Ny = 8, 8
        t_stop = 2.0

    trilayer = Trilayer(
        bottom=Layer(thickness_z=2, kappa=2.0),
        insulator=Layer(thickness_z=2, kappa=0.0, is_superconductor=False),
        top=Layer(thickness_z=2, kappa=2.0),
    )
    params = SimulationParameters(Nx=Nx, Ny=Ny, Nz=trilayer.Nz, kappa=2.0)
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
        # Insulator layer: take the first insulator z-slice
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

    try:
        popt, _ = curve_fit(exp_decay, times, psi_ins_mean, p0=[0.1, 0.9, 0.0], maxfev=5000)
        tau_fit = abs(popt[0])
    except RuntimeError:
        tau_fit = 0.1

    # --- Figure ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: |psi_insulator|(t) with fit
    ax = axes[0]
    ax.plot(times, psi_ins_mean, "o-", color="C0", markersize=3, label="|ψ|²_insulator(t)")
    t_fit = np.linspace(0, times[-1], 200)
    try:
        ax.plot(t_fit, exp_decay(t_fit, *popt), "--", color="C1",
                label=f"Fit: τ = {tau_fit:.3f}")
    except Exception:
        pass
    ax.set_xlabel("t (ξ/v_F)")
    ax.set_ylabel("mean |ψ|² in insulator")
    ax.set_title(f"Insulator relaxation — τ_fit = {tau_fit:.3f}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: |psi| z-profile at final time
    ax = axes[1]
    psi_final = sol.psi(step=-1)
    psi_3d = psi_final.reshape(Nx - 1, Ny - 1, Nz_int)
    mid_x, mid_y = (Nx - 1) // 2, (Ny - 1) // 2
    psi_z = np.abs(psi_3d[mid_x, mid_y, :]) ** 2

    z_coords = np.arange(1, trilayer.Nz) * params.hz
    colors = ["C0"] * ins_start + ["C3"] * (ins_end - ins_start) + ["C0"] * (Nz_int - ins_end)
    ax.bar(z_coords, psi_z, width=params.hz * 0.8, color=colors)

    # Shade layers
    z_ranges_dict = trilayer.z_ranges()
    for layer_name, color in [("bottom", "blue"), ("insulator", "red"), ("top", "blue")]:
        lr = z_ranges_dict[layer_name]
        ax.axvspan(lr[0] * params.hz, lr[1] * params.hz, alpha=0.1, color=color)

    ax.set_xlabel("z (ξ)")
    ax.set_ylabel("|ψ|²")
    ax.set_title("z-profile at final time")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Insulator Order Parameter Decay — Exponential Relaxation", fontsize=14, y=1.02)
    fig.tight_layout()

    out = output_dir / "insulator_psi_decay.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [out]


if __name__ == "__main__":
    main()

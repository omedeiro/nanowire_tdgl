"""Energy dissipation: monotonic decrease of Ginzburg-Landau free energy.

Demonstrates that the TDGL equation is a gradient flow, so the GL free energy
F = ∫(|ψ|²/2 + |∇ψ|²/2 + κ²B²/2) is a Lyapunov functional that decreases monotonically.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tdgl3d import AppliedField, Device, SimulationParameters, solve


def main(output_dir: Path = Path(__file__).parent, small: bool = False) -> list[Path]:
    if small:
        Nx, Ny, Nz = 12, 12, 1
        n_steps = 60
    else:
        Nx, Ny, Nz = 24, 24, 1
        n_steps = 400

    params = SimulationParameters(Nx=Nx, Ny=Ny, Nz=Nz, kappa=2.0)
    field = AppliedField(Bz=0.5, t_on_fraction=1.0)
    device = Device(params, applied_field=field)

    dt = 0.01
    t_stop = n_steps * dt
    sol = solve(device, dt=dt, t_stop=t_stop, method="euler", save_every=1, progress=False)

    # Compute energy at each saved step
    energies = []
    n = params.n_interior

    for step in range(sol.n_steps):
        state = sol.states[:, step]
        psi = state[:n]
        phi_x = state[n:2*n]
        phi_y = state[2*n:3*n]

        psi_sq = np.abs(psi) ** 2
        dpsi_dx = psi * np.conj(phi_x) - psi
        dpsi_dy = psi * np.conj(phi_y) - psi
        grad_energy = 0.5 * (np.mean(np.abs(dpsi_dx)**2) + np.mean(np.abs(dpsi_dy)**2))
        cond_energy = 0.5 * np.mean((1.0 - psi_sq)**2)
        E = cond_energy + params.kappa**2 * grad_energy
        energies.append(E)

    energies = np.array(energies)
    times = sol.times

    # dF/dt
    dFdt = np.diff(energies) / np.diff(times)

    # Expected: F decreases monotonically → dF/dt ≤ 0 everywhere
    # Count violations and max increase
    violations = dFdt > 0
    n_violations = int(np.sum(violations))
    max_increase = float(np.max(dFdt[violations])) if n_violations > 0 else 0.0
    total_decrease = float(energies[0] - energies[-1])

    # --- Figure ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: F(t) monotonic decrease
    ax = axes[0]
    ax.plot(times, energies, "o-", color="C0", markersize=2, label="F(t)")
    # Expected: F should decrease (shade regions where it increases)
    for i in range(1, len(energies)):
        if energies[i] > energies[i-1]:
            ax.axvspan(times[i-1], times[i], alpha=0.3, color="red", zorder=0)
    ax.set_xlabel("t (ξ/v_F)")
    ax.set_ylabel("GL free energy density")
    ax.set_title("Free energy decreases monotonically")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: dF/dt ≤ 0 with violation annotation
    ax = axes[1]
    ax.plot(times[1:], dFdt, "o-", color="C3", markersize=2)
    ax.axhline(0, color="gray", ls="--", alpha=0.5, label="dF/dt = 0 (expected bound)")
    # Shade violations
    if n_violations > 0:
        ax.fill_between(times[1:], 0, dFdt, where=violations, alpha=0.3, color="red",
                        label=f"{n_violations} violation(s)")

    # Annotation box
    text = (
        f"Total ΔF:     {total_decrease:.4f}\n"
        f"Violations:   {n_violations}/{len(dFdt)}\n"
        f"Max increase: {max_increase:.2e}\n"
        f"Monotonic:    {'Yes' if n_violations == 0 else 'No'}"
    )
    ax.text(
        0.97, 0.97, text, transform=ax.transAxes,
        fontsize=8, fontfamily="monospace",
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9),
    )

    ax.set_xlabel("t (ξ/v_F)")
    ax.set_ylabel("dF/dt")
    ax.set_title("Energy dissipation rate (≤ 0)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("GL Free Energy Dissipation — TDGL as Gradient Flow", fontsize=14, y=1.02)
    fig.tight_layout()

    out = output_dir / "energy_dissipation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [out]


if __name__ == "__main__":
    main()

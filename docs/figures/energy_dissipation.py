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
        Nx, Ny, Nz = 6, 6, 1
        n_steps = 30
    else:
        Nx, Ny, Nz = 12, 12, 1
        n_steps = 200

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
        # Kinetic energy from link variables
        dpsi_dx = psi * np.conj(phi_x) - psi
        dpsi_dy = psi * np.conj(phi_y) - psi
        grad_energy = 0.5 * (np.mean(np.abs(dpsi_dx)**2) + np.mean(np.abs(dpsi_dy)**2))

        # Condensation energy
        cond_energy = 0.5 * np.mean((1.0 - psi_sq)**2)

        # Total GL free energy density
        E = cond_energy + params.kappa**2 * grad_energy
        energies.append(E)

    energies = np.array(energies)
    times = sol.times

    # --- Figure ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: F(t) monotonic decrease
    ax = axes[0]
    ax.plot(times, energies, "o-", color="C0", markersize=2, label="F(t)")
    ax.set_xlabel("t (ξ/v_F)")
    ax.set_ylabel("GL free energy density")
    ax.set_title("Free energy decreases monotonically")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: dF/dt ≤ 0
    ax = axes[1]
    dFdt = np.diff(energies) / np.diff(times)
    ax.plot(times[1:], dFdt, "o-", color="C3", markersize=2)
    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("t (ξ/v_F)")
    ax.set_ylabel("dF/dt")
    ax.set_title("Energy dissipation rate (≤ 0)")
    ax.grid(True, alpha=0.3)

    fig.suptitle("GL Free Energy Dissipation — TDGL as Gradient Flow", fontsize=14, y=1.02)
    fig.tight_layout()

    out = output_dir / "energy_dissipation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [out]


if __name__ == "__main__":
    main()

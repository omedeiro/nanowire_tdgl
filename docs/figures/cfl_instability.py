"""CFL instability: catastrophic numerical collapse above the stability limit.

Demonstrates that the Forward Euler method requires dt < h²/(4κ²) for stability,
and exceeding this limit causes the order parameter to collapse.
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
    else:
        Nx, Ny, Nz = 10, 10, 1

    params = SimulationParameters(Nx=Nx, Ny=Ny, Nz=Nz, kappa=2.0)
    h = params.hx
    cfl_limit = h**2 / (4.0 * params.kappa**2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, (label, dt_factor, color) in enumerate([
        ("Stable (dt = 0.9 × CFL)", 0.9, "C0"),
        ("Unstable (dt = 3.0 × CFL)", 3.0, "C3"),
    ]):
        ax = axes[idx]
        dt = dt_factor * cfl_limit
        field = AppliedField(Bz=0.5, t_on_fraction=1.0)
        device = Device(SimulationParameters(Nx=Nx, Ny=Ny, Nz=Nz, kappa=2.0), applied_field=field)

        try:
            sol = solve(device, dt=dt, t_stop=2.0, method="euler", save_every=1, progress=False)
            psi2_t = []
            for step in range(sol.n_steps):
                psi2_t.append(np.mean(sol.psi_squared(step=step)))
            times = sol.times
        except Exception:
            times = np.array([0.0])
            psi2_t = np.array([0.0])

        psi2_t = np.array(psi2_t)
        ax.plot(times, psi2_t, "-", color=color, linewidth=1.5)
        ax.axhline(1.0, color="gray", ls=":", alpha=0.5, label="|ψ|² = 1")
        ax.set_xlabel("t (ξ/v_F)")
        ax.set_ylabel("mean |ψ|²")
        ax.set_title(f"{label}\ndt = {dt:.4f}, CFL = {cfl_limit:.4f}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.5)

    fig.suptitle("CFL Stability — dt < h²/(4κ²) required", fontsize=14, y=1.02)
    fig.tight_layout()

    out = output_dir / "cfl_instability.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [out]


if __name__ == "__main__":
    main()

"""Generate a default example plot: 2-D superconductor under applied Bz.

Runs a short Forward-Euler simulation and produces a summary PNG with
|ψ|² and B_z side by side.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import tdgl3d
from tdgl3d.visualization.plotting import plot_summary, plot_order_parameter

def main():
    # ── Parameters ──────────────────────────────────────────────────────
    # kappa > 1/sqrt(2) ≈ 0.71 → Type-II superconductor.
    # For Forward Euler stability: dt ≲ h² / (4 κ²).
    # With h=1, κ=2: dt < 1/16 ≈ 0.06, so dt=0.005 is safe.
    params = tdgl3d.SimulationParameters(
        Nx=20,
        Ny=20,
        Nz=1,
        hx=1.0,
        hy=1.0,
        hz=1.0,
        kappa=2.0,
    )

    field = tdgl3d.AppliedField(Bz=1.0, t_on_fraction=0.6)
    device = tdgl3d.Device(params, applied_field=field)

    print(device)
    print(f"  Interior nodes : {params.n_interior}")
    print(f"  State-vector   : {params.n_state}")

    # ── Solve ───────────────────────────────────────────────────────────
    # Forward Euler CFL: dt < h² / (4*κ²) = 1/16 ≈ 0.06 for these params.
    print("\nRunning Forward-Euler (dt=0.005, t_stop=20) …")
    solution = tdgl3d.solve(
        device,
        t_start=0.0,
        t_stop=20.0,
        dt=0.005,
        method="euler",
        save_every=50,
        progress=True,
    )
    print(f"  Saved {solution.n_steps} snapshots over "
          f"[{solution.times[0]:.2f}, {solution.times[-1]:.2f}]")

    psi2 = solution.psi_squared(step=-1)
    print(f"  Final |ψ|²: mean={np.mean(psi2):.4f}  min={np.min(psi2):.4f}  "
          f"max={np.max(psi2):.4f}")

    Bx, By, Bz = solution.bfield(step=-1)
    print(f"  Final B_z:  mean={np.mean(Bz):.4f}  max|B_z|={np.max(np.abs(Bz)):.4f}")

    # ── Plot ────────────────────────────────────────────────────────────
    fig = plot_summary(solution, step=-1)
    fig.savefig("default_example.png", dpi=150, bbox_inches="tight")
    print("\n✓ Saved  default_example.png")

    # Also plot the mid-simulation snapshot
    mid = solution.n_steps // 2
    fig2 = plot_summary(solution, step=mid)
    fig2.savefig("default_example_mid.png", dpi=150, bbox_inches="tight")
    print(f"✓ Saved  default_example_mid.png  (step {mid}, "
          f"t={solution.times[mid]:.2f})")


if __name__ == "__main__":
    main()

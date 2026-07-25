"""Example: simulate vortex entry in a 2-D superconductor under applied Bz.

This script demonstrates the full workflow:
1. Define simulation parameters
2. Create a device with an applied magnetic field
3. Run the simulation using Forward Euler
4. Visualize the results
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for saving figures

import tdgl3d


def main():
    # 1. Define parameters
    params = tdgl3d.SimulationParameters(
        Nx=15,
        Ny=15,
        Nz=1,      # 2-D simulation
        hx=1.0,
        hy=1.0,
        hz=1.0,
        kappa=5.0,  # Type-II regime (κ > 1/√2)
    )

    # 2. Applied magnetic field: Bz on for first 2/3 of simulation
    field = tdgl3d.AppliedField(Bz=0.5, t_on_fraction=2.0 / 3.0)

    # 3. Create device
    device = tdgl3d.Device(params, applied_field=field)
    print(device)
    print(f"State vector size: {params.n_state}")
    print(f"Interior nodes:    {params.n_interior}")

    # 4. Run simulation
    print("\nRunning Forward Euler simulation...")
    solution = tdgl3d.solve(
        device,
        t_start=0.0,
        t_stop=5.0,
        dt=0.01,
        method="euler",
        save_every=10,
        progress=True,
    )

    print(f"\nSaved {solution.n_steps} time steps.")
    print(f"Time range: [{solution.times[0]:.3f}, {solution.times[-1]:.3f}]")

    # 5. Post-processing
    psi2_init = solution.psi_squared(step=0)
    psi2_final = solution.psi_squared(step=-1)
    print(f"|ψ|² initial: mean={np.mean(psi2_init):.4f}, min={np.min(psi2_init):.4f}")
    print(f"|ψ|² final:   mean={np.mean(psi2_final):.4f}, min={np.min(psi2_final):.4f}")

    Bx, By, Bz = solution.bfield(step=-1)
    print(f"B_z final: mean={np.mean(Bz):.4f}, max={np.max(np.abs(Bz)):.4f}")

    # 6. Visualization
    from tdgl3d.visualization.plotting import plot_summary, animate

    fig = plot_summary(solution, step=-1)
    fig.savefig("example_summary.png", dpi=150, bbox_inches="tight")
    print("\nSaved example_summary.png")

    animate(solution, "example_vortices.gif", fps=10, step_stride=2)
    print("Saved example_vortices.gif")


if __name__ == "__main__":
    main()

"""Example: 3-D vortex simulation with applied field along z.

Demonstrates a small 3-D simulation and slice-based visualization.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")

import tdgl3d


def main():
    params = tdgl3d.SimulationParameters(
        Nx=6, Ny=6, Nz=6,
        hx=1.0, hy=1.0, hz=1.0,
        kappa=3.0,
    )

    field = tdgl3d.AppliedField(Bz=1.0, t_on_fraction=0.5)
    device = tdgl3d.Device(params, applied_field=field)
    print(device)
    print(f"State vector size: {params.n_state}")

    print("\nRunning 3-D Forward Euler...")
    solution = tdgl3d.solve(
        device,
        t_start=0.0,
        t_stop=1.0,
        dt=0.005,
        method="euler",
        save_every=20,
        progress=True,
    )

    print(f"Saved {solution.n_steps} steps.")

    # Visualize middle z-slice
    from tdgl3d.visualization.plotting import plot_order_parameter, plot_bfield
    import matplotlib.pyplot as plt

    middle_z = max(0, (params.Nz - 1) // 2 - 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_order_parameter(solution, step=-1, slice_z=middle_z, ax=axes[0])
    plot_bfield(solution, component="z", step=-1, slice_z=middle_z, ax=axes[1])
    fig.suptitle(f"3-D simulation, z-slice={middle_z}")
    fig.tight_layout()
    fig.savefig("example_3d_slices.png", dpi=150, bbox_inches="tight")
    print("Saved example_3d_slices.png")


if __name__ == "__main__":
    main()

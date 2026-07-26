"""Vortex entry dynamics: animated GIF showing vortex nucleation over time.

Three-panel animation:
  - Top-left:  |ψ|² heatmap with vortex core markers
  - Top-right: arg(ψ) phase heatmap
  - Bottom:    vortex count vs time with moving frame marker
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from tdgl3d import AppliedField, Device, SimulationParameters, solve
from tdgl3d.analysis.convergence import check_steady_state
from tdgl3d.analysis.vortex_counting import count_vortices_plaquette


def main(output_dir: Path = Path(__file__).parent, small: bool = False) -> list[Path]:
    if small:
        Nx, Ny, Nz = 12, 12, 1
        t_stop = 4.0
        save_every = 5
        step_stride = 5
    else:
        Nx, Ny, Nz = 100, 100, 1
        t_stop = 200.0
        save_every = 20
        step_stride = 40

    params = SimulationParameters(Nx=Nx, Ny=Ny, Nz=Nz, kappa=2.0)
    Bz_applied = 0.5
    device = Device(params, applied_field=AppliedField(Bz=Bz_applied, t_on_fraction=1.0))

    sol = solve(
        device, t_start=0.0, t_stop=t_stop, dt=0.01, method="euler",
        save_every=save_every, progress=False, log_metadata=False,
    )

    # --- Pre-compute data for all saved steps ---
    n_steps = sol.n_steps
    times = sol.times

    # Vortex count at every saved step
    vortex_counts = np.zeros(n_steps, dtype=int)
    for step in range(n_steps):
        n_v, _, _ = count_vortices_plaquette(sol, device, slice_z=0, step=step)
        vortex_counts[step] = n_v

    # --- Steady-state detection ---
    is_steady, steady_step, conv_metrics = check_steady_state(
        sol, device=device, window_size=10, psi_threshold=1e-4,
        current_threshold=1e-4, start_step=20,
    )
    if steady_step >= 0:
        t_steady = float(times[steady_step])
    else:
        # Fallback: check vortex count stability in last 20% of steps
        n_tail = max(2, n_steps // 5)
        tail = vortex_counts[-n_tail:]
        if int(np.max(tail)) - int(np.min(tail)) <= 2:
            t_steady = float(times[-n_tail])
        else:
            t_steady = None

    # Animation frame indices
    frame_steps = list(range(0, n_steps, step_stride))
    n_frames = len(frame_steps)

    # Pre-compute |ψ|² and arg(ψ) 2D arrays at animation frames
    psi2_frames = []
    phase_frames = []
    vortex_pos_frames = []
    winding_frames = []
    for s in frame_steps:
        psi2 = sol.psi_squared_2d(step=s, slice_z=0)
        psi2_frames.append(psi2)

        phase = sol.phase(step=s, mask_threshold=1e-6)
        phase_2d = sol._reshape_interior(phase, slice_z=0)
        phase_frames.append(phase_2d)

        n_v, positions, windings = count_vortices_plaquette(sol, device, slice_z=0, step=s)
        vortex_pos_frames.append(positions)
        winding_frames.append(windings)

    # Grid coordinates
    xs = np.arange(1, Nx) * params.hx
    ys = np.arange(1, Ny) * params.hy
    xx, yy = np.meshgrid(xs, ys, indexing="ij")

    # --- Figure ---
    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.3, wspace=0.3)
    ax_psi = fig.add_subplot(gs[0, 0])
    ax_phase = fig.add_subplot(gs[0, 1])
    ax_count = fig.add_subplot(gs[1, :])

    # Initial frame data
    vmax_psi2 = max(float(np.max(psi2_frames[0])), 0.1)
    im_psi = ax_psi.pcolormesh(xx, yy, psi2_frames[0], cmap="inferno",
                                vmin=0, vmax=vmax_psi2, shading="auto")
    fig.colorbar(im_psi, ax=ax_psi, fraction=0.046, pad=0.04, label="|ψ|²")
    ax_psi.set_aspect("equal")
    ax_psi.set_xlabel("x (ξ)")
    ax_psi.set_ylabel("y (ξ)")
    title_psi = ax_psi.set_title(f"|ψ|²  t = {times[frame_steps[0]]:.2f}")

    im_phase = ax_phase.pcolormesh(xx, yy, phase_frames[0], cmap="twilight",
                                    vmin=-np.pi, vmax=np.pi, shading="auto")
    fig.colorbar(im_phase, ax=ax_phase, fraction=0.046, pad=0.04, label="arg(ψ) [rad]")
    ax_phase.set_aspect("equal")
    ax_phase.set_xlabel("x (ξ)")
    ax_phase.set_ylabel("y (ξ)")
    title_phase = ax_phase.set_title("Phase winding")

    # Vortex count plot — full time series
    ax_count.plot(times, vortex_counts, "-", color="C0", linewidth=1.5, label="Vortex count")
    if t_steady is not None:
        ax_count.axvline(t_steady, color="green", linestyle="--", linewidth=1.5, alpha=0.8,
                         label=f"steady state (t={t_steady:.0f})")
    ax_count.set_xlabel("t (ξ/v_F)")
    ax_count.set_ylabel("Vortex count")
    ax_count.set_title("Vortex nucleation over time")
    ax_count.set_xlim(times[0], times[-1])
    ax_count.set_ylim(0, max(int(np.max(vortex_counts)) + 1, 1))
    ax_count.grid(True, alpha=0.3)
    ax_count.legend(loc="upper left")

    # Moving vertical marker on count plot
    marker_line = ax_count.axvline(times[frame_steps[0]], color="red", linewidth=2, alpha=0.7)

    # Vortex marker on psi plot
    (vortex_markers,) = ax_psi.plot([], [], "x", color="cyan", markersize=10, markeredgewidth=2)

    # Count text annotation on psi plot
    count_text = ax_psi.text(
        0.03, 0.03, "", transform=ax_psi.transAxes,
        fontsize=9, fontfamily="monospace",
        verticalalignment="bottom", horizontalalignment="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9),
    )

    fig.suptitle(
        f"Vortex Entry Dynamics — Bz = {Bz_applied}, κ = {params.kappa}",
        fontsize=14, y=0.98,
    )

    def update(frame_idx):
        s = frame_steps[frame_idx]

        # Update |ψ|²
        d_psi2 = psi2_frames[frame_idx]
        im_psi.set_array(d_psi2.ravel())
        title_psi.set_text(f"|ψ|²  t = {times[s]:.2f}")

        # Update phase
        d_phase = phase_frames[frame_idx]
        im_phase.set_array(d_phase.ravel())
        title_phase.set_text(f"Phase  t = {times[s]:.2f}")

        # Update vortex markers
        positions = vortex_pos_frames[frame_idx]
        if len(positions) > 0:
            vortex_markers.set_data(positions[:, 0], positions[:, 1])
        else:
            vortex_markers.set_data([], [])

        # Update count annotation
        n_v = vortex_counts[s]
        count_text.set_text(f"Vortices: {n_v}")

        # Update time marker
        marker_line.set_xdata([times[s], times[s]])

        return im_psi, im_phase, vortex_markers, marker_line, count_text

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)

    out = output_dir / "vortex_entry_dynamics.gif"
    anim.save(str(out), writer=PillowWriter(fps=10))
    plt.close(fig)
    return [out]


if __name__ == "__main__":
    main()

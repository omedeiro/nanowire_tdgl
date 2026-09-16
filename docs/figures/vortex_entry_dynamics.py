"""Vortex entry dynamics: animated GIF showing vortex nucleation over time.

Four-panel animation:
  - Top-left:  |ψ|² heatmap with vortex core markers
  - Top-right: arg(ψ) phase heatmap
  - Bottom-left:  vortex count vs time with steady-state marker
  - Bottom-right: convergence diagnostics (|ψ|² and |J_s| relative change)
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from tdgl3d import AppliedField, Device, SimulationParameters, solve
from tdgl3d.analysis.convergence import convergence_history, first_sustained_sample
from tdgl3d.analysis.vortex_counting import count_vortices_plaquette

# ``Device.initial_state`` seeds psi with 1% complex noise drawn from a
# non-deterministic RNG unless a seed is given, so an unseeded figure is a
# different realisation every time it is regenerated and cannot be compared
# against the one committed to the gallery.  Pin it.
NOISE_SEED = 7

# The order parameter settles hard; the supercurrent keeps a slow few-times-1e-4
# wobble for a few hundred tau after the count has locked, while the lattice
# anneals into place.  See ``test_vortex_entry_dynamics``.
PSI_THRESHOLD = 1e-4
CURRENT_THRESHOLD = 5e-4


def main(output_dir: Path = Path(__file__).parent, small: bool = False) -> list[Path]:
    if small:
        Nx, Ny, Nz = 12, 12, 1
        t_stop = 4.0
        save_every = 5
        step_stride = 5
    else:
        # 60 xi is the largest film that actually *finishes*: the count locks at
        # t ~ 390 and the fields stop moving by t ~ 480, leaving a visible flat
        # stretch at the end.  At 100 xi the edge barrier is still trickling
        # vortices in past t = 900, so there is no steady state to mark.
        Nx, Ny, Nz = 60, 60, 1
        t_stop = 600.0
        save_every = 50
        step_stride = 40

    # kappa = 2 and Bz = 0.5 H_c2 sit above the Bean-Livingston entry field, so
    # flux actually penetrates; below it the film simply stays Meissner and
    # there is no entry to animate.
    params = SimulationParameters(Nx=Nx, Ny=Ny, Nz=Nz, kappa=2.0)
    Bz_applied = 0.5
    device = Device(params, applied_field=AppliedField(Bz=Bz_applied, t_on_fraction=1.0))

    sol = solve(
        device, t_start=0.0, t_stop=t_stop, dt=0.01, method="euler",
        save_every=save_every, progress=False, log_metadata=False,
        noise_seed=NOISE_SEED,
    )

    # --- Pre-compute data for all saved steps ---
    n_steps = sol.n_steps
    times = sol.times

    # Vortex count at every saved step
    vortex_counts = np.zeros(n_steps, dtype=int)
    for step in range(n_steps):
        n_v, _, _ = count_vortices_plaquette(sol, device, slice_z=0, step=step)
        vortex_counts[step] = n_v

    # --- Convergence metrics, sampled across the run ---
    # Every sample costs two supercurrent evaluations, so walk the run with a
    # stride rather than step by step.  ``min_sustained`` counts samples, so
    # the steady-state mark is where the metrics first stay down for
    # min_sustained * conv_stride saved steps (10 tau_GL here) rather than
    # merely dipping once.
    window_size = 20
    conv_stride = 2
    min_sustained = 10

    conv_steps, psi2_rel_changes, current_rel_changes = convergence_history(
        sol, device, window_size=window_size, start_step=window_size, stride=conv_stride,
    )
    sample = first_sustained_sample(
        psi2_rel_changes, current_rel_changes, psi_threshold=PSI_THRESHOLD,
        current_threshold=CURRENT_THRESHOLD, min_sustained=min_sustained,
    )
    t_steady = float(times[conv_steps[sample]]) if sample >= 0 else None

    # Animation frame indices
    frame_steps = list(range(0, n_steps, step_stride))
    n_frames = len(frame_steps)

    # Pre-compute |ψ|² and arg(ψ) 2D arrays at animation frames
    psi2_frames = []
    phase_frames = []
    vortex_pos_frames = []
    for s in frame_steps:
        psi2_frames.append(sol.psi_squared_2d(step=s, slice_z=0))
        phase = sol.phase(step=s, mask_threshold=1e-6)
        phase_frames.append(sol._reshape_interior(phase, slice_z=0))
        _, positions, _ = count_vortices_plaquette(sol, device, slice_z=0, step=s)
        vortex_pos_frames.append(positions)

    # Grid coordinates
    xs = np.arange(1, Nx) * params.hx
    ys = np.arange(1, Ny) * params.hy
    xx, yy = np.meshgrid(xs, ys, indexing="ij")

    # --- Figure (2x2 grid) ---
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 2], hspace=0.35, wspace=0.35)
    ax_psi = fig.add_subplot(gs[0, 0])
    ax_phase = fig.add_subplot(gs[0, 1])
    ax_count = fig.add_subplot(gs[1, 0])
    ax_conv = fig.add_subplot(gs[1, 1])

    # --- Top-left: |ψ|² ---
    vmax_psi2 = max(float(np.max(psi2_frames[0])), 0.1)
    im_psi = ax_psi.pcolormesh(xx, yy, psi2_frames[0], cmap="inferno",
                                vmin=0, vmax=vmax_psi2, shading="auto")
    fig.colorbar(im_psi, ax=ax_psi, fraction=0.046, pad=0.04, label="|ψ|²")
    ax_psi.set_aspect("equal")
    ax_psi.set_xlabel("x (ξ)")
    ax_psi.set_ylabel("y (ξ)")
    title_psi = ax_psi.set_title(f"|ψ|²  t = {times[frame_steps[0]]:.2f}")

    # --- Top-right: Phase ---
    im_phase = ax_phase.pcolormesh(xx, yy, phase_frames[0], cmap="twilight",
                                    vmin=-np.pi, vmax=np.pi, shading="auto")
    fig.colorbar(im_phase, ax=ax_phase, fraction=0.046, pad=0.04, label="arg(ψ) [rad]")
    ax_phase.set_aspect("equal")
    ax_phase.set_xlabel("x (ξ)")
    ax_phase.set_ylabel("y (ξ)")
    title_phase = ax_phase.set_title("Phase winding")

    # --- Bottom-left: Vortex count ---
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
    ax_count.legend(loc="upper left", fontsize=8)

    marker_line = ax_count.axvline(times[frame_steps[0]], color="red", linewidth=2, alpha=0.7)

    # --- Bottom-right: Convergence diagnostics ---
    conv_times = times[conv_steps]
    ax_conv.plot(conv_times, psi2_rel_changes,
                 "-", color="C0", linewidth=1.2, label="|ψ|² rel. change")
    ax_conv.plot(conv_times, current_rel_changes,
                 "-", color="C1", linewidth=1.2, label="|J_s| rel. change")
    ax_conv.axhline(PSI_THRESHOLD, color="C0", linestyle="--", linewidth=1, alpha=0.7,
                    label=f"|ψ|² threshold = {PSI_THRESHOLD:.0e}")
    ax_conv.axhline(CURRENT_THRESHOLD, color="C1", linestyle="--", linewidth=1, alpha=0.7,
                    label=f"|J_s| threshold = {CURRENT_THRESHOLD:.0e}")
    if t_steady is not None:
        ax_conv.axvline(t_steady, color="green", linestyle="--", linewidth=1.5, alpha=0.8,
                        label=f"steady state (t={t_steady:.0f})")
    ax_conv.set_yscale("log")
    ax_conv.set_xlabel("t (ξ/v_F)")
    ax_conv.set_ylabel("Relative change")
    ax_conv.set_title("Convergence diagnostics")
    ax_conv.set_xlim(times[0], times[-1])
    ax_conv.grid(True, alpha=0.3)
    ax_conv.legend(loc="upper right", fontsize=8)

    conv_marker = ax_conv.axvline(times[frame_steps[0]], color="red", linewidth=2, alpha=0.7)

    # --- Annotations ---
    (vortex_markers,) = ax_psi.plot([], [], "x", color="cyan", markersize=10, markeredgewidth=2)
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

        im_psi.set_array(psi2_frames[frame_idx].ravel())
        title_psi.set_text(f"|ψ|²  t = {times[s]:.2f}")

        im_phase.set_array(phase_frames[frame_idx].ravel())
        title_phase.set_text(f"Phase  t = {times[s]:.2f}")

        positions = vortex_pos_frames[frame_idx]
        if len(positions) > 0:
            vortex_markers.set_data(positions[:, 0], positions[:, 1])
        else:
            vortex_markers.set_data([], [])

        count_text.set_text(f"Vortices: {vortex_counts[s]}")

        marker_line.set_xdata([times[s], times[s]])
        conv_marker.set_xdata([times[s], times[s]])

        return (im_psi, im_phase, vortex_markers, marker_line,
                count_text, conv_marker)

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)

    out = output_dir / "vortex_entry_dynamics.gif"
    # 14x10 in at the default 100 dpi lands a 31-frame GIF at ~4 MB, which is
    # a lot of repository for one figure; 80 dpi halves it and still resolves
    # the vortex markers and the axis labels.
    anim.save(str(out), writer=PillowWriter(fps=10), dpi=80)
    plt.close(fig)
    return [out]


if __name__ == "__main__":
    main()

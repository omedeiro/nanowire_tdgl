"""Flux expulsion by an S/I/S ring: how much field a hole can keep out.

A square hole is carved through both superconducting layers of an S/I/S stack,
leaving the oxide continuous.  The ring around the hole circulates a screening
current that holds the enclosed *fluxoid* at zero; above a threshold applied
field that current can no longer be sustained, a vortex crosses one of the arms,
and the fluxoid steps to a non-zero integer.

The figure shows the fluxoid staircase against applied field, the time at which
flux first enters (which diverges as the threshold is approached from above),
and the order parameter on either side of the threshold.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tdgl3d import AppliedField, Device, Layer, SimulationParameters, Trilayer
from tdgl3d.analysis.expulsion import (
    expulsion_field,
    first_entry_time,
    fluxoid_history,
    rectangular_contour,
)
from tdgl3d.core.solution import Solution
from tdgl3d.physics.applied_field import build_boundary_field_vectors
from tdgl3d.physics.rhs import BoundaryVectors
from tdgl3d.solvers.integrators import forward_euler

PHI0 = 2.0 * np.pi


def build_ring(
    applied_bz: float,
    length: float,
    hole_size: float,
    h: float,
    kappa: float,
    sc_cells: int,
    insulator_cells: int,
    hz: float = 1.0,
):
    """S/I/S stack of side *length* with a square hole through both SC layers.

    One modelling choice matters here and is easy to get wrong:

    * The superconducting layers are ``sc_cells × hz`` thick and that has to
      exceed the proximity length.  The oxide suppresses ψ over roughly a
      coherence length on each side of the interface, so a 1 ξ layer is
      pair-broken all the way through: |ψ| collapses to ~1e-4 and every phase
      measured on it is noise.
    """
    n_cells = int(round(length / h))
    trilayer = Trilayer(
        bottom=Layer(thickness_z=sc_cells, kappa=kappa),
        insulator=Layer(thickness_z=insulator_cells, kappa=kappa, is_superconductor=False),
        top=Layer(thickness_z=sc_cells, kappa=kappa),
    )
    params = SimulationParameters(
        Nx=n_cells, Ny=n_cells, Nz=trilayer.Nz, hx=h, hy=h, hz=hz, kappa=kappa
    )
    device = Device(
        params,
        applied_field=AppliedField(Bz=applied_bz, t_on_fraction=1.0),
        trilayer=trilayer,
    )

    lo, hi = 0.5 * (length - hole_size), 0.5 * (length + hole_size)
    square = [(lo, lo), (hi, lo), (hi, hi), (lo, hi)]
    z_ranges = trilayer.z_ranges()
    device.add_hole(square, z_range=z_ranges["bottom"])
    device.add_hole(square, z_range=z_ranges["top"])
    return params, device, trilayer, (lo, hi, lo, hi)


def relax(params, device, applied_bz, t_stop, n_save=25):
    """Integrate to *t_stop* at a fixed applied field and wrap in a Solution."""
    idx = device.idx
    boundary = BoundaryVectors(
        *build_boundary_field_vectors(0.0, 0.0, applied_bz, params, idx)
    )
    # The familiar dt < h²/(4κ²) is a 2-D bound; in 3-D the curl-curl block's
    # spectral radius doubles and the limit halves.
    h_min = min(params.hx, params.hy, params.hz)
    dt = 0.9 * h_min**2 / (4.0 * params.kappa**2 * (2.0 if params.is_3d else 1.0))
    times, states = forward_euler(
        device.initial_state(noise_amplitude=0.0).data,
        params, idx, lambda t, X: boundary, 0.0, t_stop, dt,
        save_every=max(1, int(t_stop / dt / n_save)),
        progress=False, material=device.material,
    )
    return Solution(times=times, states=states, params=params, idx=idx, device=device)


def scan(fields, length, hole_size, h, kappa, sc_cells, insulator_cells, t_stop):
    """Relax at each applied field and return the fluxoid outcome of each run."""
    finals, entries, histories, solutions = [], [], [], {}
    for applied_bz in fields:
        params, device, trilayer, hole_bounds = build_ring(
            applied_bz, length, hole_size, h, kappa, sc_cells, insulator_cells
        )
        solution = relax(params, device, applied_bz, t_stop)
        contour = rectangular_contour(hole_bounds, params, margin=1.5)
        # A z-slice in the middle of the bottom superconducting layer.
        slice_z = max(trilayer.z_ranges()["bottom"][1] // 2 - 1, 0)
        history = fluxoid_history(solution, device, contour, slice_z=slice_z)

        histories.append(history)
        finals.append(float(history[-1]))
        entries.append(first_entry_time(solution.times, history))
        solutions[applied_bz] = (params, device, solution, slice_z)
    return finals, entries, histories, solutions


def main(output_dir: Path = Path(__file__).parent, small: bool = False) -> list[Path]:
    kappa = 2.0
    h = 1.0
    sc_cells, insulator_cells = 4, 2
    arm = 3.0
    if small:
        hole_size, t_stop = 2.0, 3.0
        fields = [0.2, 0.9]
        refine = []
    else:
        hole_size, t_stop = 4.0, 30.0
        fields = [0.05, 0.10, 0.15, 0.22, 0.28, 0.32, 0.45, 0.6]
        # Two fields either side of the threshold, repeated at half the in-plane
        # spacing, to show the bracket is not an artefact of the coarse grid.
        refine = [0.22, 0.32]
    length = hole_size + 2 * arm

    finals, entries, histories, solutions = scan(
        fields, length, hole_size, h, kappa, sc_cells, insulator_cells, t_stop
    )

    result = expulsion_field(fields, finals, entries, hold_time=t_stop)

    refined, fine_solutions, fine_finals = None, {}, []
    if refine:
        fine_finals, fine_entries, _, fine_solutions = scan(
            refine, length, hole_size, h / 2, kappa, sc_cells, insulator_cells, t_stop
        )
        refined = expulsion_field(refine, fine_finals, fine_entries, hold_time=t_stop)

    # Representative states either side of the threshold.  Prefer the refined
    # runs for the maps: at h = ξ the arms are only three cells wide.
    if refined is not None and refined.threshold is not None:
        map_source, map_fields = fine_solutions, refine
        map_finals = dict(zip(refine, fine_finals))
        map_h = h / 2
    else:
        map_source, map_fields = solutions, fields
        map_finals = dict(zip(fields, finals))
        map_h = h
    below = min(map_fields)
    above = max(map_fields)

    def psi_map(applied_bz):
        params, device, solution, slice_z = map_source[applied_bz]
        nx, ny, nz = params.Nx - 1, params.Ny - 1, max(params.Nz - 1, 1)
        cube = np.abs(solution.psi(step=-1)).reshape(nx, ny, nz) ** 2
        return params, cube[:, :, slice_z]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # -- (0,0) fluxoid staircase vs applied field ---------------------------
    ax = axes[0, 0]
    ax.step(fields, finals, where="mid", color="C0", marker="o", label="fluxoid n")
    ax.axhline(0, color="gray", ls=":", alpha=0.6)
    if result.threshold is not None:
        ax.axvspan(result.last_expelled, result.first_entered, color="C3", alpha=0.15)
        ax.axvline(result.threshold, color="C3", ls="--", linewidth=1.5,
                   label=f"B_exp = {result.threshold:.3f} ± {result.uncertainty:.3f}")
    ideal = [applied_bz * hole_size**2 / PHI0 for applied_bz in fields]
    ax.plot(fields, ideal, ":", color="C2", linewidth=1.5,
            label="B·A_hole/Φ₀ (unscreened)")
    ax.set_xlabel("applied Bz  (Φ₀/2πξ²)")
    ax.set_ylabel("enclosed fluxoid  (Φ₀)")
    ax.set_title("Flux stays out until the ring gives way")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # -- (0,1) entry time vs field ------------------------------------------
    ax = axes[0, 1]
    entered = [(b, t) for b, t in zip(fields, entries) if t is not None]
    if entered:
        ax.plot([b for b, _ in entered], [t for _, t in entered],
                "o-", color="C1", label="first entry")
    ax.axhline(t_stop, color="gray", ls=":", alpha=0.6, label=f"hold time = {t_stop:g}")
    if result.threshold is not None:
        ax.axvline(result.threshold, color="C3", ls="--", linewidth=1.5)
    ax.set_xlabel("applied Bz  (Φ₀/2πξ²)")
    ax.set_ylabel("time of first flux entry  (τ_GL)")
    ax.set_title("Entry time diverges at the threshold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    text = (
        f"hole:        {hole_size:.0f}×{hole_size:.0f} ξ  (A = {hole_size**2:.0f} ξ²)\n"
        f"film:        {length:.0f}×{length:.0f} ξ, arms {arm:.0f} ξ, κ = {kappa}\n"
        f"stack:       S({sc_cells:.0f} ξ)/I({insulator_cells:.0f} ξ)/S({sc_cells:.0f} ξ)\n"
        f"grid:        h = {h:g} ξ\n"
        f"hold time:   {t_stop:g} τ_GL\n"
        f"{result.summary()}\n"
        f"B_exp·A/Φ₀:  "
        + (f"{result.threshold * hole_size**2 / PHI0:.3f}"
           if result.threshold is not None else "—")
        + (f"\nat h = {h / 2:g} ξ: " + refined.summary().replace("(hold time", "\n  (hold time")
           if refined is not None and refined.threshold is not None else "")
    )
    axes[0, 0].text(
        0.03, 0.97, text, transform=axes[0, 0].transAxes, fontsize=7.5,
        fontfamily="monospace", verticalalignment="top", horizontalalignment="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9),
    )

    # -- bottom row: |ψ|² either side of the threshold -----------------------
    for ax, applied_bz, label in (
        (axes[1, 0], below, "expelled"),
        (axes[1, 1], above, "flux admitted"),
    ):
        params, field_map = psi_map(applied_bz)
        xs = np.arange(1, params.Nx) * params.hx
        ys = np.arange(1, params.Ny) * params.hy
        mesh = ax.pcolormesh(
            *np.meshgrid(xs, ys, indexing="ij"), field_map,
            cmap="inferno", vmin=0.0, vmax=1.0, shading="auto",
        )
        fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04, label="|ψ|²")
        n_here = abs(map_finals[applied_bz])
        ax.set_title(
            f"Bz = {applied_bz:g} — {label}, n = {n_here:.0f}   (h = {map_h:g} ξ)"
        )
        ax.set_xlabel("x (ξ)")
        ax.set_ylabel("y (ξ)")
        ax.set_aspect("equal")

    fig.suptitle(
        "S/I/S Ring — Flux Expulsion and the Expulsion Field", fontsize=14, y=0.98
    )
    fig.tight_layout()

    out = output_dir / "sis_hole_expulsion.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # -- fluxoid histories, one line per field ------------------------------
    fig2, ax = plt.subplots(figsize=(7, 4.5))
    for applied_bz, history in zip(fields, histories):
        params, device, solution, _ = solutions[applied_bz]
        ax.plot(solution.times, history, drawstyle="steps-post",
                label=f"Bz = {applied_bz:g}")
    ax.set_xlabel("t (τ_GL)")
    ax.set_ylabel("enclosed fluxoid  (Φ₀)")
    ax.set_title(
        "Flux enters in whole quanta, sooner the higher the field\n"
        "(pairs, because the noiseless device is C4-symmetric)",
        fontsize=11,
    )
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig2.tight_layout()
    out2 = output_dir / "sis_hole_fluxoid_history.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    print(result.summary())
    return [out, out2]


if __name__ == "__main__":
    main()

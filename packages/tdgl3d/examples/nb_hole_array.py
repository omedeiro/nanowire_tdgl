"""Example: a 3×3 array of 4 µm holes in an S/I/S Nb stack.

Geometry
--------
Nine square holes, 4 µm on a side, on an 8 µm pitch (4 µm hole, 4 µm of metal
between neighbours), with an 8 µm buffer of unbroken film around the array::

    8 µm │ 4 │ 4 │ 4 │ 4 │ 4 │ 8 µm      →  36 µm across
         └ hole   gap  hole  gap  hole ┘

The stack is S(500 nm) / I(500 nm) / S(500 nm) = 1.5 µm thick, and the holes go
through all three layers.  The oxide is given the same κ as the metal: an
insulator at κ = 0 cannot carry a magnetic field at all, and would block the
field between the layers instead of transmitting it.

Scale is what makes this expensive
----------------------------------
The grid spacing is fixed by the coherence length, but the device is fixed by
lithography, so the node count goes as (36 µm / ξ)² × (1.5 µm / ξ).  At
ξ = 100 nm — Nb near T_c, where Ginzburg-Landau applies — that is 360 × 360 × 15
and 1.8 M interior nodes; at ξ = 50 nm it is 15 M and each state vector alone is
a gigabyte.  ``--xi`` is therefore the knob that decides whether a run takes
hours or weeks, and ξ is a function of temperature: ξ(T) = ξ₀/√(1 − T/T_c).

Getting flux into the holes
---------------------------
The film screens too well for the obvious protocol to work.  Ramping a field up
from zero, nothing enters at all until about **3.15 mT** — and just above that,
hundreds of vortices enter at once.  Held at 3.6 mT for 200 τ_GL, 567 of them
pack the 8 µm buffer into a triangular lattice and the whole 3×3 array stays
fully Meissner-screened behind them: the flux front stalls at the array
perimeter and never reaches a hole.  There is no applied field at which this
film holds one or two vortices in equilibrium; it holds none, or it holds
hundreds.

``--protocol field-cool`` does what the experiment does instead.  ψ starts near
zero — the numerical stand-in for cooling through T_c — with the field already
on, so flux is trapped where it already is rather than having to cross 8 µm of
screening metal.  The field then ramps down, and the two kinds of trapped flux
part company: a vortex in the metal costs a core and is driven out through the
edge, while a hole has no core to pay for and keeps its fluxoid.  What is left
is a remanent state with quantised flux in the holes and a nearly clean film.

Use forward Euler
-----------------
``solve()`` defaults to the implicit trapezoidal integrator, which on a grid
this size is roughly an order of magnitude *slower* per unit simulated time
than forward Euler.  The Newton-GCR inner solve is unpreconditioned, so its
iteration count grows about as fast as the step size it buys — at this κ and
spacing the implicit method never earns back its cost.  This script uses Euler
at 90% of the CFL limit, as every figure in ``docs/figures/`` does.

Cost
----
Measured on 4 cores, per unit of Ginzburg-Landau time (``t_stop`` is in those
units; the published S/I/S ring figure uses ``t_stop = 60``):

===========  ==================  ==============  ============  ==========
ξ            grid                interior nodes  s per τ_GL     peak RSS
===========  ==================  ==============  ============  ==========
150 nm       240 × 240 × 9              457 k             19      0.4 GB
100 nm       360 × 360 × 15            1.80 M            113      1.5 GB
70 nm        514 × 514 × 21            5.26 M            440      3.7 GB
50 nm        720 × 720 × 30           15.0 M            1459     10.4 GB
===========  ==================  ==============  ============  ==========

So the default here — ξ = 100 nm, t_stop = 60 — is about two hours.  Sweeping
applied field is embarrassingly parallel across field values: run one process
per value rather than one long run.

Memory
------
``--save-every`` controls the only term that grows with run length.  One saved
frame is ``4 × n_interior`` complex128 — 116 MB at ξ = 100 nm, 960 MB at
ξ = 50 nm — so ask for the dozen or so frames you will actually look at, not
every step.

Usage
-----
::

    # cost estimate only, no solve
    python3 packages/tdgl3d/examples/nb_hole_array.py --dry-run

    # Meissner state at one field
    python3 packages/tdgl3d/examples/nb_hole_array.py --xi 150 --t-stop 20

    # flux trapped in the holes, with a GIF of it happening
    python3 packages/tdgl3d/examples/nb_hole_array.py --xi 150 \
        --protocol field-cool --bz-mT 2.0 --bz-final-mT 0.0 --ramp 0.22 \
        --t-stop 420 --gif
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tdgl3d
from tdgl3d.core.material import Layer, Trilayer
from tdgl3d.core.units import GLUnits

# ── Device, in nanometres — the fabrication numbers, not the grid ──────────
HOLE_NM = 4000.0     # hole side
GAP_NM = 4000.0      # metal between neighbouring holes
BUFFER_NM = 8000.0   # unbroken film around the 3×3 array
LAYER_NM = 500.0     # each of S, I, S
N_HOLES = 3          # per side of the square array

KAPPA = 2.0          # λ/ξ for Nb: λ ≈ 200 nm against ξ = 100 nm near T_c

#: Forward Euler is stable below h²/(4κ²(d−1)); in 3-D the curl-curl term picks
#: up a second transverse direction and the familiar 2-D bound halves.
CFL_SAFETY = 0.9

#: Measured ``(interior nodes, seconds per Euler step)`` for this device on 4
#: cores.  Cost is not quite linear in the node count — the per-node cost nearly
#: doubles from the smallest grid to the largest as the working set outgrows
#: cache — so the estimate interpolates these rather than scaling one of them.
_MEASURED_STEP_COST = [
    (456_968, 0.54),
    (1_804_334, 3.16),
    (5_263_380, 12.38),
    (14_991_869, 41.02),
]

#: How much more the implicit integrator costs per unit simulated time, measured
#: on the ξ = 100 nm grid.  Its Newton-GCR inner solve is unpreconditioned, so
#: the iteration count grows about as fast as the step size it buys.
TRAPEZOIDAL_PENALTY = 8.0


def _seconds_per_step(n_interior: int) -> float:
    """Interpolate the measured per-step cost, log-log, in the node count."""
    xs = np.log([n for n, _ in _MEASURED_STEP_COST])
    ys = np.log([t for _, t in _MEASURED_STEP_COST])
    return float(np.exp(np.interp(np.log(n_interior), xs, ys)))


def plan(xi_nm: float, h_xi: float) -> dict:
    """Grid, node count and cost estimate for this device at a given ξ."""
    units = GLUnits(xi_nm=xi_nm, kappa=KAPPA)
    hole = units.length(HOLE_NM)
    gap = units.length(GAP_NM)
    buffer_xi = units.length(BUFFER_NM)
    side = N_HOLES * hole + (N_HOLES - 1) * gap + 2 * buffer_xi

    n_side = int(round(side / h_xi))
    n_layer = max(int(round(units.length(LAYER_NM) / h_xi)), 1)
    n_z = 3 * n_layer
    n_interior = (n_side - 1) ** 2 * max(n_z - 1, 1)

    dt = CFL_SAFETY * h_xi**2 / (4.0 * KAPPA**2 * 2.0)
    return {
        "units": units,
        "hole_xi": hole,
        "gap_xi": gap,
        "buffer_xi": buffer_xi,
        "side_xi": side,
        "side_um": units.length_nm(side) / 1000.0,
        "n_side": n_side,
        "n_layer": n_layer,
        "n_z": n_z,
        "n_interior": n_interior,
        "dt": dt,
        "MB_per_frame": 4 * n_interior * 16 / 1e6,
        "sec_per_step": _seconds_per_step(n_interior),
        # Euler spends exactly one right-hand-side evaluation per step, so its
        # cost per unit simulated time is the step cost over the step size.
        "sec_per_tau_GL": _seconds_per_step(n_interior) / dt,
    }


def hole_rects(spec: dict) -> list[tuple[float, float, float, float]]:
    """``(x0, y0, x1, y1)`` of every hole, in ξ units."""
    pitch = spec["hole_xi"] + spec["gap_xi"]
    rects = []
    for row in range(N_HOLES):
        for col in range(N_HOLES):
            x0 = spec["buffer_xi"] + col * pitch
            y0 = spec["buffer_xi"] + row * pitch
            rects.append((x0, y0, x0 + spec["hole_xi"], y0 + spec["hole_xi"]))
    return rects


#: Width of the field-cool down-ramp, as a fraction of ``t_stop``.  Everything
#: after it is settling time, and settling is what clears the film: the metal's
#: vortices leave through the edge over tens of τ_GL, so a protocol that ramps
#: late leaves a film still full of them.
FIELD_COOL_RAMP_WIDTH = 0.1


def field_cool_window(t_stop: float, ramp_start_fraction: float) -> tuple:
    """``(t_start, t_end)`` of the down-ramp."""
    start = ramp_start_fraction * t_stop
    return start, start + FIELD_COOL_RAMP_WIDTH * t_stop


def field_schedule(
    units, bz_mT: float, bz_final_mT: float, t_stop: float, protocol: str,
    ramp_start_fraction: float = 0.35,
):
    """``f(t, t_stop) -> (Bx, By, Bz)`` for the requested protocol.

    ``constant`` and ``ramp-up`` are what :class:`AppliedField` already does;
    ``field-cool`` needs a schedule of its own, because trapping flux in the
    holes and clearing it out of the metal are two different fields.  The film
    screens so well that nothing enters below about 3 mT, and at that field
    hundreds of vortices enter at once — so there is no field at which the
    metal holds one or two vortices in equilibrium.  Cooling in a *low* field
    instead traps flux everywhere at once, and the down-ramp then drives the
    film's vortices out through the edge while the holes keep theirs: a hole
    has no core to pay for, so its trapped fluxoid survives a field the film
    cannot hold a vortex in.
    """
    bz = units.field(bz_mT)
    bz_final = units.field(bz_final_mT)
    t_ramp_start, t_ramp_end = field_cool_window(t_stop, ramp_start_fraction)

    def evaluate(t: float, _t_stop: float) -> tuple[float, float, float]:
        if t <= t_ramp_start:
            return 0.0, 0.0, bz
        if t >= t_ramp_end:
            return 0.0, 0.0, bz_final
        frac = (t - t_ramp_start) / (t_ramp_end - t_ramp_start)
        return 0.0, 0.0, bz + frac * (bz_final - bz)

    if protocol != "field-cool":
        return None
    return evaluate


def normal_initial_state(device, spec: dict, seed: int, amplitude: float = 0.02):
    """A near-normal start: |ψ| ≈ *amplitude*, random phase, φ = 0.

    This is the numerical stand-in for cooling through T_c in a field.  Starting
    from the fully-formed condensate instead makes the film screen from the
    first step, and flux can then only enter from the outside edge — which on a
    36 µm film means it never reaches the array at all.  Growing ψ from nothing
    with the field already on lets flux be trapped where it already is.
    """
    state = device.initial_state(noise_amplitude=0.0, seed=seed)
    rng = np.random.default_rng(seed)
    n = spec["n_interior"]
    phase = rng.uniform(0.0, 2.0 * np.pi, n)
    state.psi[:] *= amplitude * np.exp(1j * phase)
    return state


def build(
    spec: dict, bz_mT: float, t_on_fraction: float, ramp_fraction: float = 0.0,
    field_func=None,
) -> tdgl3d.Device:
    """The trilayer with all nine holes carved through it.

    A non-zero *ramp_fraction* raises the field linearly over that fraction of
    the run and holds it after — which is how vortices are made to enter a few
    at a time instead of all at once from a step change.  *field_func* overrides
    both and supplies the field directly.
    """
    units = spec["units"]
    trilayer = Trilayer(
        bottom=Layer(thickness_z=spec["n_layer"], kappa=KAPPA, is_superconductor=True),
        # The oxide keeps the metal's κ on purpose.  At κ = 0 its φ-equation
        # degenerates and it blocks the field rather than transmitting it.
        insulator=Layer(
            thickness_z=spec["n_layer"], kappa=KAPPA, is_superconductor=False
        ),
        top=Layer(thickness_z=spec["n_layer"], kappa=KAPPA, is_superconductor=True),
    )
    params = tdgl3d.SimulationParameters(
        Nx=spec["n_side"],
        Ny=spec["n_side"],
        Nz=trilayer.Nz,
        hx=spec["side_xi"] / spec["n_side"],
        hy=spec["side_xi"] / spec["n_side"],
        hz=spec["side_xi"] / spec["n_side"],
        kappa=KAPPA,
    )
    device = tdgl3d.Device(
        params,
        applied_field=tdgl3d.AppliedField(
            Bz=units.field(bz_mT),
            t_on_fraction=t_on_fraction,
            ramp=ramp_fraction > 0.0,
            ramp_fraction=ramp_fraction or 0.5,
            field_func=field_func,
        ),
        trilayer=trilayer,
    )

    for x0, y0, x1, y1 in hole_rects(spec):
        device.add_hole(
            [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], z_range=(0, trilayer.Nz)
        )
    return device


def sc_slice(spec: dict) -> int:
    """Interior z-slice strictly inside the bottom superconductor.

    Interior arrays run k = 1 … Nz-1, so interior slice s is full-grid node
    s + 1.  The bottom superconductor owns full-grid nodes k < n_layer (both
    oxide interfaces belong to the insulator), so the deepest interior slice
    still inside the metal is n_layer - 2.
    """
    return max(min(spec["n_layer"] // 2, spec["n_layer"] - 2), 0)


def vortex_census(solution, device, spec: dict, step: int, margin: float = 2.0) -> dict:
    """Vortices sitting in the metal, and the fluxoid trapped in each hole.

    The two are counted differently because they are different things.  A
    vortex in the film is a core — a plaquette carrying 2π of gauge-invariant
    phase winding, which ``count_vortices_plaquette`` finds directly.  A hole
    has no core to find: what it holds is a fluxoid, and the only way to read
    it is the winding on a contour drawn in the metal *around* the hole, which
    is an exact integer regardless of how much field actually threads the
    opening.
    """
    from tdgl3d.analysis.vortex_counting import (
        count_vortices_plaquette,
        count_vortices_polygon,
    )

    slice_z = sc_slice(spec)
    rects = hole_rects(spec)

    _, positions, windings = count_vortices_plaquette(
        solution, device, slice_z=slice_z, step=step
    )
    in_film = 0
    for (px, py), winding in zip(positions, windings):
        # Plaquette (i, j) spans nodes i…i+1, so its centre sits at node i + 1
        # in full-grid coordinates.
        node_x, node_y = px + 1.0, py + 1.0
        inside_a_hole = any(
            x0 - margin <= node_x <= x1 + margin
            and y0 - margin <= node_y <= y1 + margin
            for x0, y0, x1, y1 in rects
        )
        if not inside_a_hole:
            in_film += abs(int(round(winding)))

    trapped = []
    for x0, y0, x1, y1 in rects:
        contour = np.array([
            [x0 - margin, y0 - margin], [x1 + margin, y0 - margin],
            [x1 + margin, y1 + margin], [x0 - margin, y1 + margin],
        ])
        trapped.append(
            int(round(count_vortices_polygon(
                solution, device, contour, slice_z=slice_z, step=step
            )))
        )

    return {"film": in_film, "holes": trapped, "hole_total": int(sum(trapped))}


def write_gif(
    solution, device, spec: dict, out_path: Path, field_of_t, fps: int = 8,
) -> dict:
    """Animate |ψ|² through the run, labelling each hole with its fluxoid.

    The fluxoid is the point of the annotation: a hole shows no core and no
    dark spot, so without the number there is nothing on screen to say how much
    flux it is holding.
    """
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Rectangle

    units = spec["units"]
    per_xi = units.xi_nm / 1000.0  # µm per ξ
    slice_z = sc_slice(spec)
    extent = [0, spec["side_um"], 0, spec["side_um"]]
    census = [
        vortex_census(solution, device, spec, step)
        for step in range(solution.n_steps)
    ]

    fig, ax = plt.subplots(figsize=(6.6, 6.0), constrained_layout=True)
    image = ax.imshow(
        solution.psi_squared_2d(0, slice_z=slice_z).T,
        origin="lower", extent=extent, cmap="inferno", vmin=0.0, vmax=1.0,
    )
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02, label=r"$|\psi|^2$")

    labels = []
    for x0, y0, x1, y1 in hole_rects(spec):
        ax.add_patch(Rectangle(
            (x0 * per_xi, y0 * per_xi),
            (x1 - x0) * per_xi, (y1 - y0) * per_xi,
            fill=False, edgecolor="#7fd4ff", lw=0.8, alpha=0.7,
        ))
        labels.append(ax.text(
            0.5 * (x0 + x1) * per_xi, 0.5 * (y0 + y1) * per_xi, "",
            color="#7fd4ff", ha="center", va="center", fontsize=13, fontweight="bold",
        ))
    title = ax.set_title("")

    def update(step: int):
        image.set_data(solution.psi_squared_2d(step, slice_z=slice_z).T)
        counts = census[step]
        for label, n in zip(labels, counts["holes"]):
            label.set_text(str(n) if n else "")
        t = float(solution.times[step])
        title.set_text(
            f"t = {t:6.1f} τ$_{{GL}}$    B = {field_of_t(t):.3f} mT    "
            f"{counts['film']} in film, {counts['hole_total']} in holes"
        )
        return [image, title, *labels]

    FuncAnimation(fig, update, frames=solution.n_steps, blit=False).save(
        str(out_path), writer=PillowWriter(fps=fps)
    )
    plt.close(fig)
    return census[-1]


def summarise(solution, spec: dict, out_dir: Path) -> dict:
    """Write a |ψ|² / B_z figure and return the numbers worth printing."""
    units = spec["units"]
    # Interior arrays run k = 1 … Nz-1, so interior slice s is full-grid node
    # s + 1.  The bottom superconductor owns full-grid nodes k < n_layer (both
    # oxide interfaces belong to the insulator), so the deepest interior slice
    # still inside the metal is n_layer - 2.
    slice_z = max(min(spec["n_layer"] // 2, spec["n_layer"] - 2), 0)

    psi2 = solution.psi_squared_2d(-1, slice_z=slice_z)
    bz = solution.bfield(-1)[2]
    bz_plane = np.asarray(bz).reshape(-1)

    extent = [0, spec["side_um"], 0, spec["side_um"]]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    im0 = axes[0].imshow(psi2.T, origin="lower", extent=extent, cmap="inferno",
                         vmin=0.0, vmax=1.0)
    axes[0].set_title(r"$|\psi|^2$, mid-plane of the bottom Nb layer")
    fig.colorbar(im0, ax=axes[0])

    n_side = int(round(np.sqrt(bz_plane.size / max(spec["n_z"] - 1, 1))))
    try:
        bz_slice = bz_plane.reshape(n_side, n_side, -1)[:, :, slice_z]
        im1 = axes[1].imshow(bz_slice.T, origin="lower", extent=extent, cmap="coolwarm")
        fig.colorbar(im1, ax=axes[1])
    except ValueError:
        axes[1].text(0.5, 0.5, "B_z slice unavailable", ha="center")
    axes[1].set_title(r"$B_z$")
    for ax in axes:
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")

    figure_path = out_dir / "nb_hole_array.png"
    fig.savefig(figure_path, dpi=140)
    plt.close(fig)

    psi_abs_max = float(np.abs(solution.psi(-1)).max())
    return {
        "figure": str(figure_path),
        "psi_abs_max": psi_abs_max,
        # The layer can be pair-broken by the adjacent oxide while still
        # producing plausible-looking screening, so this is the number to check
        # before trusting anything phase-derived.  Well below 1 means the metal
        # is suppressed, not merely that the run is young.
        "pair_broken": psi_abs_max < 1e-2,
        "suppressed": psi_abs_max < 0.5,
        "psi2_mean": float(np.mean(solution.psi_squared(-1))),
        "Bz_absmax_mT": float(units.field_to_mT(np.max(np.abs(bz_plane)))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--xi", type=float, default=100.0,
                        help="coherence length ξ(T) in nm (default: 100)")
    parser.add_argument("--h", type=float, default=1.0,
                        help="grid spacing in units of ξ (default: 1.0)")
    parser.add_argument("--bz-mT", type=float, default=1.0,
                        help="applied field in mT (default: 1.0)")
    parser.add_argument("--t-stop", type=float, default=60.0,
                        help="simulated time in τ_GL (default: 60, as in the "
                             "published S/I/S ring figure)")
    parser.add_argument("--t-on", type=float, default=1.0,
                        help="AppliedField.t_on_fraction (default: 1.0, field on "
                             "from the start)")
    parser.add_argument("--ramp", type=float, default=0.0, metavar="FRACTION",
                        help="raise the field linearly over this fraction of the "
                             "run, then hold (default: 0, field on at full "
                             "strength from t=0). Use ~0.6 to watch vortices "
                             "enter a few at a time.")
    parser.add_argument("--protocol",
                        choices=["constant", "ramp-up", "field-cool"],
                        default="constant",
                        help="constant: field on at full strength from t=0. "
                             "ramp-up: raise it over --ramp of the run (needs a "
                             "field above the entry threshold, ~3 mT here). "
                             "field-cool: start from a near-normal state at "
                             "--bz-mT, then ramp to --bz-final-mT — the way to "
                             "leave flux trapped in the holes and a clean film. "
                             "--ramp sets when the down-ramp starts (default "
                             "0.35); everything after it is settling time, and "
                             "settling is what empties the film.")
    parser.add_argument("--bz-final-mT", type=float, default=0.0,
                        help="field to ramp down to under --protocol field-cool "
                             "(default: 0)")
    parser.add_argument("--gif", action="store_true",
                        help="also write an animated GIF of |psi|^2 with each "
                             "hole labelled by its trapped fluxoid")
    parser.add_argument("--fps", type=int, default=8, help="GIF frame rate")
    parser.add_argument("--save-every", type=int, default=0,
                        help="save every n-th step; 0 picks ~12 frames")
    parser.add_argument("--method", choices=["euler", "trapezoidal"], default="euler",
                        help="integrator (default: euler — see the module docstring)")
    parser.add_argument("--seed", type=int, default=42, help="noise seed")
    parser.add_argument("--out", type=Path, default=Path(".tdgl3d-run"),
                        help="output directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the cost estimate and exit without solving")
    args = parser.parse_args()

    spec = plan(args.xi, args.h)
    units = spec["units"]
    ramp_start = args.ramp if args.ramp else 0.35
    dt = spec["dt"] if args.method == "euler" else 0.05
    n_steps = max(int(round(args.t_stop / dt)), 1)
    save_every = args.save_every or max(n_steps // 12, 1)

    print(units.summary())
    print(
        f"device {spec['side_um']:.1f} × {spec['side_um']:.1f} µm × "
        f"{units.length_nm(spec['n_z'] * args.h) / 1000:.2f} µm, "
        f"{N_HOLES}×{N_HOLES} holes of "
        f"{units.length_nm(spec['hole_xi']) / 1000:.1f} µm"
    )
    print(
        f"grid {spec['n_side']}×{spec['n_side']}×{spec['n_z']} "
        f"= {spec['n_interior']:,} interior nodes, "
        f"{spec['MB_per_frame']:.0f} MB per saved frame"
    )
    print(
        f"{args.method}, dt = {dt:.4g}, {n_steps:,} steps to t = {args.t_stop:g} τ_GL"
    )
    rate = spec["sec_per_tau_GL"]
    if args.method != "euler":
        rate *= TRAPEZOIDAL_PENALTY
    hours = rate * args.t_stop / 3600.0
    print(f"estimated ≈ {hours:.1f} h of solve time ({rate:.0f} s per τ_GL"
          f", measured on 4 cores)")
    if args.method != "euler":
        print(f"note: trapezoidal costs about {TRAPEZOIDAL_PENALTY:.0f}× Euler per "
              "unit simulated time on a grid this size")
    print(f"saving every {save_every} steps → ~{n_steps // save_every + 1} frames, "
          f"{spec['MB_per_frame'] * (n_steps // save_every + 1) / 1000:.1f} GB")
    if args.protocol == "ramp-up" and args.ramp:
        print(f"field ramps 0 → {args.bz_mT:g} mT over t = 0 … "
              f"{args.t_stop * args.ramp:g} τ_GL, then holds")
    elif args.protocol == "field-cool":
        lo, hi = field_cool_window(args.t_stop, ramp_start)
        print(f"field-cool: ψ grows from ~0 at {args.bz_mT:g} mT, "
              f"field ramps to {args.bz_final_mT:g} mT over "
              f"t = {lo:g} … {hi:g} τ_GL, then holds")

    # A superconducting layer thinner than about 3 ξ is pair-broken by the oxide
    # beside it and still produces plausible-looking screening, so this has to be
    # said before the run rather than diagnosed after it.  500 nm layers put a
    # ceiling on ξ, which is a floor on how close to T_c the device can be posed.
    layer_xi = units.length(LAYER_NM)
    if layer_xi < 3.0:
        print(
            f"WARNING: each layer is only {layer_xi:.1f} ξ thick at ξ = {args.xi:g} nm. "
            f"Below ~3 ξ the oxide pair-breaks the metal (|ψ| → 0) while B_z still "
            f"looks right. Use ξ ≲ {LAYER_NM / 3.0:.0f} nm — i.e. further from T_c — "
            f"for these 500 nm layers."
        )

    if args.dry_run:
        return

    args.out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    field_func = field_schedule(
        units, args.bz_mT, args.bz_final_mT, args.t_stop, args.protocol,
        ramp_start_fraction=ramp_start,
    )
    ramp_fraction = args.ramp if args.protocol == "ramp-up" else 0.0
    device = build(
        spec, args.bz_mT, args.t_on, ramp_fraction=ramp_fraction,
        field_func=field_func,
    )
    print(f"device built in {time.perf_counter() - t0:.1f} s")

    x0 = (
        normal_initial_state(device, spec, args.seed)
        if args.protocol == "field-cool"
        else None
    )

    t0 = time.perf_counter()
    solution = tdgl3d.solve(
        device,
        t_stop=args.t_stop,
        dt=dt,
        method=args.method,
        x0=x0,
        save_every=save_every,
        noise_seed=args.seed,
        progress=True,
        log_metadata=False,
    )
    wall = time.perf_counter() - t0

    result = {
        "xi_nm": args.xi,
        "h_xi": args.h,
        "bz_mT": args.bz_mT,
        "grid": [spec["n_side"], spec["n_side"], spec["n_z"]],
        "n_interior": spec["n_interior"],
        "method": args.method,
        "dt": dt,
        "t_stop": args.t_stop,
        "wall_seconds": round(wall, 1),
        "sec_per_tau_GL": round(wall / args.t_stop, 1),
        "frames": int(solution.n_steps),
    }
    result["protocol"] = args.protocol
    result["ramp_fraction"] = args.ramp
    result["bz_final_mT"] = args.bz_final_mT
    result.update(summarise(solution, spec, args.out))
    result.update(vortex_census(solution, device, spec, -1))

    if args.gif:
        if args.protocol == "field-cool":
            def field_of_t(t):
                return field_func(t, args.t_stop)[2] * units.field_unit_mT
        elif args.protocol == "ramp-up" and args.ramp:
            def field_of_t(t):
                return args.bz_mT * min(t / (args.t_stop * args.ramp), 1.0)
        else:
            def field_of_t(t):
                return args.bz_mT

        gif_path = args.out / "nb_hole_array.gif"
        t0 = time.perf_counter()
        write_gif(solution, device, spec, gif_path, field_of_t, fps=args.fps)
        result["gif"] = str(gif_path)
        result["gif_seconds"] = round(time.perf_counter() - t0, 1)

    h5_path = args.out / "nb_hole_array.h5"
    solution.save(str(h5_path))
    result["hdf5"] = str(h5_path)
    print(json.dumps(result, indent=2))

    if result["pair_broken"]:
        raise SystemExit(
            "max|ψ| < 1e-2: the superconductor is fully pair-broken — the layers "
            "are too thin at this ξ for anything phase-derived to be trusted."
        )


if __name__ == "__main__":
    main()

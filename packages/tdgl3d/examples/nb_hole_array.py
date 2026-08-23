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

    python3 packages/tdgl3d/examples/nb_hole_array.py --dry-run   # cost estimate only
    python3 packages/tdgl3d/examples/nb_hole_array.py --xi 150 --t-stop 20
    python3 packages/tdgl3d/examples/nb_hole_array.py --bz-mT 5.0 --out run5mT
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


def build(spec: dict, bz_mT: float, t_on_fraction: float) -> tdgl3d.Device:
    """The trilayer with all nine holes carved through it."""
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
            Bz=units.field(bz_mT), t_on_fraction=t_on_fraction
        ),
        trilayer=trilayer,
    )

    pitch = spec["hole_xi"] + spec["gap_xi"]
    for row in range(N_HOLES):
        for col in range(N_HOLES):
            x0 = spec["buffer_xi"] + col * pitch
            y0 = spec["buffer_xi"] + row * pitch
            x1, y1 = x0 + spec["hole_xi"], y0 + spec["hole_xi"]
            device.add_hole(
                [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], z_range=(0, trilayer.Nz)
            )
    return device


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
    device = build(spec, args.bz_mT, args.t_on)
    print(f"device built in {time.perf_counter() - t0:.1f} s")

    t0 = time.perf_counter()
    solution = tdgl3d.solve(
        device,
        t_stop=args.t_stop,
        dt=dt,
        method=args.method,
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
    result.update(summarise(solution, spec, args.out))

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

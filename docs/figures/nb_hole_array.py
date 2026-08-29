"""Flux trapped in a 3x3 array of 4 µm holes in an S/I/S Nb stack.

The device is what lithography fixes rather than what the grid would prefer:
nine 4 µm square holes on an 8 µm pitch, with an 8 µm buffer of unbroken film
around the array, through a 500/500/500 nm S/I/S stack.  That is 36 µm across,
which at ξ = 150 nm is a 240 x 240 x 9 grid — 457 k interior nodes.

Two figures, from one run each:

``nb_hole_array_entry.png``
    Ramping the field up.  Nothing enters until ~3.15 mT and just above it
    hundreds enter at once; held at 3.6 mT the vortex lattice packs the buffer
    while the array stays fully Meissner-screened behind it.  The flux front
    stalls at the array perimeter and never reaches a hole.

``nb_hole_array_trapped.png`` / ``.gif``
    Field-cooling instead.  ψ grows from near zero with the field already on,
    so flux is trapped where it is rather than having to cross 8 µm of
    screening metal; the field then drops below the entry threshold and the
    state settles with quantised flux in the holes and a couple of vortices in
    the metal between them.  The exact count depends on the noise seed — 3 and
    7 quanta here, 2 and 6 at another seed — so the run prints its census over
    time rather than only the last frame.

Full resolution takes about 25 min per run on four cores.  ``small=True`` runs
a quarter-size device for the smoke test.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle
from tdgl3d import AppliedField, Device, SimulationParameters, solve
from tdgl3d.analysis.vortex_counting import (
    count_vortices_plaquette,
    count_vortices_polygon,
)
from tdgl3d.core.material import Layer, Trilayer
from tdgl3d.core.units import GLUnits

KAPPA = 2.0
HOLE_NM = 4000.0
GAP_NM = 4000.0
BUFFER_NM = 8000.0
LAYER_NM = 500.0
N_HOLES = 3

#: 0.9 of the 3-D Forward-Euler limit h²/(4κ²(d−1)).
DT = 0.9 / (4 * KAPPA**2 * 2)

#: Field the ramp climbs to, and where the field-cool protocol starts and ends.
#: Entry begins at 3.15 mT; 2.0 mT is below that, so nothing new enters while
#: the cooled state settles.
RAMP_TO_MT = 3.6
COOL_AT_MT = 4.0
HOLD_AT_MT = 2.0


def _geometry(xi_nm: float, scale: float):
    """Grid, hole rectangles and unit conversion for this device at *xi_nm*."""
    units = GLUnits(xi_nm=xi_nm, kappa=KAPPA)
    hole = units.length(HOLE_NM) * scale
    gap = units.length(GAP_NM) * scale
    buffer_xi = units.length(BUFFER_NM) * scale
    side = N_HOLES * hole + (N_HOLES - 1) * gap + 2 * buffer_xi
    n_side = int(round(side))
    n_layer = max(int(round(units.length(LAYER_NM) * scale)), 1)

    pitch = hole + gap
    rects = [
        (buffer_xi + col * pitch, buffer_xi + row * pitch,
         buffer_xi + col * pitch + hole, buffer_xi + row * pitch + hole)
        for row in range(N_HOLES) for col in range(N_HOLES)
    ]
    return units, side, n_side, n_layer, rects, (buffer_xi, buffer_xi + side - 2 * buffer_xi)


def _build(xi_nm, scale, field):
    units, side, n_side, n_layer, rects, array_span = _geometry(xi_nm, scale)
    trilayer = Trilayer(
        bottom=Layer(thickness_z=n_layer, kappa=KAPPA, is_superconductor=True),
        # A non-superconducting layer still needs κ > 0: at κ = 0 its φ-equation
        # degenerates and the oxide blocks the field instead of transmitting it.
        insulator=Layer(thickness_z=n_layer, kappa=KAPPA, is_superconductor=False),
        top=Layer(thickness_z=n_layer, kappa=KAPPA, is_superconductor=True),
    )
    h = side / n_side
    params = SimulationParameters(
        Nx=n_side, Ny=n_side, Nz=trilayer.Nz, hx=h, hy=h, hz=h, kappa=KAPPA
    )
    device = Device(params, applied_field=field, trilayer=trilayer)
    for x0, y0, x1, y1 in rects:
        device.add_hole([(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
                        z_range=(0, trilayer.Nz))
    return units, device, n_layer, rects, side


def _sc_slice(n_layer: int) -> int:
    """Interior z-slice strictly inside the bottom superconductor.

    Interior arrays run k = 1 … Nz-1, so interior slice s is full-grid node
    s + 1; the bottom layer owns nodes k < n_layer, both oxide interfaces
    belonging to the insulator.
    """
    return max(min(n_layer // 2, n_layer - 2), 0)


def _census(solution, spec, step, margin=2.0):
    """Vortices in the metal (split by region) and the fluxoid in each hole.

    A vortex in the film is a core — a plaquette carrying 2π of gauge-invariant
    winding.  A hole has no core: what it holds is a fluxoid, read from the
    winding on a contour in the metal around it, which is an exact integer
    however little field threads the opening.
    """
    n_layer, rects, array_lo, array_hi = spec
    slice_z = _sc_slice(n_layer)

    _, positions, windings = count_vortices_plaquette(
        solution, None, slice_z=slice_z, step=step
    )
    in_array = in_buffer = 0
    for (px, py), winding in zip(positions, windings):
        node_x, node_y = px + 1.0, py + 1.0
        if any(x0 - margin <= node_x <= x1 + margin
               and y0 - margin <= node_y <= y1 + margin for x0, y0, x1, y1 in rects):
            continue
        charge = abs(int(round(winding)))
        if array_lo <= node_x <= array_hi and array_lo <= node_y <= array_hi:
            in_array += charge
        else:
            in_buffer += charge

    trapped = []
    for x0, y0, x1, y1 in rects:
        contour = np.array([
            [x0 - margin, y0 - margin], [x1 + margin, y0 - margin],
            [x1 + margin, y1 + margin], [x0 - margin, y1 + margin],
        ])
        trapped.append(int(round(count_vortices_polygon(
            solution, None, contour, slice_z=slice_z, step=step
        ))))
    return {"in_array": in_array, "in_buffer": in_buffer, "holes": trapped,
            "quanta": int(sum(abs(n) for n in trapped))}


def _panels(solution, units, n_layer, rects, side, side_um, title, path):
    """|ψ|² beside B_z, with the holes outlined."""
    slice_z = _sc_slice(n_layer)
    per_xi = side_um / side
    extent = [0, side_um, 0, side_um]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), constrained_layout=True)
    psi2 = solution.psi_squared_2d(-1, slice_z=slice_z)
    im0 = axes[0].imshow(psi2.T, origin="lower", extent=extent, cmap="inferno",
                         vmin=0.0, vmax=1.0)
    axes[0].set_title(r"$|\psi|^2$, mid-plane of the bottom Nb layer")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.02)

    bz = np.asarray(solution.bfield(-1)[2]).reshape(-1)
    n_side = solution.params.Nx - 1
    bz_slice = bz.reshape(n_side, n_side, -1)[:, :, slice_z]
    limit = float(np.max(np.abs(bz_slice))) or 1.0
    im1 = axes[1].imshow(units.field_to_mT(bz_slice).T, origin="lower",
                         extent=extent, cmap="coolwarm",
                         vmin=-units.field_to_mT(limit),
                         vmax=units.field_to_mT(limit))
    axes[1].set_title(r"$B_z$ (mT)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.02)

    for ax in axes:
        for x0, y0, x1, y1 in rects:
            ax.add_patch(Rectangle((x0 * per_xi, y0 * per_xi),
                                   (x1 - x0) * per_xi, (y1 - y0) * per_xi,
                                   fill=False, edgecolor="#7fd4ff", lw=0.7, alpha=0.6))
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")
    fig.suptitle(title)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _animate(solution, spec, units, side, side_um, rects, field_of_t, path, fps=10):
    """|ψ|² through the run, each hole labelled with the fluxoid it holds.

    The label is the point: a hole shows no core and no dark spot, so without
    the number there is nothing on screen to say what it is holding.
    """
    n_layer = spec[0]
    slice_z = _sc_slice(n_layer)
    per_xi = side_um / side
    extent = [0, side_um, 0, side_um]
    census = [_census(solution, spec, k) for k in range(solution.n_steps)]

    fig, ax = plt.subplots(figsize=(6.6, 6.0), constrained_layout=True)
    image = ax.imshow(solution.psi_squared_2d(0, slice_z=slice_z).T, origin="lower",
                      extent=extent, cmap="inferno", vmin=0.0, vmax=1.0)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02, label=r"$|\psi|^2$")

    labels = []
    for x0, y0, x1, y1 in rects:
        ax.add_patch(Rectangle((x0 * per_xi, y0 * per_xi),
                               (x1 - x0) * per_xi, (y1 - y0) * per_xi,
                               fill=False, edgecolor="#7fd4ff", lw=0.8, alpha=0.7))
        labels.append(ax.text(0.5 * (x0 + x1) * per_xi, 0.5 * (y0 + y1) * per_xi, "",
                              color="#7fd4ff", ha="center", va="center",
                              fontsize=13, fontweight="bold"))
    title = ax.set_title("")

    def update(step):
        image.set_data(solution.psi_squared_2d(step, slice_z=slice_z).T)
        counts = census[step]
        for label, n in zip(labels, counts["holes"]):
            label.set_text(str(n) if n else "")
        t = float(solution.times[step])
        title.set_text(
            f"t = {t:6.1f} τ$_{{GL}}$    B = {field_of_t(t):.2f} mT    "
            f"{counts['in_array']} between holes, {counts['quanta']} trapped in them"
        )
        return [image, title, *labels]

    FuncAnimation(fig, update, frames=solution.n_steps, blit=False).save(
        str(path), writer=PillowWriter(fps=fps)
    )
    plt.close(fig)
    return census[-1]


def main(output_dir: Path = Path(__file__).parent, small: bool = False) -> list[Path]:
    output_dir = Path(output_dir)
    if small:
        xi_nm, scale, t_ramp, t_cool, frames = 150.0, 0.25, 1.5, 2.0, 4
    else:
        xi_nm, scale, t_ramp, t_cool, frames = 150.0, 1.0, 250.0, 400.0, 60
    saved: list[Path] = []

    # --- 1. Ramping the field up: the flux front stalls at the array ---------
    units, device, n_layer, rects, side = _build(
        xi_nm, scale,
        AppliedField(Bz=GLUnits(xi_nm=xi_nm, kappa=KAPPA).field(RAMP_TO_MT),
                     ramp=True, ramp_fraction=0.2, t_on_fraction=1.0),
    )
    side_um = units.length_nm(side) / 1000.0
    solution = solve(device, t_stop=t_ramp, dt=DT, method="euler",
                     save_every=max(int(round(t_ramp / DT)) // frames, 1),
                     noise_seed=17, progress=False, log_metadata=False)
    saved.append(_panels(
        solution, units, n_layer, rects, side, side_um,
        f"Field ramped to {RAMP_TO_MT:g} mT — the lattice fills the buffer, "
        "the array stays screened",
        output_dir / "nb_hole_array_entry.png",
    ))

    # --- 2. Field-cooling: flux locked into the holes ------------------------
    cool = units.field(COOL_AT_MT)
    hold = units.field(HOLD_AT_MT)
    t_lo, t_hi = 0.14 * t_cool, 0.24 * t_cool

    def schedule(t, _t_stop):
        if t <= t_lo:
            return 0.0, 0.0, cool
        if t >= t_hi:
            return 0.0, 0.0, hold
        return 0.0, 0.0, cool + (t - t_lo) / (t_hi - t_lo) * (hold - cool)

    units, device, n_layer, rects, side = _build(
        xi_nm, scale, AppliedField(field_func=schedule)
    )
    # ψ from near zero with the field already on — the numerical stand-in for
    # cooling through T_c.  Starting from the formed condensate makes the film
    # screen from the first step, and flux can then only enter from the outside
    # edge, which on a 36 µm film means it never reaches the array.
    state = device.initial_state(noise_amplitude=0.0, seed=31)
    rng = np.random.default_rng(31)
    state.psi[:] *= 0.02 * np.exp(
        1j * rng.uniform(0.0, 2.0 * np.pi, device.params.n_interior)
    )

    solution = solve(device, t_stop=t_cool, dt=DT, method="euler", x0=state,
                     save_every=max(int(round(t_cool / DT)) // frames, 1),
                     noise_seed=31, progress=False, log_metadata=False)

    array_lo = rects[0][0]
    array_hi = rects[-1][2]
    spec = (n_layer, rects, array_lo, array_hi)

    # The census over time is what says the state is settled rather than caught
    # mid-transient: a hole that fills and then empties again looks identical,
    # at the last frame, to one that never filled.
    print("    t   between   buffer   quanta   per hole")
    for step in range(solution.n_steps):
        c = _census(solution, spec, step)
        print(f"{float(solution.times[step]):7.1f} {c['in_array']:7d} "
              f"{c['in_buffer']:8d} {c['quanta']:8d}   {c['holes']}")
    counts = _census(solution, spec, -1)
    saved.append(_panels(
        solution, units, n_layer, rects, side, side_um,
        f"Field-cooled at {COOL_AT_MT:g} mT, held at {HOLD_AT_MT:g} mT — "
        f"{counts['in_array']} vortices between the holes, "
        f"{counts['quanta']} flux quanta trapped in them",
        output_dir / "nb_hole_array_trapped.png",
    ))

    gif_path = output_dir / "nb_hole_array_trapped.gif"
    _animate(solution, spec, units, side, side_um, rects,
             lambda t: units.field_to_mT(schedule(t, t_cool)[2]), gif_path)
    saved.append(gif_path)
    return saved


if __name__ == "__main__":
    for path in main():
        print(f"wrote {path}")

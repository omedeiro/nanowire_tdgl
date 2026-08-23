"""Cross-sections of B and |ψ| against closed-form Ginzburg-Landau solutions.

Two limits of the coupled equations have exact solutions, and between them they
exercise both equations separately:

* **London limit** (|ψ| = 1, so the ψ-equation drops out): a square with the
  field pinned on its boundary obeys ``∇²B = B/λ²``, whose exact Fourier
  solution is :func:`tdgl3d.london_square_2d`.
* **Pair-breaking wall** (zero field, so the gauge field drops out): against an
  insulator, ``ψ'' = -ψ + ψ³`` gives ``tanh((x - x₀)/√2)`` with the offset fixed
  by matching to the insulator relaxation — no free parameters
  (:func:`tdgl3d.gl_wall_profile`).

Each is run at three grid spacings so the residual can be shown to be
discretisation error rather than disagreement. The bottom row applies the same
two models to the micron-scale S/I/S ring, where neither holds exactly.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tdgl3d import (
    AppliedField,
    Device,
    GLUnits,
    Layer,
    SimulationParameters,
    Trilayer,
    gl_wall_profile,
    london_square_2d,
)
from tdgl3d.physics.analytic import london_domain_width, plaquette_positions
from tdgl3d.physics.applied_field import build_boundary_field_vectors
from tdgl3d.physics.bfield import eval_bfield_full
from tdgl3d.physics.rhs import (
    BoundaryVectors,
    _apply_boundary_conditions,
    _expand_interior_to_full,
)
from tdgl3d.solvers.integrators import forward_euler

KAPPA = 2.0
SPACINGS = (1.0, 0.5, 0.25)
COLOURS = {1.0: "C0", 0.5: "C1", 0.25: "C2"}


def _dt(params):
    h_min = min(params.hx, params.hy, params.hz)
    return 0.9 * h_min**2 / (4.0 * params.kappa**2 * (2.0 if params.is_3d else 1.0))


def _relax(params, device, boundary, t_stop):
    _, states = forward_euler(
        device.initial_state(noise_amplitude=0.0).data, params, device.idx,
        lambda t, X: boundary, 0.0, t_stop, _dt(params),
        save_every=10**9, progress=False, material=device.material,
    )
    return states[:, -1]


def _expand(state, params, device, boundary):
    n = params.n_interior
    psi = _expand_interior_to_full(state[:n], params, device.idx)
    phi = [
        _expand_interior_to_full(state[(k + 1) * n : (k + 2) * n], params, device.idx)
        for k in range(3 if params.is_3d else 2)
    ]
    while len(phi) < 3:
        phi.append(np.zeros(params.dim_x, dtype=np.complex128))
    return _apply_boundary_conditions(psi, phi[0], phi[1], phi[2], params, device.idx, boundary)


# ---------------------------------------------------------------------------
# London limit
# ---------------------------------------------------------------------------


def london_case(h: float, length: float = 16.0, bz: float = 0.02, t_stop: float = 15.0):
    """Mid-line Bz cross-section of a square film in a weak perpendicular field."""
    n_cells = int(round(length / h))
    params = SimulationParameters(Nx=n_cells, Ny=n_cells, Nz=1, hx=h, hy=h, kappa=KAPPA)
    device = Device(params, applied_field=AppliedField(Bz=bz, t_on_fraction=1.0))
    boundary = BoundaryVectors(
        *build_boundary_field_vectors(0.0, 0.0, bz, params, device.idx)
    )
    state = _relax(params, device, boundary, t_stop)
    _, phi_x, phi_y, phi_z = _expand(state, params, device, boundary)

    nx_int, ny_int = params.Nx - 1, params.Ny - 1
    field = eval_bfield_full(phi_x, phi_y, phi_z, params, device.idx)[2].reshape(nx_int, ny_int)

    xs = plaquette_positions(params, "x")
    width = london_domain_width(params, "x")
    mid = ny_int // 2
    simulated = field[:, mid]
    model = london_square_2d(xs, np.full_like(xs, xs[mid]), width, lam=KAPPA, b0=bz)

    psi_min = float(np.abs(state[: params.n_interior]).min())
    return {
        "h": h, "x": xs, "width": width, "b0": bz,
        "sim": simulated, "model": model,
        "error": simulated - model, "psi_min": psi_min,
    }


# ---------------------------------------------------------------------------
# Pair-breaking wall
# ---------------------------------------------------------------------------


def wall_case(h: float, length: float = 24.0, wall: float = 8.0, t_stop: float = 40.0):
    """|ψ| cross-section running out of a half-plane hole into the bulk."""
    n_cells = int(round(length / h))
    params = SimulationParameters(Nx=n_cells, Ny=6, Nz=1, hx=h, hy=h, kappa=KAPPA)
    device = Device(params, applied_field=AppliedField(Bz=0.0))
    device.add_hole(
        [(-1.0, -1.0), (wall, -1.0), (wall, length + 1.0), (-1.0, length + 1.0)]
    )
    zeros = np.zeros(params.dim_x, dtype=np.float64)
    boundary = BoundaryVectors(zeros, zeros.copy(), zeros.copy())
    state = _relax(params, device, boundary, t_stop)

    nx_int, ny_int = params.Nx - 1, params.Ny - 1
    psi = np.abs(state[: params.n_interior]).reshape(nx_int, ny_int)
    mask = device.material.interior_sc_mask.reshape(nx_int, ny_int)
    row = ny_int // 2
    profile, is_sc = psi[:, row], mask[:, row] > 0

    xs = np.arange(1, params.Nx) * h
    # The material coefficient jumps *between* nodes, so the effective interface
    # is the midpoint of the last insulator node and the first superconducting
    # one.  Anchoring on either node instead costs a factor of h in accuracy.
    interface = 0.5 * (xs[~is_sc].max() + xs[is_sc].min())
    offsets = xs - interface
    model = gl_wall_profile(offsets)

    window = (offsets >= -1.5) & (offsets <= 8.0)
    return {
        "h": h, "x": offsets[window], "sim": profile[window], "model": model[window],
        "error": profile[window] - model[window],
    }


# ---------------------------------------------------------------------------
# The micron-scale ring, where neither model holds exactly
# ---------------------------------------------------------------------------


def ring_case(units: GLUnits, applied_bz: float = 0.25, t_stop: float = 60.0):
    """Cut through the middle of the 1 µm hole / 4 µm plane device."""
    plane, hole = units.length(4000.0), units.length(1000.0)
    trilayer = Trilayer(
        bottom=Layer(thickness_z=5, kappa=units.kappa),
        insulator=Layer(thickness_z=2, kappa=units.kappa, is_superconductor=False),
        top=Layer(thickness_z=5, kappa=units.kappa),
    )
    n_cells = int(round(plane))
    params = SimulationParameters(
        Nx=n_cells, Ny=n_cells, Nz=trilayer.Nz, hx=1.0, hy=1.0, hz=1.0, kappa=units.kappa
    )
    device = Device(
        params, applied_field=AppliedField(Bz=applied_bz, t_on_fraction=1.0), trilayer=trilayer
    )
    lo, hi = 0.5 * (plane - hole), 0.5 * (plane + hole)
    square = [(lo, lo), (hi, lo), (hi, hi), (lo, hi)]
    ranges = trilayer.z_ranges()
    device.add_hole(square, z_range=ranges["bottom"])
    device.add_hole(square, z_range=ranges["top"])

    boundary = BoundaryVectors(
        *build_boundary_field_vectors(0.0, 0.0, applied_bz, params, device.idx)
    )
    state = _relax(params, device, boundary, t_stop)
    _, phi_x, phi_y, phi_z = _expand(state, params, device, boundary)

    nx, ny, nz = params.Nx - 1, params.Ny - 1, params.Nz - 1
    slice_z = max(ranges["bottom"][1] // 2 - 1, 0)
    psi = np.abs(state[: params.n_interior]).reshape(nx, ny, nz)[:, ny // 2, slice_z]
    field = eval_bfield_full(phi_x, phi_y, phi_z, params, device.idx)[2]
    field = field.reshape(nx, ny, nz)[:, ny // 2, slice_z]
    mask = device.material.interior_sc_mask.reshape(nx, ny, nz)[:, ny // 2, slice_z] > 0

    xs = np.arange(1, params.Nx) * params.hx
    # Wall model anchored on the low-x edge of the hole.
    hole_x = xs[~mask]
    interface = 0.5 * (xs[xs < hole_x.min()].max() + hole_x.min())
    wall_model = gl_wall_profile(interface - xs)  # superconductor at x < interface

    return {
        "x_um": units.length_nm(xs) / 1000.0,
        "psi": psi, "field": field, "mask": mask,
        "wall_model": wall_model,
        "interface_um": units.length_nm(interface) / 1000.0,
        "applied_bz": applied_bz, "units": units,
        "hole_lo_um": units.length_nm(hole_x.min()) / 1000.0,
        "hole_hi_um": units.length_nm(hole_x.max()) / 1000.0,
    }


def _order(spacings, errors):
    """Observed order of accuracy from the rms errors.

    Returns ``nan`` for the order when there is only one spacing (the smoke-test
    path), since a single point fixes no slope.
    """
    hs = np.array(spacings, dtype=float)
    rms = np.array([np.sqrt(np.mean(e**2)) for e in errors])
    if hs.size < 2:
        return float("nan"), rms
    return float(np.polyfit(np.log(hs), np.log(rms), 1)[0]), rms


def main(output_dir: Path = Path(__file__).parent, small: bool = False) -> list[Path]:
    units = GLUnits(xi_nm=100.0, kappa=KAPPA)
    if small:
        spacings, ring_t = (1.0,), 1.0
    else:
        spacings, ring_t = SPACINGS, 60.0

    london = [london_case(h) for h in spacings]
    walls = [wall_case(h) for h in spacings]
    ring = ring_case(units, t_stop=ring_t)

    london_order, london_rms = _order(spacings, [c["error"] / c["b0"] for c in london])
    wall_order, wall_rms = _order(spacings, [c["error"] for c in walls])

    fig, axes = plt.subplots(3, 2, figsize=(13, 13.5))

    # -- row 0: London ------------------------------------------------------
    ax = axes[0, 0]
    finest = london[-1]
    ax.plot(finest["x"] / finest["width"],
            london_square_2d(finest["x"], np.full_like(finest["x"], 0.5 * finest["width"]),
                             finest["width"], KAPPA, finest["b0"]) / finest["b0"],
            "-", color="k", linewidth=2.0, alpha=0.5, label="exact London series")
    for case in london:
        ax.plot(case["x"] / case["width"], case["sim"] / case["b0"], "o",
                color=COLOURS[case["h"]], markersize=3.5, label=f"solver, h = {case['h']:g} ξ")
    ax.set_yscale("log")
    ax.set_xlabel("x / W  (mid-line of the square)")
    ax.set_ylabel("Bz / B₀")
    ax.set_title(f"London limit: ∇²B = B/λ², λ = κ = {KAPPA:g} ξ")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    ax = axes[0, 1]
    for case in london:
        ax.plot(case["x"] / case["width"], 100 * case["error"] / case["b0"], "o-",
                color=COLOURS[case["h"]], markersize=3, linewidth=1,
                label=f"h = {case['h']:g} ξ")
    ax.axhline(0, color="gray", ls=":", alpha=0.7)
    ax.set_xlabel("x / W")
    ax.set_ylabel("(solver − model) / B₀   (%)")
    ax.set_title("Residual shrinks with the grid, so it is discretisation error")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.text(
        0.03, 0.03,
        "rms error / B₀\n"
        + "\n".join(f"  h = {h:<5g} {r:.2e}" for h, r in zip(spacings, london_rms))
        + f"\norder in h: {london_order:.2f}",
        transform=ax.transAxes, fontsize=7.5, fontfamily="monospace",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9),
    )

    # -- row 1: pair-breaking wall -----------------------------------------
    ax = axes[1, 0]
    dense = np.linspace(-1.5, 8.0, 400)
    ax.plot(dense, gl_wall_profile(dense), "-", color="k", linewidth=2.0, alpha=0.5,
            label="exact  tanh((x − x₀)/√2)")
    for case in walls:
        ax.plot(case["x"], case["sim"], "o", color=COLOURS[case["h"]], markersize=3.5,
                label=f"solver, h = {case['h']:g} ξ")
    ax.axvline(0.0, color="C3", ls="--", linewidth=1.2, alpha=0.7, label="interface")
    ax.set_xlabel("distance from the interface (ξ)")
    ax.set_ylabel("|ψ|")
    ax.set_title("Pair-breaking wall: healing length √2 ξ, no free parameters")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    for case in walls:
        ax.plot(case["x"], case["error"], "o-", color=COLOURS[case["h"]],
                markersize=3, linewidth=1, label=f"h = {case['h']:g} ξ")
    ax.axhline(0, color="gray", ls=":", alpha=0.7)
    ax.axvline(0.0, color="C3", ls="--", linewidth=1.2, alpha=0.7)
    ax.set_xlabel("distance from the interface (ξ)")
    ax.set_ylabel("solver − model   (|ψ|)")
    ax.set_title("Residual is concentrated at the interface")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.text(
        0.97, 0.03,
        "rms error\n"
        + "\n".join(f"  h = {h:<5g} {r:.2e}" for h, r in zip(spacings, wall_rms))
        + f"\norder in h: {wall_order:.2f}",
        transform=ax.transAxes, fontsize=7.5, fontfamily="monospace",
        verticalalignment="bottom", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9),
    )

    # -- row 2: the real device --------------------------------------------
    ax = axes[2, 0]
    ax.axvspan(ring["hole_lo_um"], ring["hole_hi_um"], color="gray", alpha=0.18)
    ax.plot(ring["x_um"], ring["psi"], "o-", color="C0", markersize=3, linewidth=1,
            label="|ψ| (solver)")
    # The wall model describes the left arm running into the hole; drawing it
    # across the whole cut would imply it says something about the right arm.
    left = ring["x_um"] <= ring["hole_hi_um"]
    ax.plot(ring["x_um"][left], ring["wall_model"][left], "--", color="k", alpha=0.8,
            label="wall model (left hole edge)")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("|ψ|")
    ax.set_ylim(-0.05, 1.15)
    twin = ax.twinx()
    twin.plot(ring["x_um"], ring["units"].field_to_mT(ring["field"]), "s-", color="C3",
              markersize=3, linewidth=1, label="Bz (solver)")
    twin.set_ylabel("Bz (mT)", color="C3")
    twin.tick_params(axis="y", labelcolor="C3")
    ax.set_title(
        f"Micron ring cut, Bz = {ring['units'].field_to_mT(ring['applied_bz']):.2f} mT\n"
        "(shaded: the hole)", fontsize=11,
    )
    handles, labels = ax.get_legend_handles_labels()
    h2, l2 = twin.get_legend_handles_labels()
    ax.legend(handles + h2, labels + l2, fontsize=8, loc="center left")
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    arm = ring["x_um"] <= ring["hole_lo_um"]
    ax.plot(ring["x_um"][arm], (ring["psi"] - ring["wall_model"])[arm], "o-",
            color="C0", markersize=3, linewidth=1, label="|ψ| − wall model")
    ax.axhline(0, color="gray", ls=":", alpha=0.7)
    ax.set_xlim(ring["x_um"][0], ring["hole_hi_um"])
    ax.axvspan(ring["hole_lo_um"], ring["hole_hi_um"], color="gray", alpha=0.18)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("solver − model   (|ψ|)")
    ax.set_title("Where the 1-D model stops applying", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.text(
        0.03, 0.03,
        f"|ψ| plateau in the arm: {ring['psi'][arm].max():.3f}\n\n"
        "the wall model is a *zero-field, semi-\n"
        "infinite* solution asymptoting to 1.\n"
        "Two effects it cannot contain:\n"
        "  • the screening current suppresses ψ\n"
        "    at the outer edge (left of the plot)\n"
        "  • the oxide suppresses ψ from below,\n"
        "    capping the plateau under 1\n"
        "Within ~1 ξ of the hole edge, where\n"
        "neither dominates, it holds.",
        transform=ax.transAxes, fontsize=7.5, fontfamily="monospace",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9),
    )

    fig.suptitle(
        "Cross-sections against closed-form Ginzburg-Landau solutions",
        fontsize=15, y=0.995,
    )
    fig.tight_layout()
    out = output_dir / "analytic_cross_sections.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("London limit (Bz on the mid-line of a square):")
    for case, rms in zip(london, london_rms):
        print(f"  h = {case['h']:<5g} rms/B₀ = {rms:.3e}  "
              f"max/B₀ = {np.max(np.abs(case['error'])) / case['b0']:.3e}  "
              f"min|ψ| = {case['psi_min']:.4f}")
    print(f"  observed order in h: {london_order:.2f}")
    print("Pair-breaking wall (|ψ| across a hole edge):")
    for case, rms in zip(walls, wall_rms):
        print(f"  h = {case['h']:<5g} rms = {rms:.3e}  "
              f"max = {np.max(np.abs(case['error'])):.3e}")
    print(f"  observed order in h: {wall_order:.2f}")
    return [out]


if __name__ == "__main__":
    main()

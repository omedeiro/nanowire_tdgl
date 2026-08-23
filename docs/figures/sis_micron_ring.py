"""A micron-scale S/I/S ring: 1 µm hole in a 4 µm plane, 500 nm layers.

Specified in SI and converted through :class:`tdgl3d.GLUnits`.  With
ξ = 100 nm — Nb (ξ₀ ≈ 38 nm) at T/T_c ≈ 0.86, comfortably inside the
Ginzburg-Landau regime — the device is 40 ξ across with a 10 ξ hole and 5 ξ
layers, and one unit of the solver's field is 32.9 mT.

The result is *not* the small-ring result scaled up.  A 4 µm plane is 20 λ
across and screens almost completely, so the field reaching the hole is a
percent or two of the applied field and the hole is nowhere near its
fluxoid limit.  What fails first is vortex penetration into the 1.5 µm-wide
superconducting arms.  The expulsion field of this device is therefore set by
the plane, not by the hole.

Runtime: about 12 minutes for the full scan (each field point is ~65 s on a
40×40×12 grid).  Not part of the gallery's regenerate-everything loop.
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
)
from tdgl3d.analysis.expulsion import (
    expulsion_field,
    first_entry_time,
    fluxoid_history,
    rectangular_contour,
)
from tdgl3d.analysis.vortex_counting import plaquette_vorticity
from tdgl3d.core.solution import Solution
from tdgl3d.physics.applied_field import build_boundary_field_vectors
from tdgl3d.physics.bfield import eval_bfield_full
from tdgl3d.physics.rhs import (
    BoundaryVectors,
    _apply_boundary_conditions,
    _expand_interior_to_full,
    eval_f,
)
from tdgl3d.solvers.integrators import forward_euler

# Nb at T/T_c ≈ 0.86.  ξ(T) = ξ₀/sqrt(1 − T/T_c) with ξ₀ = 38 nm gives 100 nm;
# κ = λ/ξ is temperature-independent and ≈ 2 for a moderately dirty Nb film.
UNITS = GLUnits(xi_nm=100.0, kappa=2.0)

PLANE_NM = 4000.0
HOLE_NM = 1000.0
LAYER_NM = 500.0
OXIDE_NM = 200.0
H = 1.0  # in-plane and out-of-plane grid spacing, in ξ

# A perturbation is seeded so that flux entry is triggered by the physics rather
# than by round-off.  With a C4-symmetric device and a noiseless start the
# Meissner state is an *exact* fixed point (residual ~1e-14), and any
# instability has to grow out of 1e-16 — which delays entry by an amount set by
# floating-point precision, not by the barrier.  The threshold below is checked
# against a perturbation a thousand times smaller.
NOISE = 1e-3
SEED = 1


def _cells(nm: float) -> int:
    return int(round(UNITS.length(nm) / H))


def build_device(applied_bz: float):
    """The S/I/S ring, with the hole carved through both metal layers."""
    plane = UNITS.length(PLANE_NM)
    hole = UNITS.length(HOLE_NM)
    trilayer = Trilayer(
        bottom=Layer(thickness_z=_cells(LAYER_NM), kappa=UNITS.kappa),
        insulator=Layer(
            # A non-superconducting layer still needs κ > 0: at κ = 0 its
            # φ-equation degenerates and the oxide blocks the field instead of
            # transmitting it.
            thickness_z=_cells(OXIDE_NM), kappa=UNITS.kappa, is_superconductor=False,
        ),
        top=Layer(thickness_z=_cells(LAYER_NM), kappa=UNITS.kappa),
    )
    n_cells = int(round(plane / H))
    params = SimulationParameters(
        Nx=n_cells, Ny=n_cells, Nz=trilayer.Nz,
        hx=H, hy=H, hz=H, kappa=UNITS.kappa,
    )
    device = Device(
        params,
        applied_field=AppliedField(Bz=applied_bz, t_on_fraction=1.0),
        trilayer=trilayer,
    )
    lo, hi = 0.5 * (plane - hole), 0.5 * (plane + hole)
    square = [(lo, lo), (hi, lo), (hi, hi), (lo, hi)]
    z_ranges = trilayer.z_ranges()
    device.add_hole(square, z_range=z_ranges["bottom"])
    device.add_hole(square, z_range=z_ranges["top"])
    return params, device, trilayer, (lo, hi, lo, hi)


def relax(params, device, applied_bz, t_stop, noise=NOISE, n_save=12):
    idx = device.idx
    boundary = BoundaryVectors(
        *build_boundary_field_vectors(0.0, 0.0, applied_bz, params, idx)
    )
    # dt < h²/(4κ²(d−1)): the familiar h²/(4κ²) is the 2-D bound and halves in 3-D.
    h_min = min(params.hx, params.hy, params.hz)
    dt = 0.9 * h_min**2 / (4.0 * params.kappa**2 * (2.0 if params.is_3d else 1.0))
    times, states = forward_euler(
        device.initial_state(noise_amplitude=noise, seed=SEED).data,
        params, idx, lambda t, X: boundary, 0.0, t_stop, dt,
        save_every=max(1, int(t_stop / dt / n_save)),
        progress=False, material=device.material,
    )
    solution = Solution(times=times, states=states, params=params, idx=idx, device=device)
    return solution, boundary


def measure(applied_bz: float, t_stop: float, noise: float = NOISE) -> dict:
    """Relax at one applied field and report everything the study needs."""
    params, device, trilayer, hole_bounds = build_device(applied_bz)
    solution, boundary = relax(params, device, applied_bz, t_stop, noise=noise)

    slice_z = max(trilayer.z_ranges()["bottom"][1] // 2 - 1, 0)
    contour = rectangular_contour(hole_bounds, params, margin=3.0)
    history = fluxoid_history(solution, device, contour, slice_z=slice_z)

    vorticity, psi2_min = plaquette_vorticity(solution, slice_z=slice_z, step=-1)
    charged = np.rint(vorticity).astype(int)
    resolved = psi2_min >= 1e-6
    n_vortices = int(np.abs(charged[resolved]).sum())

    nx, ny, nz = params.Nx - 1, params.Ny - 1, params.Nz - 1
    n = params.n_interior
    final = solution.states[:, -1]
    psi_full = _expand_interior_to_full(final[:n], params, device.idx)
    phi = [
        _expand_interior_to_full(final[(k + 1) * n : (k + 2) * n], params, device.idx)
        for k in range(3)
    ]
    _, phi_x, phi_y, phi_z = _apply_boundary_conditions(
        psi_full, phi[0], phi[1], phi[2], params, device.idx, boundary
    )
    field = eval_bfield_full(phi_x, phi_y, phi_z, params, device.idx)[2].reshape(nx, ny, nz)
    centre = nx // 2
    hole_field = float(field[centre, centre, slice_z])

    residual = float(np.max(np.abs(eval_f(
        solution.states[:, -1], params, device.idx, boundary, material=device.material
    ))))
    psi2 = np.abs(solution.psi(step=-1)).reshape(nx, ny, nz) ** 2

    return {
        "applied_bz": applied_bz,
        "applied_mT": UNITS.field_to_mT(applied_bz),
        "fluxoid": float(history[-1]),
        "history": history,
        "times": solution.times,
        "entry_time": first_entry_time(solution.times, history),
        "n_vortices": n_vortices,
        "hole_field": hole_field,
        "hole_field_mT": UNITS.field_to_mT(hole_field),
        "screening": hole_field / applied_bz if applied_bz else 0.0,
        "hole_flux_quanta": UNITS.flux_quanta(UNITS.length(HOLE_NM) ** 2, hole_field),
        "residual": residual,
        "psi2_slice": psi2[:, :, slice_z],
        "field_slice": field[:, :, slice_z],
        "params": params,
        "expelled": n_vortices == 0 and abs(float(history[-1])) < 0.5,
    }


def main(output_dir: Path = Path(__file__).parent, small: bool = False) -> list[Path]:
    if small:
        fields, t_stop, noise_check = [0.25], 1.0, []
    else:
        fields = [0.06, 0.13, 0.20, 0.25, 0.27, 0.29, 0.31, 0.33, 0.45, 0.6]
        t_stop = 60.0
        # The two fields either side of the threshold, repeated with a
        # perturbation a thousand times smaller.  Agreement means the threshold
        # is where the barrier vanishes, not where the seed happens to be large
        # enough to cross it.
        noise_check = [0.27, 0.29]

    results = [measure(bz, t_stop) for bz in fields]

    robust = {bz: measure(bz, t_stop, noise=NOISE * 1e-3) for bz in noise_check}

    expelled = [r["applied_bz"] for r in results if r["expelled"]]
    admitted = [r["applied_bz"] for r in results if not r["expelled"]]
    bracket = expulsion_field(
        fields,
        [0.0 if r["expelled"] else 1.0 for r in results],
        [r["entry_time"] for r in results],
        hold_time=t_stop,
    )

    plane = UNITS.length(PLANE_NM)
    hole = UNITS.length(HOLE_NM)
    naive_mT = 2.067833848e-15 / (HOLE_NM * 1e-9) ** 2 * 1e3

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.5))

    # -- (0,0) what is expelled, and what breaks it -------------------------
    ax = axes[0, 0]
    applied_mT = [r["applied_mT"] for r in results]
    ax.plot(applied_mT, [r["n_vortices"] for r in results], "o-", color="C1",
            label="vortices in the plane")
    ax.plot(applied_mT, [abs(r["fluxoid"]) for r in results], "s-", color="C0",
            label="fluxoid through the hole")
    if bracket.threshold is not None:
        ax.axvspan(UNITS.field_to_mT(bracket.last_expelled),
                   UNITS.field_to_mT(bracket.first_entered), color="C3", alpha=0.15)
        ax.axvline(UNITS.field_to_mT(bracket.threshold), color="C3", ls="--",
                   linewidth=1.5, label="expulsion field")
    ax.axvline(naive_mT, color="C2", ls=":", linewidth=1.5,
               label=f"Φ₀/A_hole = {naive_mT:.2f} mT")
    ax.set_xlabel("applied Bz (mT)")
    ax.set_ylabel("count")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylim(bottom=0)
    ax.set_title("Vortices enter the plane before the hole gives way")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # -- (0,1) how much field reaches the hole ------------------------------
    ax = axes[0, 1]
    ax.plot(applied_mT, [r["hole_field_mT"] for r in results], "o-", color="C0",
            label="Bz at the hole centre")
    ax.plot(applied_mT, applied_mT, ":", color="gray", label="no screening")
    if bracket.threshold is not None:
        ax.axvline(UNITS.field_to_mT(bracket.threshold), color="C3", ls="--", linewidth=1.5)
    ax.set_xlabel("applied Bz (mT)")
    ax.set_ylabel("Bz at the hole centre (mT)")
    ax.set_title("A 4 µm plane screens the hole almost completely")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    text = (
        f"{UNITS.summary()}\n"
        f"plane:  {PLANE_NM / 1000:g}×{PLANE_NM / 1000:g} µm = {plane:g}×{plane:g} ξ\n"
        f"hole:   {HOLE_NM / 1000:g}×{HOLE_NM / 1000:g} µm = {hole:g}×{hole:g} ξ\n"
        f"stack:  S({LAYER_NM:g})/I({OXIDE_NM:g})/S({LAYER_NM:g}) nm\n"
        f"grid:   {results[0]['params'].Nx}×{results[0]['params'].Ny}"
        f"×{results[0]['params'].Nz}, h = {UNITS.length_nm(H):g} nm\n"
        f"hold:   {t_stop:g} τ_GL, seeded perturbation {NOISE:g}\n"
    )
    if expelled and admitted:
        text += (
            f"\nfully expelled up to {UNITS.field_to_mT(max(expelled)):.2f} mT\n"
            f"first failure at      {UNITS.field_to_mT(min(admitted)):.2f} mT\n"
            f"B_exp = {UNITS.field_to_mT(bracket.threshold):.2f} "
            f"± {UNITS.field_to_mT(bracket.uncertainty):.2f} mT"
        )
    if robust:
        agree = all(
            robust[bz]["expelled"] == next(r["expelled"] for r in results if r["applied_bz"] == bz)
            for bz in robust
        )
        text += (
            f"\nseed {NOISE * 1e-3:g} gives the same"
            if agree else f"\nseed {NOISE * 1e-3:g} DISAGREES"
        ) + " bracket"
    axes[0, 1].text(
        0.03, 0.97, text, transform=axes[0, 1].transAxes, fontsize=7.5,
        fontfamily="monospace", verticalalignment="top", horizontalalignment="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9),
    )

    # -- bottom row: |ψ|² either side of the threshold ----------------------
    below = max(expelled) if expelled else fields[0]
    above = min(admitted) if admitted else fields[-1]
    for ax, target, label in (
        (axes[1, 0], below, "fully expelled"),
        (axes[1, 1], above, "vortices in the plane"),
    ):
        record = next(r for r in results if r["applied_bz"] == target)
        params = record["params"]
        xs = np.arange(1, params.Nx) * UNITS.length_nm(params.hx) / 1000.0
        ys = np.arange(1, params.Ny) * UNITS.length_nm(params.hy) / 1000.0
        mesh = ax.pcolormesh(
            *np.meshgrid(xs, ys, indexing="ij"), record["psi2_slice"],
            cmap="inferno", vmin=0.0, vmax=1.0, shading="auto",
        )
        fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04, label="|ψ|²")
        ax.set_title(
            f"Bz = {record['applied_mT']:.2f} mT — {label}\n"
            f"{record['n_vortices']} vortices, fluxoid n = {record['fluxoid']:.0f}",
            fontsize=11,
        )
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")
        ax.set_aspect("equal")

    fig.suptitle(
        "1 µm hole in a 4 µm S/I/S plane — what limits flux expulsion",
        fontsize=14, y=0.98,
    )
    fig.tight_layout()
    out = output_dir / "sis_micron_ring.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    for record in results:
        print(
            f"Bz = {record['applied_mT']:6.2f} mT: "
            f"vortices={record['n_vortices']:4d}  fluxoid={record['fluxoid']:+.0f}  "
            f"Bz(hole)={record['hole_field_mT']:6.3f} mT "
            f"({record['screening']:.1%} of applied)  "
            f"flux in hole={record['hole_flux_quanta']:.3f} Φ₀  "
            f"residual={record['residual']:.1e}"
        )
    for bz, record in robust.items():
        baseline = next(r for r in results if r["applied_bz"] == bz)
        print(
            f"seed {NOISE * 1e-3:g} at {record['applied_mT']:.2f} mT: "
            f"vortices={record['n_vortices']} expelled={record['expelled']} "
            f"(baseline expelled={baseline['expelled']})"
        )
    if expelled and admitted:
        print(
            f"\nfully expelled up to {UNITS.field_to_mT(max(expelled)):.2f} mT, "
            f"first failure at {UNITS.field_to_mT(min(admitted)):.2f} mT "
            f"(hold time {t_stop:g} τ_GL)"
        )
    return [out]


if __name__ == "__main__":
    main()

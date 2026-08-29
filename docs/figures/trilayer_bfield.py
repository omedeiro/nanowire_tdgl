"""S/I/S in a perpendicular field, with vacuum around the stack.

The applied field enters this solver as prescribed flux through the
plaquettes on the *wall of the box*.  That is the right statement
only where the wall is far-field vacuum.  With the stack filling the
box the same condition lands on the superconductor's own surface, so
the film's outermost nodes are handed the applied field rather than
solving for it, and flux expelled from the film has nowhere to go.

Padding the stack with vacuum moves the condition off the metal.
Then all three things the device does are visible at once:

* the metal screens, and its own edge is screened too;
* the oxide transmits — it has no condensate, so no screening
  current, so the field passes through it;
* the expelled flux crowds into the vacuum beside the film, where
  the field *exceeds* the applied one, and relaxes back to the
  applied value out at the wall.

The right-hand column refines the grid, so the profile can be
shown to be resolved rather than asserted to be.

Runtime: about 7 minutes at the default size.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tdgl3d import AppliedField, Device, Layer, SimulationParameters, Trilayer
from tdgl3d.physics.applied_field import build_boundary_field_vectors
from tdgl3d.physics.bfield import eval_bfield_full
from tdgl3d.physics.rhs import (
    BoundaryVectors,
    _apply_boundary_conditions,
    _expand_interior_to_full,
    eval_f,
)
from tdgl3d.solvers.integrators import forward_euler

KAPPA = 2.0        # lambda = 2 xi
B0 = 0.02          # applied Bz, in Phi_0 / 2 pi xi^2

# Geometry in xi, given as the thickness the mesh actually realises
# rather than the layer's declared cell count -- see _cells_for below.
METAL = 4.0        # thickness of each metal layer  (2 lambda)
GAP = 3.0          # metal-to-metal gap across the oxide
WIDTH = 16.0       # film width in x and y
MARGIN = 6.0       # vacuum either side of the film
PAD = 6.0          # vacuum above and below the stack


def _cells_for(metal, gap, h):
    """Cell counts whose *realised* metal span and oxide gap are as asked.

    ``build_material_map`` hands both S/I interface nodes to the
    insulator, so a metal layer declared ``n`` cells spans only
    ``n-1`` cells of nodes and the gap between the two metals comes
    out ``m+2`` cells rather than ``m``.  Declaring cell counts
    directly therefore makes the *device* change when the mesh is
    refined — each interface walks one cell — and a refinement study
    would be comparing three different stacks.

    Inverting the offsets pins the realised geometry instead, so a
    refinement changes only the mesh.  It also puts a floor on the
    spacing: a gap of ``GAP`` needs ``GAP/h >= 3``, so a 3 xi gap
    cannot be represented at all above h = 1.
    """
    metal_cells = int(round(metal / h)) + 1
    oxide_cells = int(round(gap / h)) - 2
    if oxide_cells < 1:
        raise ValueError(
            f"h = {h} cannot realise a {gap} xi gap: the insulator claims "
            f"both interface nodes, so the gap is at least 2h = {2 * h} xi."
        )
    return metal_cells, oxide_cells


def _build(h, *, margin=MARGIN, pad=PAD, metal=METAL, gap=GAP, width=WIDTH):
    """Lay the device out on a grid of spacing *h*."""
    def cells(length):
        return max(1, int(round(length / h)))

    metal_cells, oxide_cells = _cells_for(metal, gap, h)
    trilayer = Trilayer(
        bottom=Layer(thickness_z=metal_cells, kappa=KAPPA),
        insulator=Layer(thickness_z=oxide_cells, kappa=0.0,
                        is_superconductor=False),
        top=Layer(thickness_z=metal_cells, kappa=KAPPA),
        vacuum_below=cells(pad),
        vacuum_above=cells(pad),
        lateral_margin=cells(margin) if margin else 0,
    )
    n_lateral = cells(width) + 2 * (cells(margin) if margin else 0)
    params = SimulationParameters(
        Nx=n_lateral, Ny=n_lateral, Nz=trilayer.Nz,
        hx=h, hy=h, hz=h, kappa=KAPPA,
    )
    device = Device(
        params, applied_field=AppliedField(Bz=B0, t_on_fraction=1.0),
        trilayer=trilayer,
    )
    return params, device, trilayer


def _realised(params, device, trilayer):
    """Metal span and oxide gap the mesh actually carries, in xi."""
    mid = (params.Nx + 1) // 2
    column = device.material.sc_mask.reshape(
        params.Nz + 1, params.Ny + 1, params.Nx + 1
    )[:, mid, mid]
    metal = np.flatnonzero(column)
    lower = metal[metal < len(column) // 2]
    upper = metal[metal >= len(column) // 2]
    span = (lower.max() - lower.min()) * params.hz
    gap = (upper.min() - lower.max()) * params.hz
    return float(span), float(gap)


def _relax(params, device, t_stop):
    """Relax to the steady state; return Bz/B0 and max |dX/dt|."""
    boundary = BoundaryVectors(
        *build_boundary_field_vectors(0.0, 0.0, B0, params, device.idx)
    )
    # dt < h^2 / (4 kappa^2 (d-1)): the familiar 2-D bound halves in 3-D.
    h = min(params.hx, params.hy, params.hz)
    dt = 0.5 * h**2 / (4.0 * params.kappa**2 * 2.0)
    _, states = forward_euler(
        device.initial_state(noise_amplitude=0.0).data,
        params, device.idx, lambda t, X: boundary,
        0.0, t_stop, dt, save_every=10**9, progress=False,
        material=device.material,
    )
    state = states[:, -1]
    residual = float(
        np.abs(eval_f(state, params, device.idx, boundary, device.material)).max()
    )

    n = params.n_interior
    full = [
        _expand_interior_to_full(state[i * n : (i + 1) * n], params, device.idx)
        for i in range(4)
    ]
    _, phi_x, phi_y, phi_z = _apply_boundary_conditions(
        *full, params, device.idx, boundary
    )
    shape = (params.Nx - 1, params.Ny - 1, params.Nz - 1)
    bz = np.real(
        eval_bfield_full(phi_x, phi_y, phi_z, params, device.idx)[2]
    ).reshape(shape)
    return bz / B0, residual


def _profiles(params, trilayer, bz):
    """Centre-line cuts through the relaxed field, in xi."""
    cx = (params.Nx - 1) // 2
    cy = (params.Ny - 1) // 2
    ranges = trilayer.z_ranges()
    k_metal = max(ranges["bottom"][0], 1) - 1

    # Bz lives on plaquettes: Bz[i] spans nodes i and i+1, so it sits
    # half a cell above the node coordinate.
    x = (np.arange(1, params.Nx) + 0.5) * params.hx
    z = (np.arange(1, params.Nz) + 0.5) * params.hz
    return {
        "x": x,
        "z": z,
        "along_x": bz[:, cy, k_metal],
        "along_z": bz[cx, cy, :],
        "map_xz": bz[:, cy, :],
        "k_metal": k_metal,
    }


def main(output_dir: Path = Path(__file__).parent, small: bool = False) -> list[Path]:
    if small:
        main_h, t_stop = 1.0, 4.0
        refine = [(1.0, 4.0)]
        main_geom = conv_geom = dict(width=4.0, margin=1.0, pad=2.0)
    else:
        main_h, t_stop = 1.0, 60.0
        refine = [(1.0, 40.0), (0.5, 40.0)]
        main_geom = {}
        # Smaller box, so the refined run stays affordable; the
        # realised metal span and oxide gap match the main device,
        # so the two panels describe the same stack.
        conv_geom = dict(width=8.0, margin=2.0, pad=4.0)

    # -- the device, at the working resolution ------------------------
    params, device, trilayer = _build(main_h, **main_geom)
    bz, residual = _relax(params, device, t_stop)
    prof = _profiles(params, trilayer, bz)
    ranges = trilayer.z_ranges()

    # -- the same device without padding, for contrast ---------------
    bare_params, bare_device, bare_tri = _build(
        main_h, margin=0.0, pad=0.0, **{k: v for k, v in main_geom.items()
                                          if k not in ('margin', 'pad')}
    )
    bare_bz, _ = _relax(bare_params, bare_device, t_stop)
    bare_prof = _profiles(bare_params, bare_tri, bare_bz)

    # -- grid refinement ---------------------------------------------
    refined = []
    for h, t in refine:
        p_r, d_r, tri_r = _build(h, **conv_geom)
        bz_r, res_r = _relax(p_r, d_r, t)
        pr = _profiles(p_r, tri_r, bz_r)
        refined.append({
            "h": h, "residual": res_r, "params": p_r,
            "realised": _realised(p_r, d_r, tri_r),
            "centre": float(bz_r[(p_r.Nx - 1) // 2, (p_r.Ny - 1) // 2,
                                  pr["k_metal"]]),
            "peak": float(pr["along_x"].max()),
            "x": pr["x"] - pr["x"].mean(),
            "along_x": pr["along_x"],
        })

    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))

    # ── (0,0) Bz through the stack, padded vs not ──────────────
    ax = axes[0, 0]
    shade = [
        ("vacuum_below", "0.85", "vacuum"),
        ("bottom", "#7fb3d5", "metal"),
        ("insulator", "#e59866", "oxide"),
        ("top", "#7fb3d5", None),
        ("vacuum_above", "0.85", None),
    ]
    for name, colour, label in shade:
        if name not in ranges:
            continue
        z0, z1 = ranges[name]
        ax.axvspan(z0 * params.hz, z1 * params.hz, color=colour, alpha=0.55,
                   lw=0, label=label)
    ax.axhline(1.0, color="gray", ls=":", lw=1.4, label="applied field")
    ax.plot(prof["z"], prof["along_z"], "o-", color="C0", ms=4,
            label="with vacuum padding")
    ax.plot(
        bare_prof["z"] + ranges["bottom"][0] * params.hz,
        bare_prof["along_z"], "s--", color="C3", ms=4, alpha=0.85,
        label="stack fills the box",
    )
    ax.set_xlabel("z (ξ)")
    ax.set_ylabel("Bz / B_applied")
    ax.set_title("Through the stack: the oxide transmits, the metal screens")
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(True, alpha=0.3)

    # ── (0,1) refinement of the same cut ──────────────────────
    ax = axes[0, 1]
    for rec, style in zip(refined, ("o-", "s--", "^:")):
        ax.plot(rec["x"], rec["along_x"], style, ms=3.5,
                label=f"h = {rec['h']:.3g} ξ  ({rec['params'].Nx}³ cells)")
    ax.axhline(1.0, color="gray", ls=":", lw=1.4)
    ax.set_xlabel("x − x_centre (ξ)")
    ax.set_ylabel("Bz / B_applied")
    ax.set_title("Same cut at three grid spacings")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    span_r, gap_r = refined[0]["realised"]
    note = f"metal span {span_r:g} ξ, gap {gap_r:g} ξ — held fixed as h changes"
    if len(refined) > 1:
        drift = abs(refined[-1]["centre"] - refined[0]["centre"])
        note += (
            "\ncentre Bz/B₀: "
            + " → ".join(f"{r['centre']:.3f}" for r in refined)
            + f"  (moves {drift:.3f})"
        )
    ax.text(
        0.03, 0.04, note,
        transform=ax.transAxes, fontsize=7.5, fontfamily="monospace",
        va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor="gray", alpha=0.9),
    )

    # ── (1,0) Bz across the film ───────────────────────────────
    ax = axes[1, 0]
    film_lo = trilayer.lateral_margin * params.hx
    film_hi = (params.Nx - trilayer.lateral_margin) * params.hx
    ax.axvspan(0, film_lo, color="0.85", alpha=0.55, lw=0, label="vacuum")
    ax.axvspan(film_hi, params.Nx * params.hx, color="0.85", alpha=0.55, lw=0)
    ax.axvspan(film_lo, film_hi, color="#7fb3d5", alpha=0.55, lw=0,
               label="metal")
    ax.axhline(1.0, color="gray", ls=":", lw=1.4, label="applied field")
    ax.plot(prof["x"], prof["along_x"], "o-", color="C0", ms=4,
            label="with vacuum padding")
    ax.plot(bare_prof["x"] + film_lo, bare_prof["along_x"], "s--", color="C3",
            ms=4, alpha=0.85, label="stack fills the box")
    peak = float(prof["along_x"].max())
    ax.annotate(
        f"flux crowds here:\n{peak:.3f} × applied",
        xy=(prof["x"][int(np.argmax(prof["along_x"]))], peak),
        xytext=(0.42, 0.80), textcoords="axes fraction", fontsize=8,
        ha="center",
        arrowprops=dict(arrowstyle="->", color="0.3", lw=1.1),
    )
    ax.set_xlabel("x (ξ)")
    ax.set_ylabel("Bz / B_applied")
    ax.set_title("Across the film: screened inside, crowded just outside")
    ax.legend(fontsize=7.5, loc="lower center")
    ax.grid(True, alpha=0.3)

    # ── (1,1) the field in the x–z plane ───────────────────────
    ax = axes[1, 1]
    xx, zz = np.meshgrid(prof["x"], prof["z"], indexing="ij")
    levels = np.linspace(
        min(0.0, prof["map_xz"].min()), max(1.05, prof["map_xz"].max()), 25
    )
    mesh = ax.contourf(xx, zz, prof["map_xz"], levels=levels, cmap="magma")
    fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04,
                 label="Bz / B_applied")
    ax.contour(xx, zz, prof["map_xz"], levels=[1.0], colors="w",
               linewidths=1.2, linestyles="--")
    for name in ("bottom", "top"):
        z0, z1 = ranges[name]
        ax.add_patch(plt.Rectangle(
            (film_lo, z0 * params.hz), film_hi - film_lo,
            (z1 - z0) * params.hz, fill=False, edgecolor="cyan", lw=1.3,
        ))
    ax.set_xlabel("x (ξ)")
    ax.set_ylabel("z (ξ)")
    ax.set_title("Bz around the stack (cyan = metal, dashed = applied)")

    fig.suptitle(
        f"S/I/S in a perpendicular field, κ = {KAPPA}, λ = {KAPPA:g} ξ — "
        f"the field in the vacuum around the stack",
        fontsize=13.5, y=0.99,
    )
    fig.tight_layout()

    out = output_dir / "trilayer_bfield.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    span, gap = _realised(params, device, trilayer)
    print(f"grid {params.Nx}×{params.Ny}×{params.Nz} at h = {main_h} ξ, "
          f"max |dX/dt| = {residual:.1e}")
    print(f"realised metal span {span:g} ξ, oxide gap {gap:g} ξ")
    print(f"Bz at the film centre     : {prof['along_z'][prof['k_metal']]:.4f} × applied")
    print(f"peak Bz beside the film   : {peak:.4f} × applied")
    for rec in refined:
        print(f"h = {rec['h']:.3g}: centre {rec['centre']:.4f}, "
              f"peak {rec['peak']:.4f}, |dX/dt| = {rec['residual']:.1e}, "
              f"realised span/gap {rec['realised'][0]:g}/{rec['realised'][1]:g} ξ")
    return [out]


if __name__ == "__main__":
    main()

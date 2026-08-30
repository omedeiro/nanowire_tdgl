"""A vortex trapped in one layer of an S/I/S stack, and where its flux goes.

The stack is superconductor / oxide / superconductor with vacuum around it.
A columnar non-superconducting defect is carved through the **bottom** layer
only, and a single vortex is seeded on it; the top layer starts, and stays,
in the vortex-free state.  What the figures show is the field that
configuration produces.

Why the defect is there.  At zero applied field a lone vortex in a finite
film is pulled towards its own image in the edge, so it leaves.  Seeded at
the exact centre of a noiseless square it survives only because every escape
direction is degenerate — a fixed point held by symmetry, which is the trap
``AGENTS.md`` warns about, not a trapped vortex.  The defect removes the core
condensation energy at one spot and pins it for real: a vortex seeded 3 ξ off
axis migrates onto the defect and stays, and the same run without the defect
expels it (``test_vortex_is_pinned_by_the_columnar_defect``).

What the oxide thickness does.  The two layers here are coupled only through
**A** — this model has no Josephson term, so nothing but the magnetic field
crosses the oxide.  The trapped vortex carries one flux quantum, but that
flux is not confined to a tube: on leaving the bottom layer it spreads over
the scale λ = κ ξ, so the fraction of it still inside the vortex radius by
the time it reaches the top layer falls off as the gap widens.  Sweeping the
gap is therefore a direct measurement of how the two layers decouple, and it
is the reason the top layer can sit in the vortex-free state at all while
its neighbour holds a quantum of flux.

Note that the fluxoid is exactly 1 in the bottom layer and exactly 0 in the
top one at every gap.  Those are topological integers and no amount of
magnetic leakage changes them; what the sweep moves is the *field*, not the
vorticity.

Two figures:

``sis_vortex_trapping_3d.png``
    Isometric views of the stack at the thinnest and the thickest oxide,
    with |ψ|² painted on the mid-plane of each superconducting layer and
    B field lines traced from the trapped core.  The dark core in the
    bottom sheet and the flat top sheet are the trapping; the flare of the
    field lines across the oxide is the bending.

``sis_vortex_trapping_sweep.png``
    The same runs read quantitatively: B_z on the vortex axis through the
    stack, the flux still inside a fixed radius as a function of height,
    how much of it survives to the top layer, and how far it has spread
    laterally once it gets there.

Runtime: about 6 minutes at the default size.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tdgl3d
from matplotlib.colors import Normalize
from scipy.interpolate import RegularGridInterpolator
from tdgl3d import AppliedField, Device, Layer, SimulationParameters, Trilayer
from tdgl3d.analysis.vortex_counting import count_vortices_polygon
from tdgl3d.physics.applied_field import build_boundary_field_vectors
from tdgl3d.physics.rhs import BoundaryVectors, eval_f

# --------------------------------------------------------------------------- #
# Device, in xi
# --------------------------------------------------------------------------- #
KAPPA = 2.0        # lambda = 2 xi
METAL = 4.0        # realised thickness of each superconducting layer
WIDTH = 20.0       # film width in x and y
MARGIN = 4.0       # vacuum either side of the film
PAD = 5.0          # vacuum above and below the stack
DEFECT = 2.0       # side of the square defect column, in xi

# Metal-to-metal oxide gaps to sweep.  These are *realised* gaps: the
# declared cell count is inverted for them in _cells_for.
GAPS = (3.0, 4.0, 6.0, 10.0)

# Radius of the disc the flux is integrated over, and of the fluxoid contour.
FLUX_RADIUS = 6.0
FLUXOID_RADIUS = 7.0


def _cells_for(metal: float, gap: float, h: float) -> tuple[int, int]:
    """Cell counts whose *realised* metal span and oxide gap are as asked.

    ``build_material_map`` hands both S/I interface nodes to the insulator,
    so a metal layer declared ``n`` cells spans only ``n-1`` cells of nodes
    and the metal-to-metal gap comes out ``m+2`` rather than ``m``.
    Declaring cell counts directly would make the *device* change whenever
    the mesh changed; inverting the offsets pins the geometry instead.
    """
    metal_cells = int(round(metal / h)) + 1
    oxide_cells = int(round(gap / h)) - 2
    if oxide_cells < 1:
        raise ValueError(
            f"h = {h} cannot realise a {gap} xi gap: the insulator claims both "
            f"interface nodes, so the gap is at least 2h = {2 * h} xi."
        )
    return metal_cells, oxide_cells


def _build(gap: float, h: float, *, width=WIDTH, margin=MARGIN, pad=PAD):
    """Lay an S/I/S stack with the given metal-to-metal gap on a mesh of size *h*."""
    def cells(length: float) -> int:
        return max(1, int(round(length / h)))

    metal_cells, oxide_cells = _cells_for(METAL, gap, h)
    trilayer = Trilayer(
        bottom=Layer(thickness_z=metal_cells, kappa=KAPPA),
        insulator=Layer(thickness_z=oxide_cells, kappa=KAPPA,
                        is_superconductor=False),
        top=Layer(thickness_z=metal_cells, kappa=KAPPA),
        vacuum_below=cells(pad), vacuum_above=cells(pad),
        lateral_margin=cells(margin),
    )
    n_lateral = cells(width) + 2 * cells(margin)
    params = SimulationParameters(
        Nx=n_lateral, Ny=n_lateral, Nz=trilayer.Nz,
        hx=h, hy=h, hz=h, kappa=KAPPA,
    )
    device = Device(
        params,
        # No applied field: the only flux in the box is the trapped quantum.
        applied_field=AppliedField(Bz=0.0, t_on_fraction=1.0),
        trilayer=trilayer,
    )
    return params, device, trilayer


def _carve_defect(params, device, trilayer, *, side=DEFECT):
    """Carve the pinning column through the bottom layer only.

    ``MaterialMap.carve_hole_polygon`` is called directly rather than
    ``Device.add_hole``: this is a non-superconducting *inclusion*, which the
    solver handles through the same insulator path as the oxide, not a
    geometric void needing the hole boundary condition (see
    ``docs/notes/HOLE_BC_STATUS.md``, where that BC is still open).
    """
    cx, cy = _axis(params)
    r = side / 2.0
    k0, k1 = trilayer.z_ranges()["bottom"]
    device.material.carve_hole_polygon(
        [(cx - r, cy - r), (cx + r, cy - r), (cx + r, cy + r), (cx - r, cy + r)],
        (k0, k1 - 1), params, device.idx,
    )


def _radii(params, device, trilayer):
    """Flux-disc and fluxoid-contour radii that actually fit this device.

    The fluxoid contour has to run *in the metal* -- the phase is not defined
    where psi = 0 -- so it is kept inside the film's own footprint.  The flux
    disc may extend past the film, since the vortex's return flux does too,
    but not past the wall of the box.
    """
    cx, _ = _axis(params)
    k_bottom = _layer_midplanes(trilayer)[0]
    i0, i1, _, _ = _film_extent(params, device, k_bottom)
    xs = _interior_axes(params)[0]
    half_film = min(cx - xs[i0], xs[i1 - 1] - cx)
    half_box = min(cx, params.Nx * params.hx - cx) - params.hx
    return (round(min(FLUX_RADIUS, half_box), 6),
            round(min(FLUXOID_RADIUS, half_film), 6))


def _axis(params) -> tuple[float, float]:
    """Centre of the film, in xi."""
    return params.Nx * params.hx / 2.0, params.Ny * params.hy / 2.0


def _interior_axes(params):
    """Physical coordinates of the interior nodes along each axis, in xi."""
    return (
        np.arange(1, params.Nx) * params.hx,
        np.arange(1, params.Ny) * params.hy,
        np.arange(1, params.Nz) * params.hz,
    )


def _interior_shape(params):
    return params.Nx - 1, params.Ny - 1, max(params.Nz - 1, 1)


def _seed_bottom_vortex(params, device, trilayer, *, offset=0.0, noise=0.0, seed=None):
    """Uniform state carrying one +1 phase winding in the bottom layer only."""
    state = device.initial_state(noise_amplitude=noise, seed=seed)
    nx, ny, nz = _interior_shape(params)
    psi = state.psi.reshape(nx, ny, nz).copy()

    cx, cy = _axis(params)
    xs, ys, _ = _interior_axes(params)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    vx, vy = cx + offset, cy
    # tanh(r/xi) is the standard Ginzburg-Landau core profile; the phase is
    # the winding itself.  The relaxation fixes up the details.
    winding = np.tanh(np.hypot(X - vx, Y - vy) / 1.5) * np.exp(
        1j * np.arctan2(Y - vy, X - vx)
    )
    for k_full in range(*trilayer.z_ranges()["bottom"]):
        psi[:, :, k_full - 1] = winding

    state.psi = psi.ravel()
    return state


def _relax(params, device, x0, t_stop, *, dt=None, save_every=10**9):
    """Relax to the steady state.  Returns the Solution and max |dX/dt|.

    The applied field is zero, so the boundary vectors are zero too and the
    only flux in the box is the quantum the seeded winding carries.
    """
    h = min(params.hx, params.hy, params.hz)
    # dt < h^2 / (4 kappa^2 (d-1)): the familiar 2-D bound halves in 3-D.
    if dt is None:
        dt = 0.5 * h**2 / (4.0 * params.kappa**2 * 2.0)
    solution = tdgl3d.solve(
        device, t_start=0.0, t_stop=t_stop, dt=dt, method="euler", x0=x0,
        save_every=save_every, progress=False, log_metadata=False,
    )
    boundary = BoundaryVectors(
        *build_boundary_field_vectors(0.0, 0.0, 0.0, params, device.idx)
    )
    residual = float(
        np.abs(
            eval_f(solution.states[:, -1], params, device.idx, boundary,
                   device.material)
        ).max()
    )
    return solution, residual


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def _layer_midplanes(trilayer) -> tuple[int, int]:
    """Interior z-indices of the mid-plane of the bottom and top layers."""
    ranges = trilayer.z_ranges()
    bottom = sum(ranges["bottom"]) // 2 - 1
    top = sum(ranges["top"]) // 2 - 1
    return bottom, top


def _fluxoid(solution, device, params, slice_z: int, radius: float) -> float:
    """Fluxoid enclosed by a square contour of half-width *radius* about the axis.

    Topologically quantised, so this is the quantity that says whether the
    layer holds a vortex -- unlike the magnetic flux, which is not quantised.
    """
    cx, cy = _axis(params)
    r = radius
    contour = np.array([[cx - r, cy - r], [cx + r, cy - r],
                        [cx + r, cy + r], [cx - r, cy + r]])
    return float(count_vortices_polygon(solution, device, contour, slice_z=slice_z))


def _profiles(solution, params, trilayer, *, radius=FLUX_RADIUS):
    """Height-resolved field diagnostics for one relaxed stack."""
    nx, ny, nz = _interior_shape(params)
    xs, ys, zs = _interior_axes(params)
    _, _, Bz = solution.bfield(step=-1, full_interior=True)
    Bz = Bz.reshape(nx, ny, nz)

    cx, cy = _axis(params)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    rho = np.hypot(X - cx, Y - cy)
    disc = rho <= radius
    cell = params.hx * params.hy

    # Flux through a fixed disc about the axis, one entry per interior z-plane,
    # in units of Phi_0 = 2 pi.  A flux tube that stayed a tube would hold this
    # constant with height; the fall-off *is* the spreading.
    flux = np.array([Bz[:, :, k][disc].sum() * cell / (2.0 * np.pi)
                     for k in range(nz)])

    # B_z on the vortex axis.  The axis falls between nodes on an even grid,
    # so average the four nodes straddling it.
    i0 = int(np.argmin(np.abs(xs - cx)))
    j0 = int(np.argmin(np.abs(ys - cy)))
    i1 = i0 + 1 if xs[i0] < cx else i0 - 1
    j1 = j0 + 1 if ys[j0] < cy else j0 - 1
    i1, j1 = np.clip([i1, j1], 0, [nx - 1, ny - 1])
    axis_Bz = 0.25 * (Bz[i0, j0, :] + Bz[i1, j0, :] + Bz[i0, j1, :] + Bz[i1, j1, :])

    # Azimuthally averaged radial profile of B_z, one row per z-plane.
    edges = np.arange(0.0, radius + 2.0 * params.hx, params.hx)
    which = np.digitize(rho.ravel(), edges) - 1
    keep = (which >= 0) & (which < len(edges) - 1)
    counts = np.bincount(which[keep], minlength=len(edges) - 1)
    radial = np.array([
        np.bincount(which[keep], weights=Bz[:, :, k].ravel()[keep],
                    minlength=len(edges) - 1) / np.maximum(counts, 1)
        for k in range(nz)
    ])
    return {
        "z": zs, "flux": flux, "axis_Bz": axis_Bz,
        "radial": radial, "radial_r": 0.5 * (edges[:-1] + edges[1:]),
    }


def _vortex_position(solution, params, slice_z: int, step: int = -1):
    """(x, y) of the vorticity centroid in a z-plane, in xi, or None if empty.

    Located from the gauge-invariant plaquette winding rather than from the
    |psi|^2 minimum.  The trapped vortex sits *on* the defect column, whose
    nodes carry psi = 0 by construction, so a |psi|^2 minimum would be
    degenerate there -- and the film corners, suppressed by vacuum on two
    sides, would win it outright.  The winding has no such ambiguity: it is
    an integer per plaquette, and the total over the layer is the number of
    vortices in it.
    """
    from tdgl3d.analysis.vortex_counting import plaquette_vorticity

    vorticity, _ = plaquette_vorticity(solution, slice_z=slice_z, step=step)
    weight = np.where(np.abs(vorticity) > 0.5, vorticity, 0.0)
    total = weight.sum()
    if abs(total) < 0.5:
        return None
    xs, ys, _ = _interior_axes(params)
    # Plaquette (i, j) is centred between interior nodes i and i+1.
    xc = 0.5 * (xs[:-1] + xs[1:])
    yc = 0.5 * (ys[:-1] + ys[1:])
    X, Y = np.meshgrid(xc, yc, indexing="ij")
    return float((weight * X).sum() / total), float((weight * Y).sum() / total)


# --------------------------------------------------------------------------- #
# Field lines
# --------------------------------------------------------------------------- #
def _trace_field_lines(solution, params, *, z_start, n_azimuth=10, radii=(0.8, 2.0),
                       step=0.25, max_len=4000):
    """Trace B field lines up and down from a ring of seeds about the axis.

    Plain RK4 on the trilinearly interpolated field.  The discrete field
    satisfies div B = 0 exactly (``test_verification_conservation``), so the
    lines are genuine flux lines and not an artefact of the interpolation.
    """
    nx, ny, nz = _interior_shape(params)
    xs, ys, zs = _interior_axes(params)
    Bx, By, Bz = solution.bfield(step=-1, full_interior=True)
    field = np.stack([Bx.reshape(nx, ny, nz), By.reshape(nx, ny, nz),
                      Bz.reshape(nx, ny, nz)], axis=-1)
    interp = RegularGridInterpolator((xs, ys, zs), field, bounds_error=False,
                                     fill_value=None)
    lo = np.array([xs[0], ys[0], zs[0]])
    hi = np.array([xs[-1], ys[-1], zs[-1]])

    def direction(p):
        b = interp(p)[0]
        n = np.linalg.norm(b)
        return b / n if n > 1e-14 else None

    def march(p0, sign):
        pts, p = [p0.copy()], p0.copy()
        for _ in range(max_len):
            k1 = direction(p)
            if k1 is None:
                break
            k2 = direction(p + sign * step * 0.5 * k1)
            if k2 is None:
                break
            k3 = direction(p + sign * step * 0.5 * k2)
            if k3 is None:
                break
            k4 = direction(p + sign * step * k3)
            if k4 is None:
                break
            p = p + sign * step * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            if np.any(p < lo) or np.any(p > hi):
                break
            pts.append(p.copy())
        return np.array(pts)

    cx, cy = _axis(params)
    lines = []
    for radius in radii:
        for a in np.linspace(0, 2 * np.pi, n_azimuth, endpoint=False):
            seed = np.array([cx + radius * np.cos(a), cy + radius * np.sin(a),
                             z_start])
            up, down = march(seed, +1.0), march(seed, -1.0)
            line = np.vstack([down[::-1], up[1:]]) if len(down) > 1 else up
            if len(line) > 3:
                lines.append(line)
    return lines


# --------------------------------------------------------------------------- #
# Isometric rendering
# --------------------------------------------------------------------------- #
def _film_extent(params, device, slice_z: int):
    """(i0, i1, j0, j1) index bounds of the metal at an interior z-plane."""
    nx, ny, nz = _interior_shape(params)
    sc = device.material.interior_sc_mask.reshape(nx, ny, nz)[:, :, slice_z] > 0
    ii = np.flatnonzero(sc.any(axis=1))
    jj = np.flatnonzero(sc.any(axis=0))
    return int(ii[0]), int(ii[-1]) + 1, int(jj[0]), int(jj[-1]) + 1


def _paint_layer(ax, params, device, solution, slice_z, cmap, norm, *, alpha=1.0,
                 z_offset=0.0):
    """Paint |psi|^2 of one layer as a horizontal sheet at its own height."""
    nx, ny, nz = _interior_shape(params)
    xs, ys, zs = _interior_axes(params)
    i0, i1, j0, j1 = _film_extent(params, device, slice_z)
    psi2 = np.abs(solution.psi(-1).reshape(nx, ny, nz)[i0:i1, j0:j1, slice_z]) ** 2
    X, Y = np.meshgrid(xs[i0:i1], ys[j0:j1], indexing="ij")
    ax.plot_surface(X, Y, np.full_like(X, zs[slice_z] - z_offset),
                    facecolors=cmap(norm(psi2)), shade=False,
                    rstride=1, cstride=1, alpha=alpha, zorder=1)
    return xs[i0], xs[i1 - 1], ys[j0], ys[j1 - 1]


def _outline(ax, x0, x1, y0, y1, z, **kw):
    ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], [z] * 5, **kw)


def _draw_oxide_box(ax, x0, x1, y0, y1, z0, z1, **kw):
    """Wireframe the oxide slab between the two metal sheets."""
    _outline(ax, x0, x1, y0, y1, z0, **kw)
    _outline(ax, x0, x1, y0, y1, z1, **kw)
    for xc, yc in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
        ax.plot([xc, xc], [yc, yc], [z0, z1], **kw)


def plot_isometric(runs, output_dir: Path) -> Path:
    """Isometric views of the stack at the thinnest and thickest oxide gap."""
    shown = [runs[0], runs[-1]]
    fig = plt.figure(figsize=(15.5, 7.9))
    cmap = plt.get_cmap("inferno")
    norm = Normalize(vmin=0.0, vmax=1.0)

    # Height is measured from the mid-plane of the bottom metal in both
    # panels, and both are given the same limits, so the two stacks can be
    # read against each other: the bottom sheet sits at the same place and
    # only the top one moves.
    def shift(run):
        return _interior_axes(run["params"])[2][_layer_midplanes(run["trilayer"])[0]]

    z_lo = min((_interior_axes(r["params"])[2][0] - shift(r)) for r in shown)
    z_hi = max((_interior_axes(r["params"])[2][-1] - shift(r)) for r in shown)

    for col, run in enumerate(shown):
        params, device, trilayer = run["params"], run["device"], run["trilayer"]
        solution = run["solution"]
        kb, kt = _layer_midplanes(trilayer)
        dz = shift(run)
        zs = _interior_axes(params)[2] - dz
        ax = fig.add_subplot(1, 2, col + 1, projection="3d")

        x0, x1, y0, y1 = _paint_layer(ax, params, device, solution, kb, cmap, norm,
                                      z_offset=dz)
        # The top sheet is left a little transparent so the core pinned in
        # the sheet below still reads through it at the narrowest gap.
        _paint_layer(ax, params, device, solution, kt, cmap, norm, alpha=0.8,
                     z_offset=dz)

        ranges = trilayer.z_ranges()
        z_ins0 = ranges["insulator"][0] * params.hz - dz
        z_ins1 = ranges["insulator"][1] * params.hz - dz
        _draw_oxide_box(ax, x0, x1, y0, y1, z_ins0, z_ins1,
                        color="0.45", lw=0.8, ls=":", zorder=2)
        for z in (zs[kb], zs[kt]):
            _outline(ax, x0, x1, y0, y1, z, color="0.25", lw=1.0, zorder=3)

        # The pinning column, through the bottom layer only.
        cx, cy = _axis(params)
        r = DEFECT / 2.0
        zb0 = ranges["bottom"][0] * params.hz - dz
        zb1 = ranges["bottom"][1] * params.hz - dz
        for dx, dy in ((-r, -r), (r, -r), (r, r), (-r, r)):
            ax.plot([cx + dx] * 2, [cy + dy] * 2, [zb0, zb1],
                    color="#00e5ff", lw=1.4, zorder=6)

        # Field lines, seeded on the trapped core.
        lines = _trace_field_lines(solution, params,
                                   z_start=_interior_axes(params)[2][kb],
                                   n_azimuth=run["n_azimuth"], radii=(0.7, 1.8, 3.0))
        for line in lines:
            ax.plot(line[:, 0], line[:, 1], line[:, 2] - dz,
                    color="#1f77ff", lw=1.0, alpha=0.75, zorder=5)

        # A flat annotation rather than 3-D text: with only two sheets there
        # is no ambiguity about which is which, and nothing lands on the core.
        ax.text2D(
            0.02, 0.92,
            "upper sheet — top metal, empty\nlower sheet — bottom metal, holds the vortex",
            transform=ax.transAxes, fontsize=9.5, color="0.15", va="top",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="0.8", pad=3.0),
        )

        ax.set_xlabel("x (ξ)", labelpad=2)
        ax.set_ylabel("y (ξ)", labelpad=2)
        ax.set_zlabel("z − z(bottom metal)  (ξ)", labelpad=2)
        ax.view_init(elev=22, azim=-58)
        ax.set_box_aspect((1, 1, 0.85))
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_zlim(z_lo, z_hi)
        ax.set_title(
            f"oxide gap {run['gap']:g} ξ = {run['gap'] / KAPPA:g} λ\n"
            f"fluxoid: bottom {round(run['fluxoid_bottom']):+d}, "
            f"top {round(run['fluxoid_top']):+d}"
            f"   —   flux at the top layer {run['flux_top']:.3f} Φ₀",
            fontsize=11, pad=14,
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=fig.axes, fraction=0.014, pad=0.02, shrink=0.6)
    cbar.set_label("|ψ|²  on each metal mid-plane")

    fig.suptitle(
        "One trapped vortex in the bottom layer of an S/I/S stack — "
        f"κ = {KAPPA:g}, λ = {KAPPA:g} ξ, no applied field.\n"
        "Cyan = the pinning column (bottom layer only); blue = B field lines "
        "from the trapped core.",
        fontsize=12.5, y=0.99,
    )
    out = output_dir / "sis_vortex_trapping_3d.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------- #
def plot_sweep(runs, output_dir: Path) -> Path:
    """The same runs read quantitatively, against the oxide gap."""
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0))
    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.8, len(runs)))
    r_flux = runs[0]["r_flux"]

    for run, colour in zip(runs, colors):
        params, trilayer = run["params"], run["trilayer"]
        prof = run["profiles"]
        # Height measured from the top of the bottom metal, so the bottom
        # layer lines up across gaps and only the oxide stretches.
        z0 = trilayer.z_ranges()["bottom"][1] * params.hz
        z = prof["z"] - z0
        label = f"{run['gap']:g} ξ"
        _, kt = _layer_midplanes(trilayer)
        axes[0, 0].plot(z, prof["axis_Bz"], color=colour, label=label, lw=1.6)
        axes[0, 1].plot(z, prof["flux"], color=colour, label=label, lw=1.6)
        # Where this stack's top metal sits on the curve.  The curves very
        # nearly coincide -- the field above the bottom layer is the trapped
        # vortex's own, and is not much changed by moving the second film.
        # What the gap decides is which point of it the top layer samples.
        axes[0, 0].plot(z[kt], prof["axis_Bz"][kt], "o", color=colour, ms=7,
                        mec="k", mew=0.6, zorder=5)
        axes[0, 1].plot(z[kt], prof["flux"][kt], "o", color=colour, ms=7,
                        mec="k", mew=0.6, zorder=5)
        radial = prof["radial"][kt]
        axes[1, 1].plot(prof["radial_r"], radial / max(radial[0], 1e-30),
                        color=colour, label=label, lw=1.6)

    for ax, ylabel, title in (
        (axes[0, 0], "B$_z$ on the vortex axis",
         "(a) the axial field decays away from the trapped core"),
        (axes[0, 1], f"flux within r ≤ {r_flux:g} ξ  (Φ₀)",
         "(b) flux leaks out of the tube as it climbs"),
    ):
        ax.axvspan(-METAL, 0.0, color="0.85", zorder=0)
        ax.annotate("bottom metal\n(the vortex is here)", (-METAL / 2, 0.03),
                    xycoords=("data", "axes fraction"), ha="center", va="bottom",
                    fontsize=8.5, color="0.35")
        ax.set_xlabel("height above the top of the bottom metal  (ξ)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(title="oxide gap", fontsize=9)
        ax.grid(alpha=0.3)
    axes[0, 0].annotate(
        "● = the top metal's mid-plane", (0.97, 0.62), xycoords="axes fraction",
        ha="right", fontsize=8.5, color="0.25",
    )

    ax = axes[1, 0]
    gaps = np.array([r["gap"] for r in runs])
    top = np.array([r["flux_top"] for r in runs])
    bottom = np.array([r["flux_bottom"] for r in runs])
    ax.semilogy(gaps, top, "o-", color="#1f77b4", label="top layer")
    ax.semilogy(gaps, bottom, "s--", color="#d62728", label="bottom layer")
    for g, t in zip(gaps, top):
        ax.annotate(f"{t:.3f}", (g, t), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8.5)
    ax.set_xlabel("metal-to-metal oxide gap  (ξ)")
    ax.set_ylabel(f"flux within r ≤ {r_flux:g} ξ at the mid-plane  (Φ₀)")
    ax.set_title("(c) so the layers decouple as the oxide thickens")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1, 1]
    ax.set_xlabel("radius from the vortex axis  (ξ)")
    ax.set_ylabel("B$_z$ / B$_z$(0) at the top-layer mid-plane")
    ax.set_title("(d) and what arrives is spread wider")
    ax.legend(title="oxide gap", fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle(
        "Where the trapped vortex's flux goes, against oxide thickness — "
        f"κ = {KAPPA:g}, metal {METAL:g} ξ each, no applied field",
        fontsize=13, y=0.995,
    )
    fig.tight_layout()
    out = output_dir / "sis_vortex_trapping_sweep.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main(output_dir: Path | str = Path("."), small: bool = False, *,
         gaps=None, t_stop=None, width=None, h=1.0):
    """Relax the stack at each oxide gap and draw both figures.

    ``small`` is the smoke-test size.  ``gaps`` (metal-to-metal, in xi),
    ``t_stop`` and ``width`` override the defaults for an exploratory sweep;
    every gap must satisfy ``gap/h >= 3``, since the insulator claims both
    interface nodes.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if small:
        default_gaps, default_t_stop = (3.0, 6.0), 12.0
        geometry = dict(width=10.0, margin=2.0, pad=3.0)
        n_azimuth = 4
    else:
        default_gaps, default_t_stop = GAPS, 60.0
        geometry = dict(width=WIDTH, margin=MARGIN, pad=PAD)
        n_azimuth = 10
    gaps = tuple(default_gaps if gaps is None else gaps)
    t_stop = default_t_stop if t_stop is None else t_stop
    if width is not None:
        geometry["width"] = width
    if len(gaps) < 2:
        raise ValueError("at least two oxide gaps are needed for the sweep")

    runs = []
    for gap in gaps:
        params, device, trilayer = _build(gap, h, **geometry)
        _carve_defect(params, device, trilayer)
        x0 = _seed_bottom_vortex(params, device, trilayer)
        solution, residual = _relax(params, device, x0, t_stop)

        kb, kt = _layer_midplanes(trilayer)
        r_flux, r_fluxoid = _radii(params, device, trilayer)
        profiles = _profiles(solution, params, trilayer, radius=r_flux)
        run = {
            "gap": gap, "params": params, "device": device, "trilayer": trilayer,
            "solution": solution, "profiles": profiles, "residual": residual,
            "n_azimuth": n_azimuth, "r_flux": r_flux,
            "fluxoid_bottom": _fluxoid(solution, device, params, kb, r_fluxoid),
            "fluxoid_top": _fluxoid(solution, device, params, kt, r_fluxoid),
            "flux_bottom": float(profiles["flux"][kb]),
            "flux_top": float(profiles["flux"][kt]),
            "psi_max": float(np.abs(solution.psi(-1)).max()),
            "vortex_bottom": _vortex_position(solution, params, kb),
            "vortex_top": _vortex_position(solution, params, kt),
        }
        runs.append(run)
        print(
            f"gap {gap:5.1f} ξ  grid {params.Nx}×{params.Ny}×{params.Nz}  "
            f"|dX/dt| {residual:.1e}  max|ψ| {run['psi_max']:.3f}  "
            f"fluxoid {run['fluxoid_bottom']:+.3f}/{run['fluxoid_top']:+.3f}  "
            f"flux {run['flux_bottom']:.3f} → {run['flux_top']:.3f} Φ₀  "
            f"core {run['vortex_bottom']}"
        )

    saved = [plot_isometric(runs, output_dir), plot_sweep(runs, output_dir)]
    transfer = [r["flux_top"] / r["flux_bottom"] for r in runs]
    print("flux reaching the top layer, as a fraction of the bottom layer's: "
          + ", ".join(f"{g:g} ξ: {t:.1%}" for g, t in zip(gaps, transfer)))
    for p in saved:
        print(f"wrote {p}")
    return saved


def _cli(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--gaps", type=float, nargs="+", default=list(GAPS), metavar="XI",
        help="metal-to-metal oxide gaps to sweep, in xi (each must be >= 3h)",
    )
    parser.add_argument("--t-stop", type=float, default=60.0,
                        help="relaxation time, in tau_GL")
    parser.add_argument("--width", type=float, default=WIDTH,
                        help="film width in x and y, in xi")
    parser.add_argument("--h", type=float, default=1.0, help="grid spacing, in xi")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent,
                        help="directory to write the PNGs into")
    parser.add_argument("--small", action="store_true",
                        help="smoke-test size: a small stack and two gaps")
    args = parser.parse_args(argv)
    return main(args.out, small=args.small, gaps=args.gaps, t_stop=args.t_stop,
                width=args.width, h=args.h)


if __name__ == "__main__":
    _cli()

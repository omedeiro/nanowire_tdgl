"""Field-cooled S/I/S film with a grid of moats: where the flux ends up.

The device is two superconducting films separated by an oxide, with a square
grid of moats (square cut-outs) etched through the whole stack and vacuum
above and below.  It is the geometry of a ground plane in an SFQ process,
where the moats exist to take flux that would otherwise be pinned in the
film next to a circuit.

Every run starts from a **random order parameter** — random amplitude and
random phase at every superconducting node, |ψ| ≤ ``PSI0_AMPLITUDE`` — with
the applied field already uniform through the box.  That is the numerical
stand-in for cooling through T_c in a field: the condensate forms with the
flux already inside it, at a density set by the field, and the vortices then
sort themselves into moats, into the film between the moats, or out through
the edge.  The link variables are seeded in the symmetric gauge so the field
is uniform at t = 0; started from φ = 0 the field would have to diffuse in
from the box walls, which on this size of film takes longer than the
condensate takes to form, and the film would screen it out.

The two metal layers are coupled only through the field — there is no
Josephson term — so each layer nucleates its own vortex configuration from
its own random start.  A vortex pinned in the film on one layer therefore
need not have a partner on the other, and the section view picks one such
vortex out.

Per applied field, four files:

``sis_moat_array_B{B}_layers.png``
    |ψ|², arg ψ and the in-plane supercurrent on the mid-plane of each metal
    layer, with the fluxoid every moat holds written on it and the vortices
    pinned in the film marked.

``sis_moat_array_B{B}_tracks.png``
    The path of every vortex from the moment the condensate has formed,
    coloured by where it ended: captured by a moat, pinned in the film,
    annihilated with an antivortex, or out through the edge.

``sis_moat_array_B{B}_section.png``
    A vertical section through a vortex that is in the film on one layer
    only, with |ψ|² and B_z through the whole stack, and the two layers'
    film vortices overlaid on one map.

``sis_moat_array_B{B}.gif``
    |ψ|² on both layers through the run, moat fluxoids labelled.

And across the sweep, ``sis_moat_array_sweep.png``: how many vortices sit in
the film and how many quanta the moats hold against applied field, on each
layer, with the per-moat occupancy — the field at which the moats fill up is
where the film count leaves the 1–10 band.

Full resolution is 92 × 92 × 16 (124 k interior nodes) and takes roughly
5 minutes per field on four cores, four of them solving.  ``small=True`` runs a 22 × 22 × 16 device
for the smoke test.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle
from tdgl3d import (
    AppliedField,
    Device,
    Layer,
    SimulationParameters,
    Solution,
    Trilayer,
    solve,
)
from tdgl3d.analysis.expulsion import rectangular_contour
from tdgl3d.analysis.vortex_counting import (
    count_vortices_plaquette,
    count_vortices_polygon,
)
from tdgl3d.visualization.plotting import imshow_extent

KAPPA = 2.0
#: In-plane grid spacing (ξ).  A vortex core is ξ across, so this puts a bit
#: more than one node in it; the winding count is topological and does not
#: need more, the pictures do.
H = 0.75
#: Out-of-plane spacing (ξ).  The layers are thick in cells, not in ξ.
HZ = 1.0
#: Superconductor / oxide / vacuum thicknesses in cells.  The oxide suppresses
#: ψ over about a coherence length on each side, so a 4 ξ metal layer keeps
#: |ψ|² ≈ 0.75 on its mid-plane; a 1 ξ layer would be pair-broken through.
SC_CELLS, INS_CELLS, VAC_CELLS = 4, 2, 3
#: Applied fields in units of Φ₀/(2πξ²) = H_c2.
FIELDS = (0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
T_STOP = 150.0
FRAMES = 75
#: |ψ| at t = 0 is uniform on [0, PSI0_AMPLITUDE] with a uniformly random phase.
PSI0_AMPLITUDE = 0.1
SEED = 7
#: 0.9 of the 3-D forward-Euler limit h²/(4κ²(d−1)) at the smallest spacing.
DT = 0.9 * min(H, HZ) ** 2 / (4 * KAPPA**2 * 2)
#: A vortex this close to a moat (ξ) belongs to the moat, not the film, and the
#: contour that reads each moat's fluxoid runs at this distance from its edge.
MOAT_MARGIN = 1.5
#: Vortex paths are traced from here on; before it the condensate is still
#: forming and the plaquette count is a soup of transient pairs.
T_TRACK_FROM = 10.0
#: Vortices within this distance (ξ) on the two layers count as the same one.
LAYER_MATCH = 2.5

LAYERS = ("bottom", "top")


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MoatGrid:
    """A square array of square moats, centred in a square film."""

    n: int = 4            #: moats per side
    side: float = 6.0     #: moat side (ξ)
    pitch: float = 15.0   #: centre-to-centre spacing (ξ)
    buffer: float = 9.0   #: unbroken film between the array and the box wall (ξ)

    @property
    def film(self) -> float:
        return (self.n - 1) * self.pitch + self.side + 2 * self.buffer

    def rects(self) -> list[tuple[float, float, float, float]]:
        """``(x0, y0, x1, y1)`` of every moat in ξ, row-major from the origin."""
        out = []
        for row in range(self.n):
            for col in range(self.n):
                x0 = self.buffer + col * self.pitch
                y0 = self.buffer + row * self.pitch
                out.append((x0, y0, x0 + self.side, y0 + self.side))
        return out


@dataclass
class Spec:
    """Everything the census and the plots need to know about one device."""

    grid: MoatGrid
    params: SimulationParameters
    trilayer: Trilayer
    rects: list[tuple[float, float, float, float]]
    contours: list[np.ndarray]     #: node-coordinate contours around each moat
    slices: dict[str, int]         #: interior z-slice of each metal mid-plane

    @property
    def side(self) -> float:
        return self.params.Nx * self.params.hx

    def xs(self) -> np.ndarray:
        return np.arange(1, self.params.Nx) * self.params.hx

    def ys(self) -> np.ndarray:
        return np.arange(1, self.params.Ny) * self.params.hy

    def zs(self) -> np.ndarray:
        return np.arange(1, self.params.Nz) * self.params.hz

    def in_moat_zone(self, xy: np.ndarray) -> np.ndarray:
        """True for points inside any moat's counting contour."""
        if len(xy) == 0:
            return np.zeros(0, dtype=bool)
        hx, hy = self.params.hx, self.params.hy
        inside = np.zeros(len(xy), dtype=bool)
        for c in self.contours:
            x_lo, x_hi = c[0, 0] * hx, c[1, 0] * hx
            y_lo, y_hi = c[0, 1] * hy, c[2, 1] * hy
            inside |= (
                (xy[:, 0] > x_lo) & (xy[:, 0] < x_hi)
                & (xy[:, 1] > y_lo) & (xy[:, 1] < y_hi)
            )
        return inside

    def near_edge(self, xy: np.ndarray, margin: float = 2.0) -> np.ndarray:
        s = self.side
        return (
            (xy[:, 0] < margin) | (xy[:, 0] > s - margin)
            | (xy[:, 1] < margin) | (xy[:, 1] > s - margin)
        )

    def sc_mask_3d(self, device: Device) -> np.ndarray:
        p = self.params
        return device.material.interior_sc_mask.reshape(p.Nx - 1, p.Ny - 1, p.Nz - 1)


def build(bz: float, grid: MoatGrid) -> tuple[Device, Spec]:
    """The S/I/S stack with the moats carved through it, at applied field *bz*."""
    trilayer = Trilayer(
        bottom=Layer(thickness_z=SC_CELLS, kappa=KAPPA),
        # κ on a non-superconducting layer carries no physics: the Maxwell
        # coefficient is the field energy and takes params.kappa everywhere.
        insulator=Layer(thickness_z=INS_CELLS, kappa=KAPPA, is_superconductor=False),
        top=Layer(thickness_z=SC_CELLS, kappa=KAPPA),
        vacuum_below=VAC_CELLS,
        vacuum_above=VAC_CELLS,
    )
    n_side = int(round(grid.film / H))
    params = SimulationParameters(
        Nx=n_side, Ny=n_side, Nz=trilayer.Nz, hx=H, hy=H, hz=HZ, kappa=KAPPA
    )
    device = Device(
        params,
        applied_field=AppliedField(Bz=bz, t_on_fraction=1.0),
        trilayer=trilayer,
    )
    rects = grid.rects()
    for x0, y0, x1, y1 in rects:
        device.add_hole([(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
                        z_range=(0, trilayer.Nz))
    contours = [
        rectangular_contour((x0, x1, y0, y1), params, margin=MOAT_MARGIN)
        for x0, y0, x1, y1 in rects
    ]
    # Mid-plane of each metal layer.  Interior arrays run k = 1 … Nz-1, so
    # interior slice s is full-grid plane s + 1.  Both oxide interfaces belong
    # to the insulator, so the metal layers are mirror images and so are the
    # two mid-planes.
    r = trilayer.z_ranges()
    k_bottom = (r["bottom"][0] + r["bottom"][1]) // 2
    k_top = r["top"][0] + (r["top"][1] - r["top"][0]) // 2
    slices = {"bottom": k_bottom - 1, "top": k_top - 1}
    return device, Spec(grid, params, trilayer, rects, contours, slices)


def field_cooled_state(device: Device, bz: float, seed: int, amplitude: float):
    """Random ψ in the metal, and links carrying a uniform field B_z = *bz*.

    The link variable on a bond is ∫A·dl along it; in the symmetric gauge
    about the box centre, A = (−B y, B x, 0)/2, so every plaquette carries
    B·hx·hy from the first step and the box wall condition (which fixes the
    flux through the ghost plaquettes to the applied value) is met exactly.
    """
    p = device.params
    state = device.initial_state(noise_amplitude=0.0)
    i, j, k = np.arange(1, p.Nx), np.arange(1, p.Ny), np.arange(1, p.Nz)
    ii, jj, _ = np.meshgrid(i, j, k, indexing="ij")
    x = (ii - 0.5 * p.Nx) * p.hx
    y = (jj - 0.5 * p.Ny) * p.hy
    state.phi_x[:] = (-0.5 * bz * y * p.hx).ravel()
    state.phi_y[:] = (0.5 * bz * x * p.hy).ravel()

    rng = np.random.default_rng(seed)
    n = p.n_interior
    state.psi[:] = (
        device.material.interior_sc_mask
        * amplitude
        * rng.uniform(0.0, 1.0, n)
        * np.exp(1j * rng.uniform(-np.pi, np.pi, n))
    )
    return state


# ---------------------------------------------------------------------------
# Census: what is in the film, what is in the moats
# ---------------------------------------------------------------------------

def census(solution: Solution, spec: Spec, step: int) -> dict[str, dict]:
    """Vortices in the film and the fluxoid in every moat, per layer.

    A vortex in the film is a core — a plaquette carrying 2π of gauge-invariant
    winding.  A moat has no core: what it holds is a fluxoid, read from the
    winding on a contour in the metal around it, which is an exact integer.
    A core inside that contour is counted with the moat, not the film.
    """
    p = spec.params
    out = {}
    for layer, sz in spec.slices.items():
        _, pos, wind = count_vortices_plaquette(solution, None, slice_z=sz, step=step)
        xy = (np.asarray(pos, dtype=float) + 1.0) * np.array([p.hx, p.hy])
        charge = np.rint(np.asarray(wind, dtype=float)).astype(int)
        in_zone = spec.in_moat_zone(xy)
        moats = [
            int(round(count_vortices_polygon(solution, None, c, slice_z=sz, step=step)))
            for c in spec.contours
        ]
        out[layer] = {
            "all_xy": xy,
            "all_q": charge,
            "film_xy": xy[~in_zone],
            "film_q": charge[~in_zone],
            "n_film": int(np.abs(charge[~in_zone]).sum()) if len(charge) else 0,
            "moats": moats,
            "n_moats": int(sum(abs(m) for m in moats)),
        }
    return out


def census_history(solution: Solution, spec: Spec) -> list[dict[str, dict]]:
    return [census(solution, spec, s) for s in range(solution.n_steps)]


def _jsonable(c: dict[str, dict]) -> dict:
    return {
        layer: {
            "n_film": d["n_film"],
            "n_moats": d["n_moats"],
            "moats": d["moats"],
            "film_xy": np.asarray(d["film_xy"]).round(3).tolist(),
            "film_q": [int(q) for q in d["film_q"]],
        }
        for layer, d in c.items()
    }


# ---------------------------------------------------------------------------
# Vortex tracking
# ---------------------------------------------------------------------------

def track_vortices(
    history: list[dict[str, dict]],
    times: np.ndarray,
    spec: Spec,
    layer: str,
    t_from: float = T_TRACK_FROM,
    max_jump: float = 4.0,
) -> list[dict]:
    """Link vortex cores frame to frame into paths, and say how each ended.

    Greedy nearest-neighbour linking, same charge only, no jump longer than
    *max_jump* ξ between saved frames.  A path that ends inside a moat's
    contour was captured by it; one that ends at the box wall left; one that
    ends elsewhere annihilated with an antivortex.  A path still alive in the
    last frame is pinned in the film (or sits on a moat rim, if inside the
    contour).
    """
    tracks: list[dict] = []
    last = len(times) - 1
    for s, t in enumerate(times):
        if t < t_from:
            continue
        xy = history[s][layer]["all_xy"]
        q = history[s][layer]["all_q"]
        active = [tr for tr in tracks if tr["alive"]]
        matched_tr: set[int] = set()
        matched_pt: set[int] = set()
        if active and len(xy):
            prev = np.array([tr["xy"][-1] for tr in active])
            d = np.linalg.norm(prev[:, None, :] - xy[None, :, :], axis=-1)
            qa = np.array([tr["q"] for tr in active])
            d[qa[:, None] != q[None, :]] = np.inf
            order = np.argsort(d, axis=None)
            for flat in order:
                a, b = np.unravel_index(flat, d.shape)
                if d[a, b] > max_jump:
                    break
                if a in matched_tr or b in matched_pt:
                    continue
                matched_tr.add(a)
                matched_pt.add(b)
                active[a]["xy"].append(xy[b])
                active[a]["steps"].append(s)
        for a, tr in enumerate(active):
            if a not in matched_tr:
                tr["alive"] = False
        for b in range(len(xy)):
            if b not in matched_pt:
                tracks.append({"q": int(q[b]), "xy": [xy[b]], "steps": [s], "alive": True})

    for tr in tracks:
        tr["xy"] = np.array(tr["xy"])
        end = tr["xy"][-1:]
        if tr["steps"][-1] == last:
            tr["fate"] = "moat" if spec.in_moat_zone(end)[0] else "film"
        elif spec.in_moat_zone(end)[0]:
            tr["fate"] = "moat"
        elif spec.near_edge(end)[0]:
            tr["fate"] = "edge"
        else:
            tr["fate"] = "annihilated"
    return tracks


def layer_comparison(c: dict[str, dict], match: float = LAYER_MATCH) -> dict[str, np.ndarray]:
    """Film vortices present on both layers, on the bottom only, on the top only."""
    b, t = c["bottom"]["film_xy"], c["top"]["film_xy"]
    if len(b) == 0 or len(t) == 0:
        return {"both": np.empty((0, 2)), "bottom_only": b, "top_only": t}
    d = np.linalg.norm(b[:, None, :] - t[None, :, :], axis=-1)
    b_has = d.min(axis=1) <= match
    t_has = d.min(axis=0) <= match
    return {"both": b[b_has], "bottom_only": b[~b_has], "top_only": t[~t_has]}


def lonely_vortex(c: dict[str, dict]) -> Optional[tuple[str, np.ndarray, float]]:
    """The film vortex furthest from any vortex on the other layer.

    Returns ``(layer, xy, distance)``; ``None`` when neither layer has a film
    vortex.
    """
    best = None
    for layer, other in (("bottom", "top"), ("top", "bottom")):
        mine, theirs = c[layer]["film_xy"], c[other]["all_xy"]
        for xy in mine:
            dist = (
                float(np.linalg.norm(theirs - xy, axis=1).min()) if len(theirs) else np.inf
            )
            if best is None or dist > best[2]:
                best = (layer, xy, dist)
    return best


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _slice(arr: np.ndarray, spec: Spec, sz: int) -> np.ndarray:
    p = spec.params
    return np.asarray(arr).reshape(p.Nx - 1, p.Ny - 1, p.Nz - 1)[:, :, sz]


def _draw_moats(ax, spec: Spec, labels=None, color="#7fd4ff", lw=0.8, fontsize=10):
    texts = []
    for m, (x0, y0, x1, y1) in enumerate(spec.rects):
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                               edgecolor=color, lw=lw, alpha=0.8))
        if labels is not None:
            n = labels[m]
            texts.append(ax.text(0.5 * (x0 + x1), 0.5 * (y0 + y1), str(n) if n else "",
                                 color=color, ha="center", va="center",
                                 fontsize=fontsize, fontweight="bold"))
    return texts


def _mark_film_vortices(ax, xy, q, **kw):
    if len(xy) == 0:
        return
    pos, neg = q > 0, q < 0
    style = dict(s=70, facecolors="none", linewidths=1.3)
    style.update(kw)
    if pos.any():
        ax.scatter(xy[pos, 0], xy[pos, 1], marker="o", edgecolors="#ff9f43", **style)
    if neg.any():
        ax.scatter(xy[neg, 0], xy[neg, 1], marker="^", edgecolors="#54a0ff", **style)


def plot_layers(solution: Solution, device: Device, spec: Spec, c: dict, bz: float,
                path: Path, step: int = -1) -> Path:
    """|ψ|², arg ψ and |J_s| on both metal mid-planes, moats labelled."""
    extent = imshow_extent(spec.xs(), spec.ys())
    mask3 = spec.sc_mask_3d(device)
    phase = solution.phase(step=step)
    jx, jy, _ = solution.supercurrent_density(step=step)

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 10.5), constrained_layout=True)
    for row, layer in enumerate(LAYERS):
        sz = spec.slices[layer]
        d = c[layer]
        hole = mask3[:, :, sz] == 0

        psi2 = solution.psi_squared_2d(step, slice_z=sz)
        ax = axes[row, 0]
        im = ax.imshow(psi2.T, origin="lower", extent=extent, cmap="inferno",
                       vmin=0.0, vmax=1.0)
        _draw_moats(ax, spec, labels=d["moats"])
        _mark_film_vortices(ax, d["film_xy"], d["film_q"])
        ax.set_title(f"{layer} layer — $|\\psi|^2$: {d['n_film']} in film, "
                     f"{d['n_moats']} quanta in moats")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

        th = _slice(phase, spec, sz)
        ax = axes[row, 1]
        im = ax.imshow(th.T, origin="lower", extent=extent, cmap="twilight",
                       vmin=-np.pi, vmax=np.pi)
        _draw_moats(ax, spec, color="white")
        ax.set_title(f"{layer} layer — arg $\\psi$")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.set_ticks([-np.pi, 0, np.pi])
        cb.set_ticklabels(["−π", "0", "π"])

        jxs, jys = _slice(jx, spec, sz), _slice(jy, spec, sz)
        jmag = np.hypot(jxs, jys)
        jmag_plot = np.where(hole, np.nan, jmag)
        ax = axes[row, 2]
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("#202020")
        im = ax.imshow(jmag_plot.T, origin="lower", extent=extent, cmap=cmap,
                       vmin=0.0, vmax=float(np.nanpercentile(jmag_plot, 99.5)) or 1.0)
        if jmag.max() > 1e-10:
            ax.streamplot(spec.xs(), spec.ys(), jxs.T, jys.T, color="white",
                          density=1.4, linewidth=0.5, arrowsize=0.6)
        _draw_moats(ax, spec)
        ax.set_title(f"{layer} layer — $|J_s|$ with streamlines")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    for ax in axes.ravel():
        ax.set_xlim(0, spec.side)
        ax.set_ylim(0, spec.side)
        ax.set_xlabel("x (ξ)")
        ax.set_ylabel("y (ξ)")
    t = float(solution.times[step])
    fig.suptitle(
        f"Field-cooled S/I/S film with {len(spec.rects)} moats, "
        f"$B_z$ = {bz:g} $H_{{c2}}$, t = {t:.0f} τ$_{{GL}}$   "
        f"(○ +1 vortex, △ −1 vortex in the film; number = fluxoid held by the moat)",
        fontsize=13,
    )
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


FATE_STYLE = {
    "moat": ("#7fd4ff", "captured by a moat"),
    "film": ("#ff9f43", "pinned in the film"),
    "annihilated": ("#b0b0b0", "annihilated"),
    "edge": ("#c44dff", "left through the edge"),
}


def plot_tracks(solution: Solution, spec: Spec, tracks: dict[str, list], c: dict,
                bz: float, path: Path) -> Path:
    """Every vortex path from t = T_TRACK_FROM on, coloured by its fate."""
    extent = imshow_extent(spec.xs(), spec.ys())
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.4), constrained_layout=True)
    for ax, layer in zip(axes, LAYERS):
        sz = spec.slices[layer]
        ax.imshow(solution.psi_squared_2d(-1, slice_z=sz).T, origin="lower",
                  extent=extent, cmap="gray", vmin=0.0, vmax=1.0, alpha=0.75)
        _draw_moats(ax, spec, labels=c[layer]["moats"], color="#7fd4ff")
        counts = {k: 0 for k in FATE_STYLE}
        for tr in tracks[layer]:
            color, _ = FATE_STYLE[tr["fate"]]
            counts[tr["fate"]] += 1
            xy = tr["xy"]
            if len(xy) > 1:
                ax.plot(xy[:, 0], xy[:, 1], color=color, lw=1.4, alpha=0.9)
            ax.plot(xy[0, 0], xy[0, 1], marker=".", color=color, ms=4)
            end_marker = "o" if tr["q"] > 0 else "^"
            ax.plot(xy[-1, 0], xy[-1, 1], marker=end_marker, color=color,
                    ms=7 if tr["fate"] == "film" else 4,
                    mfc="none" if tr["fate"] == "film" else color)
        for fate, (color, label) in FATE_STYLE.items():
            ax.plot([], [], color=color, lw=2, label=f"{label} ({counts[fate]})")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
        ax.set_xlim(0, spec.side)
        ax.set_ylim(0, spec.side)
        ax.set_xlabel("x (ξ)")
        ax.set_ylabel("y (ξ)")
        ax.set_title(f"{layer} layer — {c[layer]['n_film']} pinned in the film, "
                     f"{c[layer]['n_moats']} quanta in moats")
    fig.suptitle(
        f"Vortex paths from t = {T_TRACK_FROM:g} τ$_{{GL}}$ to the end, "
        f"$B_z$ = {bz:g} $H_{{c2}}$   (dot: start; ○/△: end of a ±1 vortex)",
        fontsize=13,
    )
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_section(solution: Solution, device: Device, spec: Spec, c: dict, bz: float,
                 path: Path, half_width: float = 9.0) -> Path:
    """A vertical cut through a vortex that is in the film on one layer only."""
    p = spec.params
    xs, ys, zs = spec.xs(), spec.ys(), spec.zs()
    mask3 = spec.sc_mask_3d(device)
    psi2 = solution.psi_squared(-1).reshape(p.Nx - 1, p.Ny - 1, p.Nz - 1)
    bz3 = np.asarray(solution.bfield(-1)[2]).reshape(p.Nx - 1, p.Ny - 1, p.Nz - 1)
    comp = layer_comparison(c)

    pick = lonely_vortex(c)
    if pick is None:
        layer, xy, dist = "bottom", np.array([0.5 * spec.side, 0.5 * spec.side]), np.nan
        caption = "no vortex pinned in the film on either layer"
    else:
        layer, xy, dist = pick
        other = "top" if layer == "bottom" else "bottom"
        caption = (f"vortex pinned in the {layer} layer at ({xy[0]:.1f}, {xy[1]:.1f}) ξ; "
                   f"nearest vortex on the {other} layer is {dist:.1f} ξ away")
    j = int(np.clip(np.rint(xy[1] / p.hy) - 1, 0, p.Ny - 2))
    x_lo, x_hi = max(xy[0] - half_width, 0.0), min(xy[0] + half_width, spec.side)
    i_lo = int(np.clip(np.floor(x_lo / p.hx) - 1, 0, p.Nx - 2))
    i_hi = int(np.clip(np.ceil(x_hi / p.hx) - 1, 0, p.Nx - 2)) + 1

    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0])
    ax_psi = fig.add_subplot(gs[0, 0:2])
    ax_bz = fig.add_subplot(gs[1, 0:2])
    ax_map = fig.add_subplot(gs[0:2, 2])

    ext = imshow_extent(xs[i_lo:i_hi], zs)
    r = spec.trilayer.z_ranges()
    for ax, cube, cmap, vmin, vmax, label in (
        (ax_psi, psi2, "inferno", 0.0, 1.0, "$|\\psi|^2$"),
        (ax_bz, bz3, "coolwarm", None, None, "$B_z$"),
    ):
        sec = cube[i_lo:i_hi, j, :].copy()
        if label.startswith("$|"):
            sec[mask3[i_lo:i_hi, j, :] == 0] = np.nan
        else:
            lim = float(np.nanmax(np.abs(sec))) or 1.0
            vmin, vmax = -lim, lim
        cm = plt.get_cmap(cmap).copy()
        cm.set_bad("#3a3a3a")
        im = ax.imshow(sec.T, origin="lower", extent=ext, cmap=cm, vmin=vmin, vmax=vmax,
                       aspect="auto")
        for name in ("bottom", "insulator", "top"):
            k0, k1 = r[name]
            ax.axhline(k0 * p.hz, color="white", lw=0.6, ls=":")
            ax.axhline(k1 * p.hz, color="white", lw=0.6, ls=":")
            ax.text(x_lo + 0.6, 0.5 * (k0 + k1) * p.hz, name, color="white", fontsize=9,
                    va="center", clip_on=True)
        ax.axvline(xy[0], color="#ff9f43", lw=1.0, ls="--")
        ax.set_ylabel("z (ξ)")
        ax.set_title(f"{label} on the x–z plane at y = {ys[j]:.1f} ξ")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label=label)
    ax_bz.set_xlabel("x (ξ)")

    # Both layers' film vortices on one map, so the asymmetry is visible at a glance.
    ax = ax_map
    both = np.minimum(psi2[:, :, spec.slices["bottom"]], psi2[:, :, spec.slices["top"]])
    ax.imshow(both.T, origin="lower", extent=imshow_extent(xs, ys), cmap="gray",
              vmin=0.0, vmax=1.0, alpha=0.7)
    _draw_moats(ax, spec, color="#7fd4ff")
    for key, color, marker, label in (
        ("both", "white", "o", "in the film on both layers"),
        ("bottom_only", "#ff5e5e", "s", "bottom layer only"),
        ("top_only", "#5ea8ff", "D", "top layer only"),
    ):
        pts = comp[key]
        ax.scatter(pts[:, 0], pts[:, 1], s=80, facecolors="none", edgecolors=color,
                   marker=marker, linewidths=1.5, label=f"{label} ({len(pts)})")
    ax.plot([x_lo, x_hi], [ys[j], ys[j]], color="#ff9f43", lw=1.2, ls="--",
            label="section line")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    ax.set_xlim(0, spec.side)
    ax.set_ylim(0, spec.side)
    ax.set_xlabel("x (ξ)")
    ax.set_ylabel("y (ξ)")
    ax.set_title("film vortices on the two layers")

    fig.suptitle(f"Section view, $B_z$ = {bz:g} $H_{{c2}}$ — {caption}", fontsize=13)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def animate(solution: Solution, spec: Spec, history: list[dict], bz: float, path: Path,
            fps: int = 8) -> Path:
    """|ψ|² on both metal mid-planes through the run, moat fluxoids labelled."""
    extent = imshow_extent(spec.xs(), spec.ys())
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.2), constrained_layout=True)
    images, labels, titles = [], {}, {}
    for ax, layer in zip(axes, LAYERS):
        sz = spec.slices[layer]
        im = ax.imshow(solution.psi_squared_2d(0, slice_z=sz).T, origin="lower",
                       extent=extent, cmap="inferno", vmin=0.0, vmax=1.0)
        images.append(im)
        labels[layer] = _draw_moats(ax, spec, labels=[0] * len(spec.rects), fontsize=11)
        titles[layer] = ax.set_title("")
        ax.set_xlabel("x (ξ)")
        ax.set_ylabel("y (ξ)")
    fig.colorbar(images[-1], ax=axes, fraction=0.03, pad=0.02, label="$|\\psi|^2$")
    sup = fig.suptitle("")

    def update(step):
        for im, layer in zip(images, LAYERS):
            im.set_data(solution.psi_squared_2d(step, slice_z=spec.slices[layer]).T)
            d = history[step][layer]
            for text, n in zip(labels[layer], d["moats"]):
                text.set_text(str(n) if n else "")
            titles[layer].set_text(
                f"{layer}: {d['n_film']} in film, {d['n_moats']} in moats")
        t = float(solution.times[step])
        sup.set_text(f"$B_z$ = {bz:g} $H_{{c2}}$    t = {t:6.1f} τ$_{{GL}}$")
        return [*images, sup, *titles.values(), *labels["bottom"], *labels["top"]]

    FuncAnimation(fig, update, frames=solution.n_steps, blit=False).save(
        str(path), writer=PillowWriter(fps=fps)
    )
    plt.close(fig)
    return path


def plot_sweep(results: dict[float, dict], spec: Spec, path: Path,
               band: tuple[int, int] = (1, 10)) -> Path:
    """Film vortices and moat quanta against applied field, and the moat occupancy."""
    fields = sorted(results)
    area = spec.side ** 2
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), constrained_layout=True)

    ax = axes[0]
    ax.axhspan(band[0], band[1], color="#2ecc71", alpha=0.15,
               label=f"{band[0]}–{band[1]} in the film")
    ax.plot(fields, [b * area / (2 * np.pi) for b in fields], color="#888", ls="--",
            label="flux through the film, $B A / \\Phi_0$")
    for layer, marker in zip(LAYERS, ("o", "s")):
        ax.plot(fields, [results[b]["final"][layer]["n_film"] for b in fields],
                marker=marker, color="#ff9f43", ls="-" if layer == "bottom" else ":",
                label=f"{layer}: vortices in the film")
        ax.plot(fields, [results[b]["final"][layer]["n_moats"] for b in fields],
                marker=marker, color="#3498db", ls="-" if layer == "bottom" else ":",
                label=f"{layer}: quanta in the moats")
    ax.set_xlabel("$B_z$ ($H_{c2}$)")
    ax.set_ylabel("count")
    ax.set_title("where the flux ends up")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    n_moats = len(spec.rects)
    for layer, color in zip(LAYERS, ("#3498db", "#e74c3c")):
        occ = np.array([results[b]["final"][layer]["moats"] for b in fields], dtype=float)
        for m in range(n_moats):
            ax.plot(fields, np.abs(occ[:, m]), color=color, alpha=0.25, lw=0.8)
        ax.plot(fields, np.abs(occ).mean(axis=1), color=color, marker="o", lw=2,
                label=f"{layer}: mean per moat")
        ax.plot(fields, np.abs(occ).max(axis=1), color=color, marker="^", lw=1, ls="--",
                label=f"{layer}: fullest moat")
    ax.set_xlabel("$B_z$ ($H_{c2}$)")
    ax.set_ylabel("fluxoid per moat ($\\Phi_0$)")
    ax.set_title(f"moat occupancy ({n_moats} moats of {spec.grid.side:g} ξ, "
                 f"thin lines: each moat)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    cmap = plt.get_cmap("plasma")
    for k, b in enumerate(fields):
        color = cmap(k / max(len(fields) - 1, 1))
        t = results[b]["times"]
        ax.plot(t, results[b]["n_film"]["bottom"], color=color, lw=1.5, label=f"B = {b:g}")
        ax.plot(t, results[b]["n_film"]["top"], color=color, lw=1.0, ls=":")
    ax.axvline(T_TRACK_FROM, color="#888", lw=0.8, ls="--")
    ax.set_xlabel("t (τ$_{GL}$)")
    ax.set_ylabel("vortices in the film")
    ax.set_title("settling (solid: bottom, dotted: top)")
    ax.set_yscale("symlog", linthresh=10)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("Field-cooled S/I/S film with a moat grid — sweep over applied field",
                 fontsize=13)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_field(bz: float, grid: MoatGrid, output_dir: Path, *, t_stop: float,
              frames: int, seed: int, amplitude: float, reuse: bool,
              gif: bool = True) -> tuple[dict, list[Path]]:
    tag = f"B{bz:.2f}"
    # Not Path.with_suffix: it would take the ".10" of "B0.10" for a suffix.
    stem = output_dir / f"sis_moat_array_{tag}"
    device, spec = build(bz, grid)
    h5 = Path(f"{stem}.h5")

    if reuse and h5.exists():
        solution = Solution.load(str(h5))
        solution.device = device
        wall = float((solution.metadata or {}).get("wall_seconds", np.nan))
        print(f"[{tag}] loaded {h5.name}: {solution.n_steps} frames")
    else:
        x0 = field_cooled_state(device, bz, seed, amplitude)
        n_steps = int(round(t_stop / DT))
        print(f"[{tag}] {device}  interior nodes: {spec.params.n_interior}  "
              f"dt = {DT:.4f}  steps: {n_steps}")
        t0 = time.perf_counter()
        solution = solve(device, t_stop=t_stop, dt=DT, method="euler", x0=x0,
                         save_every=max(n_steps // frames, 1),
                         progress=False, log_metadata=False)
        wall = time.perf_counter() - t0
        solution.metadata = {"wall_seconds": wall, "bz": bz, "seed": seed,
                             "psi0_amplitude": amplitude, "dt": DT}
        solution.save(str(h5))
        print(f"[{tag}] solved in {wall:.0f} s, wrote {h5.name}")

    history = census_history(solution, spec)
    final = history[-1]
    print(f"[{tag}]     t   bottom(film/moats)   top(film/moats)   per-moat (bottom)")
    for s in range(0, solution.n_steps, max(solution.n_steps // 15, 1)):
        c = history[s]
        print(f"[{tag}] {float(solution.times[s]):6.1f}   "
              f"{c['bottom']['n_film']:4d} / {c['bottom']['n_moats']:<4d}       "
              f"{c['top']['n_film']:4d} / {c['top']['n_moats']:<4d}     {c['bottom']['moats']}")
    c = final
    print(f"[{tag}] final: bottom {c['bottom']['n_film']} in film / "
          f"{c['bottom']['n_moats']} in moats; top {c['top']['n_film']} in film / "
          f"{c['top']['n_moats']} in moats; max|ψ|² = "
          f"{float(solution.psi_squared_2d(-1, slice_z=spec.slices['bottom']).max()):.2f}")

    tracks = {layer: track_vortices(history, solution.times, spec, layer)
              for layer in LAYERS}
    saved = [
        plot_layers(solution, device, spec, final, bz, stem.parent / f"{stem.name}_layers.png"),
        plot_tracks(solution, spec, tracks, final, bz, stem.parent / f"{stem.name}_tracks.png"),
        plot_section(solution, device, spec, final, bz, stem.parent / f"{stem.name}_section.png"),
    ]
    if gif:
        saved.append(animate(solution, spec, history, bz, Path(f"{stem}.gif")))

    result = {
        "bz": bz,
        "wall_seconds": wall,
        "times": [float(t) for t in solution.times],
        "n_film": {layer: [h[layer]["n_film"] for h in history] for layer in LAYERS},
        "n_moats": {layer: [h[layer]["n_moats"] for h in history] for layer in LAYERS},
        "final": _jsonable(final),
        "layer_comparison": {k: np.asarray(v).round(2).tolist()
                             for k, v in layer_comparison(final).items()},
        "fates": {layer: {fate: sum(tr["fate"] == fate for tr in tracks[layer])
                          for fate in FATE_STYLE} for layer in LAYERS},
        "max_psi2": {layer: float(solution.psi_squared_2d(-1, slice_z=spec.slices[layer]).max())
                     for layer in LAYERS},
    }
    solution.close()
    return result, saved


def main(output_dir: Path | str = Path(__file__).parent, small: bool = False, *,
         fields=None, seed: int = SEED, t_stop: Optional[float] = None,
         frames: Optional[int] = None, amplitude: float = PSI0_AMPLITUDE,
         reuse: bool = False, gif: bool = True) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if small:
        grid = MoatGrid(n=2, side=3.0, pitch=7.5, buffer=3.0)
        fields = tuple(fields or (0.10, 0.30))
        t_stop = t_stop or 3.0
        frames = frames or 3
    else:
        grid = MoatGrid()
        fields = tuple(fields or FIELDS)
        t_stop = t_stop or T_STOP
        frames = frames or FRAMES

    saved: list[Path] = []
    results: dict[float, dict] = {}
    for bz in fields:
        result, paths = run_field(bz, grid, output_dir, t_stop=t_stop, frames=frames,
                                  seed=seed, amplitude=amplitude, reuse=reuse, gif=gif)
        results[bz] = result
        saved.extend(paths)

    _, spec = build(fields[0], grid)
    saved.append(plot_sweep(results, spec, output_dir / "sis_moat_array_sweep.png"))
    census_path = output_dir / "sis_moat_array_census.json"
    census_path.write_text(json.dumps(
        {"geometry": {"film_xi": spec.side, "grid": spec.params.Nx,
                      "moats": spec.rects, "kappa": KAPPA, "h": H, "hz": HZ,
                      "sc_cells": SC_CELLS, "ins_cells": INS_CELLS, "vac_cells": VAC_CELLS,
                      "dt": DT, "t_stop": t_stop, "seed": seed, "psi0_amplitude": amplitude},
         "fields": {f"{b:.2f}": r for b, r in results.items()}},
        indent=1))
    saved.append(census_path)
    return saved


def _cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path(__file__).parent)
    parser.add_argument("--fields", type=float, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--t-stop", type=float, default=None)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--amplitude", type=float, default=PSI0_AMPLITUDE)
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--reuse", action="store_true",
                        help="replot from an existing .h5 instead of solving again")
    parser.add_argument("--no-gif", action="store_true")
    args = parser.parse_args(argv)
    for path in main(args.out, args.small, fields=args.fields, seed=args.seed,
                     t_stop=args.t_stop, frames=args.frames, amplitude=args.amplitude,
                     reuse=args.reuse, gif=not args.no_gif):
        print(f"wrote {path}")


if __name__ == "__main__":
    _cli()

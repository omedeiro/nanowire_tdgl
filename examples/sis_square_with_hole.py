"""Example: 5 × 5 µm S/I/S square with a rectangular hole.

Geometry
--------
- 5 µm × 5 µm square centred at (0, 0) → bounds [-2.5, 2.5] µm
- S/I/S trilayer along z:  bottom SC (3 cells) / insulator (1 cell) / top SC (3 cells)
- Rectangular hole (non-superconducting) from (-1, -0.2) to (1, 0.2) µm
  punched through both SC layers

Physical scale
--------------
We choose ξ = 100 nm, so 5 µm = 50 ξ → Nx = Ny = 50.
Grid spacing hx = hy = hz = 1.0 (in ξ units).
The hole spans x ∈ [−1, 1] µm = [−10, 10] ξ and y ∈ [−0.2, 0.2] µm = [−2, 2] ξ.

Resolution
----------
Interior nodes: 49 × 49 × 6 ≈ 14 k.  Forward-Euler with dt = 0.02, t_stop = 50.0
gives 50 saved steps.  Should complete in under 30 min on a laptop.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt

import matplotlib.cm as cm
from matplotlib.colors import Normalize

import tdgl3d
from tdgl3d.core.solution import Solution


# ── Physical constants ─────────────────────────────────────────────────
XI_NM = 100.0                     # coherence length ξ = 100 nm
UM_PER_XI = XI_NM / 1000.0        # 0.1 µm per ξ

# Film dimensions in µm
LX_UM, LY_UM = 5.0, 5.0           # total size
# → in ξ units
LX_XI = LX_UM / UM_PER_XI         # 50 ξ
LY_XI = LY_UM / UM_PER_XI         # 50 ξ

# Grid
HX = HY = HZ = 1.0                # grid spacing in ξ
NX = int(LX_XI / HX)              # 50
NY = int(LY_XI / HY)              # 50
KAPPA = 2.0                        # GL parameter

# Trilayer stack: 3 SC + 1 I + 3 SC  →  Nz = 7
SC_THICKNESS = 3                   # z-cells per SC layer
INS_THICKNESS = 1                  # z-cells for insulator

# Hole bounds in µm (centred at origin)
HOLE_X_MIN_UM, HOLE_X_MAX_UM = -1.0, 1.0
HOLE_Y_MIN_UM, HOLE_Y_MAX_UM = -0.2, 0.2

# Time integration
DT = 0.02
T_STOP = 50.0
BZ = 0.5
SAVE_EVERY = 10


def um_to_grid_index(coord_um: float, L_um: float, N: int) -> int:
    """Convert a physical coordinate in µm to the nearest grid index.

    The grid spans [0, N] in index space, corresponding to
    [−L/2, +L/2] in physical space.
    """
    frac = (coord_um + L_um / 2.0) / L_um   # 0..1
    return int(round(frac * N))


def carve_hole(device: tdgl3d.Device) -> None:
    """Modify the device's MaterialMap in-place to create a rectangular hole.

    Nodes inside the hole rectangle (in x–y) within the superconducting
    z-layers are marked as insulator (sc_mask = 0, kappa = 0).
    The solver's FPSI operator will then drive ψ → 0 at those nodes via
    the fast relaxation term.
    """
    params = device.params
    material = device.material
    assert material is not None, "Device must have a MaterialMap (use a Trilayer)."

    Nx, Ny, Nz = params.Nx, params.Ny, params.Nz
    mj = Nx + 1
    mk = mj * (Ny + 1)

    # Convert hole bounds from µm to grid indices
    i_lo = um_to_grid_index(HOLE_X_MIN_UM, LX_UM, Nx)
    i_hi = um_to_grid_index(HOLE_X_MAX_UM, LX_UM, Nx)
    j_lo = um_to_grid_index(HOLE_Y_MIN_UM, LY_UM, Ny)
    j_hi = um_to_grid_index(HOLE_Y_MAX_UM, LY_UM, Ny)

    print(f"Hole grid indices: i ∈ [{i_lo}, {i_hi}], j ∈ [{j_lo}, {j_hi}]")

    # Get z-ranges for the trilayer
    z_ranges = device.trilayer.z_ranges()
    sc_z_planes = []
    for name in ("bottom", "top"):
        k_start, k_end = z_ranges[name]
        sc_z_planes.extend(range(k_start, k_end + 1))  # include boundary node
    # Also include the last node of the top layer
    sc_z_planes = sorted(set(sc_z_planes))

    # Zero out sc_mask and kappa inside the hole for SC z-planes
    count = 0
    for k in sc_z_planes:
        for j in range(j_lo, j_hi + 1):
            for i in range(i_lo, i_hi + 1):
                m = i + mj * j + mk * k
                if m < len(material.sc_mask):
                    material.sc_mask[m] = 0.0
                    material.kappa[m] = 0.0
                    count += 1

    # Rebuild interior_sc_mask from the updated full-grid sc_mask
    material.interior_sc_mask[:] = material.sc_mask[device.idx.interior_to_full]

    n_hole_interior = int(np.sum(material.interior_sc_mask == 0.0))
    print(f"Marked {count} full-grid nodes as hole (insulator).")
    print(f"Interior nodes with sc_mask=0: {n_hole_interior} / {params.n_interior}")


def plot_slices(solution: Solution, device: tdgl3d.Device) -> None:
    """Plot |ψ|² slices for the bottom SC, insulator, and top SC layers."""
    from tdgl3d.visualization.plotting import plot_order_parameter

    params = solution.params
    z_ranges = device.trilayer.z_ranges()

    # Pick representative z-slices (interior index = k - 1 for k >= 1)
    # Bottom SC midplane, insulator, top SC midplane
    k_bot = (z_ranges["bottom"][0] + z_ranges["bottom"][1]) // 2
    k_ins = (z_ranges["insulator"][0] + z_ranges["insulator"][1]) // 2
    k_top = (z_ranges["top"][0] + z_ranges["top"][1]) // 2

    # Interior z-index = k - 1 (since interior starts at k=1)
    slices = {
        f"Bottom SC (k={k_bot})": max(k_bot - 1, 0),
        f"Insulator (k={k_ins})": max(k_ins - 1, 0),
        f"Top SC (k={k_top})": max(k_top - 1, 0),
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (label, sz) in zip(axes, slices.items()):
        plot_order_parameter(solution, step=-1, slice_z=sz, ax=ax, title=label)

    fig.suptitle(
        f"S/I/S square with hole — |ψ|² at t = {solution.times[-1]:.2f}\n"
        f"5 × 5 µm, hole: ({HOLE_X_MIN_UM}, {HOLE_Y_MIN_UM}) → "
        f"({HOLE_X_MAX_UM}, {HOLE_Y_MAX_UM}) µm",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig("sis_square_with_hole_slices.png", dpi=150, bbox_inches="tight")
    print("Saved sis_square_with_hole_slices.png")
    plt.close(fig)


def plot_xy_overview(solution: Solution, device: tdgl3d.Device) -> None:
    """Side-by-side |ψ|² and phase for the bottom SC midplane."""
    params = solution.params
    z_ranges = device.trilayer.z_ranges()
    k_bot = (z_ranges["bottom"][0] + z_ranges["bottom"][1]) // 2
    sz = max(k_bot - 1, 0)

    # Physical-coordinate axes (centred at 0)
    xs_um = (np.arange(1, params.Nx) * params.hx) * UM_PER_XI - LX_UM / 2.0
    ys_um = (np.arange(1, params.Ny) * params.hy) * UM_PER_XI - LY_UM / 2.0
    XX, YY = np.meshgrid(xs_um, ys_um, indexing="ij")

    psi = solution.psi(step=-1)
    # Interior is C-order ravelled (Nx-1, Ny-1, Nz-1) with k fastest
    nz_int = max(params.Nz - 1, 1)
    psi_cube = psi.reshape(params.Nx - 1, params.Ny - 1, nz_int)
    psi_slice = psi_cube[:, :, sz]

    psi2 = np.abs(psi_slice) ** 2
    phase = np.angle(psi_slice)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # |ψ|²
    im1 = ax1.pcolormesh(XX, YY, psi2, cmap="inferno", vmin=0, vmax=1, shading="auto")
    ax1.set_aspect("equal")
    ax1.set_xlabel("x (µm)")
    ax1.set_ylabel("y (µm)")
    ax1.set_title(f"|ψ|² — bottom SC midplane (k={k_bot})")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # Draw hole outline
    for ax in (ax1, ax2):
        rect = plt.Rectangle(
            (HOLE_X_MIN_UM, HOLE_Y_MIN_UM),
            HOLE_X_MAX_UM - HOLE_X_MIN_UM,
            HOLE_Y_MAX_UM - HOLE_Y_MIN_UM,
            linewidth=1.5, edgecolor="white", facecolor="none", linestyle="--",
        )
        ax.add_patch(rect)

    # Phase
    im2 = ax2.pcolormesh(XX, YY, phase, cmap="twilight", vmin=-np.pi, vmax=np.pi, shading="auto")
    ax2.set_aspect("equal")
    ax2.set_xlabel("x (µm)")
    ax2.set_ylabel("y (µm)")
    ax2.set_title(f"arg(ψ) — bottom SC midplane (k={k_bot})")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="phase (rad)")

    fig.suptitle(
        f"S/I/S 5 × 5 µm square with 2 × 0.4 µm hole — t = {solution.times[-1]:.2f}",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig("sis_square_with_hole_overview.png", dpi=150, bbox_inches="tight")
    print("Saved sis_square_with_hole_overview.png")
    plt.close(fig)


# ── Isometric 3-D view ────────────────────────────────────────────────

def _psi_3d(solution: Solution, step: int = -1) -> np.ndarray:
    """Return ψ reshaped to (nx, ny, nz)."""
    p = solution.params
    return solution.psi(step).reshape(p.Nx - 1, p.Ny - 1, max(p.Nz - 1, 1))


def _get_psi2_slice(solution: Solution, axis: str, index: int) -> np.ndarray:
    psi3d = _psi_3d(solution)
    if axis == "z":
        return np.abs(psi3d[:, :, index]) ** 2
    elif axis == "y":
        return np.abs(psi3d[:, index, :]) ** 2
    elif axis == "x":
        return np.abs(psi3d[index, :, :]) ** 2
    raise ValueError(axis)


def _get_phase_slice(solution: Solution, axis: str, index: int) -> np.ndarray:
    psi3d = _psi_3d(solution)
    if axis == "z":
        return np.angle(psi3d[:, :, index])
    elif axis == "y":
        return np.angle(psi3d[:, index, :])
    elif axis == "x":
        return np.angle(psi3d[index, :, :])
    raise ValueError(axis)


def _draw_wireframe(ax, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi):
    corners = np.array([
        [x_lo, y_lo, z_lo], [x_hi, y_lo, z_lo],
        [x_hi, y_hi, z_lo], [x_lo, y_hi, z_lo],
        [x_lo, y_lo, z_hi], [x_hi, y_lo, z_hi],
        [x_hi, y_hi, z_hi], [x_lo, y_hi, z_hi],
    ])
    for i0, i1 in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
                    (0,4),(1,5),(2,6),(3,7)]:
        ax.plot3D(*zip(corners[i0], corners[i1]),
                  color="white", linewidth=0.8, alpha=0.6)


def _paint_surfaces(ax, solution, data_fn, cmap_obj, norm_obj, xs, ys, zs,
                    x_lo, x_hi, y_lo, y_hi, z_lo, z_hi):
    """Paint visible faces of the film onto *ax*."""
    p = solution.params
    nx_int, ny_int, nz_int = p.Nx-1, p.Ny-1, max(p.Nz-1, 1)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")

    # Top face (z = z_hi)
    d = np.clip(data_fn("z", nz_int - 1), norm_obj.vmin, norm_obj.vmax)
    ax.plot_surface(XX, YY, np.full_like(XX, z_hi),
                    facecolors=cmap_obj(norm_obj(d)),
                    shade=False, alpha=0.95, rstride=1, cstride=1)
    # Bottom face (z = z_lo)
    d = np.clip(data_fn("z", 0), norm_obj.vmin, norm_obj.vmax)
    ax.plot_surface(XX, YY, np.full_like(XX, z_lo),
                    facecolors=cmap_obj(norm_obj(d)),
                    shade=False, alpha=0.45, rstride=1, cstride=1)
    # Front face (y = y_lo)
    XF, ZF = np.meshgrid(xs, zs, indexing="ij")
    d = np.clip(data_fn("y", 0), norm_obj.vmin, norm_obj.vmax)
    ax.plot_surface(XF, np.full_like(XF, y_lo), ZF,
                    facecolors=cmap_obj(norm_obj(d)),
                    shade=False, alpha=0.85, rstride=1, cstride=1)
    # Right face (x = x_hi)
    YR, ZR = np.meshgrid(ys, zs, indexing="ij")
    d = np.clip(data_fn("x", nx_int - 1), norm_obj.vmin, norm_obj.vmax)
    ax.plot_surface(np.full_like(YR, x_hi), YR, ZR,
                    facecolors=cmap_obj(norm_obj(d)),
                    shade=False, alpha=0.85, rstride=1, cstride=1)


def _style_ax(ax, fig, cbar, label):
    ax.set_xlabel("x (µm)", fontsize=10, labelpad=6)
    ax.set_ylabel("y (µm)", fontsize=10, labelpad=6)
    ax.set_zlabel("z (ξ)", fontsize=10, labelpad=3)
    ax.view_init(elev=28, azim=-55)
    ax.set_facecolor("#1a1a2e")
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.label.set_color("white")
        a.pane.fill = False
        a.pane.set_edgecolor("gray")
    ax.tick_params(colors="white", labelsize=8)
    ax.grid(True, alpha=0.15)
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")
    cbar.set_label(label, fontsize=11, color="white")


def plot_isometric(solution: Solution, device: tdgl3d.Device,
                   filename: str = "sis_square_with_hole_isometric.png") -> None:
    """Side-by-side isometric views: |ψ|² and arg(ψ) showing the S/I/S stack."""
    p = solution.params
    nx_int, ny_int, nz_int = p.Nx - 1, p.Ny - 1, max(p.Nz - 1, 1)

    # Physical extents: x,y in µm (centred at 0), z in ξ units
    x_lo = (1 * p.hx) * UM_PER_XI - LX_UM / 2.0
    x_hi = (p.Nx * p.hx) * UM_PER_XI - LX_UM / 2.0
    y_lo = (1 * p.hy) * UM_PER_XI - LY_UM / 2.0
    y_hi = (p.Ny * p.hy) * UM_PER_XI - LY_UM / 2.0
    z_lo = 1 * p.hz
    z_hi = p.Nz * p.hz

    xs = np.linspace(x_lo, x_hi, nx_int)
    ys = np.linspace(y_lo, y_hi, ny_int)
    zs = np.linspace(z_lo, z_hi, nz_int)

    fig = plt.figure(figsize=(18, 8))
    fig.patch.set_facecolor("#1a1a2e")

    z_ranges = device.trilayer.z_ranges()
    fig.suptitle(
        f"S/I/S 5×5 µm square with hole — |ψ|² and phase\n"
        f"SC({SC_THICKNESS}) / I({INS_THICKNESS}) / SC({SC_THICKNESS}),  "
        f"κ={KAPPA},  B$_z$={BZ},  t={solution.times[-1]:.1f}",
        fontsize=13, color="white", y=0.97,
    )

    # ── Left: |ψ|² ─────────────────────────────────────────────────────
    ax1 = fig.add_subplot(121, projection="3d")
    cmap1 = cm.inferno
    norm1 = Normalize(vmin=0, vmax=1)
    _paint_surfaces(ax1, solution,
                    lambda a, i: _get_psi2_slice(solution, a, i),
                    cmap1, norm1, xs, ys, zs,
                    x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)
    _draw_wireframe(ax1, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)
    sm1 = cm.ScalarMappable(cmap=cmap1, norm=norm1); sm1.set_array([])
    cb1 = fig.colorbar(sm1, ax=ax1, fraction=0.03, pad=0.10, shrink=0.65)
    _style_ax(ax1, fig, cb1, "|ψ|²")
    ax1.set_title("|ψ|²  (superfluid density)", fontsize=12,
                  color="white", pad=12)

    # ── Right: arg(ψ) ──────────────────────────────────────────────────
    ax2 = fig.add_subplot(122, projection="3d")
    cmap2 = cm.twilight_shifted
    norm2 = Normalize(vmin=-np.pi, vmax=np.pi)
    _paint_surfaces(ax2, solution,
                    lambda a, i: _get_phase_slice(solution, a, i),
                    cmap2, norm2, xs, ys, zs,
                    x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)
    _draw_wireframe(ax2, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)
    sm2 = cm.ScalarMappable(cmap=cmap2, norm=norm2); sm2.set_array([])
    cb2 = fig.colorbar(sm2, ax=ax2, fraction=0.03, pad=0.10, shrink=0.65)
    cb2.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cb2.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])
    _style_ax(ax2, fig, cb2, "arg(ψ)")
    ax2.set_title("arg(ψ)  (phase)", fontsize=12,
                  color="white", pad=12)

    fig.savefig(filename, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved {filename}")
    plt.close(fig)


def main() -> None:
    # ── Build trilayer ──────────────────────────────────────────────────
    trilayer = tdgl3d.Trilayer(
        bottom=tdgl3d.Layer(thickness_z=SC_THICKNESS, kappa=KAPPA),
        insulator=tdgl3d.Layer(thickness_z=INS_THICKNESS, kappa=0.0, is_superconductor=False),
        top=tdgl3d.Layer(thickness_z=SC_THICKNESS, kappa=KAPPA),
    )

    params = tdgl3d.SimulationParameters(
        Nx=NX, Ny=NY,
        hx=HX, hy=HY, hz=HZ,
        kappa=KAPPA,
    )
    # Nz will be overridden by the trilayer
    field = tdgl3d.AppliedField(Bz=BZ, t_on_fraction=1.0, ramp=True, ramp_fraction=0.5)
    device = tdgl3d.Device(params, applied_field=field, trilayer=trilayer)

    print(device)
    print(f"Nz = {params.Nz} (trilayer: {SC_THICKNESS}SC / {INS_THICKNESS}I / {SC_THICKNESS}SC)")
    print(f"Interior nodes: {params.n_interior}")
    print(f"State vector length: {params.n_state}")
    print(f"Physical size: {LX_UM} × {LY_UM} µm  (ξ = {XI_NM} nm)")
    print()

    # ── Carve the hole ─────────────────────────────────────────────────
    carve_hole(device)

    # ── Initial state (ψ = 0 in insulator + hole) ─────────────────────
    x0 = device.initial_state()
    # Add small random noise to break symmetry and help vortex nucleation
    rng = np.random.default_rng(42)
    noise = 0.01 * (rng.standard_normal(params.n_interior)
                     + 1j * rng.standard_normal(params.n_interior))
    x0.psi[:] += noise * device.material.interior_sc_mask
    n_zero = int(np.sum(np.abs(x0.psi) < 1e-12))
    print(f"Initial state: {n_zero} / {params.n_interior} interior ψ nodes zeroed")
    print()

    # ── Run simulation ─────────────────────────────────────────────────
    print(f"Running Forward-Euler (dt={DT}, t_stop={T_STOP}, "
          f"Bz={BZ}, κ={KAPPA}) …")
    print(f"CFL limit: dt < h²/(4κ²) = {HX**2 / (4 * KAPPA**2):.4f}")
    print()

    solution = tdgl3d.solve(
        device,
        t_start=0.0,
        t_stop=T_STOP,
        dt=DT,
        method="euler",
        x0=x0,
        save_every=SAVE_EVERY,
        progress=True,
    )

    print(f"\nSaved {solution.n_steps} snapshots.")
    psi2 = solution.psi_squared(step=-1)
    print(f"Final |ψ|²: mean={np.mean(psi2):.4f}  "
          f"min={np.min(psi2):.4f}  max={np.max(psi2):.4f}")

    # ── Visualise ──────────────────────────────────────────────────────
    plot_slices(solution, device)
    plot_xy_overview(solution, device)
    plot_isometric(solution, device)

    # ── Animation ──────────────────────────────────────────────────────
    z_ranges = device.trilayer.z_ranges()
    k_bot = (z_ranges["bottom"][0] + z_ranges["bottom"][1]) // 2
    sz_bot = max(k_bot - 1, 0)

    from tdgl3d.visualization.plotting import animate
    gif_path = animate(
        solution,
        filename="sis_square_with_hole.gif",
        slice_z=sz_bot,
        fps=8,
        step_stride=1,
    )
    print(f"Saved animation → {gif_path}")


if __name__ == "__main__":
    main()

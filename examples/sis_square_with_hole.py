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
HX = HY = HZ = 0.75               # grid spacing in ξ (intermediate mesh)
NX = int(LX_XI / HX)              # ~67
NY = int(LY_XI / HY)              # ~67
KAPPA = 2.0                        # GL parameter

# Trilayer stack: 8 SC + 8 I + 8 SC  →  Nz = 24
SC_THICKNESS = 8                   # z-cells per SC layer
INS_THICKNESS = 8                  # z-cells for insulator

# Hole bounds in µm (centred at origin)
HOLE_X_MIN_UM, HOLE_X_MAX_UM = -1.0, 1.0
HOLE_Y_MIN_UM, HOLE_Y_MAX_UM = -0.2, 0.2

# Time integration
DT = 0.01
T_STOP = 30.0
BZ = 0.5
SAVE_EVERY = 50


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
    z-layers are marked as insulator (sc_mask = 0) so FPSI drives ψ → 0.
    kappa is left non-zero so link variables can evolve and B-field
    penetrates the hole.
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

    # Zero out sc_mask inside the hole for SC z-planes.
    # Keep kappa non-zero so the link variables (gauge field) can still
    # evolve — this allows magnetic flux to penetrate the hole.
    count = 0
    for k in sc_z_planes:
        for j in range(j_lo, j_hi + 1):
            for i in range(i_lo, i_hi + 1):
                m = i + mj * j + mk * k
                if m < len(material.sc_mask):
                    material.sc_mask[m] = 0.0
                    # Leave kappa unchanged (inherits from SC layer)
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
    phase = np.angle(psi_slice).astype(np.float64)
    phase[psi2 < 0.02] = np.nan  # mask phase where condensate is absent

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


def _get_psi2_slice(solution: Solution, axis: str, index: int, step: int = -1,
                    sc_mask_3d: np.ndarray = None) -> np.ndarray:
    psi3d = _psi_3d(solution, step=step)
    if axis == "z":
        sl = np.abs(psi3d[:, :, index]) ** 2
        if sc_mask_3d is not None:
            sl[sc_mask_3d[:, :, index] == 0] = np.nan
    elif axis == "y":
        sl = np.abs(psi3d[:, index, :]) ** 2
        if sc_mask_3d is not None:
            sl[sc_mask_3d[:, index, :] == 0] = np.nan
    elif axis == "x":
        sl = np.abs(psi3d[index, :, :]) ** 2
        if sc_mask_3d is not None:
            sl[sc_mask_3d[index, :, :] == 0] = np.nan
    else:
        raise ValueError(axis)
    return sl


def _get_phase_slice(solution: Solution, axis: str, index: int, step: int = -1,
                     mask_threshold: float = 0.02) -> np.ndarray:
    """Return phase of ψ. Insulator nodes (|ψ|² < threshold) get value 999 (sentinel)."""
    psi3d = _psi_3d(solution, step=step)
    if axis == "z":
        sl = psi3d[:, :, index]
    elif axis == "y":
        sl = psi3d[:, index, :]
    elif axis == "x":
        sl = psi3d[index, :, :]
    else:
        raise ValueError(axis)
    phase = np.angle(sl).astype(np.float64)
    # Mark insulator/hole nodes with sentinel (handled in _paint_surfaces)
    phase[np.abs(sl) ** 2 < mask_threshold] = 999.0
    return phase


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
    """Paint visible faces of the film onto *ax*.  Sentinel 999 → dark gray."""
    p = solution.params
    nx_int, ny_int, nz_int = p.Nx-1, p.Ny-1, max(p.Nz-1, 1)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")

    GRAY = np.array([0.15, 0.15, 0.15, 1.0])

    def _colors(data):
        """Map data through norm+cmap; sentinel 999 or NaN → dark gray."""
        mask = (data > 900) | np.isnan(data)
        clipped = np.where(mask, 0.0, data)
        normed = norm_obj(clipped)
        rgba = cmap_obj(normed)
        rgba[mask] = GRAY
        return rgba

    # Top face (z = z_hi)
    d = data_fn("z", nz_int - 1)
    ax.plot_surface(XX, YY, np.full_like(XX, z_hi),
                    facecolors=_colors(d),
                    shade=False, alpha=0.95, rstride=1, cstride=1)
    # Bottom face (z = z_lo)
    d = data_fn("z", 0)
    ax.plot_surface(XX, YY, np.full_like(XX, z_lo),
                    facecolors=_colors(d),
                    shade=False, alpha=0.45, rstride=1, cstride=1)
    # Front face (y = y_lo)
    XF, ZF = np.meshgrid(xs, zs, indexing="ij")
    d = data_fn("y", 0)
    ax.plot_surface(XF, np.full_like(XF, y_lo), ZF,
                    facecolors=_colors(d),
                    shade=False, alpha=0.85, rstride=1, cstride=1)
    # Right face (x = x_hi)
    YR, ZR = np.meshgrid(ys, zs, indexing="ij")
    d = data_fn("x", nx_int - 1)
    ax.plot_surface(np.full_like(YR, x_hi), YR, ZR,
                    facecolors=_colors(d),
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

    # Build 3D SC mask for insulator marking
    sc_mask_3d = device.material.interior_sc_mask.reshape(nx_int, ny_int, nz_int)

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
                    lambda a, i: _get_psi2_slice(solution, a, i, step=-1, sc_mask_3d=sc_mask_3d),
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
                    lambda a, i: _get_phase_slice(solution, a, i, step=-1),
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


def _compute_supercurrent_3d(solution: Solution, device, step: int = -1) -> np.ndarray:
    """Compute supercurrent density magnitude |J_s| on the interior grid.

    J_x[m] = Im(exp(-i*φ_x[m]) * conj(ψ[m]) * ψ[m+1]) / hx
    J_y[m] = Im(exp(-i*φ_y[m]) * conj(ψ[m]) * ψ[m+mj]) / hy
    J_z[m] = Im(exp(-i*φ_z[m]) * conj(ψ[m]) * ψ[m+mk]) / hz  (3D only)

    Returns |J| reshaped to (Nx-1, Ny-1, Nz-1).
    """
    from tdgl3d.physics.rhs import _expand_interior_to_full

    p = solution.params
    idx = solution.idx
    n = p.n_interior
    state = solution.states[:, step]

    psi_int = state[:n]
    phi_x_int = state[n:2*n]
    phi_y_int = state[2*n:3*n]

    # Expand to full grid
    x = _expand_interior_to_full(psi_int, p, idx)
    y1 = _expand_interior_to_full(phi_x_int, p, idx)
    y2 = _expand_interior_to_full(phi_y_int, p, idx)

    m = idx.interior_to_full
    mj = p.mj
    mk = p.mk

    Jx = np.imag(np.exp(-1j * y1[m]) * np.conj(x[m]) * x[m + 1]) / p.hx
    Jy = np.imag(np.exp(-1j * y2[m]) * np.conj(x[m]) * x[m + mj]) / p.hy

    if p.is_3d:
        phi_z_int = state[3*n:4*n]
        y3 = _expand_interior_to_full(phi_z_int, p, idx)
        Jz = np.imag(np.exp(-1j * y3[m]) * np.conj(x[m]) * x[m + mk]) / p.hz
        J_mag = np.sqrt(np.real(Jx)**2 + np.real(Jy)**2 + np.real(Jz)**2)
    else:
        J_mag = np.sqrt(np.real(Jx)**2 + np.real(Jy)**2)

    nx_int, ny_int, nz_int = p.Nx - 1, p.Ny - 1, max(p.Nz - 1, 1)
    return J_mag.reshape(nx_int, ny_int, nz_int)


def _compute_bz_full_3d(solution: Solution, device, step: int = -1) -> np.ndarray:
    """Compute total Bz on interior grid including applied field via BCs.

    Expands state to full grid, applies BCs (which write the applied field
    onto boundary link variables), then computes curl on interior nodes.
    Returns Bz reshaped to (Nx-1, Ny-1, Nz-1).
    """
    from tdgl3d.physics.rhs import _expand_interior_to_full, _apply_boundary_conditions, BoundaryVectors
    from tdgl3d.physics.applied_field import build_boundary_field_vectors

    p = solution.params
    idx = solution.idx
    n = p.n_interior
    state = solution.states[:, step]

    psi_int = state[:n]
    phi_x_int = state[n:2*n]
    phi_y_int = state[2*n:3*n]
    phi_z_int = state[3*n:4*n] if p.is_3d else np.zeros(n, dtype=np.complex128)

    # Expand to full grid
    x = _expand_interior_to_full(psi_int, p, idx)
    y1 = _expand_interior_to_full(phi_x_int, p, idx)
    y2 = _expand_interior_to_full(phi_y_int, p, idx)
    y3 = _expand_interior_to_full(phi_z_int, p, idx)

    # Build boundary vectors for the applied field at the final time
    t = solution.times[step]
    bx_app, by_app, bz_app = device.applied_field.evaluate(t, T_STOP)
    Bx_vec, By_vec, Bz_vec = build_boundary_field_vectors(bx_app, by_app, bz_app, p, idx)
    u = BoundaryVectors(Bx_vec, By_vec, Bz_vec)

    # Apply BCs
    x, y1, y2, y3 = _apply_boundary_conditions(x, y1, y2, y3, p, idx, u)

    # Compute Bz = (1/(hx*hy)) * (φ_x[m] - φ_x[m+mj] - φ_y[m] + φ_y[m+1])
    # on interior nodes
    m = idx.interior_to_full
    mj = p.mj
    Bz = np.real((1.0 / (p.hx * p.hy)) * (y1[m] - y1[m + mj] - y2[m] + y2[m + 1]))

    nx_int, ny_int, nz_int = p.Nx - 1, p.Ny - 1, max(p.Nz - 1, 1)
    Bz_3d = Bz.reshape(nx_int, ny_int, nz_int)
    # The last x-row and last y-row of interior nodes see boundary link
    # variables that carry the applied field directly (staggered-grid artifact).
    # Replace them with their interior neighbours to remove the edge spike.
    Bz_3d[-1, :, :] = Bz_3d[-2, :, :]
    Bz_3d[:, -1, :] = Bz_3d[:, -2, :]
    return Bz_3d


def _get_js_slice(solution: Solution, device, axis: str, index: int, step: int = -1,
                  _cache: dict = {}) -> np.ndarray:
    """Return |J_s| slice for isometric plotting. Caches per step."""
    cache_key = (id(solution), step)
    if cache_key not in _cache:
        _cache[cache_key] = _compute_supercurrent_3d(solution, device, step=step)
    js3d = _cache[cache_key]
    nx_int, ny_int, nz_int = js3d.shape
    if axis == "z":
        return js3d[:, :, min(index, nz_int - 1)]
    elif axis == "y":
        return js3d[:, min(index, ny_int - 1), :]
    elif axis == "x":
        return js3d[min(index, nx_int - 1), :, :]
    raise ValueError(axis)


def _get_bz_full_slice(solution: Solution, device, axis: str, index: int, step: int = -1,
                       _cache: dict = {}) -> np.ndarray:
    """Return total Bz slice for isometric plotting. Caches per step."""
    cache_key = (id(solution), step)
    if cache_key not in _cache:
        _cache[cache_key] = _compute_bz_full_3d(solution, device, step=step)
    bz3d = _cache[cache_key]
    nx_int, ny_int, nz_int = bz3d.shape
    if axis == "z":
        return bz3d[:, :, min(index, nz_int - 1)]
    elif axis == "y":
        return bz3d[:, min(index, ny_int - 1), :]
    elif axis == "x":
        return bz3d[min(index, nx_int - 1), :, :]
    raise ValueError(axis)


def animate_isometric(solution: Solution, device: tdgl3d.Device,
                      prefix: str = "sis_square_with_hole",
                      fps: int = 6, step_stride: int = 1) -> list[str]:
    """Create 4 animated GIFs — one per quantity — each with a single isometric cube.
    
    Returns
    -------
    list[str]
        List of generated GIF filenames: [psi2, phase, Bz, Js].
    """
    from matplotlib.animation import FuncAnimation, PillowWriter

    p = solution.params
    nx_int, ny_int, nz_int = p.Nx - 1, p.Ny - 1, max(p.Nz - 1, 1)

    x_lo = (1 * p.hx) * UM_PER_XI - LX_UM / 2.0
    x_hi = (p.Nx * p.hx) * UM_PER_XI - LX_UM / 2.0
    y_lo = (1 * p.hy) * UM_PER_XI - LY_UM / 2.0
    y_hi = (p.Ny * p.hy) * UM_PER_XI - LY_UM / 2.0
    z_lo, z_hi = 1 * p.hz, p.Nz * p.hz

    xs = np.linspace(x_lo, x_hi, nx_int)
    ys = np.linspace(y_lo, y_hi, ny_int)
    zs = np.linspace(z_lo, z_hi, nz_int)

    # Build 3D SC mask for insulator marking
    sc_mask_3d = device.material.interior_sc_mask.reshape(nx_int, ny_int, nz_int)

    steps = list(range(0, solution.n_steps, step_stride))

    # Pre-compute Bz and Js limits from the final step for consistent colorbars
    print("  Pre-computing colorbar limits...")
    bz_final = _compute_bz_full_3d(solution, device, step=-1)
    bz_max = max(abs(bz_final.max()), abs(bz_final.min()), 0.1)
    js_final = _compute_supercurrent_3d(solution, device, step=-1)
    js_max = max(js_final.max(), 0.01)

    # Panel definitions
    panels = [
        {
            "name": "psi2",
            "title": r"$|\psi|^2$",
            "cmap": "inferno",
            "vmin": 0.0,
            "vmax": 1.0,
            "data_fn": lambda a, i, _s: _get_psi2_slice(solution, a, i, step=_s, sc_mask_3d=sc_mask_3d),
        },
        {
            "name": "phase",
            "title": r"$\arg(\psi)$",
            "cmap": "twilight_shifted",
            "vmin": -np.pi,
            "vmax": np.pi,
            "data_fn": lambda a, i, _s: _get_phase_slice(solution, a, i, step=_s),
        },
        {
            "name": "Bz",
            "title": r"$B_z$ (total field)",
            "cmap": "RdBu_r",
            "vmin": -bz_max,
            "vmax": bz_max,
            "data_fn": lambda a, i, _s: _get_bz_full_slice(solution, device, a, i, step=_s),
        },
        {
            "name": "Js",
            "title": r"$|J_s|$ (supercurrent)",
            "cmap": "hot",
            "vmin": 0.0,
            "vmax": js_max,
            "data_fn": lambda a, i, _s: _get_js_slice(solution, device, a, i, step=_s),
        },
    ]

    output_files = []

    # Generate one GIF per panel
    for panel in panels:
        filename = f"{prefix}_{panel['name']}.gif"
        print(f"  Generating {filename} ({len(steps)} frames)...")

        cmap_obj = plt.get_cmap(panel["cmap"]).copy()
        cmap_obj.set_bad(color="0.25")
        norm = Normalize(vmin=panel["vmin"], vmax=panel["vmax"])

        fig = plt.figure(figsize=(10, 8))
        fig.patch.set_facecolor("#1a1a2e")
        ax = fig.add_subplot(111, projection="3d")

        def draw_frame(frame_idx):
            fig.clf()
            ax = fig.add_subplot(111, projection="3d")
            s = steps[frame_idx]
            t = solution.times[s]

            _paint_surfaces(ax, solution,
                            lambda a, i: panel["data_fn"](a, i, s),
                            cmap_obj, norm, xs, ys, zs,
                            x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)
            _draw_wireframe(ax, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)
            
            ax.set_xlabel("x (µm)", color="white", fontsize=11)
            ax.set_ylabel("y (µm)", color="white", fontsize=11)
            ax.set_zlabel("z (ξ)", color="white", fontsize=11)
            ax.set_title(f"{panel['title']}   t = {t:.2f} / {solution.times[-1]:.2f}", 
                        fontsize=14, color="white", pad=20)
            ax.view_init(elev=25, azim=-60)
            ax.set_facecolor("#1a1a2e")
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.tick_params(colors="white", labelsize=9)
            
            # Add colorbar
            sm = cm.ScalarMappable(cmap=cmap_obj, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.1, shrink=0.6)
            cbar.ax.tick_params(colors="white", labelsize=9)
            if panel["name"] == "phase":
                cbar.set_ticks([-np.pi, 0, np.pi])
                cbar.set_ticklabels(["-π", "0", "π"])
            
            fig.patch.set_facecolor("#1a1a2e")
            return []

        anim = FuncAnimation(fig, draw_frame, frames=len(steps), blit=False)
        anim.save(filename, writer=PillowWriter(fps=fps))
        plt.close(fig)
        print(f"    → saved {filename}")
        output_files.append(filename)

    # Clear per-step caches
    _get_bz_full_slice.__defaults__[1].clear()
    _get_js_slice.__defaults__[1].clear()

    return output_files


def check_superconductivity(solution: Solution, device: tdgl3d.Device, 
                            threshold: float = 0.01) -> tuple[bool, float]:
    """Check if SC regions maintain superconductivity (|ψ|² > threshold).
    
    Returns
    -------
    tuple[bool, float]
        (is_superconducting_everywhere, min_psi2_in_sc)
    """
    # Get final state
    psi_int = solution.psi(step=-1)
    psi2 = np.abs(psi_int) ** 2
    
    # Get SC mask (interior nodes)
    sc_mask_int = device.material.interior_sc_mask
    
    # Check minimum |ψ|² in SC regions
    psi2_sc = psi2[sc_mask_int > 0]
    min_psi2 = float(np.min(psi2_sc))
    
    is_sc = min_psi2 > threshold
    return is_sc, min_psi2


def run_simulation_with_field(bz_field: float) -> tuple[Solution, tdgl3d.Device]:
    """Run a simulation with given field strength.
    
    Returns
    -------
    tuple[Solution, Device]
        The solution and device objects.
    """
    # Build trilayer
    trilayer = tdgl3d.Trilayer(
        bottom=tdgl3d.Layer(thickness_z=SC_THICKNESS, kappa=KAPPA),
        insulator=tdgl3d.Layer(thickness_z=INS_THICKNESS, kappa=KAPPA, is_superconductor=False),
        top=tdgl3d.Layer(thickness_z=SC_THICKNESS, kappa=KAPPA),
    )

    params = tdgl3d.SimulationParameters(
        Nx=NX, Ny=NY,
        hx=HX, hy=HY, hz=HZ,
        kappa=KAPPA,
    )
    
    # Field profile: zero for 30% of t_stop, ramp from 30% to 50%, hold at full for rest
    def _field_profile(t: float, t_stop: float) -> tuple[float, float, float]:
        t_settle = 0.30 * t_stop
        t_ramp_end = 0.50 * t_stop
        if t <= t_settle:
            return 0.0, 0.0, 0.0
        elif t < t_ramp_end:
            scale = (t - t_settle) / (t_ramp_end - t_settle)
            return 0.0, 0.0, bz_field * scale
        else:
            return 0.0, 0.0, bz_field

    field = tdgl3d.AppliedField(Bz=bz_field, field_func=_field_profile)
    device = tdgl3d.Device(params, applied_field=field, trilayer=trilayer)

    print(device)
    print(f"Nz = {params.Nz} (trilayer: {SC_THICKNESS}SC / {INS_THICKNESS}I / {SC_THICKNESS}SC)")
    print(f"Interior nodes: {params.n_interior}")
    print(f"State vector length: {params.n_state}")
    print(f"Physical size: {LX_UM} × {LY_UM} µm  (ξ = {XI_NM} nm)")
    print(f"Applied Bz = {bz_field}\n")

    # Carve the hole
    carve_hole(device)

    # Initial state (ψ = 0 in insulator + hole)
    x0 = device.initial_state()
    # Add small random noise to break symmetry
    rng = np.random.default_rng(42)
    noise = 0.02 * (rng.standard_normal(params.n_interior)
                     + 1j * rng.standard_normal(params.n_interior))
    x0.psi[:] += noise * device.material.interior_sc_mask

    interior_zeros = int(np.sum(device.material.interior_sc_mask == 0))
    print(f"Interior nodes with sc_mask=0: {interior_zeros} / {params.n_interior}")
    print(f"Initial state: {interior_zeros} / {params.n_interior} interior ψ nodes zeroed\n")

    # Solve
    print(f"Running Forward-Euler (dt={DT}, t_stop={T_STOP}, Bz={bz_field}, κ={KAPPA}) …")
    cfl = params.hx**2 / (4 * params.kappa**2)
    print(f"CFL limit: dt < h²/(4κ²) = {cfl:.4f}\n")

    solution = tdgl3d.solve(
        device,
        x0=x0,
        dt=DT,
        t_stop=T_STOP,
        method="euler",
        save_every=SAVE_EVERY,
        progress=True,
    )

    print(f"\nSaved {solution.n_steps} snapshots.")
    return solution, device


def main() -> None:
    # Run simulation with Bz = 0.5
    print(f"\n{'='*70}")
    print(f"Running simulation with Bz = {BZ}")
    print(f"{'='*70}\n")
    
    solution, device = run_simulation_with_field(BZ)
    
    # Check if SC is still superconducting
    min_threshold = 0.01
    is_sc, min_psi2 = check_superconductivity(solution, device, threshold=min_threshold)
    
    print(f"\nSuperconductivity check:")
    print(f"  min |ψ|² in SC regions: {min_psi2:.6f}")
    print(f"  Threshold: {min_threshold}")
    print(f"  Status: {'✓ SUPERCONDUCTING' if is_sc else '✗ SUPPRESSED'}")
    
    # ── Diagnostics: verify plot data correctness ─────────────────────
    p = solution.params
    sc_mask_3d = device.material.interior_sc_mask.reshape(
        p.Nx - 1, p.Ny - 1, max(p.Nz - 1, 1))
    psi3d = _psi_3d(solution, step=-1)
    nz_int = max(p.Nz - 1, 1)

    # Check phase in SC vs insulator
    z_ranges = device.trilayer.z_ranges()
    k_bot_mid = (z_ranges["bottom"][0] + z_ranges["bottom"][1]) // 2
    k_ins_mid = (z_ranges["insulator"][0] + z_ranges["insulator"][1]) // 2
    sz_bot = max(k_bot_mid - 1, 0)
    sz_ins = max(k_ins_mid - 1, 0)

    psi_bot = psi3d[:, :, sz_bot]
    psi_ins = psi3d[:, :, sz_ins]
    print(f"\n── Diagnostic checks ──")
    print(f"Bottom SC midplane (z={sz_bot}):")
    print(f"  |ψ|² : mean={np.mean(np.abs(psi_bot)**2):.4f}  "
          f"max={np.max(np.abs(psi_bot)**2):.4f}")
    print(f"  phase: min={np.angle(psi_bot).min():.3f}  "
          f"max={np.angle(psi_bot).max():.3f}  "
          f"std={np.angle(psi_bot).std():.4f}")
    print(f"  SC nodes: {int(np.sum(sc_mask_3d[:, :, sz_bot]))}")

    print(f"Insulator midplane (z={sz_ins}):")
    print(f"  |ψ|² : mean={np.mean(np.abs(psi_ins)**2):.6f}  "
          f"max={np.max(np.abs(psi_ins)**2):.6f}")
    print(f"  SC nodes: {int(np.sum(sc_mask_3d[:, :, sz_ins]))}")

    # Check Bz (total, with BCs applied)
    bz3d_full = _compute_bz_full_3d(solution, device, step=-1)
    bz_bot = bz3d_full[:, :, sz_bot]
    print(f"Total Bz (bottom SC midplane, with BCs):")
    print(f"  min={bz_bot.min():.4f}  max={bz_bot.max():.4f}  "
          f"mean={bz_bot.mean():.4f}  std={bz_bot.std():.4f}")
    print(f"  Applied Bz={BZ:.3f} → expect Meissner: bulk Bz ≈ 0, hole region Bz ≈ {BZ:.3f}")

    # Check supercurrent
    js3d = _compute_supercurrent_3d(solution, device, step=-1)
    js_bot = js3d[:, :, sz_bot]
    print(f"|J_s| (bottom SC midplane):")
    print(f"  min={js_bot.min():.4f}  max={js_bot.max():.4f}  "
          f"mean={js_bot.mean():.4f}  std={js_bot.std():.4f}")
    print(f"  Expect high |J_s| around hole border (persistent current)")
    # Check if NaN masking is correct
    psi2_slice = _get_psi2_slice(solution, "z", sz_ins, step=-1, sc_mask_3d=sc_mask_3d)
    n_nan = int(np.sum(np.isnan(psi2_slice)))
    n_total = psi2_slice.size
    print(f"|ψ|² NaN mask at insulator z={sz_ins}: {n_nan}/{n_total} nodes are NaN")
    phase_slice = _get_phase_slice(solution, "z", sz_ins, step=-1)
    n_sentinel = int(np.sum(phase_slice > 900))
    print(f"Phase sentinel at insulator z={sz_ins}: {n_sentinel}/{n_total} nodes masked")
    print(f"── End diagnostics ──\n")

    # ── Visualise ──────────────────────────────────────────────────────
    plot_slices(solution, device)
    plot_xy_overview(solution, device)
    plot_isometric(solution, device)

    # ── Animations (4 separate GIFs) ──────────────────────────────────
    print("\n=== Generating animated GIFs ===")
    gif_files = animate_isometric(
        solution, device,
        prefix="sis_square_with_hole",
        fps=6,
        step_stride=2,
    )
    print("\nGenerated GIF files:")
    for gf in gif_files:
        print(f"  • {gf}")


if __name__ == "__main__":
    main()

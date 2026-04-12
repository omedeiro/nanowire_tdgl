"""Isometric 3-D visualisation of a 100 × 100 × 20 nm superconducting film.

Runs a Forward-Euler simulation with a **linearly ramped** B_z, then renders
two isometric views side-by-side:
  • |ψ|² (superfluid density) — vortex cores are dark
  • arg(ψ) (order-parameter phase) — 2π windings mark vortices

The ramp lets vortices nucleate gradually and settle into a more symmetric
Abrikosov-like lattice.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

import tdgl3d
from tdgl3d.core.solution import Solution
from tdgl3d.core.parameters import SimulationParameters


# ── Physical dimensions ────────────────────────────────────────────────
# Film: 100 nm × 100 nm × 20 nm
# ξ = 5 nm → 20ξ × 20ξ × 4ξ
Lx_xi, Ly_xi, Lz_xi = 20, 20, 4   # in units of ξ

hx = hy = 1.0   # grid spacing in ξ
hz = 1.0
Nx = int(Lx_xi / hx)   # 20
Ny = int(Ly_xi / hy)   # 20
Nz = int(Lz_xi / hz)   # 4

nm_per_xi = 5.0   # physical scale: 5 nm per coherence length


def run_simulation() -> Solution:
    """Run a Forward-Euler simulation with a linearly ramped B_z."""
    params = SimulationParameters(
        Nx=Nx, Ny=Ny, Nz=Nz,
        hx=hx, hy=hy, hz=hz,
        kappa=2.0,
    )

    # Applied field: ramp linearly from 0 to Bz over [0, t_stop/2],
    # then hold constant for the remainder.
    Bz_max = 1.0
    field = tdgl3d.AppliedField(Bz=Bz_max, t_on_fraction=1.0,
                                ramp=True, ramp_fraction=0.5)
    device = tdgl3d.Device(params, applied_field=field)

    dim_nm = f"{Lx_xi*nm_per_xi:.0f}×{Ly_xi*nm_per_xi:.0f}×{Lz_xi*nm_per_xi:.0f}"
    print(f"Grid: {Nx}×{Ny}×{Nz}  ({dim_nm} nm)")
    print(f"Interior nodes: {params.n_interior},  state-vector length: {params.n_state}")
    print(f"Applied field: B_z ramp 0 → {Bz_max} over t=[0, t_stop/2]")

    # CFL: dt < h²/(4κ²) = 1/16 = 0.0625 for h=1, κ=2
    dt = 0.01
    t_stop = 100.0

    # Uniform superconducting initial condition (|ψ|=1, φ=0).
    x0 = tdgl3d.StateVector.uniform_superconducting(params).data.copy()

    print(f"Running Forward-Euler (dt={dt}, t_stop={t_stop}) …")
    solution = tdgl3d.solve(
        device,
        t_start=0.0, t_stop=t_stop, dt=dt,
        method="euler", x0=x0,
        save_every=100, progress=True,
    )

    psi2 = solution.psi_squared(step=-1)
    print(f"Saved {solution.n_steps} snapshots")
    print(f"Final |ψ|²: mean={np.mean(psi2):.4f}  "
          f"min={np.min(psi2):.4f}  max={np.max(psi2):.4f}")

    # Check a mid-simulation snapshot too
    mid = solution.n_steps // 2
    psi2_mid = solution.psi_squared(step=mid)
    print(f"Mid   |ψ|²: mean={np.mean(psi2_mid):.4f}  "
          f"min={np.min(psi2_mid):.4f}  max={np.max(psi2_mid):.4f}")
    return solution


# ── Slice helpers ──────────────────────────────────────────────────────

def _psi_3d(solution: Solution, step: int = -1) -> np.ndarray:
    """Return ψ reshaped to (nx, ny, nz)."""
    p = solution.params
    return solution.psi(step).reshape(p.Nx - 1, p.Ny - 1, max(p.Nz - 1, 1))


def _get_psi2_slice(solution: Solution, axis: str, index: int) -> np.ndarray:
    """Extract |ψ|² on a 2-D slice."""
    psi3d = _psi_3d(solution)
    if axis == "z":
        return np.abs(psi3d[:, :, index]) ** 2
    elif axis == "y":
        return np.abs(psi3d[:, index, :]) ** 2
    elif axis == "x":
        return np.abs(psi3d[index, :, :]) ** 2
    raise ValueError(axis)


def _get_phase_slice(solution: Solution, axis: str, index: int) -> np.ndarray:
    """Extract arg(ψ) ∈ [-π, π] on a 2-D slice."""
    psi3d = _psi_3d(solution)
    if axis == "z":
        return np.angle(psi3d[:, :, index])
    elif axis == "y":
        return np.angle(psi3d[:, index, :])
    elif axis == "x":
        return np.angle(psi3d[index, :, :])
    raise ValueError(axis)


# ── Drawing helpers ────────────────────────────────────────────────────

def _film_extents(p: SimulationParameters):
    """Return (x_lo, x_hi, y_lo, y_hi, z_lo, z_hi) in nm."""
    return (
        1 * p.hx * nm_per_xi,  p.Nx * p.hx * nm_per_xi,
        1 * p.hy * nm_per_xi,  p.Ny * p.hy * nm_per_xi,
        1 * p.hz * nm_per_xi,  p.Nz * p.hz * nm_per_xi,
    )


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


def _style_ax(ax, fig, cbar, label):
    """Apply the dark theme to one 3-D axes."""
    ax.set_xlabel("x  (nm)", fontsize=10, labelpad=6)
    ax.set_ylabel("y  (nm)", fontsize=10, labelpad=6)
    ax.set_zlabel("z  (nm)", fontsize=10, labelpad=3)
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


def _set_equal_aspect(ax, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi):
    rng = max(x_hi - x_lo, y_hi - y_lo, z_hi - z_lo) / 2
    ax.set_xlim((x_lo+x_hi)/2 - rng, (x_lo+x_hi)/2 + rng)
    ax.set_ylim((y_lo+y_hi)/2 - rng, (y_lo+y_hi)/2 + rng)
    ax.set_zlim((z_lo+z_hi)/2 - rng, (z_lo+z_hi)/2 + rng)


# ── Main isometric plot (|ψ|² + phase) ────────────────────────────────

def _paint_surfaces(ax, solution, data_fn, cmap_obj, norm_obj, xs, ys, zs,
                    x_lo, x_hi, y_lo, y_hi, z_lo, z_hi):
    """Paint four visible faces of the film onto *ax*."""
    p = solution.params
    nx_int, ny_int, nz_int = p.Nx-1, p.Ny-1, max(p.Nz-1, 1)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")

    # Top
    d = np.clip(data_fn("z", nz_int - 1), norm_obj.vmin, norm_obj.vmax)
    ax.plot_surface(XX, YY, np.full_like(XX, z_hi),
                    facecolors=cmap_obj(norm_obj(d)),
                    shade=False, alpha=0.95, rstride=1, cstride=1)
    # Bottom
    d = np.clip(data_fn("z", 0), norm_obj.vmin, norm_obj.vmax)
    ax.plot_surface(XX, YY, np.full_like(XX, z_lo),
                    facecolors=cmap_obj(norm_obj(d)),
                    shade=False, alpha=0.45, rstride=1, cstride=1)
    # Front (y = y_lo)
    XF, ZF = np.meshgrid(xs, zs, indexing="ij")
    d = np.clip(data_fn("y", 0), norm_obj.vmin, norm_obj.vmax)
    ax.plot_surface(XF, np.full_like(XF, y_lo), ZF,
                    facecolors=cmap_obj(norm_obj(d)),
                    shade=False, alpha=0.85, rstride=1, cstride=1)
    # Right (x = x_hi)
    YR, ZR = np.meshgrid(ys, zs, indexing="ij")
    d = np.clip(data_fn("x", nx_int - 1), norm_obj.vmin, norm_obj.vmax)
    ax.plot_surface(np.full_like(YR, x_hi), YR, ZR,
                    facecolors=cmap_obj(norm_obj(d)),
                    shade=False, alpha=0.85, rstride=1, cstride=1)


def plot_isometric(solution: Solution,
                   filename: str = "isometric_film_3d.png"):
    """Side-by-side isometric views: |ψ|² and arg(ψ)."""
    p = solution.params
    nx_int, ny_int, nz_int = p.Nx-1, p.Ny-1, max(p.Nz-1, 1)
    x_lo, x_hi, y_lo, y_hi, z_lo, z_hi = _film_extents(p)

    xs = np.linspace(x_lo, x_hi, nx_int)
    ys = np.linspace(y_lo, y_hi, ny_int)
    zs = np.linspace(z_lo, z_hi, nz_int)

    fig = plt.figure(figsize=(18, 8))
    fig.patch.set_facecolor("#1a1a2e")

    dim_nm = f"{Lx_xi*nm_per_xi:.0f}×{Ly_xi*nm_per_xi:.0f}×{Lz_xi*nm_per_xi:.0f}"
    fig.suptitle(
        f"Type-II Superconductor Film  {dim_nm} nm   "
        f"(κ = {p.kappa},  B$_z$ ramped 0→1.0)",
        fontsize=14, color="white", y=0.96,
    )

    # ── Left panel: |ψ|² ────────────────────────────────────────────────
    ax1 = fig.add_subplot(121, projection="3d")
    cmap1 = cm.inferno
    norm1 = Normalize(vmin=0, vmax=1)
    _paint_surfaces(ax1, solution,
                    lambda a, i: _get_psi2_slice(solution, a, i),
                    cmap1, norm1, xs, ys, zs,
                    x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)
    _draw_wireframe(ax1, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)
    _set_equal_aspect(ax1, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)
    sm1 = cm.ScalarMappable(cmap=cmap1, norm=norm1); sm1.set_array([])
    cb1 = fig.colorbar(sm1, ax=ax1, fraction=0.03, pad=0.10, shrink=0.65)
    _style_ax(ax1, fig, cb1, "|ψ|²")
    ax1.set_title("|ψ|²  (superfluid density)", fontsize=12,
                  color="white", pad=12)

    # ── Right panel: arg(ψ) ─────────────────────────────────────────────
    ax2 = fig.add_subplot(122, projection="3d")
    cmap2 = cm.twilight_shifted
    norm2 = Normalize(vmin=-np.pi, vmax=np.pi)
    _paint_surfaces(ax2, solution,
                    lambda a, i: _get_phase_slice(solution, a, i),
                    cmap2, norm2, xs, ys, zs,
                    x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)
    _draw_wireframe(ax2, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)
    _set_equal_aspect(ax2, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)
    sm2 = cm.ScalarMappable(cmap=cmap2, norm=norm2); sm2.set_array([])
    cb2 = fig.colorbar(sm2, ax=ax2, fraction=0.03, pad=0.10, shrink=0.65)
    cb2.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cb2.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])
    _style_ax(ax2, fig, cb2, "arg(ψ)")
    ax2.set_title("arg(ψ)  (phase)", fontsize=12,
                  color="white", pad=12)

    fig.savefig(filename, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\nSaved → {filename}")
    plt.close(fig)


if __name__ == "__main__":
    sol = run_simulation()
    plot_isometric(sol)

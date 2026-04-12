"""Visualization utilities — 2-D slice plots, 3-D scatter, animations."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from ..core.parameters import SimulationParameters
from ..core.solution import Solution
from ..mesh.indices import GridIndices


def _grid_coords_2d(params: SimulationParameters) -> tuple[NDArray, NDArray]:
    """Return meshgrid arrays for the interior nodes in x–y."""
    xs = np.arange(1, params.Nx) * params.hx
    ys = np.arange(1, params.Ny) * params.hy
    return np.meshgrid(xs, ys, indexing="ij")


def plot_order_parameter(
    solution: Solution,
    step: int = -1,
    slice_z: int = 0,
    ax=None,
    cmap: str = "inferno",
    vmin: float = 0.0,
    vmax: float = 1.0,
    title: Optional[str] = None,
):
    """Plot |ψ|² on a 2-D slice.

    Parameters
    ----------
    solution : Solution
    step : int
        Time-step index.
    slice_z : int
        z-slice for 3-D data.
    ax : matplotlib Axes, optional
    cmap : str
    vmin, vmax : float
    title : str, optional

    Returns
    -------
    ax : matplotlib Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    data = solution.psi_squared_2d(step=step, slice_z=slice_z)
    data = np.clip(data / max(data.max(), 1.0), vmin, vmax)

    xx, yy = _grid_coords_2d(solution.params)
    im = ax.pcolormesh(xx, yy, data, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title is None:
        t = solution.times[step]
        title = f"|ψ|²  t = {t:.3f}"
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_bfield(
    solution: Solution,
    component: str = "z",
    step: int = -1,
    slice_z: int = 0,
    ax=None,
    cmap: str = "RdBu_r",
):
    """Plot a component of B on a 2-D slice."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    Bx, By, Bz = solution.bfield(step=step)
    comp_map = {"x": Bx, "y": By, "z": Bz}
    data = comp_map[component.lower()]

    params = solution.params
    # B-field grid is one layer smaller
    n_bx = params.Nx - 2
    n_by = params.Ny - 2
    n_bxy = n_bx * n_by
    if params.is_3d:
        n_bz = max(params.Nz - 2, 1)
        slice_z = min(slice_z, n_bz - 1)
        offset = n_bxy * slice_z
        data_2d = data[offset : offset + n_bxy].reshape(n_bx, n_by)
    else:
        data_2d = data.reshape(n_bx, n_by)

    xs = np.arange(1, n_bx + 1) * params.hx
    ys = np.arange(1, n_by + 1) * params.hy
    xx, yy = np.meshgrid(xs, ys, indexing="ij")

    vlim = max(abs(data_2d.max()), abs(data_2d.min()), 1e-10)
    im = ax.pcolormesh(xx, yy, np.real(data_2d), cmap=cmap, vmin=-vlim, vmax=vlim, shading="auto")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    t = solution.times[step]
    ax.set_title(f"B_{component}  t = {t:.3f}")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_summary(
    solution: Solution,
    step: int = -1,
    slice_z: int = 0,
    figsize: tuple[float, float] = (14, 5),
):
    """Side-by-side |ψ|² and Bz."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    plot_order_parameter(solution, step=step, slice_z=slice_z, ax=ax1)
    plot_bfield(solution, component="z", step=step, slice_z=slice_z, ax=ax2)
    fig.tight_layout()
    return fig


def animate(
    solution: Solution,
    filename: str | Path = "tdgl3d.gif",
    slice_z: int = 0,
    fps: int = 10,
    step_stride: int = 1,
):
    """Create an animated GIF of |ψ|² over time.

    Requires ``matplotlib`` with Pillow for GIF writing.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    steps = range(0, solution.n_steps, step_stride)

    # First frame
    data = solution.psi_squared_2d(step=0, slice_z=slice_z)
    data = np.clip(data / max(data.max(), 1.0), 0, 1)
    xx, yy = _grid_coords_2d(solution.params)
    mesh = ax.pcolormesh(xx, yy, data, cmap="inferno", vmin=0, vmax=1, shading="auto")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    title = ax.set_title(f"|ψ|²  t = {solution.times[0]:.3f}")
    plt.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)

    def update(frame_idx):
        s = list(steps)[frame_idx]
        d = solution.psi_squared_2d(step=s, slice_z=slice_z)
        d = np.clip(d / max(d.max(), 1.0), 0, 1)
        mesh.set_array(d.ravel())
        title.set_text(f"|ψ|²  t = {solution.times[s]:.3f}")
        return mesh, title

    anim = FuncAnimation(fig, update, frames=len(list(steps)), blit=False)
    anim.save(str(filename), writer=PillowWriter(fps=fps))
    plt.close(fig)
    return filename

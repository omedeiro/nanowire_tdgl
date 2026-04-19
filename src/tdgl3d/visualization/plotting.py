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

    # Get B-field (returns all interior nodes by default)
    Bx, By, Bz = solution.bfield(step=step, full_interior=True)
    comp_map = {"x": Bx, "y": By, "z": Bz}
    data = comp_map[component.lower()]

    params = solution.params
    
    # Reshape using Solution's helper (handles 2D and 3D correctly)
    data_2d = solution._reshape_interior(data, slice_z=slice_z)
    
    # Grid coordinates for interior nodes
    xs = np.arange(1, params.Nx) * params.hx
    ys = np.arange(1, params.Ny) * params.hy
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


def plot_current_density(
    solution: Solution,
    step: int = -1,
    slice_z: int = 0,
    streamplot: bool = True,
    stream_density: float = 1.5,
    stream_color: str = "white",
    stream_linewidth: float = 0.8,
    stream_arrowsize: float = 0.8,
    figsize: tuple[float, float] = (18, 5),
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    """Plot current densities in a 3-panel figure: supercurrent, normal, total.
    
    Creates a comprehensive visualization showing:
    - Left panel: Supercurrent density |J_s| with optional streamlines
    - Middle panel: Normal current density |J_n| with optional streamlines
    - Right panel: Total current density |J| with optional streamlines
    
    Parameters
    ----------
    solution : Solution
        The TDGL solution to visualize
    step : int, default -1
        Time-step index (default: final step)
    slice_z : int, default 0
        z-slice index for 3D data (0 for 2D)
    streamplot : bool, default True
        If True, overlay streamlines showing current flow direction
    stream_density : float, default 1.5
        Density of streamlines (higher = more lines)
    stream_color : str, default "white"
        Color of streamlines
    stream_linewidth : float, default 0.8
        Width of streamlines
    stream_arrowsize : float, default 0.8
        Size of arrows on streamlines
    figsize : tuple, default (18, 5)
        Figure size (width, height) in inches
    cmap : str, default "viridis"
        Colormap for current magnitude
    vmin, vmax : float, optional
        Color scale limits (auto if None)
        
    Returns
    -------
    fig : matplotlib Figure
    axes : array of 3 matplotlib Axes
    
    Notes
    -----
    - Supercurrent (J_s) dominates in bulk SC regions and screens B-fields
    - Normal current (J_n) appears near vortex cores and in normal regions
    - Streamlines show current flow direction; color shows magnitude
    - NaN regions (holes, vortex cores) appear as gaps
    
    Examples
    --------
    >>> fig, axes = plot_current_density(solution, streamplot=True)
    >>> fig.savefig("currents.png", dpi=150)
    """
    import matplotlib.pyplot as plt
    
    # Get current densities
    Jsx, Jsy, Jsz = solution.supercurrent_density(step=step)
    J_s_mag = solution.current_magnitude(step=step, dataset="supercurrent")
    
    J_n = solution.normal_current_density(step=step)
    if J_n is not None:
        Jnx, Jny, Jnz = J_n
        J_n_mag = solution.current_magnitude(step=step, dataset="normal")
    else:
        # No normal current available
        Jnx = np.zeros_like(Jsx)
        Jny = np.zeros_like(Jsy)
        J_n_mag = np.zeros_like(J_s_mag)
    
    # Total current
    Jx_tot = Jsx + Jnx
    Jy_tot = Jsy + Jny
    J_tot_mag = np.sqrt(Jx_tot**2 + Jy_tot**2)
    
    # Reshape to 2D grid for plotting
    params = solution.params
    def reshape_current(J_arr):
        if params.is_3d:
            Nx_int, Ny_int = params.Nx - 1, params.Ny - 1
            Nz_int = max(params.Nz - 1, 1)
            J_3d = J_arr.reshape(Nx_int, Ny_int, Nz_int)
            return J_3d[:, :, min(slice_z, Nz_int - 1)]
        else:
            return J_arr.reshape(params.Nx - 1, params.Ny - 1)
    
    Jsx_2d = reshape_current(Jsx)
    Jsy_2d = reshape_current(Jsy)
    J_s_mag_2d = reshape_current(J_s_mag)
    
    Jnx_2d = reshape_current(Jnx)
    Jny_2d = reshape_current(Jny)
    J_n_mag_2d = reshape_current(J_n_mag)
    
    Jx_tot_2d = reshape_current(Jx_tot)
    Jy_tot_2d = reshape_current(Jy_tot)
    J_tot_mag_2d = reshape_current(J_tot_mag)
    
    # Create coordinate grid
    xx, yy = _grid_coords_2d(params)
    
    # Determine color scale
    if vmax is None:
        vmax = max(J_s_mag_2d.max(), J_n_mag_2d.max(), J_tot_mag_2d.max(), 1e-10)
    if vmin is None:
        vmin = 0.0
    
    # Create 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Panel 1: Supercurrent
    ax = axes[0]
    im1 = ax.pcolormesh(xx, yy, J_s_mag_2d, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
    if streamplot and J_s_mag_2d.max() > 1e-10:
        ax.streamplot(
            xx.T, yy.T, Jsx_2d.T, Jsy_2d.T,
            color=stream_color,
            density=stream_density,
            linewidth=stream_linewidth,
            arrowsize=stream_arrowsize,
        )
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    t = solution.times[step]
    ax.set_title(f"Supercurrent |J_s|  (t = {t:.3f})")
    plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04, label="|J_s|")
    
    # Panel 2: Normal current
    ax = axes[1]
    im2 = ax.pcolormesh(xx, yy, J_n_mag_2d, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
    if streamplot and J_n_mag_2d.max() > 1e-10:
        ax.streamplot(
            xx.T, yy.T, Jnx_2d.T, Jny_2d.T,
            color=stream_color,
            density=stream_density,
            linewidth=stream_linewidth,
            arrowsize=stream_arrowsize,
        )
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Normal Current |J_n|  (t = {t:.3f})")
    plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04, label="|J_n|")
    
    # Panel 3: Total current
    ax = axes[2]
    im3 = ax.pcolormesh(xx, yy, J_tot_mag_2d, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
    if streamplot and J_tot_mag_2d.max() > 1e-10:
        ax.streamplot(
            xx.T, yy.T, Jx_tot_2d.T, Jy_tot_2d.T,
            color=stream_color,
            density=stream_density,
            linewidth=stream_linewidth,
            arrowsize=stream_arrowsize,
        )
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Total Current |J|  (t = {t:.3f})")
    plt.colorbar(im3, ax=ax, fraction=0.046, pad=0.04, label="|J|")
    
    fig.tight_layout()
    return fig, axes

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


def plot_bfield_streamlines(
    solution: Solution,
    step: int = -1,
    slice_z: int = 0,
    streamplot: bool = True,
    stream_density: float = 1.5,
    stream_color: str = "white",
    stream_linewidth: float = 0.8,
    stream_arrowsize: float = 1.0,
    component: str = "z",
    cmap: str = "RdBu_r",
    ax=None,
    figsize: tuple[float, float] = (8, 6),
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    """Plot B-field with optional magnetic field line streamlines.
    
    Creates a visualization showing the magnetic field with optional 
    streamlines indicating field line direction.
    
    Parameters
    ----------
    solution : Solution
        The TDGL solution to visualize
    step : int, default -1
        Time-step index (default: final step)
    slice_z : int, default 0
        z-slice index for 3D data (0 for 2D)
    streamplot : bool, default True
        If True, overlay streamlines of in-plane field (Bx, By)
    stream_density : float, default 1.5
        Density of streamlines (higher = more lines)
    stream_color : str, default "white"
        Color for streamlines
    stream_linewidth : float, default 0.8
        Width of streamlines
    stream_arrowsize : float, default 1.0
        Size of arrows on streamlines
    component : str, default "z"
        Component for heatmap background: 'x', 'y', 'z', or 'magnitude'
    cmap : str, default "RdBu_r"
        Colormap for field magnitude (diverging for components, sequential for magnitude)
    ax : matplotlib Axes, optional
        Axes to plot on (creates new if None)
    figsize : tuple, default (8, 6)
        Figure size if creating new axes
    vmin, vmax : float, optional
        Color scale limits (auto if None)
    
    Returns
    -------
    fig : matplotlib Figure
    ax : matplotlib Axes
    
    Notes
    -----
    - Heatmap shows the selected B-field component or total magnitude
    - Streamlines show in-plane field direction (Bx, By)
    - White streamlines provide good contrast on diverging colormaps
    - NaN regions (vortex cores, holes) appear as gaps
    
    Examples
    --------
    >>> fig, ax = plot_bfield_streamlines(solution, component='z')
    >>> fig.savefig("bfield.png", dpi=150)
    
    >>> # Show total field magnitude with streamlines
    >>> fig, ax = plot_bfield_streamlines(solution, component='magnitude', cmap='plasma')
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Get B-field (full interior to include holes)
    Bx, By, Bz = solution.bfield(step=step, full_interior=True)
    
    params = solution.params
    
    # Reshape to 2D slice
    Bx_2d = solution._reshape_interior(Bx, slice_z=slice_z)
    By_2d = solution._reshape_interior(By, slice_z=slice_z)
    Bz_2d = solution._reshape_interior(Bz, slice_z=slice_z)
    
    # Select component for heatmap
    if component.lower() == 'x':
        data = Bx_2d
        label = r"$B_x$"
        diverging = True
    elif component.lower() == 'y':
        data = By_2d
        label = r"$B_y$"
        diverging = True
    elif component.lower() == 'z':
        data = Bz_2d
        label = r"$B_z$"
        diverging = True
    elif component.lower() == 'magnitude':
        data = np.sqrt(Bx_2d**2 + By_2d**2 + Bz_2d**2)
        label = r"$|B|$"
        diverging = False
    else:
        raise ValueError(f"Unknown component '{component}'. Use 'x', 'y', 'z', or 'magnitude'.")
    
    # Grid coordinates for interior nodes
    xs = np.arange(1, params.Nx) * params.hx
    ys = np.arange(1, params.Ny) * params.hy
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    
    # Auto-scale color limits
    if vmin is None or vmax is None:
        if diverging:
            vlim = max(np.nanmax(np.abs(data)), 1e-10)
            vmin_auto, vmax_auto = -vlim, vlim
        else:
            vmin_auto = np.nanmin(data)
            vmax_auto = np.nanmax(data)
        vmin = vmin if vmin is not None else vmin_auto
        vmax = vmax if vmax is not None else vmax_auto
    
    # Heatmap
    im = ax.pcolormesh(xx, yy, np.real(data), cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
    
    # Streamlines of in-plane field
    if streamplot:
        # Mask NaN values for streamplot
        Bx_stream = np.where(np.isnan(Bx_2d), 0.0, Bx_2d)
        By_stream = np.where(np.isnan(By_2d), 0.0, By_2d)
        
        # Only plot streamlines where field magnitude is significant
        B_mag = np.sqrt(Bx_stream**2 + By_stream**2)
        threshold = 1e-8 * np.nanmax(B_mag) if np.nanmax(B_mag) > 0 else 0
        
        # Create masked arrays
        Bx_masked = np.ma.array(Bx_stream, mask=(B_mag < threshold))
        By_masked = np.ma.array(By_stream, mask=(B_mag < threshold))
        
        try:
            ax.streamplot(
                xx.T, yy.T,  # Note: transpose for streamplot
                Bx_masked.T, By_masked.T,
                color=stream_color,
                density=stream_density,
                linewidth=stream_linewidth,
                arrowsize=stream_arrowsize,
            )
        except (ValueError, IndexError) as e:
            # Streamplot can fail if field is too uniform or has NaNs
            # Just skip streamlines in this case
            pass
    
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$ ($\xi$)")
    ax.set_ylabel(r"$y$ ($\xi$)")
    t = solution.times[step]
    ax.set_title(f"B-field ({label}) at t = {t:.3f}")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=label)
    
    return fig, ax


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
    hole_polygon: Optional[list[tuple[float, float]]] = None,
    hole_color: str = "red",
    hole_linestyle: str = "--",
    hole_linewidth: float = 1.5,
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
    hole_polygon : list of (x, y) tuples, optional
        If provided, draws the hole outline on all panels.
        Coordinates should be in the same units as the grid (ξ).
    hole_color : str, default "red"
        Color of hole outline
    hole_linestyle : str, default "--"
        Line style for hole outline (e.g., "--", "-", ":")
    hole_linewidth : float, default 1.5
        Width of hole outline
        
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
    - Hole outline helps visualize where zero-current BCs are enforced
    
    Examples
    --------
    >>> # Without hole outline
    >>> fig, axes = plot_current_density(solution, streamplot=True)
    >>> fig.savefig("currents.png", dpi=150)
    >>> 
    >>> # With hole outline
    >>> hole = [(10.0, 10.0), (20.0, 10.0), (20.0, 20.0), (10.0, 20.0)]
    >>> fig, axes = plot_current_density(solution, hole_polygon=hole)
    >>> fig.savefig("currents_with_hole.png", dpi=150)
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
    
    # Draw hole outline on all panels if provided
    if hole_polygon is not None:
        import matplotlib.patches as mpatches
        # Close the polygon
        poly_closed = list(hole_polygon) + [hole_polygon[0]]
        xs = [p[0] for p in poly_closed]
        ys = [p[1] for p in poly_closed]
        
        for ax in axes:
            ax.plot(xs, ys, color=hole_color, linestyle=hole_linestyle, 
                   linewidth=hole_linewidth, label="Hole boundary")
    
    fig.tight_layout()
    return fig, axes

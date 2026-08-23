"""Vortex detection and flux quantization for TDGL simulations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..core.device import Device
    from ..core.solution import Solution


def _interior_slice(
    flat: NDArray,
    params,
    slice_z: int,
) -> NDArray:
    """Reshape a flat *interior* array and take the ``slice_z`` z-plane.

    The interior numbering is i-slowest / k-fastest (see
    :mod:`tdgl3d.mesh.indices`), so the C-order reshape below is
    ``(Nx-1, Ny-1, Nz-1)`` and the result is indexed ``[i, j]``.
    """
    nx_int, ny_int, nz_int = params.Nx - 1, params.Ny - 1, max(params.Nz - 1, 1)
    return flat.reshape(nx_int, ny_int, nz_int)[:, :, slice_z]


def plaquette_vorticity(
    solution: Solution,
    slice_z: int = 0,
    step: int = -1,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Gauge-invariant vorticity of every elementary plaquette in a z-slice.

    For the link-variable discretisation the vorticity of the plaquette whose
    corners are the interior nodes ``(i,j) → (i+1,j) → (i+1,j+1) → (i,j+1)`` is

    .. math::
        n = \frac{1}{2\pi}\Big(\sum_{\text{links}}
            \operatorname{wrap}(\Delta\theta - \varphi) + \Phi\Big),

    where :math:`\operatorname{wrap}(\cdot)` folds into :math:`(-\pi, \pi]`,
    :math:`\varphi` is the link variable traversed in the direction of travel and
    :math:`\Phi = h_x h_y B_z` is the flux through the plaquette.  Because the
    bare phases cancel pairwise around a closed loop, ``n`` is an **exact
    integer** up to floating-point round-off, and it is invariant under
    ``ψ → ψ e^{iχ}``, ``φ → φ + Δχ``.

    This is the lattice statement of fluxoid quantisation; it is the quantity
    the continuum expression :math:`\oint(\nabla\theta - A)\cdot dl = 2\pi n`
    converges to.

    Parameters
    ----------
    solution : Solution
        The simulation result.
    slice_z : int, default 0
        Interior z-plane index (``0 … Nz-2``).
    step : int, default -1
        Which saved time step to analyse.

    Returns
    -------
    vorticity : ndarray, shape (Nx-2, Ny-2)
        Vorticity of each plaquette; entry ``[i, j]`` is the plaquette whose
        lower-left corner is interior node ``(i, j)``.
    psi2_min : ndarray, shape (Nx-2, Ny-2)
        Smallest ``|ψ|²`` among the four corners of each plaquette, for masking
        out holes and insulator regions.
    """
    params = solution.params
    n = params.n_interior
    state = solution.states[:, step]

    psi = _interior_slice(state[:n], params, slice_z)
    phi_x = np.real(_interior_slice(state[n : 2 * n], params, slice_z))
    phi_y = np.real(_interior_slice(state[2 * n : 3 * n], params, slice_z))

    theta = np.angle(psi)
    psi2 = np.abs(psi) ** 2

    # Corner slices: 00 = (i,j), 10 = (i+1,j), 11 = (i+1,j+1), 01 = (i,j+1)
    t00, t10, t11, t01 = theta[:-1, :-1], theta[1:, :-1], theta[1:, 1:], theta[:-1, 1:]
    # Links traversed: +x at (i,j), +y at (i+1,j), -x at (i,j+1), -y at (i,j)
    px_b, py_r = phi_x[:-1, :-1], phi_y[1:, :-1]
    px_t, py_l = phi_x[:-1, 1:], phi_y[:-1, :-1]

    gauge_sum = (
        _wrap_phase(t10 - t00 - px_b)
        + _wrap_phase(t11 - t10 - py_r)
        + _wrap_phase(t01 - t11 + px_t)
        + _wrap_phase(t00 - t01 + py_l)
    )
    flux = px_b + py_r - px_t - py_l

    vorticity = (gauge_sum + flux) / (2.0 * np.pi)
    psi2_min = np.minimum(
        np.minimum(psi2[:-1, :-1], psi2[1:, :-1]),
        np.minimum(psi2[1:, 1:], psi2[:-1, 1:]),
    )
    return vorticity, psi2_min


def count_vortices_plaquette(
    solution: Solution,
    device: Device,
    slice_z: int = 0,
    step: int = -1,
    winding_threshold: float = 0.8,
    mask_threshold: float = 1e-6,
) -> tuple[int, NDArray[np.float64], NDArray[np.float64]]:
    """Count vortices from the gauge-invariant winding around each plaquette.

    Thin wrapper over :func:`plaquette_vorticity` that drops plaquettes touching
    a hole/insulator and reports the ones carrying a topological charge.

    Parameters
    ----------
    solution : Solution
        The simulation result
    device : Device
        The device (unused; kept for backwards compatibility of the call site)
    slice_z : int, default 0
        Which z-slice to analyze (interior index 0 to Nz-2)
    step : int, default -1
        Which saved time step to analyze
    winding_threshold : float, default 0.8
        Detect vortex if |winding_number| > threshold (1.0 = full 2π winding)
    mask_threshold : float, default 1e-6
        Ignore plaquettes where any corner has |ψ|² < threshold (insulator/hole)

    Returns
    -------
    n_vortices : int
        Number of plaquettes carrying non-zero vorticity
    vortex_positions : ndarray, shape (n_vortices, 2)
        (x, y) grid coordinates of vortex centers (plaquette centers)
    winding_numbers : ndarray, shape (n_vortices,)
        Winding number for each vortex (±1 for a singly-quantised vortex)

    Notes
    -----
    The winding is computed from *gauge-invariant* link phases, so the count does
    not change under a gauge transformation and each entry of
    ``winding_numbers`` is an integer to machine precision.  A multiply-quantised
    core shows up as a single entry with ``|winding| > 1``, so ``n_vortices``
    counts cores rather than flux quanta.
    """
    vorticity, psi2_min = plaquette_vorticity(solution, slice_z=slice_z, step=step)

    detected = (np.abs(vorticity) > winding_threshold) & (psi2_min >= mask_threshold)
    ii, jj = np.nonzero(detected)

    n_vortices = int(ii.size)
    if n_vortices == 0:
        return 0, np.empty((0, 2)), np.empty(0)

    vortex_positions = np.column_stack([ii + 0.5, jj + 0.5]).astype(float)
    winding_numbers = vorticity[ii, jj]
    return n_vortices, vortex_positions, winding_numbers


def _rectilinear_lattice_loop(
    polygon: NDArray[np.float64],
    i_lo: int,
    i_hi: int,
    j_lo: int,
    j_hi: int,
) -> NDArray[np.intp]:
    """Snap a polygon to a closed staircase path of unit steps on the grid.

    Vertices are rounded to the nearest node and clipped to
    ``[i_lo, i_hi] × [j_lo, j_hi]``; consecutive vertices are joined by moving
    along x first and then along y.  Returns the ``(n_nodes, 2)`` array of
    ``(i, j)`` node indices, with the first node repeated at the end.
    """
    verts = np.asarray(polygon, dtype=float)
    if verts.ndim != 2 or verts.shape[1] != 2:
        raise ValueError("polygon_points must have shape (n_points, 2)")
    if len(verts) > 1 and np.allclose(verts[0], verts[-1]):
        verts = verts[:-1]
    if len(verts) < 3:
        raise ValueError("polygon_points must contain at least 3 distinct vertices")

    nodes = np.empty((len(verts), 2), dtype=np.intp)
    nodes[:, 0] = np.clip(np.rint(verts[:, 0]), i_lo, i_hi)
    nodes[:, 1] = np.clip(np.rint(verts[:, 1]), j_lo, j_hi)

    path: list[tuple[int, int]] = [(int(nodes[0, 0]), int(nodes[0, 1]))]
    for target in list(nodes[1:]) + [nodes[0]]:
        ti, tj = int(target[0]), int(target[1])
        ci, cj = path[-1]
        while ci != ti:
            ci += 1 if ti > ci else -1
            path.append((ci, cj))
        while cj != tj:
            cj += 1 if tj > cj else -1
            path.append((ci, cj))
    return np.array(path, dtype=np.intp)


def count_vortices_polygon(
    solution: Solution,
    device: Device,
    polygon_points: NDArray[np.float64],
    slice_z: int = 0,
    step: int = -1,
) -> float:
    r"""Fluxoid number enclosed by a closed contour, in units of Φ₀.

    The contour is snapped to a closed staircase of unit lattice steps and the
    fluxoid is accumulated link by link:

    .. math::
        2\pi n = \sum_{\text{links}}
            \operatorname{wrap}(\Delta\theta - \varphi) \; + \; \oint A\cdot dl ,

    where the first sum is the gauge-invariant phase gradient (the discrete
    :math:`\oint(\nabla\theta - A)\cdot dl`, i.e. the :math:`\lambda^2 J_s`
    term of the continuum fluxoid) and the second is the enclosed flux
    :math:`\sum \varphi` along the loop.  The bare phases cancel pairwise around
    the closed path, so ``n`` is an **exact integer** to floating-point
    round-off, independent of gauge, contour shape and field strength.

    Parameters
    ----------
    solution : Solution
        The simulation result
    device : Device
        The device (unused; kept for backwards compatibility of the call site)
    polygon_points : ndarray, shape (n_points, 2)
        (x, y) vertices of the contour in **full-grid node** coordinates.
        Vertices are rounded to the nearest node and clipped to the interior
        (``1 … Nx-1``, ``1 … Ny-1``) — the order parameter is not defined on the
        boundary ring, so a contour must not run along it.
    slice_z : int, default 0
        Which z-slice to analyze (interior index)
    step : int, default -1
        Which saved time step to analyze

    Returns
    -------
    n_vortices : float
        Number of flux quanta (vortices, or a trapped-flux quantum in a hole)
        enclosed by the contour.  Integer-valued up to round-off.

    Notes
    -----
    Unlike :func:`count_hole_flux_quanta`, which integrates the *magnetic* flux
    and is therefore not quantised, this quantity is topologically quantised: it
    counts phase singularities enclosed by the contour whether they sit in the
    superconductor (vortices) or in a hole (trapped fluxoid).

    Examples
    --------
    >>> polygon = np.array([[9, 9], [21, 9], [21, 21], [9, 21]])
    >>> fluxoid = count_vortices_polygon(sol, dev, polygon)  # doctest: +SKIP
    1.0
    """
    from ..physics.rhs import _expand_interior_to_full

    params = solution.params
    idx = solution.idx
    n = params.n_interior
    mj, mk = params.mj, params.mk

    state = solution.states[:, step]
    psi_full = _expand_interior_to_full(state[:n], params, idx)
    phi_x_full = np.real(_expand_interior_to_full(state[n : 2 * n], params, idx))
    phi_y_full = np.real(_expand_interior_to_full(state[2 * n : 3 * n], params, idx))

    k_full = (slice_z + 1) if params.is_3d else 0
    base = mk * k_full

    path = _rectilinear_lattice_loop(
        polygon_points, i_lo=1, i_hi=params.Nx - 1, j_lo=1, j_hi=params.Ny - 1
    )

    theta = np.angle(psi_full)
    total = 0.0
    for (i0, j0), (i1, j1) in zip(path[:-1], path[1:]):
        node0 = base + i0 + mj * j0
        node1 = base + i1 + mj * j1
        if i1 == i0 + 1:
            phi = phi_x_full[node0]
        elif i1 == i0 - 1:
            phi = -phi_x_full[node1]
        elif j1 == j0 + 1:
            phi = phi_y_full[node0]
        elif j1 == j0 - 1:
            phi = -phi_y_full[node1]
        else:  # pragma: no cover - _rectilinear_lattice_loop only emits unit steps
            raise AssertionError("lattice path contains a non-unit step")
        total += float(_wrap_phase(theta[node1] - theta[node0] - phi)) + phi

    return float(total / (2.0 * np.pi))


def count_hole_flux_quanta(
    solution: Solution,
    device: Device,
    hole_bounds: tuple[float, float, float, float],
    slice_z: int = 0,
    step: int = -1,
) -> float:
    """Compute total magnetic flux through a hole region.

    Integrates B_z over the hole area to determine how much magnetic flux
    penetrates the hole.

    **IMPORTANT**: This computes MAGNETIC FLUX (∫∫B·dA), which is NOT quantized.
    For a hole surrounded by superconductor, screening currents at the boundary
    will expel flux, so the penetrating flux can be much less than 1 Φ₀.

    To measure the quantized fluxoid (which includes supercurrent contribution),
    use `count_vortices_polygon()` with a contour around the hole.

    Parameters
    ----------
    solution : Solution
        The simulation result
    device : Device
        The device (needed for B-field calculation)
    hole_bounds : tuple of (x_min, x_max, y_min, y_max)
        Hole boundaries in grid index coordinates
    slice_z : int, default 0
        Which z-slice to analyze (interior index)
    step : int, default -1
        Which saved time step to analyze

    Returns
    -------
    n_flux_quanta : float
        Magnetic flux penetrating the hole, in units of Φ₀
        **NOT QUANTIZED** - can be any value (typically << 1 due to Meissner screening)

    Notes
    -----
    Magnetic flux through area A:
        Φ_B = ∬_A B_z dA

    Fluxoid (quantized) through contour C enclosing the hole:
        Φ_f = ∮_C (A + λ²J_s) · dl = n·Φ₀  (integer n)

    Relationship:
        Φ_f = Φ_B + ∮_C λ²J_s · dl

    For a hole surrounded by SC, screening currents contribute negatively,
    so Φ_B < Φ_f. In dimensionless units, Φ₀ = 2π.

    Examples
    --------
    >>> # Magnetic flux through hole (not quantized)
    >>> flux_magnetic = count_hole_flux_quanta(sol, dev, (10, 20, 10, 20))
    >>> # Typical result: 0.05 Φ₀ (small due to screening)
    >>>
    >>> # Fluxoid around hole (quantized)
    >>> polygon = np.array([[9, 9], [21, 9], [21, 21], [9, 21]])
    >>> fluxoid = count_vortices_polygon(sol, dev, polygon)
    >>> # Typical result: 1.0 Φ₀ (quantized integer)
    """
    params = solution.params

    # Get B-field at the slice (with boundary conditions applied)
    Bx, By, Bz = solution.bfield(step=step, full_interior=True)

    # Reshape to 3D grid
    nx_int, ny_int, nz_int = params.Nx - 1, params.Ny - 1, max(params.Nz - 1, 1)
    Bz_3d = Bz.reshape(nx_int, ny_int, nz_int)

    # Extract slice
    Bz_slice = Bz_3d[:, :, slice_z]

    # Extract hole region
    x_min, x_max, y_min, y_max = hole_bounds
    i_min = max(0, int(x_min))
    i_max = min(nx_int, int(x_max))
    j_min = max(0, int(y_min))
    j_max = min(ny_int, int(y_max))

    # Integrate B_z over hole area
    Bz_hole = Bz_slice[i_min:i_max, j_min:j_max]

    # Flux = ∫∫ B_z dA, where dA = hx * hy in each cell
    flux = np.sum(Bz_hole) * params.hx * params.hy

    # Convert to flux quanta (Φ₀ = 2π in dimensionless units)
    n_flux_quanta = flux / (2.0 * np.pi)

    return float(n_flux_quanta)


def find_vortex_cores(
    solution: Solution,
    device: Device,
    slice_z: int = 0,
    step: int = -1,
    threshold: float = 0.1,
    separation: int = 2,
) -> NDArray[np.float64]:
    """Locate vortex cores as local minima of |ψ|².

    Finds points where |ψ|² < threshold and is a local minimum within
    a neighborhood. This is useful for visualization but doesn't give
    winding numbers or distinguish ±1 vortices.

    Parameters
    ----------
    solution : Solution
        The simulation result
    device : Device
        The device
    slice_z : int, default 0
        Which z-slice to analyze (interior index)
    step : int, default -1
        Which saved time step to analyze
    threshold : float, default 0.1
        Maximum |ψ|² to consider as potential vortex core
    separation : int, default 2
        Minimum separation between cores (in grid points)

    Returns
    -------
    cores : ndarray, shape (n_cores, 2)
        (x, y) grid indices of vortex core positions

    Notes
    -----
    This method is less reliable than phase winding, as low |ψ|² can also
    occur at boundaries, insulators, or due to field-induced suppression.
    Use plaquette or polygon methods for quantitative vortex counting.
    """
    from scipy.ndimage import minimum_filter

    params = solution.params

    # Get |ψ|² at the slice
    psi = solution.psi(step=step)
    psi2 = np.abs(psi) ** 2

    nx_int, ny_int, nz_int = params.Nx - 1, params.Ny - 1, max(params.Nz - 1, 1)
    psi2_3d = psi2.reshape(nx_int, ny_int, nz_int)
    psi2_slice = psi2_3d[:, :, slice_z]

    # Find local minima
    # A point is a local minimum if it equals the minimum in its neighborhood
    min_filtered = minimum_filter(psi2_slice, size=separation)
    is_local_min = (psi2_slice == min_filtered) & (psi2_slice < threshold)

    # Get coordinates
    core_indices = np.argwhere(is_local_min)

    return core_indices.astype(float)


def _wrap_phase(dphi: float) -> float:
    """Wrap phase difference to [-π, π]."""
    return np.arctan2(np.sin(dphi), np.cos(dphi))

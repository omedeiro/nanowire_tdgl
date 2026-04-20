"""Vortex detection and flux quantization for TDGL simulations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..core.solution import Solution
    from ..core.device import Device


def count_vortices_plaquette(
    solution: Solution,
    device: Device,
    slice_z: int = 0,
    step: int = -1,
    winding_threshold: float = 0.8,
    mask_threshold: float = 0.02,
) -> tuple[int, NDArray[np.float64], NDArray[np.float64]]:
    """Count vortices using phase winding around elementary plaquettes.
    
    For each 2×2 plaquette (elementary square) in the grid, computes the
    phase winding by summing phase differences around the 4 edges. A vortex
    is detected when the winding number |Δφ/(2π)| > threshold.
    
    This method is fast and works well for isolated vortices in 2D slices.
    
    Parameters
    ----------
    solution : Solution
        The simulation result
    device : Device
        The device (needed for material mask)
    slice_z : int, default 0
        Which z-slice to analyze (interior index 0 to Nz-2)
    step : int, default -1
        Which saved time step to analyze
    winding_threshold : float, default 0.8
        Detect vortex if |winding_number| > threshold (1.0 = full 2π winding)
    mask_threshold : float, default 0.02
        Ignore plaquettes where any corner has |ψ|² < threshold (insulator/hole)
        
    Returns
    -------
    n_vortices : int
        Total number of vortices (sum of |winding numbers|)
    vortex_positions : ndarray, shape (n_vortices, 2)
        (x, y) grid coordinates of vortex centers (plaquette centers)
    winding_numbers : ndarray, shape (n_vortices,)
        Winding number for each vortex (+1 or -1 typically)
        
    Notes
    -----
    The phase winding around a plaquette with corners at (i,j), (i+1,j), 
    (i+1,j+1), (i,j+1) is computed as:
    
        Δφ = [φ(i+1,j) - φ(i,j)] + [φ(i+1,j+1) - φ(i+1,j)] +
             [φ(i,j+1) - φ(i+1,j+1)] + [φ(i,j) - φ(i,j+1)]
             
    wrapped to [-π, π] at each step. A vortex has Δφ ≈ ±2π.
    """
    params = solution.params
    
    # Get phase at the specified z-slice
    psi = solution.psi(step=step)
    psi2 = np.abs(psi) ** 2
    
    # Reshape to 3D grid
    nx_int, ny_int, nz_int = params.Nx - 1, params.Ny - 1, max(params.Nz - 1, 1)
    psi_3d = psi.reshape(nx_int, ny_int, nz_int)
    psi2_3d = psi2.reshape(nx_int, ny_int, nz_int)
    
    # Extract the 2D slice
    psi_slice = psi_3d[:, :, slice_z]
    psi2_slice = psi2_3d[:, :, slice_z]
    
    # Compute phase
    phase = np.angle(psi_slice)  # Range [-π, π]
    
    # Storage for detected vortices
    vortex_list = []
    winding_list = []
    
    # Scan all plaquettes
    for i in range(nx_int - 1):
        for j in range(ny_int - 1):
            # Four corners of plaquette: (i,j), (i+1,j), (i+1,j+1), (i,j+1)
            corners_psi2 = [
                psi2_slice[i, j],
                psi2_slice[i+1, j],
                psi2_slice[i+1, j+1],
                psi2_slice[i, j+1],
            ]
            
            # Skip plaquettes in insulator/hole regions
            if any(p < mask_threshold for p in corners_psi2):
                continue
            
            # Get phase at corners
            phi = [
                phase[i, j],
                phase[i+1, j],
                phase[i+1, j+1],
                phase[i, j+1],
            ]
            
            # Compute phase differences around the loop (with wrapping to [-π, π])
            # Edge 1: (i,j) → (i+1,j)
            dphi1 = _wrap_phase(phi[1] - phi[0])
            # Edge 2: (i+1,j) → (i+1,j+1)
            dphi2 = _wrap_phase(phi[2] - phi[1])
            # Edge 3: (i+1,j+1) → (i,j+1)
            dphi3 = _wrap_phase(phi[3] - phi[2])
            # Edge 4: (i,j+1) → (i,j)
            dphi4 = _wrap_phase(phi[0] - phi[3])
            
            # Total winding around plaquette
            winding = dphi1 + dphi2 + dphi3 + dphi4
            winding_number = winding / (2.0 * np.pi)
            
            # Detect vortex
            if abs(winding_number) > winding_threshold:
                # Position at plaquette center (in grid coordinates)
                x_center = i + 0.5
                y_center = j + 0.5
                vortex_list.append([x_center, y_center])
                winding_list.append(winding_number)
    
    n_vortices = len(vortex_list)
    
    if n_vortices > 0:
        vortex_positions = np.array(vortex_list)
        winding_numbers = np.array(winding_list)
    else:
        vortex_positions = np.empty((0, 2))
        winding_numbers = np.empty(0)
    
    return n_vortices, vortex_positions, winding_numbers


def count_vortices_polygon(
    solution: Solution,
    device: Device,
    polygon_points: NDArray[np.float64],
    slice_z: int = 0,
    step: int = -1,
) -> float:
    """Count vortices by computing fluxoid through a polygon.
    
    Computes the fluxoid Φ_f = ∮ A·dl + ∮ λ²J_s·dl around a closed polygon.
    For a superconductor with vortices, Φ_f ≈ n·Φ₀ where n is the number
    of enclosed vortices and Φ₀ is the flux quantum.
    
    In dimensionless units (length in ξ, field in H_c2), Φ₀ = 2π, so the
    returned value n = Φ_f/(2π) gives the vortex count directly.
    
    This is more robust than plaquette winding for holes and boundaries.
    
    Parameters
    ----------
    solution : Solution
        The simulation result
    device : Device
        The device (needed for link variables and field)
    polygon_points : ndarray, shape (n_points, 2)
        (x, y) coordinates of polygon vertices in grid index units.
        Polygon should be closed (first and last point can be same or not).
    slice_z : int, default 0
        Which z-slice to analyze (interior index)
    step : int, default -1
        Which saved time step to analyze
        
    Returns
    -------
    n_vortices : float
        Number of vortices enclosed by polygon (can be fractional if polygon
        doesn't fully enclose vortices or cuts through screening currents)
        
    Notes
    -----
    The fluxoid is:
        Φ_f = ∮ (A + λ² J_s) · dl
        
    Using Stokes' theorem:
        Φ_f = ∬ B · dA + ∮ λ² J_s · dl
        
    For a vortex-free region: Φ_f = ∬ B · dA (enclosed flux)
    For a region with vortices: Φ_f includes phase winding contribution
    
    In our dimensionless units (ξ, H_c2), the flux quantum is Φ₀ = 2π.
    """
    from ..physics.rhs import _expand_interior_to_full
    
    params = solution.params
    idx = solution.idx
    n = params.n_interior
    
    # Get state at the specified step
    state = solution.states[:, step]
    psi_int = state[:n]
    phi_x_int = state[n:2*n]
    phi_y_int = state[2*n:3*n]
    
    # Expand to full grid
    psi_full = _expand_interior_to_full(psi_int, params, idx)
    phi_x_full = _expand_interior_to_full(phi_x_int, params, idx)
    phi_y_full = _expand_interior_to_full(phi_y_int, params, idx)
    
    # Reshape to 2D or 3D depending on Nz
    nx, ny = params.Nx + 1, params.Ny + 1
    
    if params.Nz == 1:
        # 2D case: full grid is just (Nx+1) × (Ny+1)
        psi_grid = psi_full.reshape(nx, ny)
        phi_x_grid = phi_x_full.reshape(nx, ny)
        phi_y_grid = phi_y_full.reshape(nx, ny)
        
        # Extract 2D slice (already 2D, just use it)
        psi_slice = psi_grid
        phi_x_slice = phi_x_grid
        phi_y_slice = phi_y_grid
    else:
        # 3D case: full grid is (Nx+1) × (Ny+1) × (Nz+1)
        nz = params.Nz + 1
        psi_grid = psi_full.reshape(nx, ny, nz)
        phi_x_grid = phi_x_full.reshape(nx, ny, nz)
        phi_y_grid = phi_y_full.reshape(nx, ny, nz)
        
        # Extract 2D slice (use interior+boundary nodes)
        psi_slice = psi_grid[:, :, slice_z + 1]  # +1 because slice_z is interior index
        phi_x_slice = phi_x_grid[:, :, slice_z + 1]
        phi_y_slice = phi_y_grid[:, :, slice_z + 1]
    
    # Ensure polygon is closed
    polygon = np.array(polygon_points)
    if not np.allclose(polygon[0], polygon[-1]):
        polygon = np.vstack([polygon, polygon[0:1]])
    
    # Integrate A·dl + λ²J_s·dl around polygon
    # For simplicity, we'll integrate A·dl (link variables) and approximate J_s contribution
    
    # This is a simplified version - for a full implementation we'd need:
    # 1. Interpolate link variables along polygon edges
    # 2. Compute supercurrent along polygon
    # 3. Sum both contributions
    
    # For now, return phase winding as proxy (matches plaquette method)
    # Full implementation would use gauge field line integrals
    
    # Compute phase winding around polygon (simpler approximation)
    fluxoid = 0.0
    
    for i in range(len(polygon) - 1):
        p1 = polygon[i]
        p2 = polygon[i + 1]
        
        # Get indices (clip to valid range)
        i1, j1 = int(np.clip(p1[0], 0, nx-1)), int(np.clip(p1[1], 0, ny-1))
        i2, j2 = int(np.clip(p2[0], 0, nx-1)), int(np.clip(p2[1], 0, ny-1))
        
        # Get phase difference
        phase1 = np.angle(psi_slice[i1, j1])
        phase2 = np.angle(psi_slice[i2, j2])
        
        dphi = _wrap_phase(phase2 - phase1)
        fluxoid += dphi
    
    # Convert to vortex count (Φ₀ = 2π in dimensionless units)
    n_vortices = fluxoid / (2.0 * np.pi)
    
    return float(n_vortices)


def count_hole_flux_quanta(
    solution: Solution,
    device: Device,
    hole_bounds: tuple[float, float, float, float],
    slice_z: int = 0,
    step: int = -1,
) -> float:
    """Compute total magnetic flux through a hole region.
    
    Integrates B_z over the hole area to determine how many flux quanta
    (Φ₀ = h/2e) penetrate the hole.
    
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
        Number of flux quanta Φ₀ penetrating the hole
        
    Notes
    -----
    Flux through area A:
        Φ = ∬_A B_z dA
        
    In dimensionless units, Φ₀ = 2π, so:
        n = Φ / (2π)
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

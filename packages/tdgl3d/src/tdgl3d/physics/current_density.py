"""Compute supercurrent and normal current densities from the TDGL state."""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ..core.parameters import SimulationParameters
from ..mesh.indices import GridIndices


def eval_supercurrent_density(
    psi_full: NDArray[np.complexfloating],
    phi_x_full: NDArray[np.complexfloating],
    phi_y_full: NDArray[np.complexfloating],
    phi_z_full: NDArray[np.complexfloating],
    params: SimulationParameters,
    idx: GridIndices,
) -> tuple[NDArray, NDArray, NDArray]:
    """Compute supercurrent density J_s = Im[ψ* (∇ - iA) ψ] at interior nodes.
    
    The supercurrent is computed using the gauge-covariant derivative on links:
        J_s^x[i] = Im[ψ*[i] × exp(-iφ_x[i]) × ψ[i+1]]
    and similarly for y and z components.
    
    Parameters
    ----------
    psi_full : ndarray, shape (n_full,)
        Order parameter on full grid (including boundaries)
    phi_x_full : ndarray, shape (n_full,)
        x-component of link variables on full grid
    phi_y_full : ndarray, shape (n_full,)
        y-component of link variables on full grid
    phi_z_full : ndarray, shape (n_full,)
        z-component of link variables on full grid (zeros for 2D)
    params : SimulationParameters
    idx : GridIndices
        
    Returns
    -------
    Jx, Jy, Jz : ndarray, each shape (n_interior,)
        Supercurrent density components at interior nodes.
        
    Notes
    -----
    Physical interpretation:
        J_s = |ψ|² (∇θ - A)
    where θ = arg(ψ) is the phase. The gauge-covariant form ensures the
    supercurrent remains gauge-invariant.
    
    In vortex cores and holes where |ψ| → 0, the supercurrent → 0.
    """
    # Interior node indices (in full-grid numbering)
    m = idx.interior_to_full
    mj = params.mj  # Full-grid stride in j direction
    mk = params.mk  # Full-grid stride in k direction
    
    # Supercurrent on x-links: J_sx = Im[ψ*[m] × exp(-iφ_x[m]) × ψ[m+1]]
    Jx = np.imag(np.conj(psi_full[m]) * np.exp(-1j * phi_x_full[m]) * psi_full[m + 1])
    
    # Supercurrent on y-links: J_sy = Im[ψ*[m] × exp(-iφ_y[m]) × ψ[m+mj]]
    Jy = np.imag(np.conj(psi_full[m]) * np.exp(-1j * phi_y_full[m]) * psi_full[m + mj])
    
    if params.is_3d:
        # Supercurrent on z-links: J_sz = Im[ψ*[m] × exp(-iφ_z[m]) × ψ[m+mk]]
        Jz = np.imag(np.conj(psi_full[m]) * np.exp(-1j * phi_z_full[m]) * psi_full[m + mk])
    else:
        Jz = np.zeros(params.n_interior, dtype=np.float64)
    
    return Jx, Jy, Jz


def eval_normal_current_density(
    mu: NDArray[np.float64],
    params: SimulationParameters,
) -> tuple[NDArray, NDArray, NDArray]:
    """Compute normal current density J_n = -∇μ at interior nodes.
    
    The normal current follows Ohm's law in the dimensionless TDGL formulation.
    It represents dissipative current flow driven by the scalar potential gradient.
    
    Parameters
    ----------
    mu : ndarray, shape (n_interior,)
        Scalar potential at interior nodes
    params : SimulationParameters
        
    Returns
    -------
    Jnx, Jny, Jnz : ndarray, each shape (n_interior,)
        Normal current density components at interior nodes.
        
    Notes
    -----
    Uses centered finite differences:
        J_nx[i,j,k] = -(μ[i+1,j,k] - μ[i-1,j,k]) / (2×hx)
        
    At interior boundaries, one-sided differences are used automatically
    since mu is only defined on interior nodes.
    
    In superconducting regions far from vortices, J_n ≈ 0 (dissipationless flow).
    J_n becomes significant near vortex cores and in normal regions.
    """
    Nx_int = params.Nx - 1
    Ny_int = params.Ny - 1
    Nz_int = max(params.Nz - 1, 1)
    
    # Reshape μ to 3D grid for gradient computation
    if params.is_3d:
        mu_3d = mu.reshape(Nx_int, Ny_int, Nz_int)
    else:
        mu_3d = mu.reshape(Nx_int, Ny_int, 1)
    
    # Gradient using centered differences (np.gradient handles boundaries)
    grad_mu_x, grad_mu_y, grad_mu_z = np.gradient(mu_3d, params.hx, params.hy, params.hz)
    
    # Normal current: J_n = -∇μ
    Jnx = -grad_mu_x.ravel()[:params.n_interior]
    Jny = -grad_mu_y.ravel()[:params.n_interior]
    
    if params.is_3d:
        Jnz = -grad_mu_z.ravel()[:params.n_interior]
    else:
        Jnz = np.zeros(params.n_interior, dtype=np.float64)
    
    return Jnx, Jny, Jnz


def eval_current_magnitude(
    Jx: NDArray[np.float64],
    Jy: NDArray[np.float64],
    Jz: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute magnitude |J| = sqrt(Jx² + Jy² + Jz²).
    
    Parameters
    ----------
    Jx, Jy, Jz : ndarray
        Current density components
        
    Returns
    -------
    ndarray
        Current magnitude at each node
    """
    return np.sqrt(Jx**2 + Jy**2 + Jz**2)

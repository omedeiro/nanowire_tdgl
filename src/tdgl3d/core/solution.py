"""Solution container — stores simulation results and provides post-processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .parameters import SimulationParameters
from ..mesh.indices import GridIndices
from ..physics.bfield import eval_bfield, eval_bfield_full

if TYPE_CHECKING:
    from .device import Device


@dataclass
class Solution:
    """Container for the output of a TDGL simulation.

    Attributes
    ----------
    times : ndarray, shape (n_saved,)
    states : ndarray, shape (n_state, n_saved)
    params : SimulationParameters
    idx : GridIndices
    device : Device, optional
        Reference to the Device used for simulation (needed for applied field info)
    """

    times: NDArray[np.float64]
    states: NDArray[np.complex128]
    params: SimulationParameters
    idx: GridIndices
    device: Optional['Device'] = None  # Forward reference, will be resolved at runtime

    # -- convenience accessors -----------------------------------------------

    @property
    def n_steps(self) -> int:
        return self.states.shape[1]

    def psi(self, step: int = -1) -> NDArray[np.complex128]:
        """Order parameter at a given saved step."""
        n = self.params.n_interior
        return self.states[:n, step]
    
    def phase(self, step: int = -1, mask_threshold: float = 0.02) -> NDArray[np.float64]:
        """Phase of order parameter with NaN masking where |ψ|² < threshold.
        
        The phase is undefined in vortex cores, insulator regions, and holes where
        the order parameter vanishes. This method returns np.nan at such locations.
        
        Parameters
        ----------
        step : int, default -1
            Which saved time step to extract (default: final step)
        mask_threshold : float, default 0.02
            Nodes where |ψ|² < mask_threshold will have phase = NaN
            
        Returns
        -------
        ndarray, shape (n_interior,)
            Phase in radians, with NaN where |ψ| → 0
        """
        psi = self.psi(step)
        phase = np.angle(psi).astype(np.float64)
        phase[np.abs(psi) ** 2 < mask_threshold] = np.nan
        return phase

    def psi_squared(self, step: int = -1) -> NDArray[np.float64]:
        """Super-fluid density |ψ|²."""
        return np.abs(self.psi(step)) ** 2

    def phi_x(self, step: int = -1) -> NDArray[np.complex128]:
        n = self.params.n_interior
        return self.states[n : 2 * n, step]

    def phi_y(self, step: int = -1) -> NDArray[np.complex128]:
        n = self.params.n_interior
        return self.states[2 * n : 3 * n, step]

    def phi_z(self, step: int = -1) -> NDArray[np.complex128]:
        if not self.params.is_3d:
            raise AttributeError("No phi_z for 2-D simulations.")
        n = self.params.n_interior
        return self.states[3 * n : 4 * n, step]

    def bfield(self, step: int = -1, full_interior: bool = True) -> tuple[NDArray, NDArray, NDArray]:
        """Compute B = curl(A) at a saved step.
        
        Parameters
        ----------
        step : int, default -1
            Which saved time step to extract (default: final step)
        full_interior : bool, default True
            If True, compute B-field at ALL interior nodes (including holes/insulators).
            If False, compute only at safe subset one layer inward from boundary.
            
        Returns
        -------
        Bx, By, Bz : ndarray
            Magnetic field components. 
            If full_interior=True: shape (n_interior,) - all nodes including holes
            If full_interior=False: shape (len(bfield_interior),) - safe subset only
            
        Notes
        -----
        The default full_interior=True enables visualization of magnetic field 
        penetration into holes and insulator regions. This uses a full-grid 
        expansion internally to safely compute the curl stencil everywhere.
        
        For holes carved in the superconductor (sc_mask=0), the B-field should
        equal the applied field since there is no screening.
        """
        if full_interior:
            # Use full-grid method to get B everywhere including holes
            from ..physics.rhs import _expand_interior_to_full, _apply_boundary_conditions, BoundaryVectors
            from ..physics.applied_field import build_boundary_field_vectors
            
            n = self.params.n_interior
            state = self.states[:, step]
            
            # Extract interior components
            psi_int = state[:n]
            phi_x_int = state[n : 2 * n]
            phi_y_int = state[2 * n : 3 * n]
            phi_z_int = state[3 * n : 4 * n] if self.params.is_3d else np.zeros(n, dtype=np.complex128)
            
            # Expand to full grid
            psi_full = _expand_interior_to_full(psi_int, self.params, self.idx)
            phi_x_full = _expand_interior_to_full(phi_x_int, self.params, self.idx)
            phi_y_full = _expand_interior_to_full(phi_y_int, self.params, self.idx)
            phi_z_full = _expand_interior_to_full(phi_z_int, self.params, self.idx)
            
            # Apply boundary conditions to encode applied field at boundaries
            # This is essential for getting the correct total B-field in holes
            if self.device is not None and self.device.applied_field is not None:
                t = self.times[step]
                bx_app, by_app, bz_app = self.device.applied_field.evaluate(t, self.times[-1])
                Bx_vec, By_vec, Bz_vec = build_boundary_field_vectors(
                    bx_app, by_app, bz_app, self.params, self.idx
                )
                boundary_vecs = BoundaryVectors(Bx_vec, By_vec, Bz_vec)
                psi_full, phi_x_full, phi_y_full, phi_z_full = _apply_boundary_conditions(
                    psi_full, phi_x_full, phi_y_full, phi_z_full,
                    self.params, self.idx, boundary_vecs
                )
            
            return eval_bfield_full(phi_x_full, phi_y_full, phi_z_full, self.params, self.idx)
        else:
            # Original behavior: compute at safe subset
            return eval_bfield(self.states[:, step], self.params, self.idx, full_interior=False)

    # -- slicing helpers for 2-D plots ----------------------------------------

    def _reshape_interior(self, arr: NDArray, slice_z: int = 0) -> NDArray:
        """Reshape an interior-node array to (Nx-1, Ny-1) at a given z-slice.

        The interior vector is C-order ravelled from shape
        ``(Nx-1, Ny-1, Nz-1)`` with k (z) varying **fastest**.
        """
        Nx, Ny, Nz = self.params.Nx, self.params.Ny, self.params.Nz
        if self.params.is_3d:
            cube = arr.reshape(Nx - 1, Ny - 1, max(Nz - 1, 1))
            return cube[:, :, slice_z]
        return arr.reshape(Nx - 1, Ny - 1)

    def psi_squared_2d(self, step: int = -1, slice_z: int = 0) -> NDArray[np.float64]:
        """Return |ψ|² reshaped for a 2-D image at *slice_z*."""
        return np.abs(self._reshape_interior(self.psi(step), slice_z)) ** 2
    
    # -- current density methods ----------------------------------------------
    
    def supercurrent_density(self, step: int = -1) -> tuple[NDArray, NDArray, NDArray]:
        """Compute supercurrent density J_s = Im[ψ* (∇ - iA) ψ].
        
        The supercurrent is the dissipationless current carried by Cooper pairs.
        It screens applied magnetic fields (Meissner effect) and circulates around
        vortex cores.
        
        Parameters
        ----------
        step : int, default -1
            Which saved time step to extract (default: final step)
            
        Returns
        -------
        Jx, Jy, Jz : ndarray, each shape (n_interior,)
            Supercurrent density components at all interior nodes.
            J_s = 0 in vortex cores and holes where |ψ| → 0.
        """
        from ..physics.current_density import eval_supercurrent_density
        from ..physics.rhs import _expand_interior_to_full
        
        n = self.params.n_interior
        state = self.states[:, step]
        
        # Extract and expand to full grid
        psi_int = state[:n]
        phi_x_int = state[n : 2 * n]
        phi_y_int = state[2 * n : 3 * n]
        phi_z_int = state[3 * n : 4 * n] if self.params.is_3d else np.zeros(n, dtype=np.complex128)
        
        psi_full = _expand_interior_to_full(psi_int, self.params, self.idx)
        phi_x_full = _expand_interior_to_full(phi_x_int, self.params, self.idx)
        phi_y_full = _expand_interior_to_full(phi_y_int, self.params, self.idx)
        phi_z_full = _expand_interior_to_full(phi_z_int, self.params, self.idx)
        
        return eval_supercurrent_density(
            psi_full, phi_x_full, phi_y_full, phi_z_full, self.params, self.idx
        )
    
    def normal_current_density(self, step: int = -1) -> Optional[tuple[NDArray, NDArray, NDArray]]:
        """Compute normal current density J_n = -∇μ.
        
        The normal current represents dissipative current flow. It follows Ohm's
        law and generates heat. In the bulk superconductor away from vortices,
        J_n ≈ 0. It becomes significant near vortex cores and in normal regions.
        
        Parameters
        ----------
        step : int, default -1
            Which saved time step to extract (default: final step)
            
        Returns
        -------
        Jnx, Jny, Jnz : ndarray, each shape (n_interior,), or None
            Normal current density components, or None if scalar potential μ
            is not available (only computed for certain solve methods).
        """
        # Note: μ is typically not stored in the standard state vector
        # This would require extending the solver to compute and store μ
        # For now, return None - will be implemented when μ storage is added
        return None
    
    def current_density(self, step: int = -1) -> tuple[NDArray, NDArray, NDArray]:
        """Compute total current density J = J_s + J_n.
        
        In most cases, the supercurrent dominates and J ≈ J_s.
        
        Parameters
        ----------
        step : int, default -1
            Which saved time step to extract (default: final step)
            
        Returns
        -------
        Jx, Jy, Jz : ndarray, each shape (n_interior,)
            Total current density components. Currently returns J_s since
            J_n is not yet stored in the solution.
        """
        J_s = self.supercurrent_density(step)
        J_n = self.normal_current_density(step)
        
        if J_n is None:
            # Only supercurrent available
            return J_s
        
        # Total current
        return (J_s[0] + J_n[0], J_s[1] + J_n[1], J_s[2] + J_n[2])
    
    def current_magnitude(self, step: int = -1, dataset: Optional[str] = None) -> NDArray[np.float64]:
        """Compute current magnitude |J| at interior nodes.
        
        Parameters
        ----------
        step : int, default -1
            Which saved time step to extract (default: final step)
        dataset : str or None, default None
            Which current to compute:
            - None: total current |J_s + J_n|
            - "supercurrent": |J_s|
            - "normal": |J_n|
            
        Returns
        -------
        ndarray, shape (n_interior,)
            Current magnitude at each interior node
        """
        from ..physics.current_density import eval_current_magnitude
        
        if dataset == "supercurrent":
            Jx, Jy, Jz = self.supercurrent_density(step)
        elif dataset == "normal":
            J_n = self.normal_current_density(step)
            if J_n is None:
                return np.zeros(self.params.n_interior, dtype=np.float64)
            Jx, Jy, Jz = J_n
        else:  # None or "total"
            Jx, Jy, Jz = self.current_density(step)
        
        return eval_current_magnitude(Jx, Jy, Jz)

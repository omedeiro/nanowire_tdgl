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
    metadata : dict, optional
        Run metadata including timing, configuration, and diagnostics
    """

    times: NDArray[np.float64]
    states: NDArray[np.complex128]
    params: SimulationParameters
    idx: GridIndices
    device: Optional['Device'] = None  # Forward reference, will be resolved at runtime
    metadata: Optional[dict] = None

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
    
    # -- Vortex counting and convergence analysis ------------------------------
    
    def count_vortices(
        self,
        device: 'Device',
        slice_z: int = 0,
        step: int = -1,
        method: str = "plaquette",
        **kwargs,
    ):
        """Count vortices in a 2D slice.
        
        Parameters
        ----------
        device : Device
            The device (needed for material mask and field info)
        slice_z : int, default 0
            Which z-slice to analyze (interior index 0 to Nz-2)
        step : int, default -1
            Which saved time step to analyze
        method : str, default "plaquette"
            Vortex counting method:
            - "plaquette": Phase winding around elementary squares
            - "polygon": Fluxoid through specified polygon (requires polygon_points kwarg)
            - "cores": Local minima of |ψ|² (less accurate)
        **kwargs
            Additional arguments passed to vortex counting function
            
        Returns
        -------
        Depends on method:
        - "plaquette": (n_vortices, positions, winding_numbers)
        - "polygon": n_vortices (float)
        - "cores": core_positions array
        
        See Also
        --------
        tdgl3d.analysis.vortex_counting
        """
        from ..analysis.vortex_counting import (
            count_vortices_plaquette,
            count_vortices_polygon,
            find_vortex_cores,
        )
        
        if method == "plaquette":
            return count_vortices_plaquette(self, device, slice_z, step, **kwargs)
        elif method == "polygon":
            if 'polygon_points' not in kwargs:
                raise ValueError("polygon method requires polygon_points argument")
            return count_vortices_polygon(self, device, kwargs['polygon_points'], slice_z, step)
        elif method == "cores":
            return find_vortex_cores(self, device, slice_z, step, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def check_steady_state(
        self,
        device: Optional['Device'] = None,
        window_size: int = 10,
        psi_threshold: float = 1e-4,
        current_threshold: float = 1e-4,
        start_step: int = 20,
    ):
        """Check if simulation has reached steady state.
        
        Parameters
        ----------
        device : Device, optional
            The device (enables supercurrent-based convergence check)
        window_size : int, default 10
            Number of saved steps to compare over
        psi_threshold : float, default 1e-4
            Convergence threshold for |ψ|²
        current_threshold : float, default 1e-4
            Convergence threshold for supercurrent (if device provided)
        start_step : int, default 20
            Don't check before this step (allow initial transient)
            
        Returns
        -------
        is_steady : bool
            True if steady state reached
        steady_step : int
            First step where steady state achieved (-1 if never)
        metrics : dict
            Convergence diagnostics
            
        See Also
        --------
        tdgl3d.analysis.convergence.check_steady_state
        """
        from ..analysis.convergence import check_steady_state
        return check_steady_state(
            self, device, window_size, psi_threshold, 
            current_threshold, start_step
        )
    
    # -- I/O methods ---------------------------------------------------------
    
    def save(self, filename: str) -> None:
        """Save solution to HDF5 file.
        
        Saves all simulation data including times, states, parameters, grid indices,
        and metadata to an HDF5 file for later loading and visualization.
        
        Parameters
        ----------
        filename : str
            Output filename (should end with .h5 or .hdf5)
            
        Examples
        --------
        >>> solution = solve(device, ...)
        >>> solution.save("my_simulation.h5")
        >>> # Later...
        >>> loaded = Solution.load("my_simulation.h5")
        
        Notes
        -----
        The Device object is not saved (would require serializing Trilayer, AppliedField, etc.).
        If you need the Device for post-processing, save it separately or reconstruct it.
        """
        import h5py
        
        with h5py.File(filename, 'w') as f:
            # Core data
            f.create_dataset('times', data=self.times)
            f.create_dataset('states', data=self.states)
            
            # Parameters (all scalar attributes)
            grp_params = f.create_group('params')
            for key in ['Nx', 'Ny', 'Nz', 'hx', 'hy', 'hz', 'kappa']:
                grp_params.attrs[key] = getattr(self.params, key)
            
            # GridIndices (store all arrays and scalars)
            grp_idx = f.create_group('idx')
            for key in dir(self.idx):
                if key.startswith('_'):
                    continue
                val = getattr(self.idx, key)
                if isinstance(val, np.ndarray):
                    grp_idx.create_dataset(key, data=val)
                elif isinstance(val, (int, float, bool)):
                    grp_idx.attrs[key] = val
            
            # Metadata (optional)
            if self.metadata is not None:
                grp_meta = f.create_group('metadata')
                for key, val in self.metadata.items():
                    if isinstance(val, (str, int, float, bool)):
                        grp_meta.attrs[key] = val
                    elif isinstance(val, np.ndarray):
                        grp_meta.create_dataset(key, data=val)
                    elif isinstance(val, dict):
                        # Nested dict - store as subgroup
                        sub_grp = grp_meta.create_group(key)
                        for sub_key, sub_val in val.items():
                            if isinstance(sub_val, (str, int, float, bool)):
                                sub_grp.attrs[sub_key] = sub_val
                            elif isinstance(sub_val, np.ndarray):
                                sub_grp.create_dataset(sub_key, data=sub_val)
    
    @staticmethod
    def load(filename: str) -> 'Solution':
        """Load solution from HDF5 file.
        
        Loads a previously saved Solution from an HDF5 file.
        
        Parameters
        ----------
        filename : str
            Input filename (*.h5 or *.hdf5)
            
        Returns
        -------
        Solution
            Loaded solution object (device will be None)
            
        Examples
        --------
        >>> solution = Solution.load("my_simulation.h5")
        >>> psi_final = solution.psi(step=-1)
        >>> 
        >>> # Reconstruct device if needed for analysis
        >>> device = Device(solution.params, ...)
        >>> solution.device = device
        
        Notes
        -----
        The Device is not saved/loaded. If you need it for analysis functions
        (e.g., vortex counting), reconstruct it manually and assign to solution.device.
        """
        import h5py
        
        with h5py.File(filename, 'r') as f:
            # Core data
            times = f['times'][:]
            states = f['states'][:]
            
            # Parameters
            grp_params = f['params']
            params = SimulationParameters(
                Nx=grp_params.attrs['Nx'],
                Ny=grp_params.attrs['Ny'],
                Nz=grp_params.attrs['Nz'],
                hx=grp_params.attrs['hx'],
                hy=grp_params.attrs['hy'],
                hz=grp_params.attrs['hz'],
                kappa=grp_params.attrs['kappa'],
            )
            
            # GridIndices - load all arrays
            from ..mesh.indices import GridIndices
            grp_idx = f['idx']
            idx_dict = {}
            for key in grp_idx.keys():
                idx_dict[key] = grp_idx[key][:]
            
            # GridIndices is a dataclass, construct it with all saved arrays
            idx = GridIndices(**idx_dict)
            
            # Metadata (optional)
            metadata = None
            if 'metadata' in f:
                metadata = {}
                grp_meta = f['metadata']
                for key in grp_meta.attrs:
                    metadata[key] = grp_meta.attrs[key]
                for key in grp_meta.keys():
                    if isinstance(grp_meta[key], h5py.Dataset):
                        metadata[key] = grp_meta[key][:]
                    elif isinstance(grp_meta[key], h5py.Group):
                        # Nested group
                        sub_dict = {}
                        sub_grp = grp_meta[key]
                        for sub_key in sub_grp.attrs:
                            sub_dict[sub_key] = sub_grp.attrs[sub_key]
                        for sub_key in sub_grp.keys():
                            sub_dict[sub_key] = sub_grp[sub_key][:]
                        metadata[key] = sub_dict
            
            return Solution(
                times=times,
                states=states,
                params=params,
                idx=idx,
                device=None,  # Device not saved
                metadata=metadata,
            )

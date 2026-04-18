"""Solution container — stores simulation results and provides post-processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .parameters import SimulationParameters
from ..mesh.indices import GridIndices
from ..physics.bfield import eval_bfield


@dataclass
class Solution:
    """Container for the output of a TDGL simulation.

    Attributes
    ----------
    times : ndarray, shape (n_saved,)
    states : ndarray, shape (n_state, n_saved)
    params : SimulationParameters
    idx : GridIndices
    """

    times: NDArray[np.float64]
    states: NDArray[np.complex128]
    params: SimulationParameters
    idx: GridIndices

    # -- convenience accessors -----------------------------------------------

    @property
    def n_steps(self) -> int:
        return self.states.shape[1]

    def psi(self, step: int = -1) -> NDArray[np.complex128]:
        """Order parameter at a given saved step."""
        n = self.params.n_interior
        return self.states[:n, step]

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

    def bfield(self, step: int = -1) -> tuple[NDArray, NDArray, NDArray]:
        """Compute B = curl(A) at a saved step."""
        return eval_bfield(self.states[:, step], self.params, self.idx)

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

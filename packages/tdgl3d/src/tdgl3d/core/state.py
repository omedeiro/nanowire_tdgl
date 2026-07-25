"""State vector utilities — pack / unpack the TDGL field variables."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .parameters import SimulationParameters


class StateVector:
    """Wrap a flat complex state vector and provide named views into it.

    The state is laid out as ``[ψ, φ_x, φ_y, φ_z]`` where each block has
    ``n_interior`` entries.  For 2-D (``Nz == 1``) the φ_z block is omitted.

    Parameters
    ----------
    data : ndarray, shape (n_state,)
        The raw flat state vector (complex128).
    params : SimulationParameters
        Grid / physics parameters.
    """

    def __init__(self, data: NDArray[np.complexfloating], params: SimulationParameters):
        n = params.n_interior
        expected = params.n_state
        if data.shape != (expected,):
            raise ValueError(
                f"State vector length {data.shape} does not match "
                f"expected ({expected},) for grid {params.Nx}×{params.Ny}×{params.Nz}."
            )
        self._data = np.asarray(data, dtype=np.complex128)
        self._params = params

    # -- views into the flat vector ------------------------------------------

    @property
    def data(self) -> NDArray[np.complex128]:
        return self._data

    @property
    def psi(self) -> NDArray[np.complex128]:
        n = self._params.n_interior
        return self._data[:n]

    @psi.setter
    def psi(self, val: NDArray) -> None:
        n = self._params.n_interior
        self._data[:n] = val

    @property
    def phi_x(self) -> NDArray[np.complex128]:
        n = self._params.n_interior
        return self._data[n : 2 * n]

    @phi_x.setter
    def phi_x(self, val: NDArray) -> None:
        n = self._params.n_interior
        self._data[n : 2 * n] = val

    @property
    def phi_y(self) -> NDArray[np.complex128]:
        n = self._params.n_interior
        return self._data[2 * self._params.n_interior : 3 * self._params.n_interior]

    @phi_y.setter
    def phi_y(self, val: NDArray) -> None:
        n = self._params.n_interior
        self._data[2 * n : 3 * n] = val

    @property
    def phi_z(self) -> NDArray[np.complex128]:
        if not self._params.is_3d:
            raise AttributeError("phi_z is not available for 2-D simulations (Nz == 1).")
        n = self._params.n_interior
        return self._data[3 * n : 4 * n]

    @phi_z.setter
    def phi_z(self, val: NDArray) -> None:
        if not self._params.is_3d:
            raise AttributeError("phi_z is not available for 2-D simulations.")
        n = self._params.n_interior
        self._data[3 * n : 4 * n] = val

    # -- convenience ----------------------------------------------------------

    def copy(self) -> StateVector:
        return StateVector(self._data.copy(), self._params)

    @classmethod
    def uniform_superconducting(cls, params: SimulationParameters) -> StateVector:
        """Initialise with |ψ| = 1 everywhere and zero link variables."""
        data = np.zeros(params.n_state, dtype=np.complex128)
        n = params.n_interior
        data[:n] = 1.0 + 0j
        return cls(data, params)

    @classmethod
    def from_components(
        cls,
        psi: NDArray,
        phi_x: NDArray,
        phi_y: NDArray,
        phi_z: NDArray | None,
        params: SimulationParameters,
    ) -> StateVector:
        if params.is_3d:
            if phi_z is None:
                raise ValueError("phi_z required for 3-D.")
            data = np.concatenate([psi.ravel(), phi_x.ravel(), phi_y.ravel(), phi_z.ravel()])
        else:
            data = np.concatenate([psi.ravel(), phi_x.ravel(), phi_y.ravel()])
        return cls(data.astype(np.complex128), params)

    def __repr__(self) -> str:
        p = self._params
        return (
            f"StateVector(Nx={p.Nx}, Ny={p.Ny}, Nz={p.Nz}, "
            f"n_interior={p.n_interior}, |ψ|_max={np.max(np.abs(self.psi)):.4f})"
        )

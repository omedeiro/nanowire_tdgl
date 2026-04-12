"""Simulation parameters dataclass — replaces the MATLAB ``p`` struct."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class SimulationParameters:
    """All physical and discretisation parameters for a 3D TDGL simulation.

    Parameters
    ----------
    Nx, Ny, Nz : int
        Number of *interior* grid cells in each direction.  The full grid has
        ``(Nx+1) × (Ny+1) × (Nz+1)`` nodes (indices 0 … Nx).  Set ``Nz = 1``
        for a quasi-2-D simulation.
    hx, hy, hz : float
        Grid spacing in each direction (in units of the coherence length ξ).
    kappa : float
        Ginzburg-Landau parameter κ = λ/ξ.
    periodic_x, periodic_y, periodic_z : bool
        Whether to use periodic boundary conditions along each axis.
        Default is ``False`` (zero-current / applied-field BCs).
    """

    # Grid -------------------------------------------------------------------
    Nx: int = 10
    Ny: int = 10
    Nz: int = 1
    hx: float = 1.0
    hy: float = 1.0
    hz: float = 1.0

    # Physics ----------------------------------------------------------------
    kappa: float = 5.0

    # Boundary conditions ----------------------------------------------------
    periodic_x: bool = False
    periodic_y: bool = False
    periodic_z: bool = False

    # Derived (filled by ``construct_indices``) --------------------------------
    # These will be set after construction via ``mesh.construct_indices``.

    def __post_init__(self) -> None:
        if self.Nx < 2 or self.Ny < 2:
            raise ValueError("Nx and Ny must each be >= 2.")
        if self.Nz < 1:
            raise ValueError("Nz must be >= 1.")

    @property
    def is_3d(self) -> bool:
        return self.Nz > 1

    @property
    def n_interior(self) -> int:
        """Number of interior nodes (where the PDE is evaluated)."""
        return (self.Nx - 1) * (self.Ny - 1) * max(self.Nz - 1, 1)

    @property
    def n_full(self) -> int:
        """Total number of nodes on the extended grid (including ghosts)."""
        nz = (self.Nz + 1) if self.is_3d else 1
        return (self.Nx + 1) * (self.Ny + 1) * nz

    @property
    def n_state(self) -> int:
        """Length of the full state vector [ψ; φ_x; φ_y; φ_z]."""
        n = self.n_interior
        return 4 * n if self.is_3d else 3 * n

    @property
    def dim_x(self) -> int:
        """Total nodes on the (Nx+1)×(Ny+1)×(Nz+1) grid."""
        if self.is_3d:
            return (self.Nx + 1) * (self.Ny + 1) * (self.Nz + 1)
        return (self.Nx + 1) * (self.Ny + 1)

    @property
    def mj(self) -> int:
        """Stride along j (y-direction)."""
        return self.Nx + 1

    @property
    def mk(self) -> int:
        """Stride along k (z-direction)."""
        return (self.Nx + 1) * (self.Ny + 1)

    def copy(self) -> SimulationParameters:
        """Return an independent copy."""
        import copy
        return copy.deepcopy(self)

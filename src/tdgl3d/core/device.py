"""Device — combines parameters, grid indices, and applied field into a ready-to-simulate object."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .parameters import SimulationParameters
from .state import StateVector
from .material import MaterialMap, Trilayer, build_material_map
from ..mesh.indices import GridIndices, construct_indices
from ..physics.applied_field import AppliedField


@dataclass
class Device:
    """A superconducting device ready for simulation.

    Parameters
    ----------
    params : SimulationParameters
        Grid and physics parameters.
    applied_field : AppliedField, optional
        External magnetic field specification.  Defaults to zero field.
    trilayer : Trilayer, optional
        If provided, the device is a multi-layer stack.  ``params.Nz``
        must match ``trilayer.Nz``, or will be set automatically.
    """

    params: SimulationParameters
    applied_field: AppliedField = field(default_factory=AppliedField)
    trilayer: Optional[Trilayer] = None
    _idx: Optional[GridIndices] = field(default=None, repr=False, init=False)
    _material: Optional[MaterialMap] = field(default=None, repr=False, init=False)

    def __post_init__(self) -> None:
        if self.trilayer is not None:
            # Ensure Nz matches
            if self.params.Nz != self.trilayer.Nz:
                object.__setattr__(
                    self.params, "Nz", self.trilayer.Nz
                )
        self._idx = construct_indices(self.params)
        if self.trilayer is not None:
            self._material = build_material_map(
                self.params, self.trilayer, self._idx
            )

    @property
    def idx(self) -> GridIndices:
        if self._idx is None:
            self._idx = construct_indices(self.params)
        return self._idx

    @property
    def material(self) -> Optional[MaterialMap]:
        return self._material

    def initial_state(self) -> StateVector:
        """Return a uniform-superconducting initial state.

        If a trilayer is present, ψ is set to 0 in the insulator layer.
        """
        sv = StateVector.uniform_superconducting(self.params)
        if self._material is not None:
            # Kill ψ in non-SC regions
            sv.psi[:] *= self._material.interior_sc_mask
        return sv

    def rebuild_indices(self) -> None:
        """Rebuild grid indices (call after changing params)."""
        self._idx = construct_indices(self.params)
        if self.trilayer is not None:
            self._material = build_material_map(
                self.params, self.trilayer, self._idx
            )

    def add_hole(
        self,
        vertices: list[tuple[float, float]],
        z_range: Optional[tuple[int, int]] = None,
    ) -> None:
        """Add a polygon-shaped hole with zero-current boundary conditions.

        This method:
        1. Registers the hole boundary in GridIndices (for BC enforcement)
        2. Carves the hole in MaterialMap (marks interior as non-SC)

        Parameters
        ----------
        vertices : list of (x, y) tuples
            Polygon vertices in physical coordinates (ξ units).
            The polygon is automatically closed.
        z_range : (k_min, k_max), optional
            Z-layer extent (grid indices, inclusive).
            If None, hole extends through all z-layers (0, Nz).

        Notes
        -----
        **Physics: Holes vs Insulators**
        
        - **Holes** are geometric voids completely removed from the simulation.
          Zero-current boundary condition φ = 0 is enforced at all hole edges.
          Vortices CANNOT form inside holes (no superconductor = no phase winding).
          
        - **Insulators** (e.g., S/I/S layers) are part of the simulation domain
          with suppressed order parameter but no special BCs on φ.
          Use `Trilayer` for insulator layers, not this method.
        
        **Implementation details:**
        
        - Can be called multiple times to add multiple holes
        - Holes must be added **before** calling solve()
        - Zero-current BCs are enforced at hole boundaries automatically during
          time integration (dφ/dt = 0 at hole boundaries)
        - Hole interior is treated as non-superconducting material (ψ = 0)

        Examples
        --------
        >>> # Square hole from z=0 to z=5
        >>> square = [(5.0, 5.0), (15.0, 5.0), (15.0, 15.0), (5.0, 15.0)]
        >>> device.add_hole(square, z_range=(0, 5))
        >>>
        >>> # Triangle hole through all z-layers
        >>> triangle = [(10.0, 10.0), (20.0, 10.0), (15.0, 20.0)]
        >>> device.add_hole(triangle)
        """
        # Default: hole extends through all z-layers
        if z_range is None:
            z_range = (0, self.params.Nz)

        # Register hole boundary in GridIndices
        self.idx.define_hole_polygon(vertices, z_range, self.params)

        # Carve hole in MaterialMap (if material map exists)
        if self._material is not None:
            self._material.carve_hole_polygon(
                vertices, z_range, self.params, self.idx
            )
        else:
            # Create a uniform material map if one doesn't exist
            # (needed for hole carving to work on uniform devices)
            from ..mesh.indices import GridIndices
            self._material = MaterialMap(
                kappa=np.full(self.params.dim_x, self.params.kappa, dtype=np.float64),
                sc_mask=np.ones(self.params.dim_x, dtype=np.float64),
                interior_sc_mask=np.ones(self.params.n_interior, dtype=np.float64),
            )
            self._material.carve_hole_polygon(
                vertices, z_range, self.params, self.idx
            )

    def __repr__(self) -> str:
        p = self.params
        tri = ""
        if self.trilayer is not None:
            r = self.trilayer.z_ranges()
            tri = (
                f", trilayer=({r['bottom'][1]}SC / "
                f"{r['insulator'][1]-r['insulator'][0]}I / "
                f"{r['top'][1]-r['top'][0]}SC)"
            )
        return (
            f"Device(Nx={p.Nx}, Ny={p.Ny}, Nz={p.Nz}, "
            f"κ={p.kappa}, field=({self.applied_field.Bx}, "
            f"{self.applied_field.By}, {self.applied_field.Bz}){tri})"
        )

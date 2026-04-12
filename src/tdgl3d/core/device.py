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

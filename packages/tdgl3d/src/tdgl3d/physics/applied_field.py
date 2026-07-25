"""Applied magnetic field — replaces ``eval_u.m``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray

from ..core.parameters import SimulationParameters


@dataclass
class AppliedField:
    """Specification of the externally applied magnetic field.

    Constant values ``Bx``, ``By``, ``Bz`` are applied (in units of Φ₀/(2πξ²))
    at the corresponding boundary faces during the *on* interval
    ``[0, t_on_fraction * t_stop]``.  Outside that interval the applied field
    is zero.

    When ``ramp`` is ``True`` the field ramps linearly from 0 to the full
    magnitude over ``[0, ramp_fraction * t_stop]`` and remains constant
    afterwards (until ``t_on_fraction * t_stop`` if ``t_on_fraction < 1``).

    For time-varying fields supply a callable via ``field_func``.
    """

    Bx: float = 0.0
    By: float = 0.0
    Bz: float = 0.0
    t_on_fraction: float = 2.0 / 3.0

    # Linear ramp options
    ramp: bool = False
    ramp_fraction: float = 0.5  # fraction of t_stop over which to ramp

    # Optional callable: f(t, t_stop) -> (Bx, By, Bz)
    field_func: Optional[Callable[[float, float], tuple[float, float, float]]] = None

    def evaluate(self, t: float, t_stop: float) -> tuple[float, float, float]:
        """Return (Bx, By, Bz) at time *t*."""
        if self.field_func is not None:
            return self.field_func(t, t_stop)

        if t < 0 or t > t_stop * self.t_on_fraction:
            return 0.0, 0.0, 0.0

        if self.ramp:
            t_ramp = t_stop * self.ramp_fraction
            scale = min(t / t_ramp, 1.0) if t_ramp > 0 else 1.0
        else:
            scale = 1.0

        return self.Bx * scale, self.By * scale, self.Bz * scale


def build_boundary_field_vectors(
    applied_bx: float,
    applied_by: float,
    applied_bz: float,
    params: SimulationParameters,
    idx,  # GridIndices
) -> tuple[NDArray, NDArray, NDArray]:
    """Construct full-grid sparse boundary field vectors u.Bx, u.By, u.Bz.

    Mimics the second half of ``eval_u.m``: the applied field magnitude is
    placed at the appropriate boundary-face indices.
    """
    N = params.dim_x

    Bx_vec = np.zeros(N, dtype=np.float64)
    By_vec = np.zeros(N, dtype=np.float64)
    Bz_vec = np.zeros(N, dtype=np.float64)

    # Indices where each field component lives on the boundary
    # (union of the two pairs of faces perpendicular to the other axes)
    if applied_bx != 0.0:
        bc_idx = np.concatenate([idx.y_face_lo_inner, idx.z_face_lo_inner, idx.y_face_hi_inner, idx.z_face_hi_inner])
        np.add.at(Bx_vec, bc_idx, applied_bx)

    if applied_by != 0.0:
        bc_idx = np.concatenate([idx.z_face_lo_inner, idx.x_face_lo_inner, idx.z_face_hi_inner, idx.x_face_hi_inner])
        np.add.at(By_vec, bc_idx, applied_by)

    if applied_bz != 0.0:
        bc_idx = np.concatenate([idx.x_face_lo_inner, idx.y_face_lo_inner, idx.x_face_hi_inner, idx.y_face_hi_inner])
        np.add.at(Bz_vec, bc_idx, applied_bz)

    return Bx_vec, By_vec, Bz_vec

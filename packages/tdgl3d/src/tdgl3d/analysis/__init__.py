"""Analysis tools for TDGL simulations — convergence and vortex counting."""

from __future__ import annotations

from .convergence import check_steady_state, compute_convergence_metrics
from .vortex_counting import (
    count_vortices_plaquette,
    count_vortices_polygon,
    count_hole_flux_quanta,
    find_vortex_cores,
)

__all__ = [
    "check_steady_state",
    "compute_convergence_metrics",
    "count_vortices_plaquette",
    "count_vortices_polygon",
    "count_hole_flux_quanta",
    "find_vortex_cores",
]

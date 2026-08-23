"""Analysis tools for TDGL simulations — convergence and vortex counting."""

from __future__ import annotations

from .convergence import check_steady_state, compute_convergence_metrics
from .expulsion import (
    ExpulsionResult,
    expulsion_field,
    first_entry_time,
    fluxoid_history,
    rectangular_contour,
)
from .vortex_counting import (
    count_hole_flux_quanta,
    count_vortices_plaquette,
    count_vortices_polygon,
    find_vortex_cores,
    plaquette_vorticity,
)

__all__ = [
    "check_steady_state",
    "compute_convergence_metrics",
    "count_vortices_plaquette",
    "count_vortices_polygon",
    "plaquette_vorticity",
    "count_hole_flux_quanta",
    "ExpulsionResult",
    "expulsion_field",
    "first_entry_time",
    "fluxoid_history",
    "rectangular_contour",
    "find_vortex_cores",
]

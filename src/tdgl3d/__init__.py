"""
tdgl3d — 3D Time-Dependent Ginzburg-Landau Simulator
=====================================================

Solves the coupled TDGL equations for the superconducting order parameter
and gauge-invariant link variables on a 3D structured Cartesian grid.
"""

from .core.parameters import SimulationParameters
from .core.device import Device
from .core.state import StateVector
from .core.material import Layer, Trilayer, MaterialMap
from .physics.applied_field import AppliedField
from .solvers.runner import solve

# Analysis tools
from .analysis import (
    check_steady_state,
    compute_convergence_metrics,
    count_vortices_plaquette,
    count_vortices_polygon,
    count_hole_flux_quanta,
    find_vortex_cores,
)

__version__ = "0.1.0"

__all__ = [
    "SimulationParameters",
    "Device",
    "StateVector",
    "Layer",
    "Trilayer",
    "MaterialMap",
    "AppliedField",
    "solve",
    # Analysis
    "check_steady_state",
    "compute_convergence_metrics",
    "count_vortices_plaquette",
    "count_vortices_polygon",
    "count_hole_flux_quanta",
    "find_vortex_cores",
]


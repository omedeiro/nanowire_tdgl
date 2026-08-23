"""
tdgl3d — 3D Time-Dependent Ginzburg-Landau Simulator
=====================================================

Solves the coupled TDGL equations for the superconducting order parameter
and gauge-invariant link variables on a 3D structured Cartesian grid.
"""

# Analysis tools
from .analysis import (
    ExpulsionResult,
    check_steady_state,
    compute_convergence_metrics,
    count_hole_flux_quanta,
    count_vortices_plaquette,
    count_vortices_polygon,
    expulsion_field,
    find_vortex_cores,
    fluxoid_history,
    plaquette_vorticity,
    rectangular_contour,
)
from .core.device import Device
from .core.material import Layer, MaterialMap, Trilayer
from .core.parameters import SimulationParameters
from .core.solution import Solution
from .core.state import StateVector
from .core.units import PHI0_WB, GLUnits
from .physics.analytic import (
    gl_wall_interface_value,
    gl_wall_profile,
    interior_node_positions,
    london_slab_1d,
    london_square_2d,
    plaquette_positions,
)
from .physics.applied_field import AppliedField
from .physics.free_energy import gl_free_energy, gl_free_energy_terms
from .solvers.runner import solve

__version__ = "1.0.0"

__all__ = [
    "SimulationParameters",
    "Device",
    "StateVector",
    "Solution",
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
    "plaquette_vorticity",
    "count_hole_flux_quanta",
    "GLUnits",
    "gl_wall_interface_value",
    "gl_wall_profile",
    "interior_node_positions",
    "london_slab_1d",
    "london_square_2d",
    "plaquette_positions",
    "PHI0_WB",
    "ExpulsionResult",
    "expulsion_field",
    "fluxoid_history",
    "rectangular_contour",
    "gl_free_energy",
    "gl_free_energy_terms",
    "find_vortex_cores",
]


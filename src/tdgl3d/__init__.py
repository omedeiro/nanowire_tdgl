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
]

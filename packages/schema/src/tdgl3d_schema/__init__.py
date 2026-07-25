"""tdgl3d Project Object Model (POM) v0.

Pydantic models defining the shared contract between the studio frontend,
the solver job server, and AI assistant tool calls.
"""

from __future__ import annotations

from tdgl3d_schema.pom import (
    POM_VERSION,
    AppliedFieldSpec,
    DeviceSpec,
    GridSpec,
    HoleSpec,
    JobStatus,
    LayerSpec,
    Project,
    ResultMeta,
    SimulationSpec,
    SolverSettings,
    TrilayerSpec,
)

__all__ = [
    "POM_VERSION",
    "AppliedFieldSpec",
    "DeviceSpec",
    "GridSpec",
    "HoleSpec",
    "JobStatus",
    "LayerSpec",
    "Project",
    "ResultMeta",
    "SimulationSpec",
    "SolverSettings",
    "TrilayerSpec",
]

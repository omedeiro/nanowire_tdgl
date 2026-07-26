"""Project Object Model (POM) v0.

Every entity is an editable object with a stable ``id`` so the UI object tree,
undo history, and AI tool calls can address objects individually.

Design notes
------------
- Mirrors the public API of :mod:`tdgl3d` (``SimulationParameters``, ``Device``,
  ``AppliedField``, ``Layer``/``Trilayer``, ``Device.add_hole``, ``solve``).
- Forward-compatible: unknown extra fields are rejected (strict contract);
  schema evolution happens through ``POM_VERSION`` bumps.
- Callable applied fields (``field_func``) are intentionally *not* representable
  in the POM; time-varying fields will be added as declarative waveforms later.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Literal, Optional, Tuple
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

POM_VERSION = "0.1.0"


def _new_id() -> str:
    return uuid4().hex


class POMObject(BaseModel):
    """Base class for all addressable POM entities."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=_new_id)
    name: str = ""


# --------------------------------------------------------------------------- #
# Geometry / grid
# --------------------------------------------------------------------------- #
class GridSpec(BaseModel):
    """Structured grid: (Nx+1) x (Ny+1) x (Nz+1) nodes. Nz=1 => quasi-2D."""

    model_config = ConfigDict(extra="forbid")

    Nx: int = Field(10, ge=2)
    Ny: int = Field(10, ge=2)
    Nz: int = Field(1, ge=1)
    hx: float = Field(1.0, gt=0)
    hy: float = Field(1.0, gt=0)
    hz: float = Field(1.0, gt=0)


class HoleSpec(POMObject):
    """Polygonal hole (geometric void, zero-current BC at edges)."""

    vertices: List[Tuple[float, float]] = Field(min_length=3)
    z_range: Optional[Tuple[int, int]] = None


# --------------------------------------------------------------------------- #
# Materials
# --------------------------------------------------------------------------- #
class LayerSpec(POMObject):
    thickness_z: int = Field(ge=1)
    kappa: float = Field(gt=0)
    is_superconductor: bool = True


class TrilayerSpec(POMObject):
    """S/I/S stack along z (bottom SC -- insulator -- top SC)."""

    bottom: LayerSpec
    insulator: LayerSpec
    top: LayerSpec

    @model_validator(mode="after")
    def _check_insulator(self) -> "TrilayerSpec":
        if self.insulator.is_superconductor:
            raise ValueError("insulator layer must have is_superconductor=False")
        return self


# --------------------------------------------------------------------------- #
# Applied field
# --------------------------------------------------------------------------- #
class AppliedFieldSpec(POMObject):
    """Constant or linearly ramped applied field (units of Phi0 / 2*pi*xi^2)."""

    Bx: float = 0.0
    By: float = 0.0
    Bz: float = 0.0
    t_on_fraction: float = Field(2.0 / 3.0, ge=0)
    ramp: bool = False
    ramp_fraction: float = Field(0.5, gt=0, le=1)


# --------------------------------------------------------------------------- #
# Device
# --------------------------------------------------------------------------- #
class DeviceSpec(POMObject):
    grid: GridSpec = Field(default_factory=GridSpec)
    kappa: float = Field(5.0, gt=0, description="Uniform GL parameter (ignored if trilayer set)")
    trilayer: Optional[TrilayerSpec] = None
    applied_field: AppliedFieldSpec = Field(default_factory=AppliedFieldSpec)
    holes: List[HoleSpec] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Solver / simulation
# --------------------------------------------------------------------------- #
class SolverSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["euler", "trapezoidal"] = "trapezoidal"
    t_start: float = 0.0
    t_stop: float = Field(10.0, gt=0)
    dt: float = Field(0.05, gt=0)
    save_every: int = Field(1, ge=1)
    # Newton / GCR (trapezoidal only)
    newton_tol_f: float = 1e-3
    newton_tol_dx: float = 1e-3
    newton_max_iter: int = Field(20, ge=1)
    tol_gcr: float = 1e-4
    eps_mf: float = 1e-4
    adaptive: bool = True
    # Initial-state noise (symmetry breaking)
    initial_noise_amplitude: float = Field(
        0.01, ge=0,
        description="Amplitude of complex Gaussian noise on ψ in SC regions. "
                    "Set to 0 for a perfectly uniform state.",
    )
    initial_noise_seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility. None = non-deterministic.",
    )

    @model_validator(mode="after")
    def _check_window(self) -> "SolverSettings":
        if self.t_stop <= self.t_start:
            raise ValueError("t_stop must be greater than t_start")
        return self


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SimulationSpec(POMObject):
    """A runnable simulation: a device + solver settings."""

    device_id: str
    solver: SolverSettings = Field(default_factory=SolverSettings)


class ResultMeta(POMObject):
    """Metadata for a completed run; heavy data stays in artifact files."""

    simulation_id: str
    job_id: str
    status: JobStatus = JobStatus.PENDING
    n_frames: int = 0
    t_final: Optional[float] = None
    artifact_path: Optional[str] = None
    error: Optional[str] = None


# --------------------------------------------------------------------------- #
# Project (root document)
# --------------------------------------------------------------------------- #
class Project(POMObject):
    """Root POM document. All entities live here as editable objects."""

    pom_version: str = POM_VERSION
    description: str = ""
    devices: List[DeviceSpec] = Field(default_factory=list)
    simulations: List[SimulationSpec] = Field(default_factory=list)
    results: List[ResultMeta] = Field(default_factory=list)

    def get_device(self, device_id: str) -> DeviceSpec:
        for d in self.devices:
            if d.id == device_id:
                return d
        raise KeyError(f"No device with id {device_id!r}")

    def get_simulation(self, simulation_id: str) -> SimulationSpec:
        for s in self.simulations:
            if s.id == simulation_id:
                return s
        raise KeyError(f"No simulation with id {simulation_id!r}")

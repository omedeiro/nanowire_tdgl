from __future__ import annotations

import pytest
from pydantic import ValidationError

from tdgl3d_schema import (
    AppliedFieldSpec,
    DeviceSpec,
    GridSpec,
    HoleSpec,
    LayerSpec,
    Project,
    SimulationSpec,
    SolverSettings,
    TrilayerSpec,
)


def make_trilayer() -> TrilayerSpec:
    return TrilayerSpec(
        bottom=LayerSpec(thickness_z=2, kappa=2.0),
        insulator=LayerSpec(thickness_z=1, kappa=2.0, is_superconductor=False),
        top=LayerSpec(thickness_z=2, kappa=2.0),
    )


def test_project_roundtrip():
    device = DeviceSpec(
        name="sis-square",
        grid=GridSpec(Nx=20, Ny=20, Nz=5),
        trilayer=make_trilayer(),
        applied_field=AppliedFieldSpec(Bz=0.5, ramp=True),
        holes=[HoleSpec(vertices=[(5, 5), (10, 5), (10, 10), (5, 10)])],
    )
    project = Project(name="test", devices=[device])
    project.simulations.append(SimulationSpec(device_id=device.id))

    dumped = project.model_dump_json()
    restored = Project.model_validate_json(dumped)
    assert restored == project
    assert restored.get_device(device.id).name == "sis-square"


def test_insulator_must_not_be_superconductor():
    with pytest.raises(ValidationError):
        TrilayerSpec(
            bottom=LayerSpec(thickness_z=2, kappa=2.0),
            insulator=LayerSpec(thickness_z=1, kappa=2.0, is_superconductor=True),
            top=LayerSpec(thickness_z=2, kappa=2.0),
        )


def test_grid_bounds():
    with pytest.raises(ValidationError):
        GridSpec(Nx=1)
    with pytest.raises(ValidationError):
        GridSpec(hx=0)


def test_solver_time_window():
    with pytest.raises(ValidationError):
        SolverSettings(t_start=5.0, t_stop=5.0)


def test_hole_needs_polygon():
    with pytest.raises(ValidationError):
        HoleSpec(vertices=[(0, 0), (1, 1)])


def test_extra_fields_rejected():
    with pytest.raises(ValidationError):
        DeviceSpec(bogus_field=1)


def test_unique_ids():
    a, b = DeviceSpec(), DeviceSpec()
    assert a.id != b.id


def test_json_schema_exports():
    schema = Project.model_json_schema()
    assert "properties" in schema
    assert "devices" in schema["properties"]

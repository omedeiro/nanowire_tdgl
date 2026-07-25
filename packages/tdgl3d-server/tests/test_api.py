from __future__ import annotations

import importlib
import os
import time

import pytest
from fastapi.testclient import TestClient

from tdgl3d_schema import DeviceSpec, GridSpec, Project, SimulationSpec, SolverSettings


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("TDGL3D_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("TDGL3D_API_TOKEN", "test-token")
    import tdgl3d_server.app as app_module

    app_module = importlib.reload(app_module)
    with TestClient(app_module.app) as c:
        c.headers.update({"Authorization": "Bearer test-token"})
        yield c
    app_module.runner.shutdown()


def make_project() -> Project:
    # Tiny, CFL-safe run: dt < h^2/(4*kappa^2) = 1/16 for kappa=2.
    device = DeviceSpec(name="tiny", grid=GridSpec(Nx=4, Ny=4, Nz=1), kappa=2.0)
    sim = SimulationSpec(
        device_id=device.id,
        solver=SolverSettings(method="euler", t_stop=0.5, dt=0.05),
    )
    return Project(name="test-project", devices=[device], simulations=[sim])


def test_health_is_open(client):
    del client.headers["Authorization"]
    assert client.get("/health").json() == {"status": "ok"}


def test_auth_required(client):
    del client.headers["Authorization"]
    assert client.get("/projects").status_code == 401


def test_project_crud(client):
    project = make_project()
    r = client.post("/projects", json=project.model_dump(mode="json"))
    assert r.status_code == 201
    assert client.get(f"/projects/{project.id}").json()["name"] == "test-project"
    assert len(client.get("/projects").json()) == 1

    project.description = "updated"
    r = client.put(f"/projects/{project.id}", json=project.model_dump(mode="json"))
    assert r.json()["description"] == "updated"

    assert client.delete(f"/projects/{project.id}").status_code == 204
    assert client.get(f"/projects/{project.id}").status_code == 404


def test_invalid_project_rejected(client):
    r = client.post("/projects", json={"bogus": True})
    assert r.status_code == 422


def test_job_lifecycle(client):
    project = make_project()
    client.post("/projects", json=project.model_dump(mode="json"))

    r = client.post(
        f"/projects/{project.id}/jobs",
        json={"simulation_id": project.simulations[0].id},
    )
    assert r.status_code == 202
    job_id = r.json()["id"]

    deadline = time.time() + 60
    status = None
    while time.time() < deadline:
        status = client.get(f"/jobs/{job_id}").json()["status"]
        if status in ("completed", "failed"):
            break
        time.sleep(0.25)
    assert status == "completed", client.get(f"/jobs/{job_id}").json()["error"]

    r = client.get(f"/jobs/{job_id}/result")
    assert r.status_code == 200
    assert len(r.content) > 0


def test_job_unknown_simulation(client):
    project = make_project()
    client.post("/projects", json=project.model_dump(mode="json"))
    r = client.post(f"/projects/{project.id}/jobs", json={"simulation_id": "nope"})
    assert r.status_code == 404

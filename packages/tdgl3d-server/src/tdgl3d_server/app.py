"""FastAPI application exposing the tdgl3d solver as a job service."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from tdgl3d_schema import Project

from tdgl3d_server.jobs import JobRunner
from tdgl3d_server.store import Store

DATA_DIR = Path(os.environ.get("TDGL3D_DATA_DIR", "~/.tdgl3d-server")).expanduser()
API_TOKEN = os.environ.get("TDGL3D_API_TOKEN")

app = FastAPI(title="tdgl3d-server", version="0.1.0")
store = Store(DATA_DIR / "server.sqlite3")
runner = JobRunner(store, DATA_DIR / "artifacts")


def require_auth(request: Request) -> None:
    if not API_TOKEN:
        return
    header = request.headers.get("authorization", "")
    if header != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token")


class JobRequest(BaseModel):
    simulation_id: str


class JobInfo(BaseModel):
    id: str
    project_id: str
    simulation_id: str
    status: str
    error: Optional[str] = None
    artifact_path: Optional[str] = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# -- projects ----------------------------------------------------------------
@app.post("/projects", dependencies=[Depends(require_auth)], status_code=201)
def create_project(project: Project) -> Project:
    store.put_project(project.id, project.model_dump(mode="json"))
    return project


@app.get("/projects", dependencies=[Depends(require_auth)])
def list_projects() -> List[Project]:
    return [Project.model_validate(doc) for doc in store.list_projects()]


@app.get("/projects/{project_id}", dependencies=[Depends(require_auth)])
def get_project(project_id: str) -> Project:
    doc = store.get_project(project_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return Project.model_validate(doc)


@app.put("/projects/{project_id}", dependencies=[Depends(require_auth)])
def update_project(project_id: str, project: Project) -> Project:
    if store.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found")
    if project.id != project_id:
        raise HTTPException(status_code=400, detail="Project id mismatch")
    store.put_project(project_id, project.model_dump(mode="json"))
    return project


@app.delete("/projects/{project_id}", dependencies=[Depends(require_auth)], status_code=204)
def delete_project(project_id: str) -> None:
    if not store.delete_project(project_id):
        raise HTTPException(status_code=404, detail="Project not found")


# -- jobs ----------------------------------------------------------------------
@app.post("/projects/{project_id}/jobs", dependencies=[Depends(require_auth)], status_code=202)
def launch_job(project_id: str, body: JobRequest) -> JobInfo:
    doc = store.get_project(project_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Project not found")
    project = Project.model_validate(doc)
    try:
        sim = project.get_simulation(body.simulation_id)
        project.get_device(sim.device_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    job_id = store.create_job(project_id, body.simulation_id)
    runner.submit(job_id, project, body.simulation_id)
    return JobInfo(**store.get_job(job_id))


@app.get("/jobs/{job_id}", dependencies=[Depends(require_auth)])
def get_job(job_id: str) -> JobInfo:
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobInfo(**job)


@app.get("/jobs/{job_id}/events", dependencies=[Depends(require_auth)])
async def job_events(job_id: str) -> StreamingResponse:
    if store.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")

    async def stream():
        last_status = None
        while True:
            job = store.get_job(job_id)
            if job is None:
                break
            if job["status"] != last_status:
                last_status = job["status"]
                payload = {"status": job["status"], "error": job["error"]}
                yield f"data: {json.dumps(payload)}\n\n"
            if last_status in ("completed", "failed", "cancelled"):
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/jobs/{job_id}/result", dependencies=[Depends(require_auth)])
def job_result(job_id: str) -> FileResponse:
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "completed" or not job["artifact_path"]:
        raise HTTPException(status_code=409, detail=f"Job status is {job['status']}")
    return FileResponse(job["artifact_path"], filename=f"{job_id}.h5")

"""Background job execution.

Jobs run in a small worker pool inside the server process. NumPy/SciPy release
the GIL for the heavy kernels, and self-hosted deployments run one solver at a
time by default. A process pool / external queue can replace this later without
changing the HTTP contract.
"""

from __future__ import annotations

import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tdgl3d import solve
from tdgl3d_schema import Project

from tdgl3d_server.build import build_device, solve_kwargs
from tdgl3d_server.store import Store


class JobRunner:
    def __init__(self, store: Store, artifact_dir: Path, max_workers: int = 1) -> None:
        self.store = store
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self._pool = ThreadPoolExecutor(max_workers=max_workers)

    def submit(self, job_id: str, project: Project, simulation_id: str) -> None:
        self._pool.submit(self._run, job_id, project, simulation_id)

    def _run(self, job_id: str, project: Project, simulation_id: str) -> None:
        try:
            self.store.update_job(job_id, "running")
            sim = project.get_simulation(simulation_id)
            device = build_device(project.get_device(sim.device_id))
            solution = solve(device, **solve_kwargs(sim.solver))
            artifact = self.artifact_dir / f"{job_id}.h5"
            solution.save(str(artifact))
            self.store.update_job(job_id, "completed", artifact_path=str(artifact))
        except Exception:  # noqa: BLE001 - report any failure on the job record
            self.store.update_job(job_id, "failed", error=traceback.format_exc())

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

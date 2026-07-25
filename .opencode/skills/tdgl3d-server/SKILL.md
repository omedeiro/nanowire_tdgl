---
name: tdgl3d-server
description: >
  Work on the FastAPI server in packages/tdgl3d-server. Use when editing API
  endpoints, job processing, SQLite store, build.py POM translation, or
  Cloudflare Tunnel deployment.
---

# tdgl3d-server

FastAPI job service exposing the solver over HTTP. Self-hosted behind a
Cloudflare Tunnel in production, plain-local during development.

## Rules

- HTTP/persistence concerns live here; physics lives in `packages/tdgl3d`.
- `build.py` is the **only** place that translates POM specs to tdgl3d
  objects. Keep it in sync with `tdgl3d_schema.pom`.
- Persistence is SQLite (`store.py`) by design — self-host friendly. Don't add
  Redis/Celery without an explicit decision.
- Auth is a bearer token via `TDGL3D_API_TOKEN` (unset = auth disabled, dev
  only). Cloudflare Access fronts the tunnel in production.
- Tests must stay fast: tiny grids, `method="euler"`, CFL-safe dt
  (`dt < h^2 / (4*kappa^2)`).

## API Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness check |
| POST | `/projects` | Create a project (POM document) |
| GET | `/projects` | List projects |
| GET | `/projects/{id}` | Fetch project |
| PUT | `/projects/{id}` | Replace project (full POM validation) |
| DELETE | `/projects/{id}` | Delete project |
| POST | `/projects/{id}/jobs` | Launch a simulation (`{"simulation_id": ...}`) |
| GET | `/jobs/{id}` | Job status |
| GET | `/jobs/{id}/events` | SSE stream of status updates |
| GET | `/jobs/{id}/result` | Download the HDF5 result artifact |

All endpoints except `/health` require `Authorization: Bearer $TDGL3D_API_TOKEN`
when the token is configured.

## Run Locally

```bash
pip install -e "packages/tdgl3d-server[dev]"
export TDGL3D_API_TOKEN=dev-token          # optional; omit to disable auth
python3 -m uvicorn tdgl3d_server.app:app --port 8787
```

## Cloudflare Tunnel Deployment

```bash
brew install cloudflared
cloudflared tunnel login
cloudflared tunnel create tdgl3d-solver
# route e.g. solver.example.com -> localhost:8787 in ~/.cloudflared/config.yml
cloudflared tunnel route dns tdgl3d-solver solver.example.com
cloudflared tunnel run tdgl3d-solver
```

Put Cloudflare Access in front of the hostname for identity-based auth in
addition to the bearer token.

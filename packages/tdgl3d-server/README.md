# tdgl3d-server

FastAPI job service exposing the tdgl3d solver over HTTP. Designed to run
self-hosted behind a Cloudflare Tunnel (see "Deployment" below), or plain-local
during development.

## Run locally

```bash
pip install -e "packages/tdgl3d-server[dev]"
export TDGL3D_API_TOKEN=dev-token          # optional; omit to disable auth
python3 -m uvicorn tdgl3d_server.app:app --port 8787
```

## API

All endpoints (except `/health`) require `Authorization: Bearer $TDGL3D_API_TOKEN`
when the token is configured.

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

## Deployment (Cloudflare Tunnel)

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

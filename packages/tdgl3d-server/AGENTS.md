# Agent Instructions — packages/tdgl3d-server

FastAPI job service exposing the solver over HTTP. See README.md for the API
table and Cloudflare Tunnel deployment.

Rules:
- HTTP/persistence concerns live here; physics lives in `packages/tdgl3d`.
- `build.py` is the only place that translates POM specs → tdgl3d objects.
  Keep it in sync with `tdgl3d_schema.pom`.
- Persistence is SQLite (`store.py`) by design — self-host friendly. Don't add
  Redis/Celery without an explicit decision.
- Auth is a bearer token via `TDGL3D_API_TOKEN` (unset ⇒ auth disabled, dev
  only). Cloudflare Access fronts the tunnel in production.
- Tests must stay fast: tiny grids, `method="euler"`, CFL-safe dt
  (dt < h²/(4κ²)).

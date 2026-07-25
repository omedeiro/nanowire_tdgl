# Agent Instructions — tdgl3d platform (monorepo root)

AI-native scientific computing platform centered on a 3D TDGL superconductor
solver. **Read `docs/ROADMAP.md` for the full vision, architecture, and phase
plan.**

## Repository map

| Path | What it is | Details |
|---|---|---|
| `packages/tdgl3d/` | Python TDGL solver (physics core) | `packages/tdgl3d/AGENTS.md` |
| `packages/schema/` | POM (Project Object Model) Pydantic schemas | shared contract for UI/server/AI tools |
| `packages/tdgl3d-server/` | FastAPI job service wrapping the solver | `packages/tdgl3d-server/README.md` |
| `apps/` | Frontend apps (Next.js studio — Phase 2, not yet created) | — |
| `agents/` | AI agent tool definitions and prompts (Phase 3/4) | — |
| `docs/` | Roadmap, design notes | `docs/ROADMAP.md`, `docs/notes/` |

## Development workflow (applies to humans AND AI agents)

1. Never commit directly to `main`. Create a feature branch.
2. Use **Conventional Commits** (`feat:`, `fix:`, `refactor:`, `docs:`, `test:`,
   `ci:`, `chore:`; scope by package, e.g. `feat(server): ...`). Breaking
   changes: `!` suffix. release-please derives versions/changelogs from these.
3. Every change ships with tests and doc updates where relevant.
4. Open a PR; CI (ruff + pytest on 3.10/3.12) must pass before merge.
5. Versioning/CHANGELOGs are automated by release-please — never hand-edit
   version numbers or CHANGELOG files.
6. Keep binary artifacts (GIF/PNG/HDF5/logs) out of git; `.gitignore` covers
   them. Git history was rewritten 2026-07 to purge old artifacts.

## Essential commands

```bash
# Install everything (from repo root)
pip install -e "packages/tdgl3d[dev]" -e "packages/schema[dev]" -e "packages/tdgl3d-server[dev]"

# Test all packages
python3 -m pytest packages/tdgl3d/tests packages/schema/tests packages/tdgl3d-server/tests -q

# Lint (config in root ruff.toml)
ruff check packages/

# Run the solver API locally
python3 -m uvicorn tdgl3d_server.app:app --port 8787
```

**Critical:** Always use `python3`, never bare `python` (machine has no alias).

## Architecture invariants

- **POM is the contract.** UI, server, and AI tools all operate on
  `tdgl3d_schema.Project`. New solver features must be reflected in the POM
  (`packages/schema/src/tdgl3d_schema/pom.py`) and in the server's
  `build.py` translation layer.
- The solver stays framework-free (NumPy/SciPy only); FastAPI/HTTP concerns
  live only in `tdgl3d-server`.
- The frontend (future) deploys to Cloudflare Workers; the solver server is
  self-hosted behind a Cloudflare Tunnel — never assume solver code can run in
  a Worker.

## Known WIP

- 3 tests in `packages/tdgl3d/tests` are marked `xfail` (hole-BC / flux
  trapping work; see `docs/notes/HOLE_BC_STATUS.md`).
- Periodic BCs are defined in `SimulationParameters` but not implemented.

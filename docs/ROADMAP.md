# Roadmap — AI-Native Scientific Computing Platform

**Vision:** Evolve this repository from a 3D TDGL solver into a modular, scalable,
AI-native scientific computing platform, plus an AI-driven development environment
that lets the application continuously evolve through natural language.

## High-level architecture

Two interconnected systems:

```
                    AI Development Agent
                            │
         Development Tools (Git, CI/CD, PRs, Tests, Docs)
                            │
                     Application Source Code
                            │
                    React / Next.js Frontend
                            │
                  Interactive Scientific Workspace
                            │
                    Project Object Model (POM)
                            │
         Simulation Engine / Visualization / Analysis
                            ▲
                            │
                 AI Scientific Assistant (Tool Calling)
```

### Key decisions (locked in)

| Decision | Choice |
|---|---|
| Solver compute | Self-hosted FastAPI behind a **Cloudflare Tunnel** (Cloudflare Access in front) |
| Dev agent execution | **GitHub Actions** — web console triggers `workflow_dispatch`; headless agent opens PRs |
| Frontend hosting | Cloudflare Workers (static assets); note: Pages is deprecated in favor of Workers |
| Versioning | Conventional Commits + **release-please** (monorepo manifest mode), semver, automated CHANGELOG |
| Repo shape | Monorepo (pnpm workspaces + turbo added when JS apps land) |
| Git history | Rewritten with `git-filter-repo` to purge large binary artifacts (~60 MB) |

### Architectural constraints (be honest about these)

1. The Python solver **cannot** run on Cloudflare Workers/Pages — heavy NumPy/SciPy
   sparse numerics need real compute. Frontend on Cloudflare; solver jobs on the
   self-hosted server via Tunnel.
2. The AI dev agent needs git + Python + the test suite; it executes in GitHub
   Actions, not in the browser. The web console is a control plane only.
3. Guardrails: the agent never pushes to `main`; CI must pass; humans approve PRs.
   Every AI change follows the human workflow: branch → conventional commits →
   tests → PR → release-please handles version + changelog.

## Target monorepo layout

```
nanowire_tdgl/
├── packages/
│   ├── tdgl3d/            # Python solver (existing package, API unchanged)
│   ├── tdgl3d-server/     # FastAPI job service: submit/monitor/fetch results
│   └── schema/            # POM JSON Schemas → Pydantic (and later TS) models
├── apps/
│   └── studio/            # Next.js scientific workspace (Phase 2)
├── agents/                # agent tool definitions, prompts, guardrails
├── docs/                  # roadmap, notes, (later) docs site
├── .github/workflows/     # ci.yml, release-please.yml, (later) agent.yml, deploy.yml
└── AGENTS.md              # root agent instructions + per-package AGENTS.md
```

## Project Object Model (POM)

A versioned JSON document — geometry, materials, mesh, boundary conditions,
solver settings, simulations, results, visualizations — defined once in
`packages/schema` and shared by the UI, the solver server, and AI tool calls.
Everything is an editable object; the POM is the single contract all AI tools
and UI components operate on.

## Phases

### Phase 0 — Repo hygiene & engineering foundation ✅ (this milestone)
- Purge binary artifacts (GIFs, HDF5, logs, figures) from tree and history
- Monorepo restructure; `tdgl3d` continues to work via `pip install -e packages/tdgl3d`
- CI: GitHub Actions (ruff + pytest matrix + coverage), branch protection on main
- release-please + Conventional Commits for semver/changelog
- AGENTS.md hierarchy (root workflow rules + per-package physics notes)

### Phase 1 — Solver as a service ✅ (this milestone)
- `packages/schema`: POM v0 (Project → Device, Trilayer/Materials, AppliedField,
  SolverSettings, Simulation job spec, Result metadata)
- `packages/tdgl3d-server`: FastAPI — project CRUD, `POST /jobs` (runs `solve()`
  in a worker process), SSE progress, results download. SQLite job table; no
  Redis/Celery (self-host friendly). Bearer-token auth.
- Cloudflare Tunnel documented (cloudflared → e.g. solver.yourdomain.com behind
  Cloudflare Access); server also runs plain-local.

### Phase 2 — Scientific workspace (Next.js studio)
- Next.js + TypeScript + Tailwind on Cloudflare Workers
- POM state store (Zustand), object tree + inspector (editable parameters,
  materials, BCs)
- 3D viz: react-three-fiber (geometry/mesh/isosurfaces); WebGL2/instancing for
  large fields; 2D slice views with colormaps
- Import/export: POM JSON, HDF5 download, VTK/CSV
- Plugin-style module registry from day one (physics modules, visualizers
  registered, not hardcoded)

### Phase 3 — AI scientific assistant (in-app)
- Cloudflare Agents SDK worker; tools map 1:1 to POM API endpoints
  (create_geometry, assign_material, configure_solver, launch_job,
  analyze_results, produce_visualization, export_project, …)
- Chat panel in studio; every AI mutation passes the same POM validation/undo
  history as manual edits; app stays authoritative over state and rendering

### Phase 4 — AI dev console
- Web UI for natural-language dev requests (add features, refactor, redesign UI,
  new visualizations, docs, agent-instruction updates, bug fixes, tests, PRs,
  deploys)
- Request → durable queue → `workflow_dispatch` → headless agent in GitHub
  Actions → feature branch + PR (with tests + docs); status streamed to console
- Preview deployments per PR; merge ⇒ version bump, changelog, deploy

### Phase 5+ — Scale-out
- More physics modules; optimization / parameter-sweep orchestration
- Docs site with automated updates; agent self-improvement of AGENTS.md via PRs

## Operational notes

- History was rewritten (2026-07); old clones/SHAs are invalid — re-clone.
- Critical dev commands live in the root AGENTS.md; always `python3`, never `python`.

# Agent Instructions — tdgl3d platform (monorepo root)

AI-native scientific computing platform centered on a 3D TDGL superconductor
solver. **Read `docs/ROADMAP.md` for the full vision, architecture, and phase
plan.**

## Repository map

| Path | What it is | Details |
|---|---|---|
| `packages/tdgl3d/` | Python TDGL solver (physics core) | Load `tdgl3d-solver` skill |
| `packages/schema/` | POM (Project Object Model) Pydantic schemas | Load `schema-pom` skill |
| `packages/tdgl3d-server/` | FastAPI job service wrapping the solver | Load `tdgl3d-server` skill |
| `apps/` | Frontend apps (Next.js studio — Phase 2, not yet created) | — |
| `agents/` | AI agent tool definitions and prompts (Phase 3/4) | — |
| `docs/` | Roadmap, design notes | `docs/ROADMAP.md`, `docs/notes/` |
| `docs/notes/PHYSICS_CONVENTIONS.md` | Units, gauge convention, exact discrete identities, index ordering | **Read before touching the solver core** |
| `docs/notes/VERSIONING.md` | SemVer policy, Conventional Commit → Keep a Changelog mapping | Read before committing a change that moves results |

## Development workflow (applies to humans AND AI agents)

1. Never commit directly to `main`. Create a feature branch.
2. Use **Conventional Commits** (`feat:`, `fix:`, `refactor:`, `docs:`, `test:`,
   `ci:`, `chore:`; scope by package, e.g. `feat(server): ...`). Breaking
   changes: `!` suffix. release-please derives versions/changelogs from these,
   so the subject line *is* the changelog entry — see
   `docs/notes/VERSIONING.md` for the type → Keep a Changelog section mapping
   and for what counts as breaking in a package whose output is numbers (a
   physically correct result changing to a *different* physically correct one is
   breaking; correcting a result that was wrong is not).
3. Every change ships with tests and doc updates where relevant.
4. Open a PR; CI (ruff + pytest on 3.10/3.12) must pass before merge.
5. Versioning follows **SemVer** and changelogs follow **Keep a Changelog**,
   both automated by release-please — never hand-edit version numbers,
   `.release-please-manifest.json`, or any package `CHANGELOG.md`. The root
   `CHANGELOG.md` is an index, not an entry list.
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

# Run a simulation / drive the server end-to-end (diagnostics + PNG)
python3 .claude/skills/run-nanowire-tdgl/driver.py solve
python3 .claude/skills/run-nanowire-tdgl/driver.py api --start
```

To launch and drive either surface, load the `run-nanowire-tdgl` skill
(`.claude/skills/run-nanowire-tdgl/`) — it documents the driver above, the
verified setup, and the traps (CFL, `t_on_fraction`, the two HDF5 loaders).

**Critical:** Always use `python3`, never bare `python` (machine has no alias).

## Physics verification

Before changing anything in `packages/tdgl3d/src/tdgl3d/{operators,physics,mesh}/`,
read `docs/notes/PHYSICS_CONVENTIONS.md`. It records the sign and index
conventions the solver depends on, and which test fails when each one is broken.

The physics is verified by five suites in `packages/tdgl3d/tests/`:
`test_verification_{gauge,conservation,symmetry,analytic,vortex,expulsion}.py`,
plus `test_physics_validation.py` for heterostructures. They assert through the
`check_*` helpers on the `phys_log` fixture, which record measured value,
expected value and tolerance into `logs/test_*.json` for
`docs/generate_test_report.py` (every check) and `docs/generate_error_table.py`
(only the checks anchored to a known solution, with the fraction of each error
budget used, and the known solutions still uncovered). The error table's
reference list is curated in that script and it exits non-zero when a test or
check label it names no longer exists, so regenerate it after renaming either.

Two rules for adding checks there:

1. **State the expected value and the tolerance up front.** A tolerance computed
   from the measurement it is checking (`tol = max(0.01, 10 * observed)`) can
   never fail and verifies nothing.
2. **Make the test non-vacuous.** If a quantity can be trivially zero — no
   vortices nucleated, no field present, an empty index array — assert the scale
   as well as the deviation.

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
- `Layer.kappa` does **not** reach the Maxwell term: the coefficient
  multiplying `κ²∇×∇×A` is the field energy `B²/2μ₀`, so it is uniform and an
  oxide declared `kappa=0.0` still transmits the field. `Layer.magnetic_kappa`
  is the opt-in override for a genuinely varying coefficient — see
  "Heterostructures" in `docs/notes/PHYSICS_CONVENTIONS.md`.
- Superconducting layers thinner than ~2 ξ are fully pair-broken by an adjacent
  insulator (|ψ| ~ 1e-4) while still producing plausible-looking output. Check
  `max |ψ|` before trusting anything phase-derived.
- Devices given in SI go through `tdgl3d.GLUnits`, which needs ξ **at the
  temperature of interest** — the same geometry is a different simulation at a
  different temperature. See "SI units" in `docs/notes/PHYSICS_CONVENTIONS.md`.
- A noiseless symmetric device relaxes to an exact fixed point, so a metastable
  branch can only be broken by round-off. Seed a perturbation when measuring an
  entry threshold, and check the answer against a much smaller one.
- The ghost-ring corner plaquette at `(0, 0)` carries zero applied flux; it does
  not enter the dynamics (see `docs/notes/PHYSICS_CONVENTIONS.md`).

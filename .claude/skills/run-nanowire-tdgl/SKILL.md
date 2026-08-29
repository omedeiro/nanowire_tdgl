---
name: run-nanowire-tdgl
description: Build, run, and drive the tdgl3d platform — the 3D TDGL superconductor solver and its FastAPI job service. Use when asked to run, start, launch, or smoke-test the solver or the server, run a simulation, plot or screenshot simulation output, exercise the HTTP API end to end, or confirm that a solver change still produces physical results.
---

The repo holds three installable Python packages — `packages/tdgl3d` (the
solver), `packages/schema` (the POM Pydantic contract), and
`packages/tdgl3d-server` (a FastAPI job service that wraps the solver). There is
no GUI and no frontend yet; the two things you can actually *run* are a
simulation and the HTTP server.

Drive both with **`.claude/skills/run-nanowire-tdgl/driver.py`**. It runs a real
simulation in-process, prints physics diagnostics, renders a PNG you can look
at, and (separately) starts uvicorn and walks the full POM → job → HDF5 round
trip. All paths below are relative to the repo root.

Most changes here land in `packages/tdgl3d/src/tdgl3d/{physics,analysis,core,mesh,operators}/`,
so `driver.py solve` is the entry point you want; the server commands are for
API/POM changes.

## Prerequisites

**No `apt-get` is needed.** A clean Ubuntu container with Python 3.11 and pip
runs this repo as-is — numpy/scipy/matplotlib/h5py all install from wheels, and
matplotlib never needs a display (the driver and the examples force the `Agg`
backend). Verified on Python 3.11.15; `pyproject.toml` requires `>=3.10`.

Always use `python3`, never bare `python` — there is no `python` alias.

## Setup

```bash
python3 -m pip install -e "packages/tdgl3d[dev]" -e "packages/schema[dev]" -e "packages/tdgl3d-server[dev]"
```

Takes ~30 s. Then confirm:

```bash
python3 -c "import tdgl3d, tdgl3d_schema, tdgl3d_server; print('OK', tdgl3d.__version__)"
# -> OK 1.0.0
```

No build step — everything is pure Python installed editable.

## Run (agent path)

```bash
python3 .claude/skills/run-nanowire-tdgl/driver.py solve
```

~3 s. Runs a 15×15 2-D film at κ=5 under Bz=0.5, writes
`.tdgl3d-run/solution.h5` and `.tdgl3d-run/summary.png`, and prints a JSON
diagnostic block:

```
Device(Nx=15, Ny=15, Nz=1, κ=5.0, field=(0.0, 0.0, 0.5))
solve wall time: 2.42s
wrote /home/user/nanowire_tdgl/.tdgl3d-run/solution.h5
wrote /home/user/nanowire_tdgl/.tdgl3d-run/summary.png
{
  "label": "film2d",
  "n_steps": 201,
  "psi_abs_max_final": 0.9080452466509444,
  "psi2_mean_final": 0.3941889144151209,
  "Bz_absmax_final": 0.5000000000000002,
  "vortices_final": 11,
  "vortices_peak": 13,
  "vortices_trace": "t=0.00:0 t=2.32:1 t=2.35:2 ... t=4.72:13 t=4.95:10 t=4.97:11",
  "free_energy_final": 985.5971992125337,
  "steady_state": [false, -1]
}
```

**Read `summary.png`** — it is the closest thing this project has to a
screenshot. `|ψ|²` on the left, `B_z` on the right. A correct run shows dark
vortex cores on an orange background, arranged with the C4 symmetry of the
square. A blank/uniform panel means no vortices nucleated; a panel with
`B_z` in the 1e19 range means the run diverged.

`solve` **exits 1** when the result is not physical, so it works as a smoke
test: NaN/Inf in the state, `max|ψ| < 1e-2` (fully pair-broken), or
`max|B_z| > 1e3` (diverged).

| command | what it does |
|---|---|
| `solve` | in-process simulation → `solution.h5` + `summary.png` + diagnostics; exits 1 on a non-physical result |
| `solve --preset {film2d,trilayer,hole}` | plain 2-D film / S-I-S stack / film with a square hole |
| `solve --gif` | also writes `vortices.gif` (~1.7 MB, adds ~9 s) |
| `inspect PATH.h5 [--png]` | reload a saved artifact, print the `\|ψ\|²` slice as a numeric grid |
| `serve` | start uvicorn on 127.0.0.1:8787 in the background, block until `/health` answers |
| `api [--start]` | full HTTP round trip: POST project → launch job → poll → download HDF5 → summarise |
| `stop` | SIGTERM the background server |

Physics knobs on `solve`/`api`: `--nx --ny --nz --hx --hy --hz --kappa --bx
--by --bz --t-on --method {euler,trapezoidal} --t-stop --dt --save-every
--noise --seed --out`. `python3 .claude/skills/run-nanowire-tdgl/driver.py solve -h`
lists them.

Everything lands in `.tdgl3d-run/` (gitignored) — override with `--out DIR`.

### Driving the server

```bash
python3 .claude/skills/run-nanowire-tdgl/driver.py api --start
```

~2 s end to end. Output:

```
server pid=6113 on http://127.0.0.1:8787 (log: /home/user/nanowire_tdgl/.tdgl3d-run/uvicorn.log)
data dir: /home/user/nanowire_tdgl/.tdgl3d-run/serverdata  token: dev-token
POST /projects -> 201 id=driver-demo
POST /projects/driver-demo/jobs -> 202 job=9c81436406074e41a016916dd004c32f status=pending
job finished: status=completed after 0.6s
GET /jobs/.../result -> 71440 bytes -> /home/user/nanowire_tdgl/.tdgl3d-run/9c8143....h5
artifact: (147, 21) states, 21 frames, t=[0.00, 2.00]
|psi|^2 final: mean=0.9404 min=0.9169
wrote /home/user/nanowire_tdgl/.tdgl3d-run/api-summary.png
API round-trip OK
```

`api` re-runs safely against the same `--project-id`: `POST /projects` upserts.
Run `stop` when done.

To poke endpoints by hand while `serve` is up (auth is a bearer token from
`TDGL3D_API_TOKEN`, defaulting to `dev-token` in the driver):

```bash
curl -sS http://127.0.0.1:8787/health
# {"status":"ok"}
curl -sS http://127.0.0.1:8787/projects -H 'Authorization: Bearer dev-token'
curl -sS -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8787/projects   # -> 401
```

The SSE stream is a plain `curl -N` and closes itself when the job ends:

```bash
curl -sSN http://127.0.0.1:8787/jobs/$JOB_ID/events -H 'Authorization: Bearer dev-token'
# data: {"status": "running", "error": null}
#
# data: {"status": "completed", "error": null}
```

## Direct invocation

For a change to one function, skip the driver and call it:

```bash
python3 -c "
import numpy as np, tdgl3d
dev = tdgl3d.Device(tdgl3d.SimulationParameters(Nx=8, Ny=8, Nz=1, kappa=2.0),
                    applied_field=tdgl3d.AppliedField(Bz=0.5, t_on_fraction=1.0))
sol = tdgl3d.solve(dev, t_stop=2.0, dt=0.02, method='euler', save_every=5,
                   noise_seed=42, progress=False, log_metadata=False)
print(sol.states.shape, float(np.abs(sol.psi(-1)).max()))
"
# -> (147, 21) 0.9845066223553696
```

Pass `log_metadata=False` — otherwise `solve()` writes a `logs/run_*.json` into
the current directory.

Export the POM JSON Schema (takes an output path; without one it drops
`pom.schema.json` into the CWD, which is **not** gitignored):

```bash
python3 -m tdgl3d_schema.export_json_schema .tdgl3d-run/pom.schema.json
```

## Run (human path)

```bash
python3 -m uvicorn tdgl3d_server.app:app --port 8787   # Ctrl-C to stop
```

Blocks in the foreground; artifacts and the SQLite store go to
`$TDGL3D_DATA_DIR` (default `~/.tdgl3d-server`). Set `TDGL3D_API_TOKEN` to
require auth — **unset means auth is entirely disabled**.

Standalone demos live in `packages/tdgl3d/examples/`; they write PNG/GIF into
the *current* directory, so `cd` somewhere disposable first:

```bash
mkdir -p .tdgl3d-run/ex && cd .tdgl3d-run/ex && python3 ../../packages/tdgl3d/examples/vortex_entry_2d.py
# -> example_summary.png, example_vortices.gif, and a logs/ dir; ~4.5 s
```

## Test

```bash
python3 -m pytest packages/tdgl3d/tests packages/schema/tests packages/tdgl3d-server/tests -q
# 318 passed, 3 xfailed in ~1m (solver package alone)
```

**About a minute** — it used to be 5½, most of it the physics-verification
suites, before the solver's right-hand side stopped rebuilding sparse matrices
on every call. While iterating, use the fast subset:

```bash
python3 -m pytest packages/tdgl3d/tests -q -k "not verification and not physics_validation"
# 241 passed, 77 deselected, 3 xfailed in 13s

python3 -m pytest packages/schema/tests packages/tdgl3d-server/tests -q
# 14 passed in 1s
```

The 3 `xfail`s are the known hole-BC / flux-trapping work (`docs/notes/HOLE_BC_STATUS.md`).

Lint (config in the root `ruff.toml`, and it only covers `packages/`):

```bash
ruff check packages/
# All checks passed!
```

The verification suites write `logs/test_*.json` through the `phys_log`
fixture; `docs/generate_test_report.py` turns those into
`docs/physics_test_report.md`. It reads whatever is in `logs/` right then, so
regenerate only after a **full** run — the fast subset deselects the
verification tests and would silently produce a shrunken report:

```bash
python3 docs/generate_test_report.py --output .tdgl3d-run/physics_test_report.md
```

## Gotchas

- **The applied field switches itself off two thirds of the way through.**
  `AppliedField.t_on_fraction` defaults to `2/3`, so at `t = t_stop` the field
  is zero, the vortices have left, and `count_vortices(...)` correctly returns
  **0** — while `summary.png` still shows deep `|ψ|²` cores, because the order
  parameter has not finished relaxing. Two runs that look identical in the PNG
  can differ by 8 vortices. `driver.py` defaults `--t-on 1.0` (field on the
  whole window) and prints `vortices_peak` plus a `vortices_trace` for exactly
  this reason. Never read a vortex count off the last frame alone.

- **`Solution.save()` and `tdgl3d.io.hdf5.load_solution()` write and read
  different layouts.** The server saves artifacts with `Solution.save()`, which
  writes a complex `states` dataset; `load_solution()` looks for
  `states_real`/`states_imag` and dies with
  `KeyError: "... object 'states_real' doesn't exist"`. Always load with
  **`Solution.load(path)`**.

- **`Solution` has no `.order_parameter()` or `.bfield()`-returns-one-array.**
  Despite what `.opencode/skills/tdgl3d-solver/SKILL.md` says, the methods are
  `.psi(step)`, `.psi_squared(step)`, `.psi_squared_2d(step, slice_z)`, and
  `.bfield(step=...)` returns the triple `(Bx, By, Bz)`.

- **`gl_free_energy` takes a flat state vector, not a `Solution`.** The call is
  `gl_free_energy(solution.states[:, -1], device.params, device.idx)`.
  `gl_free_energy(solution, device, step=-1)` raises `TypeError: unexpected
  keyword argument 'step'`.

- **`Solution.count_vortices(device)` requires the device positionally** even
  though the docstring says it is unused.

- **A CFL violation does not always give you NaN.** Forward Euler needs
  `dt < h²/(4κ²)` — with `h=1, κ=5` that is `dt < 0.01`. At `dt=0.02` the run
  completes, `|ψ|` stays bounded and plausible-looking, and only `B_z` blows up
  to ~3e19. `driver.py` warns before solving and fails after, but a hand-rolled
  script will happily hand you garbage. `method="trapezoidal"` is implicit and
  has no such limit.

- **Cross-check a suspicious result against the other integrator.** Euler at
  `dt=0.005` and trapezoidal at `dt=0.05` on the default preset agree to 4
  digits (`free_energy_final` 985.597 vs 985.535, both 11 vortices at
  `t=5`) — a disagreement means one of them is wrong, not that the physics
  is ambiguous.

- **Several entry points litter the current directory.** `solve()` writes
  `logs/` unless `log_metadata=False`; the `examples/` scripts write their PNG
  and GIF into the CWD; `python3 -m tdgl3d_schema.export_json_schema` with no
  argument drops `pom.schema.json` at the CWD root — and `*.png` and `*.json`
  are **not** in `.gitignore` (`docs/figures/*.png` is committed). Run from
  `.tdgl3d-run/` or pass an explicit output path.

- **A thin trilayer is legitimately pair-broken.** The `trilayer` preset with
  `--nz 3` finishes with `max|ψ| = 0.70`, not ~1: SC layers under ~2 ξ next to
  an insulator are suppressed. That is the physics, not a bug — but check
  `psi_abs_max_final` before trusting anything phase-derived, and give the
  oxide layer the *same* κ as the metal (κ=0 degenerates the φ-equation
  entirely). `driver.py`'s preset does this.

## Troubleshooting

- **`KeyError: "Unable to synchronously open object (object 'states_real' doesn't exist)"`**
  — you loaded a `Solution.save()`/server artifact with
  `tdgl3d.io.hdf5.load_solution()`. Use `Solution.load(path)`.

- **`TypeError: gl_free_energy() got an unexpected keyword argument 'step'`** —
  wrong signature; see the gotcha above.

- **`"Bz_absmax_final": 3.4e+19` and a garbage-looking PNG** — CFL violation.
  Lower `--dt` below `h²/(4κ²)` or switch to `--method trapezoidal`.

- **`vortices_final: 0` on a run whose PNG clearly shows cores** — the field
  turned off at `t_on_fraction * t_stop`. Pass `--t-on 1.0`, or read
  `vortices_peak` / `vortices_trace`.

- **`server did not come up; see .tdgl3d-run/uvicorn.log`** — the driver dumps
  the last 2 KB of that log to stderr. Usually port 8787 already taken by an
  earlier run: `python3 .claude/skills/run-nanowire-tdgl/driver.py stop`, or
  `--port` something else.

- **Careful with `pkill -f "uvicorn tdgl3d_server"`** — the pattern matches the
  shell running it and kills your own session. Use the driver's `stop`, or
  `pkill -f "[u]vicorn"`.

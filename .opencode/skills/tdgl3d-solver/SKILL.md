---
name: tdgl3d-solver
description: >
  Work on the TDGL solver core in packages/tdgl3d. Use when editing solver
  physics, operators, integrators, trilayer/material system, state vectors,
  or debugging simulation output. Covers data flow, domain concepts, and
  coding conventions specific to the solver package.
---

# tdgl3d Solver Core

Python package simulating vortex dynamics in 3D Type-II superconductors using
the Time-Dependent Ginzburg-Landau (TDGL) equations. Ported from a MATLAB
codebase written for MIT 6.336.

## Data Flow

```
Device(params, applied_field, trilayer?)
  |  -> GridIndices      (mesh/indices.py)
  |  -> MaterialMap      (core/material.py)   [only if trilayer]
  v
solve(device, ...) -> forward_euler() / trapezoidal()
  |
  v  (each time step)
eval_f(X, params, idx, u, material)     (physics/rhs.py)
  +- expand interior -> full grid
  +- apply boundary conditions (link-variable BCs)
  +- LPSI_{x,y,z} . X  (Laplacians for psi)
  +- FPSI(X, material)  (nonlinear psi term + insulator relaxation)
  +- LPHI_{x,y,z}(material) . X  (curl-curl for phi, per-node kappa)
  +- FPHI_{x,y,z}(X, material)   (supercurrent, per-node kappa)
  +- strip to interior -> dX/dt
  v
Solution(times, states, params, idx)
```

## Domain Concepts

- **State vector** is `[psi, phi_x, phi_y, phi_z]`, each block has
  `n_interior` complex entries. For 2D (`Nz=1`) the `phi_z` block is omitted.
- **Full grid** has `(Nx+1)x(Ny+1)x(Nz+1)` nodes. Linear index:
  `m = i + j*(Nx+1) + k*(Nx+1)*(Ny+1)`.
- **Interior nodes** are `1 <= i <= Nx-1`, `1 <= j <= Ny-1`,
  `1 <= k <= max(Nz-1,1)`. `idx.interior_to_full` maps compact interior
  numbering to full-grid linear index.
- **Operators** are sparse CSR matrices on the full grid. `eval_f` extracts
  only interior rows for the time derivative.
- **Boundary conditions:** Zero-current on all faces. Applied B enters as
  Peierls phases written onto boundary link variables in
  `_apply_boundary_conditions()`.
- **CFL (Forward Euler):** `dt < h^2 / (4*kappa^2)`. With h=1, kappa=2:
  `dt < 0.0625`.

## Trilayer / Multi-Material System

- `Trilayer(bottom=Layer, insulator=Layer, top=Layer)` defines an S/I/S stack
  along z. `trilayer.Nz` is the total z-cells.
- `build_material_map()` creates per-node `kappa[]` and `sc_mask[]` arrays
  based on which z-plane each node lives in.
- `MaterialMap` flows: `Device` -> `solve()` -> `integrators` -> `eval_f()` ->
  individual operators. When `material is None`, everything falls back to
  uniform `params.kappa`.
- **Insulator suppression:** `construct_FPSI` adds `-psi/tau_relax`
  (tau=0.1) at insulator nodes, driving psi -> 0 smoothly.
- `Device.initial_state()` zeroes psi in the insulator via
  `interior_sc_mask`.

## Repository Layout

```
packages/tdgl3d/
src/tdgl3d/              <- importable package
  __init__.py            <- public API: SimulationParameters, Device,
                            StateVector, AppliedField, Layer, Trilayer,
                            MaterialMap, solve
  core/
    parameters.py        <- SimulationParameters: Nx,Ny,Nz, hx,hy,hz, kappa, periodic
    device.py            <- Device: bundles params + field + trilayer; builds idx & material
    state.py             <- StateVector: [psi, phi_x, phi_y, phi_z] with named views
    solution.py          <- Solution: times + states matrix, .psi(), .psi_squared_2d(), .bfield()
    material.py          <- Layer, Trilayer, MaterialMap, build_material_map()
  mesh/
    indices.py           <- GridIndices: 26 index arrays, interior_to_full mapping
  operators/
    sparse_operators.py  <- LPSI, LPHI (Laplacians), FPSI, FPHI (forcing) — CSR matrices
  physics/
    rhs.py               <- eval_f(X, params, idx, u, material): full RHS dX/dt
    applied_field.py     <- AppliedField (constant/ramp/callable) + boundary vectors
    bfield.py            <- eval_bfield(): B = curl(A)
  solvers/
    runner.py            <- solve(): high-level entry point
    integrators.py       <- forward_euler(), trapezoidal()
    newton.py            <- newton_gcr(), newton_gcr_trap()
    tgcr.py              <- tgcr_matrix_free(), tgcr_matrix_free_trap()
  visualization/
    plotting.py          <- plot_order_parameter, plot_bfield, plot_summary, animate
  io/
    hdf5.py              <- save_solution(), load_solution()
tests/
  test_parameters.py     (11 tests)
  test_indices.py        (11 tests)
  test_operators.py      (12 tests)
  test_state.py           (7 tests)
  test_physics.py        (11 tests)
  test_solvers.py         (7 tests)
  test_integration.py     (7 tests)
  test_visualization.py  (17 tests)
  test_trilayer.py       (18 tests)
  validate_analytical.py
examples/
  isometric_film_3d.py   <- dual-panel |psi|^2 + phase isometric 3D scatter
  vortex_3d.py           <- 3D vortex nucleation demo
  vortex_entry_2d.py     <- 2D thin-film entry
  check_symmetry.py      <- C4 symmetry verification
  verify_indices_bc.py   <- MATLAB index comparison
  generate_default_plot.py
```

## Coding Conventions

- All source files use `from __future__ import annotations`.
- Dataclasses for all data containers (parameters, state, material, solution).
- Type hints everywhere; `Optional[X]` from `typing` (not `X | None`) for 3.10.
- Operators return `scipy.sparse.csr_matrix`.
- Tests use `pytest`; fixtures in each test file.
- `np.testing.assert_allclose` for floating-point comparisons.
- Import style: absolute imports from `tdgl3d.*` in tests and examples.

## Commands

```bash
# Run all tests from repo root
python3 -m pytest packages/tdgl3d/tests -x -q

# Run specific test group
python3 -m pytest packages/tdgl3d/tests/test_trilayer.py -v

# Quick import check
python3 -c "from tdgl3d import Device, solve, Trilayer, Layer; print('OK')"

# Run an example
python3 packages/tdgl3d/examples/isometric_film_3d.py
```

## Known Limitations

- Periodic BCs are defined in `SimulationParameters` but **not yet wired** into
  the operator construction — only zero-current BCs are implemented.
- No adaptive mesh refinement; the grid is uniform.
- The trilayer currently supports identical SC materials in top and bottom;
  different kappa values for top vs. bottom are supported by the `MaterialMap`
  but haven't been tested extensively.
- Visualization is z-slice based; full 3D volume rendering is not implemented.

## What Was Validated

- All 26 index arrays in `GridIndices` match the MATLAB output for square grids
  (4x4x2, 6x6x3, 10x10x4). Non-square grids revealed bugs in the *MATLAB*
  code (Nx/Ny swap), not the Python code.
- Perfect C4 symmetry (< 1e-15) confirmed with uniform initial conditions.
- Applied Bz verified uniform across all boundary nodes with no double-counting.
- 101 tests passing as of the trilayer implementation.

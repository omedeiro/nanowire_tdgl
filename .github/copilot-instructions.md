# Copilot Agent Instructions — tdgl3d

## What this project is

`tdgl3d` is a **Python** package that simulates vortex dynamics in 3D Type-II
superconductors using the **Time-Dependent Ginzburg-Landau (TDGL)** equations.
It was ported from a MATLAB codebase written for MIT 6.336 (the original `.m`
files live in the parent directory `../`).

The physics: a superconducting order parameter ψ (complex scalar) and
gauge-invariant link variables φ_x, φ_y, φ_z evolve on a uniform 3D
Cartesian grid under applied magnetic field.

## Tech stack

- **Python ≥ 3.10** (tested on 3.11)
- **numpy**, **scipy** (sparse matrices, linear algebra)
- **matplotlib** (visualization)
- **h5py** (I/O), **tqdm** (progress bars)
- **pytest** (test suite — 101 tests)
- Installed as editable package: `pip install -e ".[dev]"`
- Uses `python3` on this machine (no bare `python` alias)

## Repository layout

```
tdgl3d/
├── src/tdgl3d/           ← importable package
│   ├── __init__.py       ← public API: SimulationParameters, Device, StateVector,
│   │                        AppliedField, Layer, Trilayer, MaterialMap, solve
│   ├── core/
│   │   ├── parameters.py ← SimulationParameters: Nx,Ny,Nz, hx,hy,hz, kappa, periodic
│   │   ├── device.py     ← Device: bundles params + field + trilayer; builds idx & material
│   │   ├── state.py      ← StateVector: [ψ, φ_x, φ_y, φ_z] with named .psi/.phi_x/… views
│   │   ├── solution.py   ← Solution: times + states matrix, .order_parameter(), .bfield()
│   │   └── material.py   ← Layer, Trilayer, MaterialMap, build_material_map()
│   ├── mesh/
│   │   └── indices.py    ← GridIndices: 26 index arrays, interior_to_full mapping
│   ├── operators/
│   │   └── sparse_operators.py ← LPSI, LPHI (Laplacians), FPSI, FPHI (forcing) — CSR matrices
│   ├── physics/
│   │   ├── rhs.py        ← eval_f(X, params, idx, u, material): full RHS dX/dt
│   │   ├── applied_field.py ← AppliedField (constant/ramp/callable) + boundary vectors
│   │   └── bfield.py     ← eval_bfield(): B = curl(A)
│   ├── solvers/
│   │   ├── runner.py     ← solve(): high-level entry point
│   │   ├── integrators.py ← forward_euler(), trapezoidal()
│   │   ├── newton.py     ← newton_gcr(), newton_gcr_trap()
│   │   └── tgcr.py       ← tgcr_matrix_free(), tgcr_matrix_free_trap()
│   ├── visualization/
│   │   └── plotting.py   ← plot_order_parameter, plot_bfield, plot_summary, animate
│   └── io/
│       └── hdf5.py       ← save_solution(), load_solution()
├── tests/
│   ├── test_parameters.py    (11 tests)
│   ├── test_indices.py       (11 tests)
│   ├── test_operators.py     (12 tests)
│   ├── test_state.py          (7 tests)
│   ├── test_physics.py       (11 tests)
│   ├── test_solvers.py        (7 tests)
│   ├── test_integration.py    (7 tests)
│   ├── test_visualization.py (17 tests)
│   ├── test_trilayer.py      (18 tests)
│   └── validate_analytical.py
├── examples/
│   ├── isometric_film_3d.py     ← dual-panel |ψ|² + phase isometric 3D scatter
│   ├── vortex_3d.py             ← 3D vortex nucleation demo
│   ├── vortex_entry_2d.py       ← 2D thin-film entry
│   ├── check_symmetry.py        ← C4 symmetry verification
│   ├── verify_indices_bc.py     ← MATLAB index comparison
│   └── generate_default_plot.py
└── pyproject.toml
```

## Data flow & architecture

```
Device(params, applied_field, trilayer?)
  │  → GridIndices      (mesh/indices.py)
  │  → MaterialMap      (core/material.py)   [only if trilayer]
  ▼
solve(device, ...) → forward_euler() / trapezoidal()
  │
  ▼  (each time step)
eval_f(X, params, idx, u, material)     (physics/rhs.py)
  ├─ expand interior → full grid
  ├─ apply boundary conditions (link-variable BCs)
  ├─ LPSI_{x,y,z} · X  (Laplacians for ψ)
  ├─ FPSI(X, material)  (nonlinear ψ term + insulator relaxation)
  ├─ LPHI_{x,y,z}(material) · X  (curl-curl for φ, per-node κ)
  ├─ FPHI_{x,y,z}(X, material)   (supercurrent, per-node κ)
  └─ strip to interior → dX/dt
  ▼
Solution(times, states, params, idx)
```

## Key domain concepts

- **State vector** is `[ψ, φ_x, φ_y, φ_z]`, each block has `n_interior`
  complex entries.  For 2D (`Nz=1`) the `φ_z` block is omitted.
- **Full grid** has `(Nx+1)×(Ny+1)×(Nz+1)` nodes.  Linear index:
  `m = i + j*(Nx+1) + k*(Nx+1)*(Ny+1)`.
- **Interior nodes** are `1 ≤ i ≤ Nx-1`, `1 ≤ j ≤ Ny-1`, `1 ≤ k ≤ max(Nz-1,1)`.
  `idx.interior_to_full` maps compact interior numbering → full-grid linear index.
- **Operators** are sparse CSR matrices on the full grid.  `eval_f` extracts
  only interior rows for the time derivative.
- **Boundary conditions:** Zero-current on all faces.  Applied B enters as
  Peierls phases written onto boundary link variables in `_apply_boundary_conditions()`.
- **CFL (Forward Euler):** dt < h²/(4κ²).  With h=1, κ=2: dt < 0.0625.

## Trilayer / multi-material system

- `Trilayer(bottom=Layer, insulator=Layer, top=Layer)` defines an S/I/S stack
  along z.  `trilayer.Nz` is the total z-cells.
- `build_material_map()` creates per-node `kappa[]` and `sc_mask[]` arrays
  based on which z-plane each node lives in.
- `MaterialMap` flows: `Device` → `solve()` → `integrators` → `eval_f()` →
  individual operators.  When `material is None`, everything falls back to
  uniform `params.kappa`.
- **Insulator suppression:** `construct_FPSI` adds `−ψ/τ_relax` (τ=0.1) at
  insulator nodes, driving ψ → 0 smoothly.
- `Device.initial_state()` zeroes ψ in the insulator via `interior_sc_mask`.

## Coding conventions

- All source files use `from __future__ import annotations`.
- Dataclasses for all data containers (parameters, state, material, solution).
- Type hints everywhere; `Optional[X]` from `typing` (not `X | None`) for 3.10.
- Operators return `scipy.sparse.csr_matrix`.
- Tests use `pytest`; fixtures in each test file.
- `np.testing.assert_allclose` for floating-point comparisons.
- Import style: absolute imports from `tdgl3d.*` in tests and examples.

## Common tasks

### Run all tests
```bash
python3 -m pytest tests/ -x -q
```

### Run specific test group
```bash
python3 -m pytest tests/test_trilayer.py -v
```

### Quick import check
```bash
python3 -c "from tdgl3d import Device, solve, Trilayer, Layer; print('OK')"
```

### Run an example
```bash
python3 examples/isometric_film_3d.py
```

## What was validated

- All 26 index arrays in `GridIndices` match the MATLAB output for square grids
  (4×4×2, 6×6×3, 10×10×4).  Non-square grids revealed bugs in the *MATLAB*
  code (Nx/Ny swap), not the Python code.
- Perfect C4 symmetry (< 1e-15) confirmed with uniform initial conditions.
- Applied Bz verified uniform across all boundary nodes with no double-counting.
- 101 tests passing as of the trilayer implementation.

## Known limitations / future work

- Periodic BCs are defined in `SimulationParameters` but **not yet wired** into
  the operator construction — only zero-current BCs are implemented.
- No adaptive mesh refinement; the grid is uniform.
- The trilayer currently supports identical SC materials in top and bottom;
  different κ values for top vs. bottom are supported by the `MaterialMap`
  but haven't been tested extensively.
- Visualization is z-slice based; full 3D volume rendering is not implemented.

# tdgl3d — 3D Time-Dependent Ginzburg-Landau Simulator

A Python package for simulating vortex and phase dynamics in three-dimensional
Type-II superconductors using the time-dependent Ginzburg-Landau (TDGL) model
on a structured finite-difference grid.

## Overview

`tdgl3d` solves the coupled TDGL equations for the superconducting order
parameter ψ and the gauge-invariant vector potential **A** (link variables
φ_x, φ_y, φ_z) in 3D:

```
∂ψ/∂t = (∇ − iA)²ψ + (1 − |ψ|²)ψ

∂A/∂t = κ² ∇×(∇×A) − Im[ψ* (∇ − iA) ψ]
```

The spatial discretisation uses **link variables** (Peierls phases) on a
uniform Cartesian grid, exactly as described in the MATLAB predecessor.

## Features

| Feature | Description |
|---------|-------------|
| **3D structured grid** | Uniform Cartesian mesh with configurable Nx×Ny×Nz |
| **S/I/S trilayer** | Multi-material support via per-node κ and superconductor mask |
| **Boundary conditions** | Zero-current BCs; applied B-field via link-variable BCs |
| **Applied field ramp** | Linear ramp from 0 to full magnitude over a configurable fraction |
| **Time integrators** | Forward Euler (explicit) and Trapezoidal (implicit, Newton-GCR) |
| **Matrix-free Newton-GCR** | Jacobian-free Newton-Krylov solver for the implicit step |
| **Sparse operators** | All discrete Laplacian and forcing operators built with `scipy.sparse` |
| **Post-processing** | B-field evaluation, order-parameter magnitude, vorticity |
| **Visualization** | 2D slice plots, 3D isometric scatter plots, animated GIFs |
| **HDF5 I/O** | Save/load solutions via h5py |
| **Validation suite** | 101 tests — analytical checks, convergence, conservation, trilayer |

## Installation

```bash
cd tdgl3d
pip install -e ".[dev]"
pytest          # 101 tests
```

**Requirements:** Python ≥ 3.10, numpy ≥ 1.24, scipy ≥ 1.10, matplotlib ≥ 3.7,
h5py ≥ 3.8, tqdm ≥ 4.65.  Dev extras add pytest, pytest-cov, ruff.

## Quick start

### Single-layer thin film

```python
import tdgl3d

params = tdgl3d.SimulationParameters(
    Nx=20, Ny=20, Nz=4,
    hx=1.0, hy=1.0, hz=1.0,
    kappa=5.0,
)
field = tdgl3d.AppliedField(Bz=1.0, ramp=True, ramp_fraction=0.3)
device = tdgl3d.Device(params, applied_field=field)

solution = tdgl3d.solve(device, t_stop=10.0, dt=0.05, method="trapezoidal")

solution.plot_order_parameter(slice_z=2)
```

### S/I/S trilayer

```python
import tdgl3d

trilayer = tdgl3d.Trilayer(
    bottom=tdgl3d.Layer(thickness_z=3, kappa=2.0),
    insulator=tdgl3d.Layer(thickness_z=1, kappa=0.0, is_superconductor=False),
    top=tdgl3d.Layer(thickness_z=3, kappa=2.0),
)
params = tdgl3d.SimulationParameters(Nx=20, Ny=20, kappa=2.0)
field = tdgl3d.AppliedField(Bz=0.5, ramp=True)
device = tdgl3d.Device(params, applied_field=field, trilayer=trilayer)

x0 = device.initial_state()          # ψ=0 in insulator, |ψ|=1 in SC
solution = tdgl3d.solve(device, t_stop=5.0, dt=0.02, method="euler", x0=x0)
```

## Public API

| Symbol | Module | Description |
|--------|--------|-------------|
| `SimulationParameters` | `core.parameters` | Grid size (Nx, Ny, Nz), spacing (hx, hy, hz), κ, periodic BCs |
| `Device` | `core.device` | Bundles params + field + optional trilayer; builds indices & material map |
| `StateVector` | `core.state` | Wraps flat `[ψ, φ_x, φ_y, φ_z]` vector with named views (`.psi`, `.phi_x`, …) |
| `AppliedField` | `physics.applied_field` | Constant or ramped `(Bx, By, Bz)`; optional `field_func(t, t_stop)` callable |
| `Layer` | `core.material` | Single material layer: `thickness_z`, `kappa`, `is_superconductor` |
| `Trilayer` | `core.material` | S/I/S stack of three `Layer`s; computes `Nz`, `z_ranges()` |
| `MaterialMap` | `core.material` | Per-node arrays: `kappa`, `sc_mask`, `interior_sc_mask` |
| `Solution` | `core.solution` | Stores `times` + `states` matrix; methods for B-field, order param extraction |
| `solve()` | `solvers.runner` | Main entry — runs Forward Euler or Trapezoidal integration |

## Architecture & data flow

```
User script
  │
  ▼
Device(params, applied_field, trilayer?)
  │  ├─ constructs GridIndices   (mesh/indices.py)
  │  └─ constructs MaterialMap   (core/material.py)  ← only if trilayer
  │
  ▼
solve(device, ...)                       (solvers/runner.py)
  │  ├─ builds eval_u(t, X) closure      (physics/applied_field.py)
  │  ├─ extracts device.material (or None)
  │  └─ calls forward_euler() or trapezoidal()
  │                                       (solvers/integrators.py)
  ▼
Time-step loop
  │  eval_f(X, params, idx, u, material)  (physics/rhs.py)
  │    ├─ expand interior → full grid
  │    ├─ apply boundary conditions (link-variable BCs from applied B)
  │    ├─ construct LPSI_{x,y,z} · X_full           (operators/sparse_operators.py)
  │    ├─ construct FPSI(X_full, material)           nonlinear + insulator relaxation
  │    ├─ construct LPHI_{x,y,z}(material) · X_full  per-node κ in curl-curl
  │    ├─ construct FPHI_{x,y,z}(X_full, material)  supercurrent + per-node κ
  │    └─ strip to interior rows → dX/dt
  │
  │  (Trapezoidal only)
  │  newton_gcr_trap(f_closure, ...)      (solvers/newton.py)
  │    └─ tgcr_matrix_free_trap(...)      (solvers/tgcr.py)
  │
  ▼
Solution(times, states, params, idx)     (core/solution.py)
  ├─ .order_parameter(step)  → 3D |ψ|²
  ├─ .bfield(step)           → (Bx, By, Bz)
  ├─ .plot_order_parameter() / .plot_bfield()
  └─ save_solution() / load_solution()   (io/hdf5.py)
```

### Key design decisions

- **State vector layout:** `[ψ, φ_x, φ_y, φ_z]` each of length `n_interior`.
  For 2D (`Nz=1`) the `φ_z` block is omitted.
- **Interior / full-grid duality:** PDE is evaluated on the full
  `(Nx+1)×(Ny+1)×(Nz+1)` grid (operators are full-grid sparse matrices).
  Only interior rows are extracted for the time derivative.
  `idx.interior_to_full` maps interior numbering → full linear index.
- **Link-variable BCs:** `_apply_boundary_conditions()` in `rhs.py` writes
  the applied-field Peierls phases onto boundary link variables before
  each operator evaluation.
- **Material threading:** `MaterialMap` flows from `Device` → `solve()` →
  `integrators` → `eval_f()` → individual operators.  When `material is None`
  all operators fall back to the uniform `params.kappa`.
- **Insulator suppression:** In `construct_FPSI`, insulator nodes get an
  extra `−ψ/τ_relax` (τ_relax = 0.1) driving ψ → 0 without hard discontinuity.
- **CFL condition (Forward Euler):** dt < h² / (4κ²).  With h=1, κ=2: dt < 0.0625.

## Project layout

```
tdgl3d/
├── src/tdgl3d/
│   ├── __init__.py          # Public exports
│   ├── core/
│   │   ├── parameters.py    # SimulationParameters dataclass
│   │   ├── device.py        # Device: params + field + trilayer → indices + material
│   │   ├── state.py         # StateVector: named views into [ψ, φ_x, φ_y, φ_z]
│   │   ├── solution.py      # Solution: times + states + post-processing
│   │   └── material.py      # Layer, Trilayer, MaterialMap, build_material_map()
│   ├── mesh/
│   │   └── indices.py       # GridIndices: 26 face/mask arrays, interior_to_full
│   ├── operators/
│   │   └── sparse_operators.py  # LPSI, LPHI, FPSI, FPHI — scipy.sparse CSR
│   ├── physics/
│   │   ├── rhs.py           # eval_f(): full RHS evaluation dX/dt
│   │   ├── applied_field.py # AppliedField + build_boundary_field_vectors()
│   │   └── bfield.py        # eval_bfield(): B = curl(A) at interior nodes
│   ├── solvers/
│   │   ├── runner.py        # solve(): high-level entry point
│   │   ├── integrators.py   # forward_euler(), trapezoidal()
│   │   ├── newton.py        # newton_gcr(), newton_gcr_trap()
│   │   └── tgcr.py          # tgcr_matrix_free(), tgcr_matrix_free_trap()
│   ├── visualization/
│   │   └── plotting.py      # plot_order_parameter, plot_bfield, animate
│   └── io/
│       └── hdf5.py          # save_solution(), load_solution()
├── tests/
│   ├── test_parameters.py   # 11 tests — SimulationParameters validation
│   ├── test_indices.py      # 11 tests — GridIndices construction, symmetry
│   ├── test_operators.py    # 12 tests — operator shapes, symmetry, sparsity
│   ├── test_state.py        #  7 tests — StateVector views, factory methods
│   ├── test_physics.py      # 11 tests — eval_f, BCs, applied field
│   ├── test_solvers.py      #  7 tests — Euler/Trap convergence, Newton
│   ├── test_integration.py  #  7 tests — end-to-end solve() smoke tests
│   ├── test_visualization.py# 17 tests — plotting functions
│   ├── test_trilayer.py     # 18 tests — Layer/Trilayer/MaterialMap/Device/sim
│   └── validate_analytical.py  # Analytical Jacobian comparison
├── examples/
│   ├── isometric_film_3d.py    # Dual-panel |ψ|² + phase isometric scatter
│   ├── vortex_3d.py            # 3D vortex nucleation
│   ├── vortex_entry_2d.py      # 2D thin-film vortex entry
│   ├── check_symmetry.py       # C4 symmetry verification
│   ├── verify_indices_bc.py    # Index & BC validation against MATLAB
│   └── generate_default_plot.py
└── pyproject.toml
```

## Test suite

```bash
pytest                  # all 101 tests
pytest -k trilayer      # just trilayer tests
pytest --cov=tdgl3d     # with coverage
```

## MATLAB provenance

This package is a Python rewrite of the 3D TDGL MATLAB code developed for MIT
6.336 (Spring 2021).  The original `.m` files live in the parent directory.
Index-for-index verification against the MATLAB code is documented in
`examples/verify_indices_bc.py`.
"""

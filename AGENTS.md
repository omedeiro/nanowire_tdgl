# Agent Instructions — tdgl3d

## Project type

Python package: 3D Time-Dependent Ginzburg-Landau superconductor simulator.
Ported from MATLAB (MIT 6.336). Physics-heavy numerical code.

## Essential commands

```bash
# Install (from repo root)
pip install -e ".[dev]"

# Test
python3 -m pytest tests/ -x -q        # all 101 tests, stop on first fail
python3 -m pytest -k trilayer         # subset
python3 -m pytest --cov=tdgl3d        # with coverage

# Quick smoke test
python3 -c "from tdgl3d import Device, solve, Trilayer; print('OK')"

# Run example
python3 examples/vortex_entry_2d.py
```

**Critical:** Always use `python3`, never bare `python` (machine has no alias).

## Architecture

### State vector layout
`[ψ, φ_x, φ_y, φ_z]` each of length `n_interior` (complex). For 2D (`Nz=1`), `φ_z` is omitted.

### Interior/full-grid duality
- Operators are sparse CSR matrices on the **full grid**: `(Nx+1) × (Ny+1) × (Nz+1)`
- Time derivative is computed only at **interior** nodes: `1 ≤ i ≤ Nx-1`, etc.
- `idx.interior_to_full` maps compact interior numbering → full-grid linear index
- `eval_f()` in `physics/rhs.py` expands interior → full, applies BCs, computes operators, then strips back to interior rows

### Boundary conditions
- Zero-current on all faces (only mode implemented)
- Applied B enters as Peierls phases written onto boundary link variables in `_apply_boundary_conditions()`
- Periodic BCs are **defined** in `SimulationParameters` but **not yet wired** into operator construction

### Material threading (S/I/S trilayer)
`MaterialMap` flows: `Device` → `solve()` → `integrators` → `eval_f()` → individual operators.
When `material is None`, all operators fall back to uniform `params.kappa`.

### Insulator suppression
`construct_FPSI` adds `−ψ/τ_relax` (τ=0.1) at insulator nodes to drive ψ → 0 smoothly.
`Device.initial_state()` zeroes ψ in the insulator via `interior_sc_mask`.

## CFL constraint (Forward Euler)

`dt < h² / (4κ²)`. With `h=1`, `κ=2`: `dt < 0.0625`.

## Code conventions

- All source files: `from __future__ import annotations`
- Dataclasses for all containers (`SimulationParameters`, `StateVector`, `MaterialMap`, `Solution`)
- Type hints everywhere; use `Optional[X]` from `typing` (not `X | None`) for Python 3.10 compat
- Operators return `scipy.sparse.csr_matrix`
- Tests: `pytest`, `np.testing.assert_allclose` for floats
- Imports: absolute (`from tdgl3d.*`) in tests and examples

## Public API (exported from `tdgl3d.__init__`)

| Symbol | Module | Purpose |
|--------|--------|---------|
| `SimulationParameters` | `core.parameters` | Grid size, spacing, κ, periodic (not yet used) |
| `Device` | `core.device` | Bundles params + field + trilayer; builds `GridIndices` + `MaterialMap` |
| `StateVector` | `core.state` | Wraps flat array with named views (`.psi`, `.phi_x`, …) |
| `AppliedField` | `physics.applied_field` | Constant/ramped `(Bx, By, Bz)` or callable |
| `Layer`, `Trilayer` | `core.material` | S/I/S stack definition |
| `MaterialMap` | `core.material` | Per-node `kappa[]`, `sc_mask[]` |
| `solve()` | `solvers.runner` | Main entry: runs Euler or Trapezoidal integration |

`Solution` is returned by `solve()` but not in `__all__`.

## Validation status

- All 26 index arrays in `GridIndices` match MATLAB output (square grids 4×4×2, 6×6×3, 10×10×4)
- Non-square grids revealed bugs in the **MATLAB** code (Nx/Ny swap), not Python
- C4 symmetry verified to < 1e-15
- Applied Bz uniform across boundary nodes, no double-counting
- 101 tests passing as of trilayer implementation

## Known gaps

- Periodic BCs defined but **not implemented** in operators
- No adaptive mesh refinement (uniform grid only)
- Trilayer supports different κ per layer but only extensively tested with identical SC materials
- Visualization is z-slice based; no 3D volume rendering

## Related files

Existing instructions: `.github/copilot-instructions.md` (more detailed architecture notes, data flow diagrams).

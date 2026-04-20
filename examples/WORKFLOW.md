# Development Workflow Guide

## For Physics Validation & Research

When working with long-running simulations (20+ minutes), use the **save/load workflow** to avoid re-running simulations every time you want to change visualization parameters.

---

## Recommended Workflow

### 1. Run simulation ONCE and save to HDF5

```bash
# Run the full SIS trilayer example (~30 minutes)
python3 examples/sis_square_with_hole.py
```

**Output:**
- HDF5 file: `sis_square_Bz0.50_t120.h5` (~50-100 MB)
- Initial plots: PNG images for quick validation
- Console: vortex count, steady-state detection, physics diagnostics

### 2. Visualize INSTANTLY anytime

```bash
# Basic visualization (default plots: psi, current)
python3 examples/visualize_solution.py sis_square_Bz0.50_t120.h5

# Custom plots at specific timestep
python3 examples/visualize_solution.py sis_square_Bz0.50_t120.h5 \
    --plots psi phase current bfield \
    --step 100 \
    --dpi 300

# Different z-slice (top SC layer)
python3 examples/visualize_solution.py sis_square_Bz0.50_t120.h5 \
    --slice-z 20 \
    --output-dir figures/top_layer/
```

**Runtime:** < 1 second per plot (no simulation re-run needed!)

### 3. Load and analyze programmatically

```python
from tdgl3d import Solution

# Load saved solution (instant)
solution = Solution.load("sis_square_Bz0.50_t120.h5")

# Access all data
print(f"Loaded {solution.n_steps} timesteps")
print(f"Time range: {solution.times[0]:.2f} → {solution.times[-1]:.2f}")

# Extract order parameter at final time
psi_final = solution.psi(step=-1)

# Run custom analysis
from tdgl3d.analysis.vortex_counting import count_vortices_plaquette
n_vortices, positions, windings = count_vortices_plaquette(
    solution, solution.device, slice_z=0, step=-1
)
print(f"Final vortex count: {n_vortices}")

# Generate new plots with different parameters
from tdgl3d.visualization.plotting import plot_order_parameter
fig, ax = plot_order_parameter(solution, step=50, slice_z=5)
fig.savefig("custom_plot.png", dpi=300)
```

---

## File Management

### Typical HDF5 file sizes

| Grid size | Timesteps | File size |
|-----------|-----------|-----------|
| 50×50×10  | 100       | ~20 MB    |
| 67×67×24  | 240       | ~100 MB   |
| 100×100×30| 500       | ~500 MB   |

**Tip:** HDF5 files compress well. For archival, use `gzip`:
```bash
gzip sis_square_Bz0.50_t120.h5  # → .h5.gz (~10-20% original size)
gunzip sis_square_Bz0.50_t120.h5.gz  # Decompress when needed
```

### Organization for parameter sweeps

```bash
results/
├── field_sweep/
│   ├── sis_square_Bz0.10_t120.h5
│   ├── sis_square_Bz0.30_t120.h5
│   ├── sis_square_Bz0.50_t120.h5
│   └── sis_square_Bz1.00_t120.h5
├── kappa_sweep/
│   ├── sis_kappa1.0_Bz0.5.h5
│   ├── sis_kappa2.0_Bz0.5.h5
│   └── sis_kappa5.0_Bz0.5.h5
└── figures/
    ├── field_sweep_vortex_counts.png
    └── kappa_comparison.png
```

---

## Quick Validation (Smoke Tests)

For rapid iteration during code development, use the **fast smoke tests**:

```bash
# Run all 178 tests (~14 seconds)
python3 -m pytest tests/ -q

# Run only fast smoke tests (< 1 second)
python3 -m pytest tests/test_hole_bc_smoke.py -v

# Run specific physics validation
python3 -m pytest tests/test_physics_nonz_fields.py -v -k "transverse"
```

---

## Common Workflows

### A. Parameter sweep (magnetic field)

```python
"""Sweep Bz from 0.1 to 1.0 and save all results."""
import tdgl3d
import numpy as np

fields = np.linspace(0.1, 1.0, 10)
for bz in fields:
    print(f"\n=== Running Bz = {bz:.2f} ===")
    solution, device, hole = run_simulation_with_field(bz)
    
    # Auto-saves as: sis_square_Bz{bz:.2f}_t120.h5
    print(f"Saved: sis_square_Bz{bz:.2f}_t120.h5")
```

Then analyze all results together:
```python
"""Plot vortex count vs field strength."""
fields = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
vortex_counts = []

for bz in fields:
    sol = Solution.load(f"sis_square_Bz{bz:.2f}_t120.h5")
    n_vort, _, _ = count_vortices_plaquette(sol, sol.device, slice_z=0, step=-1)
    vortex_counts.append(n_vort)

plt.plot(fields, vortex_counts, 'o-')
plt.xlabel("Applied $B_z$ (dimensionless)")
plt.ylabel("Vortex count at t=120")
plt.savefig("vortex_vs_field.png")
```

### B. Time-evolution movies

```python
"""Generate high-quality animation from saved HDF5."""
from tdgl3d import Solution
from examples.sis_square_with_hole import animate_isometric

# Load saved simulation (instant)
solution = Solution.load("sis_square_Bz0.50_t120.h5")

# Generate animations with different frame rates
animate_isometric(solution, device, prefix="high_fps", fps=15, step_stride=1)
animate_isometric(solution, device, prefix="low_fps", fps=6, step_stride=5)
```

### C. Compare different geometries

```python
"""Compare hole shapes: square vs circular."""
sol_square = Solution.load("sis_square_hole.h5")
sol_circle = Solution.load("sis_circular_hole.h5")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
plot_order_parameter(sol_square, step=-1, slice_z=0, ax=ax1, title="Square hole")
plot_order_parameter(sol_circle, step=-1, slice_z=0, ax=ax2, title="Circular hole")
fig.savefig("hole_shape_comparison.png", dpi=300)
```

---

## What NOT to do

### ❌ Don't re-run simulations for small visualization changes

**Bad:**
```bash
# Modify colormap in script
python3 examples/sis_square_with_hole.py  # Wait 30 min...
# Oops, wrong z-slice
python3 examples/sis_square_with_hole.py  # Wait 30 min again...
```

**Good:**
```bash
# Run once
python3 examples/sis_square_with_hole.py  # Wait 30 min once

# Then iterate on visualization instantly
python3 examples/visualize_solution.py sis_square_Bz0.50_t120.h5 --slice-z 5
python3 examples/visualize_solution.py sis_square_Bz0.50_t120.h5 --slice-z 10
python3 examples/visualize_solution.py sis_square_Bz0.50_t120.h5 --step 100
# Each takes < 1 second!
```

### ❌ Don't commit large HDF5 files to git

Add to `.gitignore`:
```
# HDF5 simulation results (too large for git)
*.h5
*.h5.gz

# Keep only example outputs as PNG
!docs/examples/*.png
```

Instead, share via:
- Lab server / shared storage
- Data repository (Zenodo, Dryad)
- Cloud storage (Google Drive, Dropbox)

---

## Summary

1. **Run simulation once** → Auto-saves to HDF5
2. **Visualize instantly** → Load HDF5 and plot
3. **Iterate on analysis** → No re-running needed
4. **Use smoke tests** → Fast validation during development
5. **Organize results** → Structured directories for parameter sweeps

**Key benefit:** Separate simulation (slow) from visualization (fast). Run physics once, analyze many ways.

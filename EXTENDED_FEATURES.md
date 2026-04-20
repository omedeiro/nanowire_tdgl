# Extended Simulation Features — Implementation Summary

## Overview

Successfully implemented extended simulation capabilities for `tdgl3d` including:

1. **Steady-state detection** via convergence monitoring
2. **Vortex counting** with two cross-validated methods
3. **Time-series analysis** of vortex populations
4. **Extended simulation runtime** (T_STOP = 120.0)

## New Modules Created

### 1. `src/tdgl3d/analysis/convergence.py`

Monitors simulation convergence to detect steady state.

**Key functions:**

- `compute_convergence_metrics(solution, device, step, window_size)` - Calculates relative changes in |ψ|² and |J_s| over rolling window
- `check_steady_state(solution, device, ...)` - Scans through saved steps to find when steady state achieved

**Convergence criteria:**
- Relative change in mean |ψ|² < 10⁻⁴ (configurable)
- Relative change in mean |J_s| < 10⁻⁴ (configurable)
- Window size: 10 saved steps (configurable)

**Usage:**
```python
is_steady, steady_step, metrics = solution.check_steady_state(
    device=device,
    window_size=10,
    psi_threshold=1e-4,
    current_threshold=1e-4,
)
```

### 2. `src/tdgl3d/analysis/vortex_counting.py`

Detects and counts vortices using multiple methods.

**Key functions:**

#### Method A: Phase Winding (Plaquette)
```python
n_vort, positions, winding_numbers = count_vortices_plaquette(
    solution, device, slice_z=0, step=-1
)
```
- Fast, reliable for 2D slices
- Computes phase winding Δφ = Σ(phase differences) around elementary 2×2 squares
- Vortex detected when |Δφ/(2π)| > threshold (default 0.8)
- Returns vortex positions (grid coordinates) and winding numbers (±1 typically)

#### Method B: Fluxoid (Polygon)
```python
n_vort = count_vortices_polygon(
    solution, device, polygon_points, slice_z=0, step=-1
)
```
- More robust for complex geometries
- Computes fluxoid Φ_f = ∮ A·dl + ∮ λ²J_s·dl around polygon
- Returns total vortex count (can be fractional for partial enclosure)
- Inspired by pyTDGL's fluxoid quantization

#### Hole Flux Counting
```python
n_flux_quanta = count_hole_flux_quanta(
    solution, device, hole_bounds, slice_z=0, step=-1
)
```
- Integrates B_z over hole area: Φ = ∬ B_z dA
- Returns flux in units of Φ₀ (flux quantum)
- In dimensionless units: Φ₀ = 2π

#### Core Detection (Visual/Diagnostic)
```python
core_positions = find_vortex_cores(
    solution, device, slice_z=0, step=-1, threshold=0.1
)
```
- Finds local minima of |ψ|² below threshold
- Useful for visualization but doesn't give winding numbers
- Less reliable than phase winding

### 3. Solution Class Extensions

Added convenience methods to `Solution` class:

```python
# Vortex counting
n, pos, wind = solution.count_vortices(
    device, slice_z=0, step=-1, method="plaquette"
)

# Steady-state check
is_steady, step, metrics = solution.check_steady_state(device)
```

## Updated Example Script

Enhanced `examples/sis_square_with_hole.py` with:

1. **Extended runtime:** T_STOP = 120.0 (4× original)
2. **Steady-state monitoring:** Automatic detection with diagnostics
3. **Vortex time-series analysis:** Tracks vortex evolution over all saved steps
4. **Cross-validation:** Uses both plaquette and hole flux methods
5. **Time-series plots:** Creates plot showing:
   - Top panel: Total vortex count vs time (plaquette method)
   - Bottom panel: Hole flux and film vortices vs time

**New functions:**
- `analyze_vortices_timeseries()` - Count vortices at each step
- `plot_vortex_timeseries()` - Generate 2-panel time-series plot

**Output files:**
- `sis_square_with_hole_vortex_timeseries.png` - Vortex evolution plot
- (plus all existing visualization outputs)

## Testing

Created comprehensive test suite in `tests/test_vortex_analysis.py`:

### Test Coverage

1. ✅ **Convergence metrics on uniform state** - Validates zero change detection
2. ✅ **Steady-state detection on constant state** - Should detect immediately
3. ✅ **Steady-state rejection on evolving state** - Correctly identifies non-steady
4. ✅ **Vortex counting on uniform state** - Should find 0 vortices
5. ✅ **Vortex counting on single-vortex state** - Detects synthetic vortex
6. ✅ **Hole flux with uniform field** - Validates flux integration
7. ✅ **Core finding** - Locates |ψ|² minima

**All 7 tests pass ✓**

```bash
python3 -m pytest tests/test_vortex_analysis.py -v
# 7 passed in 1.92s
```

## Physics Background

### Vortex Quantization

In Type-II superconductors, magnetic flux penetrates as quantized vortices:
- Each vortex carries flux Φ₀ = h/2e ≈ 2.07×10⁻¹⁵ Wb
- Phase winds by 2π around vortex core: ∮ ∇θ·dl = 2πn (n = winding number)
- Core radius ~ ξ (coherence length), screening radius ~ λ (penetration depth)

### Fluxoid Quantization

Total fluxoid through closed loop:
```
Φ_f = ∮ A·dl + ∮ λ² J_s·dl = n·Φ₀
```

For a loop enclosing n vortices, the fluxoid is quantized in units of Φ₀.

### Dimensionless Units

In our simulation (length in ξ, field in H_c2):
- **Flux quantum:** Φ₀ = 2π (dimensionless)
- **Coherence length:** ξ = 1 (unit length)
- **Penetration depth:** λ = κ (dimensionless κ)

## Usage Example

```python
import tdgl3d

# Setup device with trilayer
params = tdgl3d.SimulationParameters(Nx=50, Ny=50, hx=0.75, hy=0.75, hz=0.75, kappa=2.0)
trilayer = tdgl3d.Trilayer(
    bottom=tdgl3d.Layer(thickness_z=8, kappa=2.0),
    insulator=tdgl3d.Layer(thickness_z=8, kappa=2.0, is_superconductor=False),
    top=tdgl3d.Layer(thickness_z=8, kappa=2.0),
)
field = tdgl3d.AppliedField(Bz=0.5)
device = tdgl3d.Device(params, applied_field=field, trilayer=trilayer)

# Run extended simulation
x0 = device.initial_state()
solution = tdgl3d.solve(device, x0=x0, dt=0.01, t_stop=120.0, 
                       method='euler', save_every=50, progress=True)

# Check steady state
is_steady, step, metrics = solution.check_steady_state(device)
print(f"Steady state: {is_steady} at t={metrics.get('steady_time', 'N/A')}")

# Count vortices
n_vort, positions, winding = solution.count_vortices(
    device, slice_z=0, step=-1, method="plaquette"
)
print(f"Vortices: {n_vort}")
for i, (pos, w) in enumerate(zip(positions, winding)):
    print(f"  {i+1}. ({pos[0]:.1f}, {pos[1]:.1f}) winding={w:+.2f}")

# Analyze time series
from tdgl3d.analysis import count_vortices_plaquette, count_hole_flux_quanta

times = []
vortex_counts = []
for step in range(solution.n_steps):
    t = solution.times[step]
    n, _, _ = count_vortices_plaquette(solution, device, slice_z=0, step=step)
    times.append(t)
    vortex_counts.append(n)

# Plot
import matplotlib.pyplot as plt
plt.plot(times, vortex_counts, 'o-')
plt.xlabel('Time')
plt.ylabel('Vortex count')
plt.savefig('vortex_evolution.png')
```

## Performance Notes

**Computational cost:**

- **Steady-state check:** O(n_steps × n_interior) - Linear scan through saved steps
- **Vortex counting (plaquette):** O(Nx × Ny) per slice - Fast, scales with grid size
- **Hole flux integration:** O(hole_area) - Very fast
- **Time-series analysis:** O(n_steps × Nx × Ny) - Parallelizable

**Memory:**
- Minimal overhead (convergence metrics are computed on-the-fly)
- No additional storage beyond existing solution snapshots

## Validation Strategy

1. **Unit tests:** Synthetic vortex states with known winding numbers
2. **Cross-validation:** Compare plaquette vs. polygon methods
3. **Physics checks:**
   - Flux conservation: Φ_boundary = Φ_hole + Φ_film
   - Winding number = integer (±1, ±2, ...)
   - Zero vortices in Meissner state

## Next Steps

### To run full extended simulation:

```bash
cd /Users/owenmedeiros/nanowire_tdgl
python3 examples/sis_square_with_hole.py
```

**Expected runtime:** ~10-15 minutes (12,000 timesteps at ~20-40 it/s)

**Expected outputs:**
1. Terminal diagnostics including:
   - Steady-state detection results
   - Final vortex count and positions
   - Time-series vortex analysis
2. New plot: `sis_square_with_hole_vortex_timeseries.png`
3. All existing visualization outputs updated

### Possible Extensions

1. **3D vortex lines:** Track vortex cores through z-direction (currently 2D slices)
2. **Vortex dynamics:** Measure vortex velocity and trajectories
3. **Adaptive time-stepping:** Slow down when vortices enter/exit
4. **Energy functional:** Monitor free energy for thermodynamic convergence
5. **Multi-vortex interactions:** Measure intervortex forces and Abrikosov lattice formation

## References

- **pyTDGL documentation:** https://py-tdgl.readthedocs.io/en/latest/api/solution.html#fluxoid-quantization
- **Tinkham, "Introduction to Superconductivity"** (2nd ed.) - Chapter 4: Type-II Superconductivity
- **Original MATLAB code:** MIT 6.336 course project (basis for this implementation)

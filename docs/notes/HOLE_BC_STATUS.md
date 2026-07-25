## Hole Boundary Conditions - Current Status & Issue

### What We've Implemented ✅
1. **Geometry identification**: `identify_hole_nodes()` and `identify_boundary_links()` correctly find hole regions and boundary links
2. **Material carving**: `MaterialMap.carve_hole_polygon()` marks hole interior as non-SC (sc_mask = 0)
3. **RHS BC enforcement**: `_apply_boundary_conditions()` zeros φ values at hole boundaries during RHS evaluation

### The Problem ❌
**The link variables φ at hole boundaries are NOT staying at zero.**

Test results show:
- Max |φ_x| at hole boundaries: ~0.02 (should be ~0)
- Max |φ_y| at hole boundaries: ~0.02 (should be ~0)
- Max |ψ| in hole: ~0.11 (should be ~0) ← This is actually OK due to relaxation

### Root Cause Analysis

The TDGL equations evolve the state as:
```
dψ/dt = Lψ + Fψ
dφ/dt = Lφ + Fφ
```

**Current implementation:**
1. During RHS evaluation (`eval_f`), we call `_apply_boundary_conditions()` which zeros φ at hole boundaries
2. We then compute dψ/dt and dφ/dt using those zeroed values
3. The integrator updates: `φ_new = φ_old + dt * dφ/dt`

**The issue:**
Even though we zero φ during the RHS calculation, the **dφ/dt term is non-zero** at the hole boundaries! This means φ drifts away from zero over time.

### Solution Options

**Option A: Mask dφ/dt at hole boundaries** (Recommended)
After computing dφ/dt in `eval_f()`, explicitly zero it at hole boundary indices:
```python
dPhidtX[hole_x_interior_indices] = 0.0
dPhidtY[hole_y_interior_indices] = 0.0
```
This ensures φ stays constant (at zero) at hole boundaries.

**Option B: Post-integration projection**
After each time step, explicitly zero φ at hole boundaries in the state vector.
This is less elegant but guaranteed to work.

**Option C: Modify FPHI operators**
Make the FPHI forcing terms aware of holes and return zero for hole boundary links.
This is more invasive but cleaner mathematically.

### Insulator vs. Hole - Key Difference

**Insulator** (e.g., middle layer in S/I/S):
- `sc_mask = 0` → drives ψ → 0 via relaxation term
- Link variables φ can evolve freely (allows B-field penetration)
- NO boundary conditions on φ

**Hole** (geometric void):
- `sc_mask = 0` → drives ψ → 0 via relaxation term  
- **Zero-current BC**: φ = 0 at hole edges (no current flows into void)
- This requires special handling in the time integration

### Why Vortices Don't Form in Holes

Vortices are topological defects in the **superconducting order parameter ψ**:
- They require a phase winding: ∮ ∇θ · dl = 2πn
- Phase is only defined where |ψ| > 0 (i.e., in the superconductor)
- In holes: |ψ| ≈ 0 everywhere → no phase → no vortices!

**What you SHOULD see:**
✓ Magnetic flux penetrating through the hole (B ≠ 0 in hole)
✓ Persistent currents circulating around the hole (in the SC, not in the hole)
✓ Meissner screening in the bulk SC
✓ Possible vortices NEAR the hole edges (in the SC)

**What you should NOT expect:**
✗ Vortices inside the hole
✗ Phase winding in the hole
✗ Supercurrent in the hole

### Next Steps

I recommend implementing **Option A** - masking dφ/dt at hole boundaries. This requires:
1. Converting hole boundary link indices from full-grid to interior numbering
2. Zeroing the appropriate components of dφ/dt before returning from `eval_f()`
3. Adding tests to verify φ remains at zero throughout the simulation

Would you like me to proceed with this implementation?
